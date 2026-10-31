"""Signal model of one process in one category, interpolated in MH.

1. The simulation of every mass point is split into the right vertex (|dZ| <= 1 cm, RV) and wrong
   vertex (WV) scenarios. A scenario with too few events at MH = 125 GeV takes its shape from a
   replacement process x category (``replacementMap.py``).
2. The datasets are reweighted to the beamspot width in data.
3. RV and WV are fitted with the number of Gaussians from the signal fTest
   (:class:`finalfits.signal_model.SignalFit`); the parameters become splines in MH.
4. :class:`SignalModel` builds the final pdf ``hggpdfsmrel_<proc>_<year>_<cat>_13TeV`` (RV and WV
   mixed by the RV fraction), with the photon systematics as nuisances on mean, width and rate, and
   its normalisation XS * BR * eff x acc * rate, and writes it to the workspace ``wsig_13TeV``.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import pickle
import re
from dataclasses import dataclass, field

import numpy as np
import ROOT

from finalfits import analysis
from finalfits.signal_ftest import dataset_name
from finalfits.signal_model import SignalFit
from finalfits.trees2ws import SQRTS, WS_DIR, WS_NAME

logger = logging.getLogger(__name__)

MH_LOW, MH_HIGH, MH_NOMINAL = "120", "130", "125"
SYSTEMATIC_TYPES = ("scales", "scalesCorr", "scalesGlobal", "smears")
SCENARIO_CUTS = {"RV": "abs(dZ)<=1.", "WV": "abs(dZ)>1."}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Input datasets


def load_dataset(input_dir: str, proc: str, mass: str, cat: str, observables):
    """Dataset of (proc, cat) at one mass point from the Trees2WS workspaces, reduced to the observables."""
    f = ROOT.TFile(glob.glob(f"{input_dir}/output*M{mass}*{proc}.root")[0], "read")
    ws = f.Get(f"{WS_DIR}/{WS_NAME}")
    dataset = ws.data(dataset_name(proc, mass, cat)).reduce(observables)
    ws.Delete()
    f.Close()
    return dataset


def _replacement_rv(replacement_map: dict, cat: str, proc: str) -> tuple[str, str]:
    """RV replacement (proc, cat) for a category, chosen by the fiducial label of the process."""
    label = "in" if proc.endswith("_in") else "out" if proc.endswith("_out") else None
    if label is None:
        raise ValueError(f"Process name '{proc}' does not have a valid fiducial label ('in' or 'out')")
    return replacement_map["procRVMap"][cat][label], replacement_map["catRVMap"][cat][label]


@dataclass
class ShapeSource:
    """Which process x category the shape of a vertex scenario is extracted from."""

    proc: str
    cat: str
    initial_proc: str = ""
    initial_cat: str = ""
    nominal_num_entries: float = 0.0
    nominal_sum_entries: float = 0.0
    replaced: bool = False

    def report(self) -> list[str]:
        lines = [
            f"initial_shape_source_proc={self.initial_proc}",
            f"initial_shape_source_cat={self.initial_cat}",
            f"nominal_MH125_numEntries={self.nominal_num_entries:g}",
            f"nominal_MH125_sumEntries={self.nominal_sum_entries:.6f}",
            f"replaced={'yes' if self.replaced else 'no'}",
            f"final_shape_source_proc={self.proc}",
            f"final_shape_source_cat={self.cat}",
        ]
        if self.replaced:
            lines += [f"replacement_proc={self.proc}", f"replacement_cat={self.cat}"]
        return lines


def scenario_datasets(input_dir, proc, cat, mass_points, observables, scenario, replacement, threshold, nominal=None):
    """Datasets of a vertex scenario per mass point, from the replacement if too few events at 125 GeV.

    ``replacement`` is the (proc, cat) to use instead; ``nominal`` collects the full datasets.
    Returns (datasets, ShapeSource).
    """

    def load(p, c, full=None):
        datasets = {}
        for mass in mass_points:
            dataset = load_dataset(input_dir, p, mass, c, observables)
            if full is not None:
                full[mass] = dataset.Clone()
            datasets[mass] = dataset.reduce(observables, SCENARIO_CUTS[scenario]) if scenario else dataset
        return datasets

    def too_few(datasets):
        d = datasets[MH_NOMINAL]
        return d.numEntries() < threshold or d.sumEntries() <= 0.0

    source = ShapeSource(proc, cat, proc, cat)
    # the full (nominal) datasets always come from the process itself, not from a replacement
    datasets = load(proc, cat, nominal)
    source.nominal_num_entries = datasets[MH_NOMINAL].numEntries()
    source.nominal_sum_entries = datasets[MH_NOMINAL].sumEntries()
    if too_few(datasets):
        datasets = load(*replacement)
        if too_few(datasets):
            raise RuntimeError(
                f"Replacement {replacement} has too few entries ({datasets[MH_NOMINAL].numEntries()} < {threshold})"
            )
        source.proc, source.cat = replacement
        source.replaced = True
        logger.info(
            "%s: too few entries (%g < %g), shape from %s",
            scenario or "Total",
            source.nominal_num_entries,
            threshold,
            replacement,
        )
    for mass, d in datasets.items():
        logger.info(
            "%s (%s, %s) MH = %s: numEntries = %g, sumEntries = %.6f",
            scenario or "Total",
            source.proc,
            source.cat,
            mass,
            d.numEntries(),
            d.sumEntries(),
        )
    return datasets, source


def _refilled(template, xvar_name: str, x: np.ndarray, dz: np.ndarray, weights: np.ndarray):
    """Copy of ``template`` (name, variables, storage) holding the given events."""
    variables = ROOT.RooArgSet(template.get())
    variables.add(template.weightVar())
    weight_name = template.weightVar().GetName()
    events = ROOT.RooDataSet.from_numpy(
        {xvar_name: x, "dZ": dz, weight_name: weights}, variables, weight_name=weight_name
    )
    filled = template.emptyClone()
    filled.append(events)
    return filled


def beamspot_reweigh(
    dataset, width_data: float, width_mc: float, xvar_name: str = "CMS_hgg_mass", preserve_norm: bool = True
):
    """Reweight the dZ distribution from the beamspot width in simulation to the one in data.

    Events with |dZ| >= 0.1 cm are weighted by the ratio of Gaussians of width sqrt(2) * beamspot
    width; the total sum of weights is kept if ``preserve_norm``. Weights are clipped to
    [-1e5, 1e6] as by the RooRealVar they used to be filled through.
    """
    values = dataset.to_numpy()
    x, dz = values[xvar_name], values["dZ"]
    weights = values[dataset.weightVar().GetName()]
    sigma_mc, sigma_data = math.sqrt(2) * width_mc, math.sqrt(2) * width_data
    factors = [
        1.0 if abs(z) < 0.1 else ROOT.TMath.Gaus(z, 0, sigma_data, True) / ROOT.TMath.Gaus(z, 0, sigma_mc, True)
        for z in dz
    ]
    reweighted = _refilled(dataset, xvar_name, x, dz, np.clip([f * w for f, w in zip(factors, weights)], -1e5, 1e6))
    if not preserve_norm:
        return reweighted
    sum_reweighted = reweighted.sumEntries()
    scale = dataset.sumEntries() / sum_reweighted if sum_reweighted != 0.0 else 1.0
    weights = reweighted.to_numpy()[dataset.weightVar().GetName()]
    return _refilled(dataset, xvar_name, x, dz, np.clip([scale * w for w in weights], -1e5, 1e6))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Final model


def sm_cross_sections_and_br() -> dict[str, np.ndarray]:
    """SM cross sections of the production modes and the H->gg BR (combine), MH = 120-130 GeV in 0.1 GeV."""
    import HiggsAnalysis.CombinedLimit.PhysicsModel as models
    from HiggsAnalysis.CombinedLimit.Datacard import Datacard
    from HiggsAnalysis.CombinedLimit.ModelTools import ModelBuilder
    from HiggsAnalysis.CombinedLimit.SMHiggsBuilder import SMHiggsBuilder

    class Options:
        physModel = "HiggsAnalysis.CombinedLimit.PhysicsModel:floatingHiggsMass"
        physOpt = ["higgsMassRange=90,250"]
        bin = True
        fileName = "dummy.root"
        cexpr = False
        out = "wsdefault"
        verbose = 0
        mass = 125
        funcXSext = "dummy"

    options = Options()
    builder = ModelBuilder(Datacard(), options)
    physics = models.floatingHiggsMass
    physics.setPhysicsOptions(options.physOpt)
    builder.setPhysics(physics)
    builder.physics.doParametersOfInterest()
    sm = SMHiggsBuilder(builder)
    mh_var = sm.modelBuilder.out.var("MH")
    sm.makeBR(analysis.decayMode)
    for mode in analysis.productionModes:
        sm.makeXS(mode, SQRTS)

    values = {mode: [] for mode in [*analysis.productionModes, analysis.decayMode, "constant"]}
    mh = 120.0
    while mh < 130.05:
        mh_var.setVal(mh)
        for mode in analysis.productionModes:
            values[mode].append(sm.modelBuilder.out.function(f"SM_XS_{mode}_{SQRTS}").getVal())
        values[analysis.decayMode].append(sm.modelBuilder.out.function(f"SM_BR_{analysis.decayMode}").getVal())
        values["constant"].append(1.0)
        mh += 0.1
    values = {k: np.asarray(v) for k, v in values.items()}
    if "ggZH" in values and "ZH" in values:
        values["qqZH"] = values["ZH"] - values["ggZH"]
    return values


def nuisance_name(name: str) -> str:
    """Name of the nuisance parameter of a signal shape systematic in the workspace."""
    if name == "deltafracright" or name.startswith("Scale") or name.startswith("Smearing"):
        return f"CMS_HIG26007_{name}"
    return f"{analysis.outputWSNuisanceTitle__}_{name}"


@dataclass
class Nuisance:
    param: object
    mean: object
    sigma: object
    rate: object
    opts: list = field(default_factory=list)


class SignalModel:
    """Final signal model of (proc, cat) from the RV and WV fits (or a single "Total" fit)."""

    def __init__(
        self,
        fits,
        proc,
        cat,
        ext,
        year,
        datasets,
        xvar,
        mh,
        mass_points,
        xsbr_map,
        photon_syst_file,
        proc_syst,
        systematics: dict[str, str],
        skip_systematics=False,
    ):
        self.fits, self.proc, self.cat, self.ext, self.year = fits, proc, cat, ext, year
        self.proc_syst = proc_syst
        self.name = f"{proc}_{year}_{cat}_{SQRTS}"
        self.datasets, self.xvar, self.mh, self.mass_points = datasets, xvar, mh, mass_points
        self.observables = ROOT.RooArgSet(xvar)
        self.int_lumi = ROOT.RooRealVar("IntLumi", "IntLumi", 1.0, 0.0, 999999999.0)  # pb^-1
        self.xsbr_map = xsbr_map
        self.systematics = systematics
        self.skip_systematics = skip_systematics
        self.splines, self.functions, self.pdfs, self.out_datasets = {}, {}, {}, {}
        self.nuisances: dict[str, dict[str, Nuisance]] = {}
        self._keep_alive = []

        self.xsbr = sm_cross_sections_and_br()
        self._build_xs_br_splines()
        self._build_eff_acc_spline()
        if not self.skip_systematics:
            self._build_nuisances(photon_syst_file)
        title = analysis.outputWSObjectTitle__
        if "Total" in fits:
            self._build_pdf(fits["Total"], "total")
            self.pdfs["final"] = self.pdfs["total"]
        else:
            self._build_rv_fraction()
            self._build_pdf(fits["RV"], "rv")
            self._build_pdf(fits["WV"], "wv")
            name = f"{title}_{self.name}"
            self.pdfs["final"] = ROOT.RooAddPdf(
                name, name, ROOT.RooArgList(self.pdfs["rv"], self.pdfs["wv"]), ROOT.RooArgList(self.functions["fracRV"])
            )
        self._build_norm()
        for mass, dataset in datasets.items():
            self.out_datasets[mass] = dataset.Clone(f"sig_mass_m{mass}_{self.name}")
            self.out_datasets[f"{mass}_copy"] = dataset.Clone(f"sig_mass_m{mass}_{cat}")

    def _spline(self, key, name, x, y):
        self.splines[key] = ROOT.RooSpline1D(name, name, self.mh, len(x), x, y)
        return self.splines[key]

    def _two_point_padding(self, mh, values):
        # a single mass point gives a constant spline between MH_LOW and MH_HIGH
        if len(values) == 1:
            return np.asarray([float(MH_LOW), mh[0], float(MH_HIGH)]), np.asarray([values[0]] * 3)
        return np.asarray(mh), np.asarray(values)

    def _build_xs_br_splines(self):
        mh = np.linspace(120.0, 130.0, 101)
        xs = self.xsbr_map[self.proc].get("factor", 1.0) * self.xsbr[self.xsbr_map[self.proc]["mode"]]
        self._spline("xs", f"fxs_{self.proc}_{SQRTS}", mh, xs)
        br = self.xsbr_map["decay"].get("factor", 1.0) * self.xsbr[self.xsbr_map["decay"]["mode"]]
        self._spline("br", f"fbr_{SQRTS}", mh, br)

    def _build_eff_acc_spline(self):
        # In HiggsDNA the sum of weights is eff x acc
        mh, ea = self._two_point_padding(
            [float(m) for m in self.mass_points], [self.datasets[m].sumEntries() for m in self.mass_points]
        )
        self._spline("ea", f"fea_{self.name}", mh, ea)

    def _add_nuisance(self, kind, name, mean, sigma, rate, opts=()):
        param = ROOT.RooRealVar(nuisance_name(name), nuisance_name(name), 0.0, -5.0, 5.0)
        param.setConstant(True)
        consts = [
            ROOT.RooConstVar(f"const_{self.name}_{x}_{name}", f"const_{self.name}_{x}_{name}", v)
            for x, v in (("mean", mean), ("sigma", sigma), ("rate", rate))
        ]
        self.nuisances[kind][name] = Nuisance(param, *consts, list(opts))

    def _build_nuisances(self, photon_syst_file):
        for kind in SYSTEMATIC_TYPES:
            if self.systematics.get(kind, ""):
                self.nuisances[kind] = {}
        if not os.path.exists(photon_syst_file):
            raise FileNotFoundError(f"Photon systematics {photon_syst_file} do not exist: run CalcPhotonSyst first")
        with open(photon_syst_file, "rb") as f:
            data = pickle.load(f)
        row = data[data["proc"] == self.proc_syst]
        if len(row) == 0:
            logger.warning("Process %s is not in %s: skipping the systematics", self.proc, photon_syst_file)
            self.skip_systematics = True
            return
        for kind in SYSTEMATIC_TYPES:
            for syst in self.systematics.get(kind, "").split(","):
                if syst == "":
                    continue
                options = syst.split(":")
                name = options[0]
                nuisance = name
                # Zmmg scales: correlated between 2022 and 2023, separate for 2024
                if kind == "scalesCorr" and name in ("ScaleEBZmmg", "ScaleEEZmmg"):
                    year = str(self.year).strip()
                    if year.startswith(("2022", "2023", "2223")):
                        nuisance = f"{name}_2022_2023"
                    elif year.startswith("2024"):
                        nuisance = f"{name}_2024"
                # scales and smears are decorrelated between years
                suffix = "" if kind in ("scalesCorr", "scalesGlobal") else f"_{self.year}"
                if kind == "scalesGlobal":
                    mean = sigma = rate = 0.0
                else:
                    mean, sigma, rate = (row[f"{name}_{x}"].values[0] for x in ("mean", "sigma", "rate"))
                self._add_nuisance(kind, f"{nuisance}{suffix}", mean, sigma, rate, options[1:])

    def _build_rv_fraction(self):
        fractions = []
        for mass in self.mass_points:
            n_rv, n_wv = (
                max(0.0, self.datasets[mass].reduce(self.observables, SCENARIO_CUTS[s]).sumEntries())
                for s in ("RV", "WV")
            )
            fraction = 1.0 if n_rv + n_wv == 0.0 else n_rv / (n_rv + n_wv)
            fractions.append(1.0 if fraction != fraction else fraction)
        mh, fractions = self._two_point_padding([float(m) for m in self.mass_points], fractions)
        name = f"{analysis.outputWSObjectTitle__}_{self.name}"
        spline = self._spline("fracRV", f"{name}_rvFracSpline", mh, np.clip(fractions, 0.0, 1.0))
        if self.skip_systematics:
            self.functions["fracRV"] = ROOT.RooFormulaVar(
                f"{name}_rvFrac", f"{name}_rvFrac", "TMath::Max(0.0, TMath::Min(@0, 1.0))", ROOT.RooArgList(spline)
            )
        else:
            self.nuisances["other"] = {}
            self._add_nuisance("other", "deltafracright", 1.0, 1.0, 1.0)
            param = self.nuisances["other"]["deltafracright"].param
            self.functions["fracRV"] = ROOT.RooFormulaVar(
                f"{name}_rvFrac",
                f"{name}_rvFrac",
                "TMath::Max(0.0, TMath::Min(@0 + @1, 1.0))",
                ROOT.RooArgList(spline, param),
            )

    def _shape_nuisances(self):
        for kind in ("scales", "scalesCorr", "smears"):
            yield from self.nuisances.get(kind, {}).values()

    def _formula(self, name, formula, dependents):
        self.functions[name] = ROOT.RooFormulaVar(name, name, formula, dependents)
        self._keep_alive.append(dependents)
        return self.functions[name]

    def _mean(self, dm_spline: str):
        """MH + dm, scaled by (1 + sum of global scales + mean variation x nuisance)."""
        dependents = ROOT.RooArgList(self.mh, self.splines[dm_spline])
        formula = "(@0+@1)"
        if not self.skip_systematics:
            formula += "*(1."
            for nuisance in self.nuisances.get("scalesGlobal", {}).values():
                formula += f"+@{dependents.getSize()}"
                for option in nuisance.opts:
                    if f"factor_{self.cat}" in option:
                        formula += "*%3.1f" % float(option.split("=")[-1])
                dependents.add(nuisance.param)
            for nuisance in self._shape_nuisances():
                if abs(nuisance.mean.getVal()) >= 5.0e-5:
                    formula += f"+@{dependents.getSize()}*@{dependents.getSize() + 1}"
                    dependents.add(nuisance.mean)
                    dependents.add(nuisance.param)
            formula += ")"
        return self._formula(re.sub("dm", "mean", dm_spline), formula, dependents)

    def _sigma(self, sigma_spline: str):
        """sigma x max(0.01, 1 + sigma variation x nuisance)."""
        dependents = ROOT.RooArgList(self.splines[sigma_spline])
        formula = "@0"
        if not self.skip_systematics:
            formula += "*TMath::Max(1.e-2,(1."
            for nuisance in self._shape_nuisances():
                if nuisance.sigma.getVal() >= 1e-4:
                    formula += f"+@{dependents.getSize()}*@{dependents.getSize() + 1}"
                    dependents.add(nuisance.sigma)
                    dependents.add(nuisance.param)
            formula += "))"
        return self._formula(sigma_spline, formula, dependents)

    def _rate(self, name: str):
        dependents = ROOT.RooArgList()
        formula = "(1."
        if not self.skip_systematics:
            for nuisance in self._shape_nuisances():
                if nuisance.rate.getVal() >= 5.0e-4:
                    formula += f"+@{dependents.getSize()}*@{dependents.getSize() + 1}"
                    dependents.add(nuisance.rate)
                    dependents.add(nuisance.param)
        formula += ")"
        return self._formula(name, formula, dependents)

    def _fit_spline(self, fit: SignalFit, key: str, suffix: str):
        spline = fit.splines[key].Clone()
        spline.SetName(f"{re.sub('sigma', 'sigma_fit', key)}_{suffix}")
        self.splines[f"{key}_{suffix}"] = spline
        return spline

    def _build_pdf(self, fit: SignalFit, ext: str):
        suffix = self.name if ext == "total" else f"{self.name}_{ext}"
        pdfs, fractions = ROOT.RooArgList(), ROOT.RooArgList()
        if "dcb" in fit.pdfs:
            for p in ("dm", "sigma", "n1", "n2", "a1", "a2"):
                self._fit_spline(fit, f"{p}_dcb", suffix)
            mean = self._mean(f"dm_dcb_{suffix}")
            sigma = self._sigma(f"sigma_dcb_{suffix}")
            s = {p: self.splines[f"{p}_dcb_{suffix}"] for p in ("a1", "n1", "a2", "n2")}
            self.pdfs[f"dcb_{suffix}"] = ROOT.RooDoubleCBFast(
                f"dcb_{suffix}", f"dcb_{suffix}", self.xvar, mean, sigma, s["a1"], s["n1"], s["a2"], s["n2"]
            )
            spline = fit.splines["sigma_gaus"].Clone()
            spline.SetName(f"sigma_fit_gaus_{suffix}")
            self.splines[f"sigma_gaus_{suffix}"] = spline
            sigma_gaus = self._sigma(f"sigma_gaus_{suffix}")
            self.pdfs[f"gaus_{suffix}"] = ROOT.RooGaussian(
                f"gaus_{suffix}", f"gaus_{suffix}", self.xvar, mean, sigma_gaus
            )
            fraction = fit.splines["frac_constrained"].Clone()
            fraction.SetName(f"frac_{suffix}")
            self.splines[f"frac_{suffix}"] = fraction
            pdfs.add(self.pdfs[f"dcb_{suffix}"])
            pdfs.add(self.pdfs[f"gaus_{suffix}"])
            fractions.add(fraction)
        else:
            for g in range(fit.n_gaussians):
                self._fit_spline(fit, f"dm_g{g}", suffix)
                self._fit_spline(fit, f"sigma_g{g}", suffix)
                mean = self._mean(f"dm_g{g}_{suffix}")
                sigma = self._sigma(f"sigma_g{g}_{suffix}")
                name = f"gaus_g{g}_{suffix}"
                self.pdfs[name] = ROOT.RooGaussian(name, name, self.xvar, mean, sigma)
                pdfs.add(self.pdfs[name])
                if g < fit.n_gaussians - 1:
                    # the fraction splines of the fit are renamed, not copied
                    fraction = fit.splines[f"frac_g{g}_constrained"]
                    fraction.SetName(f"frac_g{g}_{suffix}")
                    self.splines[f"frac_g{g}_{suffix}"] = fraction
                    fractions.add(fraction)
        name = f"{analysis.outputWSObjectTitle__}_{suffix}"
        self.pdfs[ext] = ROOT.RooAddPdf(name, name, pdfs, fractions, True)
        self._keep_alive += [pdfs, fractions]

    def _build_norm(self):
        rate = self._rate(f"rate_{self.name}")
        final = self.pdfs["final"].GetName()
        factors = [self.splines["xs"], self.splines["br"], self.splines["ea"], rate]
        self.functions["final_norm"] = ROOT.RooFormulaVar(
            f"{final}_norm", f"{final}_norm", "@0*@1*@2*@3", ROOT.RooArgList(*factors)
        )
        self.functions["final_normThisLumi"] = ROOT.RooFormulaVar(
            f"{final}_normThisLumi", f"{final}_normThisLumi", "@0*@1*@2*@3*@4", ROOT.RooArgList(*factors, self.int_lumi)
        )
        self.pdfs["final_extend"] = ROOT.RooExtendPdf(
            f"extend{final}", f"extend{final}", self.pdfs["final"], self.functions["final_norm"]
        )
        self.pdfs["final_extendThisLumi"] = ROOT.RooExtendPdf(
            f"extend{final}ThisLumi", f"extend{final}ThisLumi", self.pdfs["final"], self.functions["final_normThisLumi"]
        )

    def save(self, ws) -> None:
        recycle = ROOT.RooFit.RecycleConflictNodes()
        ws.Import(self.pdfs["final"], recycle)
        ws.Import(self.functions["final_norm"], recycle)
        ws.Import(self.functions["final_normThisLumi"], recycle)
        ws.Import(self.pdfs["final_extend"], recycle)
        ws.Import(self.pdfs["final_extendThisLumi"], recycle)
        for dataset in self.out_datasets.values():
            ws.Import(dataset)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@dataclass
class SignalFitOptions:
    mass_points: tuple[str, ...] = ("120", "125", "130")
    analysis_xsbr: str = "STXS"
    analysis_rm: str = "STXS"
    systematics: dict = field(default_factory=dict)  # scales, scalesCorr, scalesGlobal, smears
    replacement_threshold: int = 0
    beamspot_width_data: float = 3.5
    beamspot_width_mc: float = 3.7
    skip_beamspot_reweigh: bool = False
    skip_vertex_scenario_split: bool = False
    skip_systematics: bool = False
    skip_zeroes: bool = False
    use_dcb: bool = False  # double crystal ball + Gaussian instead of the fTest Gaussians
    mh_poly_order: int = 1
    n_bins: int = 80
    minimizer_method: str = "TNC"


def run_signal_fit(
    input_dir, ingredients_dir, output_file, proc, cat, year, ext, opts: SignalFitOptions, report_file=None
):
    """Fit the signal model of (proc, cat) and write it to ``output_file``.

    ``ingredients_dir`` holds outdir_<ext>/fTest (number of Gaussians) and outdir_<ext>/calcPhotonSyst.
    Returns (SignalModel, fits) or None if skip_zeroes and there are no events.
    """
    mass_points = list(opts.mass_points)
    mh_poly_order = opts.mh_poly_order
    if len(mass_points) == 1 and mh_poly_order > 0:
        logger.warning("Single mass point: the polynomials in MH are of order 0")
        mh_poly_order = 0
    if opts.analysis_rm not in analysis.globalReplacementMap:
        raise KeyError(f"No replacement map for {opts.analysis_rm} in Signal/tools/replacementMap.py")
    if opts.analysis_xsbr not in analysis.globalXSBRMap:
        raise KeyError(f"No XS x BR map for {opts.analysis_xsbr} in commonTools/XSBRMap.py")
    replacement_map = analysis.globalReplacementMap[opts.analysis_rm]

    f0 = ROOT.TFile(glob.glob(f"{input_dir}/output*M{MH_NOMINAL}*{proc}.root")[0], "read")
    ws0 = f0.Get(f"{WS_DIR}/{WS_NAME}")
    xvar, dz = ws0.var("CMS_hgg_mass"), ws0.var("dZ")
    observables = ROOT.RooArgSet(xvar, dz)
    mh = ROOT.RooRealVar("MH", "m_{H}", int(MH_LOW), int(MH_HIGH))
    mh.setUnit("GeV")
    mh.setConstant(True)

    if opts.skip_zeroes:
        d = load_dataset(input_dir, proc, MH_NOMINAL, cat, observables)
        if d.numEntries() == 0 or d.sumEntries() <= 0.0:
            logger.info("(%s, %s) has zero events: no signal model", proc, cat)
            return None

    nominal = {}
    sources, datasets = {}, {}
    rv_scenario = None if opts.skip_vertex_scenario_split else "RV"
    datasets["RV"], sources["RV"] = scenario_datasets(
        input_dir,
        proc,
        cat,
        mass_points,
        observables,
        rv_scenario,
        _replacement_rv(replacement_map, cat, proc),
        opts.replacement_threshold,
        nominal,
    )
    if not opts.skip_vertex_scenario_split:
        datasets["WV"], sources["WV"] = scenario_datasets(
            input_dir,
            proc,
            cat,
            mass_points,
            observables,
            "WV",
            (replacement_map["procWV"], replacement_map["catWV"]),
            opts.replacement_threshold,
        )

    if not opts.skip_beamspot_reweigh:
        widths = (opts.beamspot_width_data, opts.beamspot_width_mc)
        for scenario in datasets:
            datasets[scenario] = {m: beamspot_reweigh(d, *widths) for m, d in datasets[scenario].items()}
        nominal = {m: beamspot_reweigh(d, *widths) for m, d in nominal.items()}

    fits = {}
    for scenario, source in sources.items():
        name = "Total" if opts.skip_vertex_scenario_split else scenario
        fit = SignalFit(
            name, proc, cat, datasets[scenario], xvar.Clone(), mh, opts.n_bins, mh_poly_order, opts.minimizer_method
        )
        if opts.use_dcb:
            fit.build_dcb_plus_gaussian()
        else:
            path = os.path.join(ingredients_dir, f"outdir_{ext}", "fTest", "json", f"nGauss_{source.cat}.json")
            with open(path) as f:
                fit.build_gaussians(int(json.load(f)[f"{source.proc}__{source.cat}"][f"n{scenario}"]))
        fit.fit()
        fit.build_splines()
        fits[name] = fit
        logger.info("%s fit (%d Gaussians): chi2/ndof = %.4f", name, fit.n_gaussians, fit.reduced_chi2())

    photon_syst = os.path.join(ingredients_dir, f"outdir_{ext}", "calcPhotonSyst", "pkl", f"{cat}.pkl")
    model = SignalModel(
        fits,
        proc,
        cat,
        ext,
        year,
        nominal,
        xvar,
        mh,
        mass_points,
        analysis.globalXSBRMap[opts.analysis_xsbr],
        photon_syst,
        proc,
        opts.systematics,
        opts.skip_systematics,
    )
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        f = ROOT.TFile(output_file, "RECREATE")
        ws_name = f"{analysis.outputWSName__}_{SQRTS}"
        ws = ROOT.RooWorkspace(ws_name, ws_name)
        model.save(ws)
        ws.Write()
        f.Close()
    if report_file:
        _write_report(report_file, proc, cat, year, ext, opts, sources)
    f0.Close()
    return model, fits, sources


def _write_report(path, proc, cat, year, ext, opts, sources):
    lines = [
        "SignalFit replacement report",
        f"proc={proc}",
        f"cat={cat}",
        f"year={year}",
        f"ext={ext}",
        f"replacementThreshold={opts.replacement_threshold}",
        f"analysisRM={opts.analysis_rm}",
        "",
        "[RV_or_Total]",
        *sources["RV"].report(),
        "",
        "[WV]",
    ]
    lines += sources["WV"].report() if "WV" in sources else ["status=skipped (skipVertexScenarioSplit=True)"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
