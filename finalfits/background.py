"""Background modelling with the discrete profiling method.

For every analysis category, the data in the m_gg sidebands (100-115 and 135-180 GeV) are fitted
with the function families of :mod:`finalfits.pdfs` in increasing order:

* F-test: the order of each family is increased until the improvement of the likelihood is no
  longer significant (p > 0.05); the last significant order is the "truth" order of the family.
* Envelope: orders are added while the improvement is p < 0.1, keeping those with an acceptable
  goodness of fit (p > 0.01, or the truth order).
* The envelope functions form a RooMultiPdf ``CMS_hgg_<cat>_<ext>_bkgshape`` with the index
  ``pdfindex_<cat>_<ext>`` set to the best fit (lowest NLL + number of parameters) and the
  normalisation ``CMS_hgg_<cat>_<ext>_bkgshape_norm`` (sideband yield scaled to 100-180 GeV).

The fits use RooFit's global random generator (randomised restarts, toys): running every category
in its own process makes the results independent of the other categories.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import ROOT

from finalfits import background_plots as plots
from finalfits.pdfs import DAT_NAMES, FAMILIES, BackgroundPdfs

logger = logging.getLogger(__name__)
RF = ROOT.RooFit

MGG_LOW, MGG_HIGH = 100, 180
BLIND_LOW, BLIND_HIGH = 115, 135
N_BINS = 4 * (MGG_HIGH - MGG_LOW)  # 0.25 GeV bins
N_BINS_SIDEBANDS = 4 * (MGG_HIGH - MGG_LOW - (BLIND_HIGH - BLIND_LOW))
FIT_RANGE = "low,high"  # the sidebands

MAX_ORDER = 6
FTEST_THRESHOLD = 0.05
ENVELOPE_THRESHOLD = 0.1
GOF_THRESHOLD = 0.01
FIT_TRIES = 3


def owned(obj):
    """Take ownership of an object that RooFit returns as a pointer for the caller to delete."""
    ROOT.SetOwnership(obj, True)
    return obj


def fit(pdf, data, max_tries: int = FIT_TRIES) -> tuple[float, int]:
    """Fit in the sidebands, restarting from randomised parameters if needed; returns (NLL, status)."""
    params = pdf.getParameters(ROOT.RooArgSet())
    nll, status = 10e8, 1
    for _ in range(max_tries):
        result = pdf.fitTo(
            data, RF.Save(True), RF.Minimizer("Minuit2", "minimize"), RF.SumW2Error(True), RF.Range(FIT_RANGE)
        )
        nll, status = result.minNll(), result.status()
        if status == 0:
            break
        params.assignValueOnly(result.randomizePars())
    return nll, status


@dataclass
class GoodnessOfFit:
    chi2_reduced: float = 0.0
    chi2: float = 0.0
    prob: float = 0.0
    n_points: int = 0
    n_fit_params: int = 0
    ndof: int = 0


def _is_sideband(x: float) -> bool:
    return MGG_LOW <= x < BLIND_LOW or BLIND_HIGH <= x <= MGG_HIGH


def plot_chi2(plot, pdf_name: str, data_name: str, n_fit_params: int) -> GoodnessOfFit:
    """Chi2 between the pdf and data curves of a RooPlot, counting only points in the sidebands."""
    gof = GoodnessOfFit(chi2_reduced=plot.chiSquare(pdf_name, data_name, n_fit_params), n_fit_params=n_fit_params)
    data = plot.findObject(data_name)
    if data:
        gof.n_points = sum(_is_sideband(data.GetPointX(i)) for i in range(data.GetN()))
    if gof.n_points <= 0:
        gof.n_points = N_BINS_SIDEBANDS
    gof.ndof = gof.n_points - n_fit_params
    if gof.ndof > 0:
        gof.chi2 = gof.chi2_reduced * gof.ndof
        gof.prob = ROOT.TMath.Prob(gof.chi2, gof.ndof)
    else:
        logger.warning("Non-positive ndof = %d (%d points, %d fit parameters)", gof.ndof, gof.n_points, n_fit_params)
    return gof


def goodness_of_fit(mass, pdf, data, plot_path: str, rng: ROOT.TRandom3, n_toys: int = 500):
    """Goodness of fit of a fitted pdf in the sidebands; returns (p-value, GoodnessOfFit).

    With fewer than 5 events per bin on average, the p-value is taken from toys (plotted to
    ``<plot_path>_gofTest.pdf``) instead of the asymptotic chi2 distribution.
    """
    norm = ROOT.RooRealVar("norm", "norm", data.sumEntries(), 0, 10e6)
    extended = ROOT.RooExtendPdf("ext", "ext", pdf, norm)
    sideband_entries = data.sumEntries()

    frame = owned(mass.frame())
    data.plotOn(frame, RF.Binning(N_BINS), RF.Name("data"), RF.Range(FIT_RANGE))
    extended.plotOn(
        frame,
        RF.Name("pdf"),
        RF.Range(FIT_RANGE),
        RF.NormRange(FIT_RANGE),
        RF.Normalization(sideband_entries, ROOT.RooAbsReal.NumEvent),
    )
    n_params = extended.getParameters(data).getSize()
    gof = plot_chi2(frame, "pdf", "data", n_params)
    logger.debug("Goodness of fit of %s with %d fitted parameters", pdf.GetName(), n_params)

    if data.sumEntries() / N_BINS_SIDEBANDS >= 5:
        prob = gof.prob
    else:
        logger.info("Low statistics: running %d toys for the goodness of fit of %s", n_toys, pdf.GetName())
        params = extended.getParameters(data)
        best_fit = ROOT.RooArgSet()
        params.snapshot(best_fit)
        n_data = int(data.sumEntries())
        n_pass, toy_chi2 = 0, []
        for _ in range(n_toys):
            params.assignValueOnly(best_fit)
            toy = owned(
                extended.generateBinned(
                    ROOT.RooArgSet(mass),
                    RF.NumEvents(rng.Poisson(n_data)),
                    RF.Range(FIT_RANGE),
                    RF.ExpectedData(False),
                    RF.Extended(False),
                )
            )
            extended.fitTo(
                toy,
                RF.Minimizer("Minuit2", "minimize"),
                RF.Minos(False),
                RF.Hesse(False),
                RF.PrintLevel(-1),
                RF.Strategy(0),
                RF.SumW2Error(True),
                RF.Range(FIT_RANGE),
            )
            toy_frame = owned(mass.frame())
            toy.plotOn(toy_frame, RF.Binning(N_BINS), RF.Name("toy_data"), RF.Range(FIT_RANGE))
            extended.plotOn(
                toy_frame,
                RF.Name("toy_pdf"),
                RF.Range(FIT_RANGE),
                RF.NormRange(FIT_RANGE),
                RF.Normalization(toy.sumEntries(), ROOT.RooAbsReal.NumEvent),
            )
            toy_gof = plot_chi2(toy_frame, "toy_pdf", "toy_data", n_params)
            n_pass += toy_gof.chi2 >= gof.chi2
            toy_chi2.append(toy_gof.chi2)
        prob = n_pass / n_toys
        plots.plot_gof_toys(toy_chi2, gof.chi2, extended.GetName(), f"{plot_path}_gofTest.pdf")
        params.assignValueOnly(best_fit)

    logger.info(
        "GoF %s: chi2/ndof = %g (%g / %d), p-value = %g", pdf.GetName(), gof.chi2_reduced, gof.chi2, gof.ndof, prob
    )
    return prob, gof


def _toy_fit(pdf, toy, fallback, params, max_tries: int = 2) -> tuple[float, int]:
    nll, status = 0.0, 1
    for _ in range(max_tries):
        result = pdf.fitTo(
            toy,
            RF.Save(True),
            RF.Strategy(1),
            RF.SumW2Error(True),
            RF.Minimizer("Minuit2", "minimize"),
            RF.Minos(False),
            RF.Hesse(False),
            RF.PrintLevel(-1),
            RF.Range(FIT_RANGE),
        )
        nll, status = result.minNll(), result.status()
        if status == 0:
            break
        params.assignValueOnly(fallback.randomizePars())
    return nll, status


def ftest_toys(chi2, ndof, prob_asymptotic, pdf_null, pdf_test, mass, data, plot_path: str, n_toys: int = 5000):
    """Check the asymptotic F-test p-value with toys of the lower order; plots to ``plot_path``."""

    def fit_data(pdf):
        return pdf.fitTo(
            data,
            RF.Save(True),
            RF.Strategy(1),
            RF.Minimizer("Minuit2", "minimize"),
            RF.SumW2Error(True),
            RF.PrintLevel(-1),
            RF.Range(FIT_RANGE),
        )

    fit_null, fit_test = fit_data(pdf_null), fit_data(pdf_test)
    params_null, params_test = pdf_null.getParameters(ROOT.RooArgSet()), pdf_test.getParameters(ROOT.RooArgSet())
    start_null, start_test = ROOT.RooArgSet(), ROOT.RooArgSet()
    params_null.snapshot(start_null)
    params_test.snapshot(start_test)

    n_data = int(data.sumEntries())
    mass.setBins(N_BINS)
    toy_chi2, status_null, status_test = [], [], []
    for _ in range(n_toys):
        params_null.assignValueOnly(start_null)
        params_test.assignValueOnly(start_test)
        toy = owned(
            pdf_null.generateBinned(
                ROOT.RooArgSet(mass),
                RF.NumEvents(n_data),
                RF.Range(FIT_RANGE),
                RF.ExpectedData(False),
                RF.Extended(False),
            )
        )
        nll_null, stat_null = _toy_fit(pdf_null, toy, fit_null, params_null)
        nll_test, stat_test = _toy_fit(pdf_test, toy, fit_test, params_test)
        status_null.append(stat_null)
        status_test.append(stat_test)
        if stat_null == 0 and stat_test == 0:
            toy_chi2.append(2 * (nll_null - nll_test))
    prob = sum(c >= chi2 for c in toy_chi2) / len(toy_chi2) if toy_chi2 else 0.0
    plots.plot_ftest_toys(
        toy_chi2, chi2, ndof, prob, prob_asymptotic, status_null, status_test, pdf_null, pdf_test, plot_path
    )
    params_null.assignValueOnly(start_null)
    params_test.assignValueOnly(start_test)
    return prob


def ftest_probability(chi2, ndof, pdf_null, pdf_test, mass, data, plot_path: str, with_toys: bool) -> float:
    """p-value of the likelihood ratio between two orders (asymptotic chi2 distribution).

    With ``with_toys``, the distribution is also checked with toys (plotted to ``plot_path``).
    """
    prob = ROOT.TMath.Prob(chi2, ndof)
    if with_toys:
        ftest_toys(chi2, ndof, prob, pdf_null, pdf_test, mass, data, plot_path)
    return prob


def sideband_scaled_yield(pdf, mass, sideband_yield: float) -> float:
    """Yield in 100-180 GeV from the sideband yield and the shape of the pdf."""
    observables = ROOT.RooArgSet(mass)
    sideband = owned(pdf.createIntegral(observables, RF.NormSet(observables), RF.Range(FIT_RANGE))).getVal()
    full = owned(pdf.createIntegral(observables, RF.NormSet(observables), RF.Range("fullRange"))).getVal()
    if sideband <= 0.0:
        return sideband_yield
    return sideband_yield * ((full if full > 0.0 else 1.0) / sideband)


def best_fit_index(multipdf, data, index) -> int:
    """Index of the pdf with the lowest NLL + penalty; the parameters are left at that fit."""
    params = multipdf.getParameters(ROOT.RooArgSet())
    params.remove(index)
    best, start = ROOT.RooArgSet(), ROOT.RooArgSet()
    params.snapshot(best)
    params.snapshot(start)
    best_index, best_nll = 0, 1e10
    for i in range(index.numTypes()):
        params.assignValueOnly(start)
        index.setIndex(i)
        nll, _ = fit(multipdf.getCurrentPdf(), data)
        nll += multipdf.getCorrection()
        logger.debug("%s: NLL + correction = %g", multipdf.getCurrentPdf().GetName(), nll)
        if nll < best_nll:
            best_nll = nll
            best.assignValueOnly(params)
            best_index = i
    index.setIndex(best_index)
    params.assignValueOnly(best)
    logger.info("Best fit function: %s", multipdf.getCurrentPdf().GetName())
    return best_index


@dataclass
class CategoryResult:
    """Outcome of the fTest of one category."""

    truth_orders: dict[str, int] = field(default_factory=dict)
    envelope_orders: dict[str, list[int]] = field(default_factory=dict)
    table: list[str] = field(default_factory=list)  # rows of fTestResults.txt


class FTest:
    """Runs the fTest for the categories of one data workspace (see the module docstring)."""

    def __init__(self, mass, out_dir: str, label: str, blind: bool = True, ftest_toys: bool = False):
        self.mass = mass
        self.out_dir = out_dir
        self.label = label  # <year>_13TeV, part of the pdf names
        self.blind = blind
        self.ftest_toys = ftest_toys
        self.pdfs = BackgroundPdfs(mass)
        self.rng = ROOT.TRandom3()  # Poisson numbers of events of the goodness-of-fit toys

    @staticmethod
    def _fit(pdf, data):
        nll, status = fit(pdf, data)
        if status != 0:
            logger.warning("Fit status for %s is %d", pdf.GetName(), status)
        return nll, status

    def truth_order(self, family, data, cat_number):
        """F-test: returns (truth order, its pdf, results table row)."""
        nll, prev_nll, chi2, prob = 0.0, 0.0, 0.0, 0.0
        order, prev_order, truth = 1, 0, 0
        prev_pdf = truth_pdf = None
        while prob < FTEST_THRESHOLD and order <= MAX_ORDER:
            pdf = self.pdfs.make(family, order, f"ftest_pdf_{cat_number}_{self.label}")
            if pdf is None:
                order += 1
                continue
            nll, status = self._fit(pdf, data)
            chi2 = 2.0 * (prev_nll - nll)
            if chi2 < 0.0 and order > 1:
                chi2 = 0.0
            prob = 0.0
            if prev_pdf is not None:
                path = f"{self.out_dir}/Ftest_from_{family}{order}_cat{cat_number}.pdf"
                prob = ftest_probability(
                    chi2, order - prev_order, prev_pdf, pdf, self.mass, data, path, self.ftest_toys
                )
            self.plot_fit(pdf, data, f"{self.out_dir}/{family}{order}_cat{cat_number}.pdf", status)
            logger.info("F-test %s %d: NLL %g -> %g, chi2 = %g, p = %g", family, order, prev_nll, nll, chi2, prob)
            prev_nll = nll
            truth, truth_pdf = prev_order, prev_pdf
            prev_order, prev_pdf = order, pdf
            order += 1
        return truth, truth_pdf, f"{family:>15s} & {truth + 1} & {chi2:5.2f} & {prob:5.2f} \\\\\n"

    def envelope_orders(self, family, data, cat_number, truth, stored_pdfs):
        """Orders of a family in the envelope (their pdfs are added to stored_pdfs); returns (orders, table row)."""
        nll, prev_nll, chi2, prob = 0.0, 0.0, 0.0, 0.0
        order, prev_order, last = 1, 0, 0
        orders = []
        while prob < ENVELOPE_THRESHOLD:
            pdf = self.pdfs.make(family, order, f"env_pdf_{cat_number}_{self.label}")
            if pdf is None:
                if order > MAX_ORDER:
                    logger.warning("No more orders of %s allowed", family)
                    break
                order += 1
                continue
            nll, status = self._fit(pdf, data)
            chi2 = 2.0 * (prev_nll - nll)
            if chi2 < 0.0 and order > 1:
                chi2 = 0.0
            prob = ROOT.TMath.Prob(chi2, order - prev_order)
            logger.info("Envelope %s %d: NLL %g -> %g, chi2 = %g, p = %g", family, order, prev_nll, nll, chi2, prob)
            prev_nll = nll
            last = prev_order
            gof_prob = self.plot_fit(pdf, data, f"{self.out_dir}/{family}{order}_cat{cat_number}.pdf", status)
            if prob < ENVELOPE_THRESHOLD and (gof_prob > GOF_THRESHOLD or order == truth):
                logger.info("Adding %s to the envelope (GoF p = %g)", pdf.GetName(), gof_prob)
                stored_pdfs.add(pdf)
                orders.append(order)
            prev_order = order
            order += 1
        return orders, f"{family:>15s} & {last + 1} & {chi2:5.2f} & {prob:5.2f} \\\\\n"

    def plot_fit(self, pdf, data, path, status) -> float:
        """Goodness of fit of a fitted pdf and its plot; returns the goodness-of-fit p-value."""
        prob, gof = goodness_of_fit(self.mass, pdf, data, path, self.rng)
        plots.plot_fit(self.mass, pdf, data, path, gof, prob, status, self.blind, FIT_RANGE)
        return prob

    def run_category(self, cat: str, cat_number: int, data, envelope: bool) -> tuple[CategoryResult, object]:
        """fTest (and envelope) of one category; returns the result and the list of envelope pdfs."""
        result = CategoryResult()
        stored_pdfs = ROOT.RooArgList("store")
        truth_pdfs = {}
        for family in FAMILIES:
            truth, truth_pdf, row = self.truth_order(family, data, cat_number)
            result.table.append(row)
            result.truth_orders[family] = truth
            truth_pdfs[f"{family}{truth}"] = truth_pdf
            if envelope:
                logger.info("Determining the envelope functions of %s for %s", family, cat)
                orders, row = self.envelope_orders(family, data, cat_number, truth, stored_pdfs)
                result.table.append(row)
                result.envelope_orders[family] = orders
        plots.plot_truths(
            self.mass, truth_pdfs, data, f"{self.out_dir}/truths_cat{cat_number}", cat, self.blind, FIT_RANGE
        )
        return result, stored_pdfs


def _transfer_macros(in_file, out_file) -> None:
    for key in in_file.GetListOfKeys():
        if key.ReadObj().ClassName() == "TMacro":
            macro = in_file.Get(key.GetName())
            out_file.cd()
            macro.Write()


def run_ftest(
    input_file: str,
    categories: list[str],
    out_dir: str,
    dat_file: str,
    year: str,
    cat_offset: int = 0,
    multipdf_file: str | None = None,
    is_data: bool = True,
    blind: bool = True,
    ftest_toys: bool = False,
) -> list[CategoryResult]:
    """Run the fTest for the categories of a data workspace (``Data_13TeV_<cat>`` in allData.root).

    Writes the plots and fTestResults.txt to ``out_dir``, the chosen orders to ``dat_file`` and,
    if ``multipdf_file`` is given, the workspace ``multipdf`` with the RooMultiPdf of every category.
    """
    os.makedirs(out_dir, exist_ok=True)
    in_file = ROOT.TFile.Open(input_file)
    in_ws = in_file.Get("tagsDumper/cms_hgg_13TeV" if is_data else "cms_hgg_workspace")
    label = "13TeV" if year == "all" else f"{year}_13TeV"

    if multipdf_file is not None:
        out_file = ROOT.TFile(multipdf_file, "RECREATE")
        out_ws = ROOT.RooWorkspace()
        out_ws.SetName("multipdf")
        _transfer_macros(in_file, out_file)
        out_ws.Import(ROOT.RooRealVar("IntLumi", "hacked int lumi", 1000.0))
        out_ws.Import(in_ws.var("SqrtS") or ROOT.RooRealVar("SqrtS", "SqrtS", 13))

    mass = in_ws.var("CMS_hgg_mass")
    mass.setRange("low", MGG_LOW, BLIND_LOW)
    mass.setRange("high", BLIND_HIGH, MGG_HIGH)
    mass.setRange("fullRange", MGG_LOW, MGG_HIGH)
    ftest = FTest(mass, out_dir, label, blind, ftest_toys)

    results = []
    with open(os.path.join(out_dir, "fTestResults.txt"), "w") as table:
        table.write("Truth Model & d.o.f & $\\Delta NLL_{N+1}$ & $p(\\chi^{2}>\\chi^{2}_{(N\\rightarrow N+1)})$ \\\\\n")
        table.write("\\hline\n")
        for i, cat in enumerate(categories):
            cat_number = i + cat_offset
            data_full = in_ws.data(f"Data_13TeV_{cat}" if is_data else f"data_mass_{cat}")
            mass.setBins(N_BINS)
            sideband = data_full.reduce(RF.Cut(f"CMS_hgg_mass<{BLIND_LOW} || CMS_hgg_mass>{BLIND_HIGH}"))
            if sideband:
                owned(sideband)
            data = ROOT.RooDataHist(f"roohist_data_mass_{cat}", "data", ROOT.RooArgSet(mass), sideband or data_full)

            table.write(f"\\multicolumn{{4}}{{|c|}}{{\\textbf{{Category {i}}}}} \\\\\n")
            table.write("\\hline\n")
            result, stored_pdfs = ftest.run_category(cat, cat_number, data, envelope=multipdf_file is not None)
            table.writelines(result.table)
            table.write("\\hline\n")
            results.append(result)

            if multipdf_file is not None:
                index = ROOT.RooCategory(f"pdfindex_{cat}_{label}", "c")
                multipdf = ROOT.RooMultiPdf(f"CMS_hgg_{cat}_{label}_bkgshape", "all pdfs", index, stored_pdfs)
                best = best_fit_index(multipdf, data, index)
                scaled_yield = sideband_scaled_yield(stored_pdfs.at(best), mass, data.sumEntries())
                if scaled_yield <= 0.0:
                    raise RuntimeError(f"Non-positive background yield {scaled_yield} in {cat}")
                norm = ROOT.RooRealVar(
                    f"CMS_hgg_{cat}_{label}_bkgshape_norm",
                    "nbkg",
                    scaled_yield,
                    0,
                    max(3.0 * scaled_yield, scaled_yield + 10.0),
                )
                logger.info(
                    "RooMultiPdf %s with %d pdfs, best fit %s",
                    multipdf.GetName(),
                    index.numTypes(),
                    stored_pdfs.at(best).GetName(),
                )
                mass.setBins(N_BINS)
                # all data (also in the blinded region), binned; the index comes with the multipdf
                data_binned = ROOT.RooDataHist(f"roohist_data_mass_{cat}", "data", ROOT.RooArgSet(mass), data_full)
                for obj in (multipdf, norm, data_binned):
                    out_ws.Import(obj)
                plots.plot_multipdf(
                    mass, multipdf, index, data, f"{out_dir}/multipdf_{cat}", cat, i, best, ftest.blind, FIT_RANGE
                )

    if multipdf_file is not None:
        out_file.cd()
        out_ws.Write()
        out_file.Close()

    with open(dat_file, "w") as dat:
        for i, result in enumerate(results):
            dat.write(f"cat={i + cat_offset}\n")
            for family, order in sorted(result.truth_orders.items()):
                dat.write(f"truth={family}:{order}:{DAT_NAMES[family]}{order}\n")
            for family, orders in sorted(result.envelope_orders.items()):
                for order in orders:
                    dat.write(f"paul={family}:{order}:{DAT_NAMES[family]}{order}\n")
            dat.write("\n")
    in_file.Close()
    return results
