"""Signal fTest: number of Gaussians of the signal model of every process in one category.

The simulation at one mass point is split into the right vertex (|dZ| <= 1 cm, RV) and wrong vertex
(WV) scenarios. Each is fitted with 1 to ``n_gauss_max`` Gaussians (see :mod:`finalfits.signal_model`)
and the number with the lowest chi2/ndof is chosen. Scenarios with fewer than ``threshold`` events
keep one Gaussian. The result is written as json, read by the signal fit:
``{"<proc>__<cat>": {"nRV": n, "nWV": n}, ...}``.
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass

import ROOT

from finalfits import analysis, signal_plots
from finalfits.signal_model import SignalFit
from finalfits.trees2ws import SQRTS, WS_DIR, WS_NAME

logger = logging.getLogger(__name__)

# MH range of the fTest fits (a single mass point is fitted)
MH_LOW, MH_HIGH = 120, 130


def workspace_file(input_dir: str, proc: str, mass: str) -> str:
    return glob.glob(f"{input_dir}/output*M{mass}*{proc}.root")[0]


def dataset_name(proc: str, mass: str, cat: str) -> str:
    """Dataset of a process in the Trees2WS workspaces, e.g. ggh_NJ_0p0_1p0_in -> ggh_125_13TeV_<cat>."""
    parts = proc.split("_")
    if len(parts) <= 2 and parts[-1] in ("in", "out"):
        # fiducial in/out processes keep their tag in the dataset name
        return f"{analysis.procToData(parts[0])}_{analysis.procToData(parts[-1])}_{mass}_{SQRTS}_{cat}"
    return f"{analysis.procToData(parts[0])}_{mass}_{SQRTS}_{cat}"


@dataclass
class ProcessResult:
    proc: str
    sum_entries: float
    n_rv: int = 1
    n_wv: int = 1


def best_n_gaussians(name, proc, cat, mass, dataset, xvar, mh, n_bins, n_gauss_max, minimizer_method):
    """Fits with 1..n_gauss_max Gaussians; returns (n with the lowest chi2/ndof, fits by n)."""
    fits, best, best_chi2 = {}, 1, 999
    for n in range(1, n_gauss_max + 1):
        fit = SignalFit(f"fTest_{name}_{n}", proc, cat, {mass: dataset}, xvar.Clone(), mh, n_bins, 0, minimizer_method)
        fit.build_gaussians(n)
        fit.fit()
        if fit.ndof < 1:
            continue
        fits[n] = fit
        reduced_chi2 = fit.reduced_chi2()
        if reduced_chi2 < best_chi2:
            best_chi2, best = reduced_chi2, n
        logger.info("(%s, %s, %s): nGauss = %d, chi2/ndof = %.4f", proc, cat, name, n, reduced_chi2)
    return best, fits


def run_ftest(
    input_dir: str,
    out_dir: str,
    cat: str,
    procs: list[str],
    mass: str = "125",
    n_bins: int = 80,
    threshold: int = 30,
    n_gauss_max: int = 5,
    n_procs_to_ftest: int = -1,
    skip_wv: bool = False,
    minimizer_method: str = "TNC",
    plots: set[str] | None = None,
    plot_labels: dict[str, str] | None = None,
    write_json: bool = True,
) -> list[ProcessResult]:
    """fTest of the processes ``procs`` in category ``cat``.

    Writes ``<out_dir>/json/nGauss_<cat>.json`` and, for the scenarios in ``plots`` (RV, RVResults,
    WV, WVResults), the plots to ``<out_dir>/Plots``. Only the ``n_procs_to_ftest`` processes with the
    largest yields are tested (-1: all), the others keep one Gaussian.
    """
    plots = plots or set()
    plot_labels = plot_labels or {}
    f0 = ROOT.TFile(glob.glob(f"{input_dir}/output*")[0], "read")
    ws0 = f0.Get(f"{WS_DIR}/{WS_NAME}")
    xvar = ws0.var("CMS_hgg_mass")
    observables = ROOT.RooArgSet(xvar, ws0.var("dZ"))
    mh = ROOT.RooRealVar("MH", "m_{H}", MH_LOW, MH_HIGH)
    mh.setUnit("GeV")
    mh.setConstant(True)

    def read(proc):
        f = ROOT.TFile(workspace_file(input_dir, proc, mass), "read")
        dataset = f.Get(f"{WS_DIR}/{WS_NAME}").data(dataset_name(proc, mass, cat)).reduce(observables)
        f.Close()
        return dataset

    results = []
    for proc in procs:
        results.append(ProcessResult(proc, read(proc).sumEntries()))
    if n_procs_to_ftest == -1 or n_procs_to_ftest > len(procs):
        to_test = results
    else:
        to_test = sorted(results, key=lambda r: r.sum_entries, reverse=True)[:n_procs_to_ftest]

    for result in to_test:
        logger.info("Process %s", result.proc)
        dataset = read(result.proc)
        for scenario, cut in (("RV", "abs(dZ)<=1."), ("WV", "abs(dZ)>1.")):
            if scenario == "WV" and skip_wv:
                continue
            scenario_data = dataset.reduce(observables, cut)
            if scenario_data.numEntries() < threshold:
                continue
            best, fits = best_n_gaussians(
                scenario, result.proc, cat, mass, scenario_data, xvar, mh, n_bins, n_gauss_max, minimizer_method
            )
            setattr(result, f"n_{scenario.lower()}", best)
            label = plot_labels.get(scenario, "")
            plot_dir = os.path.join(out_dir, "Plots")
            if fits and scenario in plots:
                signal_plots.plot_ftest(fits, best, plot_dir, scenario, result.proc, cat, mass, label)
            if fits and f"{scenario}Results" in plots:
                signal_plots.plot_ftest_results(fits, best, plot_dir, scenario, result.proc, cat, mass, label)
    f0.Close()

    if write_json:
        os.makedirs(os.path.join(out_dir, "json"), exist_ok=True)
        lines = []
        for r in sorted(results, key=lambda r: r.sum_entries, reverse=True):
            key = f'"{r.proc}__{cat}"'
            lines.append(f'    {key:<90s} : {{"nRV":{r.n_rv},"nWV":{r.n_wv}}}')
        with open(os.path.join(out_dir, "json", f"nGauss_{cat}.json"), "w") as f:
            f.write("{\n" + ",\n".join(lines) + "\n}")
    return results
