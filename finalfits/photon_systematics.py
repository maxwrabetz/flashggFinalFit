"""Photon energy scale and smearing systematics of the signal shape.

For every signal process in a category, the m_gg distribution of each up/down variation
(``<proc>_125_13TeV_<cat>_<syst>{Up,Down}01sigma`` from Trees2WS) is compared to the nominal one.
The relative change of the mean, of the effective width and of the rate, averaged over up and
down, become the nuisance parameters of the signal model (``<syst>_mean``, ``_sigma``, ``_rate``).
"""

from __future__ import annotations

import glob
import logging
import os
import pickle

import pandas as pd
import ROOT

from finalfits.signal_ftest import dataset_name
from finalfits.trees2ws import WS_DIR, WS_NAME

logger = logging.getLogger(__name__)

SYSTEMATIC_TYPES = ("scales", "scalesCorr", "smears")


def eff_sigma(hist) -> float:
    """Half width of the narrowest window around the mean containing 68.3% of the histogram."""
    n_bins, bin_width, x_min = hist.GetXaxis().GetNbins(), hist.GetXaxis().GetBinWidth(1), hist.GetXaxis().GetXmin()
    mean, rms, total = hist.GetMean(), hist.GetRMS(), hist.Integral()
    # scan the window centre around the mean by up to RMS (at most 10% of the bins)
    n_window = int(rms / bin_width) if (rms / bin_width) < 0.1 * n_bins else int(0.1 * n_bins)
    limit = 0.683 * total
    best = 9999999
    for shift in range(-n_window, n_window + 1):
        i_centre = int((mean - x_min) / bin_width + 1 + shift)
        x_up = x_down = (i_centre - 0.5) * bin_width + x_min
        i_up = i_down = i_centre
        y = hist.GetBinContent(i_centre)
        content = y
        reached = False
        for _ in range(1, n_bins):
            if reached:
                continue
            if i_up < n_bins:
                i_up += 1
                x_up += bin_width
                y = hist.GetBinContent(i_up)
                content += y
                reached = content > limit
            else:
                logger.debug("Reached the last bin in the effective sigma of %s: returning 0", hist.GetName())
                return 0
            if not reached:
                if i_down > 0:
                    i_down -= 1
                    x_down -= bin_width
                    y = hist.GetBinContent(i_down)
                    content += y
                    reached = content > limit
                else:
                    logger.debug("Reached bin 0 in the effective sigma of %s: returning 0", hist.GetName())
                    return 0
        # the last bin contributes linearly up to the limit
        dx = 0.0 if y == 0.0 else (content - limit) * (bin_width / y)
        width = (x_up - x_down + bin_width - dx) * 0.5
        best = min(best, width)
    return best


def _histograms(ws, nominal_name: str, syst: str, mgg, n_bins: int) -> dict:
    hists = {}
    names = {
        "nominal": nominal_name,
        "up": f"{nominal_name}_{syst}Up01sigma",
        "down": f"{nominal_name}_{syst}Down01sigma",
    }
    for kind, name in names.items():
        hist_name = "nominal" if kind == "nominal" else f"{syst}_{kind}"
        hists[kind] = ROOT.TH1F(hist_name, hist_name, n_bins, 100, 180)
        data = ws.data(name)
        if not data:
            raise KeyError(f"Missing dataset {name}")
        data.fillHistogram(hists[kind], ROOT.RooArgList(mgg))
    return hists


def _relative_variation(values: dict, reference: float, threshold: float) -> float:
    """Mean of |up - ref| / ref and |down - ref| / ref, capped at threshold (0 if undefined)."""
    if reference == 0:
        return 0
    variation = (abs((values["up"] - reference) / reference) + abs((values["down"] - reference) / reference)) / 2
    return 0 if variation != variation else min(variation, threshold)


def variations(hists: dict, thresholds: dict) -> tuple[float, float, float]:
    """Relative (mean, sigma, rate) variations of the up/down histograms with respect to nominal."""
    if hists["nominal"].Integral() == 0:
        return 0, 0, 0
    mean = _relative_variation(
        {k: h.GetMean() for k, h in hists.items()}, hists["nominal"].GetMean(), thresholds["mean"]
    )
    sigmas = {k: eff_sigma(h) for k, h in hists.items()}
    sigma = _relative_variation(sigmas, sigmas["nominal"], thresholds["sigma"])
    # the variations can both be one-sided: the rate is compared to their midpoint
    rates = {k: h.Integral() for k, h in hists.items()}
    rate = _relative_variation(rates, 0.5 * (rates["up"] + rates["down"]), thresholds["rate"])
    return mean, sigma, rate


def calc_photon_systematics(
    input_dir: str,
    cat: str,
    procs: list[str],
    systematics: dict[str, list[str]],
    xvar: str = "CMS_hgg_mass",
    n_bins: int = 80,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """Variations of every process (rows) and systematic (columns <syst>_mean/_sigma/_rate).

    ``systematics`` maps the types scales, scalesCorr and smears to their systematics.
    """
    thresholds = thresholds or {"mean": 0.05, "sigma": 0.5, "rate": 0.05}
    names = [s for stype in SYSTEMATIC_TYPES for s in systematics.get(stype, []) if s]
    columns = ["proc", "cat", "inputWSFile", "nominalDataName"]
    columns += [f"{s}_{x}" for s in names for x in ("mean", "sigma", "rate")]
    rows = []
    mgg = ROOT.RooRealVar(xvar, xvar, 125)
    for proc in procs:
        logger.info("Processing (%s, %s)", proc, cat)
        path = glob.glob(f"{input_dir}/output*M125*{proc}.root")[0]
        row = {"proc": proc, "cat": cat, "inputWSFile": path, "nominalDataName": dataset_name(proc, "125", cat)}
        f = ROOT.TFile(path)
        ws = f.Get(f"{WS_DIR}/{WS_NAME}")
        for syst in names:
            hists = _histograms(ws, row["nominalDataName"], syst, mgg, n_bins)
            for key, value in zip(("mean", "sigma", "rate"), variations(hists, thresholds)):
                row[f"{syst}_{key}"] = value
            for hist in hists.values():
                hist.Delete()
        ws.Delete()
        f.Close()
        rows.append(row)
    # object columns with python numbers, as read by the signal fit since flashgg
    return pd.DataFrame(rows, columns=columns, dtype=object)


def write_pickle(data: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
