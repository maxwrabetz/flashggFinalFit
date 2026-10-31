"""Correlation and covariance matrices of the POIs from the robustHesse output of combine (ROOT)."""

from __future__ import annotations

import json
import logging
import os
from array import array

import ROOT

from finalfits.analysis import differentialProcTable_, luminosity

logger = logging.getLogger(__name__)

HIGGS_MASS = "125.07"
# yellow - white - blue
PALETTE = {"stops": [0.000, 0.500, 1.000], "red": [0.992, 1.000, 0.110], "green": [0.749, 1.000, 0.196],
           "blue": [0.106, 1.000, 0.867]}  # fmt: skip


def set_palette(ncontours: int = 256, alpha: float = 1.0) -> None:
    stops, red, green, blue = (array("d", PALETTE[key]) for key in ("stops", "red", "green", "blue"))
    ROOT.TColor.CreateGradientColorTable(len(stops), stops, red, green, blue, ncontours, alpha)


def load_pois(inputs_json: str, mode: str, drop_thq: bool = False) -> tuple[list[str], str]:
    """POIs of a fit mode in the inputs json and the fit type (e.g. syst, from fits = robustHesse:syst:<poi>)."""
    with open(inputs_json) as f:
        inputs = json.load(f)[mode]
    pois = inputs["pois"].split(",")
    if drop_thq and "r_tHq" in pois:
        pois.remove("r_tHq")
    fit = inputs["fits"].split("+")[0].split(":")
    return pois, f"{fit[0]}_{fit[1]}".split("_")[-1]


def read_matrix(hesse_file: str, cov: bool = False) -> tuple[dict, int]:
    """{(poi_i, poi_j): value} of the correlation (or covariance) matrix and the number of POIs in the file."""
    in_file = ROOT.TFile(hesse_file, "READ")
    matrix = in_file.Get("h_covariance" if cov else "h_correlation")
    params = in_file.Get("floatParsFinal")
    # the POIs (r_*) and their index in the matrix; the last parameter of the list is never a POI
    pars = {par.GetName(): i for i, par in enumerate(params) if i < len(params) - 1 and par.GetName().startswith("r_")}
    logger.info(f"Processing the following {len(pars)} parameters: {', '.join(pars)}")
    values = {(a, b): matrix.GetBinContent(i + 1, j + 1) for a, i in pars.items() for b, j in pars.items()}
    in_file.Close()
    return values, len(pars)


def _format_axes(hist) -> None:
    for axis in (hist.GetXaxis(), hist.GetYaxis(), hist.GetZaxis()):
        axis.SetTitleSize(0.0462874993682)
        axis.SetLabelSize(0.0462874993682)
    hist.GetXaxis().SetTitleOffset(1.0)
    hist.GetXaxis().SetLabelOffset(0.0100567853078)
    hist.GetYaxis().SetTitleOffset(1.32249999046)
    hist.GetYaxis().SetLabelOffset(0.005)
    hist.GetZaxis().SetTitleOffset(1.14999997616)


def _canvas(mode: str, cov: bool):
    # (width, height, top, right, left, bottom margin); top None: the style default
    if mode.count("stage1p2") or mode.count("mu_reco"):
        layout = (800, 900, None, 0.14, 0.18, 0.2)
    elif mode.count("PTH"):
        layout = (1200, 1100, 0.11, 0.16, 0.24, 0.27)
    elif cov:
        layout = (800, 495, 0.12, 0.18, 0.23, 0.26)
    else:
        layout = (800, 800, None, 0.14, 0.20, 0.20)
    width, height, top, right, left, bottom = layout
    canvas = ROOT.TCanvas("can", "can", width, height)
    pad = canvas.GetPad(0)
    if top is not None:
        pad.SetTopMargin(top)
    pad.SetRightMargin(right)
    pad.SetLeftMargin(left)
    pad.SetBottomMargin(bottom)
    pad.SetTicks(1, 1)
    return canvas


def _style_hist(hist, mode: str, cov: bool) -> None:
    """Label and text sizes depending on the fit mode."""
    _format_axes(hist)
    x, y = hist.GetXaxis(), hist.GetYaxis()
    x.SetTickLength(0.0)
    x.SetLabelSize(0.06)
    x.SetLabelFont(42)
    y.SetTickLength(0.0)
    y.SetLabelSize(0.06)
    y.SetLabelOffset(0.003)
    y.SetLabelFont(42)
    hist.GetZaxis().SetRangeUser(*((-0.5, 0.5) if cov else (-1.0, 1.0)))
    hist.GetZaxis().SetTickLength(0.0)
    hist.GetZaxis().SetLabelSize(0.03)
    hist.GetZaxis().SetDecimals(True)

    def sizes(label_size, marker_size):
        x.SetLabelSize(label_size)
        y.SetLabelSize(label_size)
        hist.SetMarkerSize(marker_size)

    if mode.count("stage1p2"):
        x.SetLabelOffset(0.003)
        x.LabelsOption("v")
        if mode.count("extended"):
            sizes(0.025, 0.55)
        elif mode.count("minimal"):
            sizes(0.02, 0.6)
        else:
            sizes(0.03, 0.8)
    elif mode.count("stage0"):
        x.LabelsOption("h")
        y.SetLabelOffset(0.007)
        hist.SetMarkerSize(1.5)
        x.SetLabelSize(0.05)
        y.SetLabelSize(0.05)
    elif mode.count("mu_reco"):
        x.SetLabelOffset(0.003)
        x.LabelsOption("v")
        sizes(0.025, 0.55)
    elif mode in differentialProcTable_:
        x.SetLabelOffset(0.003)
        x.LabelsOption("v")
        if cov:
            sizes(*((0.035, 0.95) if mode.count("PTH") else (0.06, 2)))
        else:
            if mode.count("PTH"):
                label_size = 0.035
            elif mode.count("rapidity") or mode.count("NJ") or mode.count("PTJ0"):
                label_size = 0.06
            else:
                label_size = 0.03
            sizes(label_size, 0.85 if mode.count("PTH") else 1.5)
    else:
        y.SetLabelOffset(0.007)
        hist.SetMarkerSize(1.5)


def _draw_labels(canvas, mode: str, translate: dict, year: str, observed: bool, preliminary: bool) -> None:
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextAlign(12)
    top_text_size, info_text_size = (0.03, 0.032) if mode.count("PTH") else (0.045, 0.04)
    latex.SetTextSize(top_text_size)
    top_y = 1.00 - canvas.GetTopMargin() + 0.025
    extra = ("" if observed else "Simulation") + (" Preliminary" if preliminary else "")
    cms = "#bf{CMS}" + (f" #it{{{extra.strip()}}}" if extra else "")
    latex.DrawLatex(canvas.GetLeftMargin() + 0.005, top_y, cms)
    latex.SetTextSize(top_text_size)
    latex.SetTextAlign(32)
    latex.DrawLatex(1.00 - canvas.GetRightMargin(), top_y, "%0.1f fb^{-1} (13.6 TeV)" % luminosity(year))
    latex.SetTextSize(info_text_size)
    x = 1.00 - canvas.GetRightMargin() - 0.02
    latex.DrawLatex(x, 1.00 - canvas.GetTopMargin() - 0.04, f"{translate.get(mode, mode)}")
    latex.DrawLatex(x, 1.00 - canvas.GetTopMargin() - 0.10, "H #rightarrow #gamma#gamma")
    latex.DrawLatex(x, 1.00 - canvas.GetTopMargin() - 0.15, "#font[52]{m}_{H} = %s GeV" % HIGGS_MASS)


def plot_matrix(
    hesse_file: str,
    inputs_json: str,
    mode: str,
    out_dir: str,
    translate: dict | None = None,
    cov: bool = False,
    observed: bool = False,
    preliminary: bool = True,
    year: str = "2022",
    ext: str = "",
    drop_thq: bool = False,
) -> list[str]:
    """Lower triangle of the correlation (or covariance) matrix of the POIs of a fit mode, as png and pdf.

    translate: display names of the POIs and of the mode (e.g. Plots/poi_differential_hesse_noLabels.json).
    """
    translate = translate or {}
    pois, fit_type = load_pois(inputs_json, mode, drop_thq)
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetNumberContours(500)
    values, n_pars = read_matrix(hesse_file, cov)

    hist = ROOT.TH2F(f"corr_{mode}", "", n_pars, -0.5, n_pars - 0.5, n_pars, -0.5, n_pars - 0.5)
    for i, poi_i in enumerate(pois):
        for j, poi_j in enumerate(reversed(pois)):
            hist.GetXaxis().SetBinLabel(i + 1, translate.get(poi_i, poi_i))
            hist.GetYaxis().SetBinLabel(j + 1, translate.get(poi_j, poi_j))
            if i <= hist.GetNbinsX() - 1 - j:
                hist.Fill(i, j, values[(poi_i, poi_j)])
    logger.info(f"Final correlation map used is: {values}")

    set_palette()
    ROOT.gStyle.SetNumberContours(500)
    ROOT.gStyle.SetPaintTextFormat("1.2f")
    ROOT.gStyle.SetTextFont(42)
    canvas = _canvas(mode, cov)
    _style_hist(hist, mode, cov)
    hist.Draw("colz,text")
    _draw_labels(canvas, mode, translate, year, observed, preliminary)
    # white text on the dark (> 0.8) cells
    for binx in range(1, hist.GetNbinsX() + 1):
        for biny in range(1, hist.GetNbinsY() + 1):
            if hist.GetBinContent(binx, biny) > 0.8:
                latex = ROOT.TLatex()
                latex.SetTextAlign(22)
                latex.SetTextSize(0.03)
                latex.SetTextColor(ROOT.kWhite)
                x, y = hist.GetXaxis().GetBinCenter(binx), hist.GetYaxis().GetBinCenter(biny)
                latex.DrawLatex(x, y, f"{hist.GetBinContent(binx, biny):.2f}")

    name = f"{'covMatrix' if cov else 'corrMatrix'}_{mode}_{fit_type}{'_obs' if observed else ''}{ext}"
    paths = [os.path.join(out_dir, f"{name}.{suffix}") for suffix in ("png", "pdf")]
    for path in paths:
        canvas.Print(path)
    return paths
