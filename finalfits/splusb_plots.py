"""Signal + background model plots of the diphoton mass per category, their sum and the S/(S+B) weighted sum (ROOT).

The models and the data are taken from a combine workspace (the datacard workspace for the pre-fit plots, or a
post-fit workspace saved with --saveWorkspace together with its best-fit snapshot). The +-1/2 sigma bands of the
background model come from toys in ./SplusBModels<ext>/toys/toy_*.root. All outputs are written relative to the
current directory: ./SplusBModels<ext>/ (plots, toy yields) and ./jsons/ (category weights).
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import ROOT
import scipy.stats

logger = logging.getLogger(__name__)

HH_FIX_CATS = ["DoubleHTag_10_13TeV", "DoubleHTag_11_13TeV"]  # HH categories with the Mjj_90GeV variable


@dataclass
class SplusBOptions:
    """Options of the S+B plots, as on the command line of scripts/plot_splusb.py."""

    input_ws_file: str
    cats: str  # comma separated, or all
    load_snapshot: str | None = None
    unblind: bool = False
    blinding_region: str = "116,134"
    data_scaler: float = 1.0
    do_bkg_renormalization: bool = False
    do_bands: bool = False
    do_toy_veto: bool = False
    load_toy_yields: str = ""
    save_toy_yields: bool = False
    do_zeroes: bool = False
    skip_individual_cat_plots: bool = False
    do_sum_categories: bool = False
    do_cat_weights: bool = False
    load_weights: str = ""
    save_weights: bool = False
    parameter_map: str | None = None  # name:value,...
    ext: str = ""
    mass: str | None = None
    xvar: str = "CMS_hgg_mass,m_{#gamma#gamma},GeV"  # name,title,unit
    n_bins: int = 80
    pdf_n_bins: int = 3200
    translate_cats: str | None = None
    problematic_cats: str = ""
    do_hh_mjj_fix: bool = False
    poi: str = ""
    lumi_label: str = "61.9 fb^{-1} (13.6 TeV)"

    @property
    def xvar_name(self) -> str:
        return self.xvar.split(",")[0]

    @property
    def blinding(self) -> list[float]:
        return [float(self.blinding_region.split(",")[0]), float(self.blinding_region.split(",")[1])]

    def selected(self, cat: str) -> bool:
        # the problematic categories are left out everywhere: plots, sums, weights and toy bands
        if cat in self.problematic_cats.split(","):
            return False
        return self.cats == "all" or cat in self.cats.split(",")

    @property
    def several_cats(self) -> bool:
        return len(self.cats.split(",")) > 1 or self.cats == "all"


def eff_sigma(hist) -> float:
    """Half width of the narrowest window holding 68.3% of the histogram (0 if it reaches the histogram edge)."""
    nbins, binw, xmin = hist.GetXaxis().GetNbins(), hist.GetXaxis().GetBinWidth(1), hist.GetXaxis().GetXmin()
    mu, rms, total = hist.GetMean(), hist.GetRMS(), hist.Integral()
    # scan the window centre around the mean by up to RMS / bin width (at most 10% of the bins)
    n_window = int(rms / binw) if (rms / binw) < 0.1 * nbins else int(0.1 * nbins)
    rlim = 0.683 * total
    wmin = 9999999
    for iscan in range(-1 * n_window, n_window + 1):
        i_centre = int((mu - xmin) / binw + 1 + iscan)
        x_centre = (i_centre - 0.5) * binw + xmin
        x_up, x_down = x_centre, x_centre
        i_up, i_down = i_centre, i_centre
        y = hist.GetBinContent(i_centre)
        r = y
        reached_limit = False
        for _ in range(1, nbins):
            if reached_limit:
                continue
            if i_up < nbins and not reached_limit:
                i_up += 1
                x_up += binw
                y = hist.GetBinContent(i_up)
                r += y
                if r > rlim:
                    reached_limit = True
            else:
                logger.warning(f"Reach nBins in effSigma calc: {hist.GetName()}. Returning 0 for effSigma")
                return 0
            if not reached_limit:
                if i_down > 0:
                    i_down -= 1
                    x_down -= binw
                    y = hist.GetBinContent(i_down)
                    r += y
                    if r > rlim:
                        reached_limit = True
                else:
                    logger.warning(f"Reach 0 in effSigma calc: {hist.GetName()}. Returning 0 for effSigma")
                    return 0
        # fraction of the last bin above the limit (linear)
        dx = 0.0 if y == 0.0 else (r - rlim) * (binw / y)
        w = (x_up - x_down + binw - dx) * 0.5
        if w < wmin:
            wmin = w
    return wmin


def band_properties(toys: pd.DataFrame, cat: str, ibin: int) -> dict:
    """Median and +-1/2 sigma quantiles of the toy yields in a bin (cat all/wall: the (weighted) sum)."""
    column = toys["%s_%g" % ({"all": "sum", "wall": "wsum"}.get(cat, cat), ibin)].values
    quantile = {n: 50 * (1 + math.erf(n / math.sqrt(2))) for n in (1.0, -1.0, 2.0, -2.0)}
    return {
        "median": np.median(column),
        "up1sigma": np.percentile(column, quantile[1.0]),
        "down1sigma": np.percentile(column, quantile[-1.0]),
        "up2sigma": np.percentile(column, quantile[2.0]),
        "down2sigma": np.percentile(column, quantile[-2.0]),
    }


def _data_graph(hD, hist):
    """Data points of hist with the 68% Poisson intervals of the counts in hD."""
    graph = ROOT.TGraphAsymmErrors()
    for ibin in range(1, hist.GetNbinsX() + 1):
        graph.SetPoint(ibin - 1, hist.GetBinCenter(ibin), hist.GetBinContent(ibin))
        low = scipy.stats.gamma.interval(0.68, hD.GetBinContent(ibin))[0]
        if low != low:  # NaN for zero entries
            low = 0
        up = scipy.stats.gamma.interval(0.68, hD.GetBinContent(ibin) + 1)[1]
        graph.SetPointError(ibin - 1, 0, 0, abs(hD.GetBinContent(ibin) - low), abs(hD.GetBinContent(ibin) - up))
    graph.SetMarkerColor(1)
    graph.SetMarkerStyle(20)
    graph.SetFillColor(2)
    graph.Draw("Same PE")
    return graph


def _draw_pdfs(hSB, hB, hS, options: SplusBOptions) -> None:
    if options.unblind:
        hSB.SetLineWidth(3)
        hSB.SetLineColor(2)
        hSB.Draw("Hist same c")
        hB.SetLineWidth(3)
        hB.SetLineColor(2)
        hB.SetLineStyle(2)
        hB.Draw("Hist same c")
    else:
        hS.SetLineWidth(3)
        hS.SetLineColor(9)
        hS.SetFillColor(38)
        hS.SetFillStyle(1001)
        hS.GetXaxis().SetRangeUser(*options.blinding)
        hS.Draw("Hist same cf")
        hB.SetLineWidth(3)
        hB.SetLineColor(2)
        hB.Draw("Hist same c")


def plot_splusb(
    workspace, hD, hSB, hB, hS, hDr, hBr, hSr, cat: str, options: SplusBOptions, toys=None, reduce_range=None
):
    """Data, S+B and B model (signal model when blinded) and the background subtracted ratio of one category.

    hSB, hB, hS: {"pdfNBins": finely binned model, "nBins": model in the data binning}; toys: toy yields per bin
    for the bands (None: no bands).
    """
    translate_cats = {}
    if options.translate_cats is not None:
        with open(options.translate_cats) as f:
            translate_cats = json.load(f)
    do_bands = toys is not None
    if reduce_range is not None:
        for h in [hD, hDr, hBr, hSr, *hSB.values(), *hB.values(), *hS.values()]:
            h.GetXaxis().SetRangeUser(reduce_range[0], reduce_range[1])

    canv = ROOT.TCanvas(f"canv_{cat}", f"canv_{cat}", 700, 700)
    pad1 = ROOT.TPad(f"pad1_{cat}", f"pad1_{cat}", 0, 0.25, 1, 1)
    pad1.SetTickx()
    pad1.SetTicky()
    pad1.SetBottomMargin(0.18)
    pad1.SetLeftMargin(0.12)
    pad1.Draw()
    pad2 = ROOT.TPad(f"pad2_{cat}", f"pad2_{cat}", 0, 0, 1, 0.35)
    pad2.SetTickx()
    pad2.SetTicky()
    pad2.SetTopMargin(0.03)
    pad2.SetBottomMargin(0.25)
    pad2.SetLeftMargin(0.12)
    pad2.Draw()
    pad_size_ratio = 0.75 / 0.35
    ROOT.TGaxis.SetMaxDigits(4)
    ROOT.TGaxis.SetExponentOffset(-0.05, 0.00, "y")

    pad1.cd()
    h_axes = hD.Clone()
    h_axes.Reset()
    h_axes.SetMaximum((hD.GetMaximum() + hD.GetBinError(hD.GetMaximumBin())) * (1.4 if do_bands else 1.3))
    h_axes.SetMinimum(0.0)
    h_axes.SetTitle("")
    h_axes.GetXaxis().SetTitle("")
    h_axes.GetXaxis().SetLabelSize(0)
    h_axes.GetYaxis().SetTitleSize(0.05)
    h_axes.GetYaxis().SetTitle("S/(S+B) Weighted Events / GeV" if cat == "wall" else "Events / GeV")
    h_axes.GetYaxis().SetTitleOffset(1.1)
    h_axes.GetYaxis().SetLabelSize(0.035)
    h_axes.GetYaxis().SetLabelOffset(0.007)
    h_axes.Draw()
    if do_bands:
        bands = {key: ROOT.TGraphAsymmErrors() for key in ("1sig", "2sig", "1sig_r", "2sig_r")}
        for i, ibin in enumerate(range(h_axes.GetXaxis().GetFirst(), h_axes.GetNbinsX() + 1)):
            xval = h_axes.GetXaxis().GetBinCenter(ibin)
            xerr = 0.5 * (h_axes.GetXaxis().GetBinWidth(ibin))
            bkgval = hB["nBins"].GetBinContent(ibin)
            props = band_properties(toys, cat, ibin)
            for n in ("1", "2"):
                down, up = props["median"] - props[f"down{n}sigma"], props[f"up{n}sigma"] - props["median"]
                bands[f"{n}sig"].SetPoint(i, xval, props["median"])
                bands[f"{n}sig_r"].SetPoint(i, xval, props["median"] - bkgval)
                bands[f"{n}sig"].SetPointError(i, xerr, xerr, down, up)
                bands[f"{n}sig_r"].SetPointError(i, xerr, xerr, down, up)
        for key, band in bands.items():
            band.SetFillColor(ROOT.kGreen + 1 if key.startswith("1") else ROOT.kOrange)
            band.SetFillStyle(1001)
        bands["2sig"].Draw("LE3SAME")
        bands["1sig"].Draw("LE3SAME")
    _draw_pdfs(hSB["pdfNBins"], hB["pdfNBins"], hS["pdfNBins"], options)
    data_graphs = {"data": _data_graph(hD, hD)}  # alive until the canvas is saved

    leg = ROOT.TLegend(0.58, 0.46, 0.86, 0.76) if do_bands else ROOT.TLegend(0.58, 0.52, 0.86, 0.76)
    leg.SetFillStyle(0)
    leg.SetLineColor(0)
    leg.SetTextSize(0.045)
    leg.AddEntry(data_graphs["data"], "Data", "ep")
    if options.unblind:
        leg.AddEntry(hSB["pdfNBins"], "S+B fit", "l")
        leg.AddEntry(hB["pdfNBins"], "B component", "l")
    else:
        leg.AddEntry(hB["pdfNBins"], "B fit", "l")
        leg.AddEntry(hS["pdfNBins"], "S model", "fl")
    if do_bands:
        leg.AddEntry(bands["1sig"], "#pm1 #sigma", "F")
        leg.AddEntry(bands["2sig"], "#pm2 #sigma", "F")
    leg.Draw("Same")

    lat0 = ROOT.TLatex()
    lat0.SetTextFont(42)
    lat0.SetTextAlign(11)
    lat0.SetNDC()
    lat0.SetTextSize(0.06)
    lat0.DrawLatex(0.12, 0.92, "#bf{CMS} #it{Preliminary}")
    lat0.DrawLatex(0.565, 0.92, options.lumi_label)
    lat0.DrawLatex(0.6, 0.8, "#scale[0.6]{%s}" % translate_cats.get(cat, cat))
    lat0.DrawLatex(0.15, 0.83, "#scale[0.75]{H #rightarrow #gamma#gamma, m_{H} = 125.07 GeV}")
    if "PseudoToy" in options.input_ws_file:
        lat0.DrawLatex(0.15, 0.76, "#scale[0.75]{Pseudo data}")
    if options.load_snapshot is not None:
        if options.poi != "":
            lat0.DrawLatex(0.15, 0.77, "#scale[0.75]{#hat{#mu} = %.2f}" % workspace.var(options.poi).getVal())
        else:
            lat0.DrawLatex(0.15, 0.77, "#scale[0.6]{#vec{#alpha} = (#mu_{ggH}, #mu_{VBF}, #mu_{VH}, #mu_{top})}")
    else:
        lat0.DrawLatex(0.15, 0.77, "#scale[0.75]{#mu = 1.0}")

    # background subtracted
    pad2.cd()
    h_axes_ratio = hDr.Clone()
    h_axes_ratio.Reset()
    h_axes_ratio.SetMaximum(
        max((hDr.GetMaximum() + hDr.GetBinError(hDr.GetMaximumBin())) * 1.5, hSr.GetMaximum() * 1.2)
    )
    h_axes_ratio.SetMinimum((hDr.GetMinimum() - hDr.GetBinError(hDr.GetMinimumBin())) * 1.3)
    h_axes_ratio.SetTitle("")
    h_axes_ratio.GetXaxis().SetTitleSize(0.05 * pad_size_ratio)
    h_axes_ratio.GetXaxis().SetLabelSize(0.035 * pad_size_ratio)
    h_axes_ratio.GetXaxis().SetLabelOffset(0.007)
    h_axes_ratio.GetXaxis().SetTickLength(0.03 * pad_size_ratio)
    h_axes_ratio.GetYaxis().SetLabelSize(0.035 * pad_size_ratio)
    h_axes_ratio.GetYaxis().SetLabelOffset(0.007)
    h_axes_ratio.GetYaxis().SetTitle("")
    h_axes_ratio.Draw()
    if do_bands:
        bands["2sig_r"].Draw("LE3SAME")
        bands["1sig_r"].Draw("LE3SAME")
    _draw_pdfs(hSr, hBr, hSr, options)
    data_graphs["ratio"] = _data_graph(hD, hDr)

    lat1 = ROOT.TLatex()
    lat1.SetTextFont(42)
    lat1.SetTextAlign(33)
    lat1.SetNDC(1)
    lat1.SetTextSize(0.045 * pad_size_ratio)
    lat1.DrawLatex(0.87, 0.91, "B component subtracted")

    canv.Update()
    ext_label = options.ext.lstrip("_")
    prefix = f"{ext_label}_" if ext_label else ""
    for suffix in ("png", "pdf"):
        canv.SaveAs(f"./SplusBModels{options.ext}/{prefix}{cat}_{options.xvar_name}.{suffix}")


class _MassVariables:
    """The mass variable of the categories, with the Mjj_90GeV variable of the HH categories for --doHHMjjFix."""

    def __init__(self, workspace, options: SplusBOptions):
        name, title, unit = options.xvar.split(",")[0], options.xvar.split(",")[1], options.xvar.split(",")[2]
        self.weight = ROOT.RooRealVar("weight", "weight", 0)
        self.xvar = self._setup(workspace.var(name), title, unit)
        self.fix = None
        if options.do_hh_mjj_fix:
            self.fix = self._setup(workspace.var(f"{name}_90GeV"), title, unit)
            ratio = (self.fix.getMax() - self.fix.getMin()) / (self.xvar.getMax() - self.xvar.getMin())
            logger.info(f"HH fix: using Mjj_90GeV variable for cats: {','.join(HH_FIX_CATS)}")
            for option in ("n_bins", "pdf_n_bins"):
                if not (ratio * getattr(options, option)) % 1 == 0:
                    raise ValueError(f"{option} for Mjj_90GeV is not an integer. Please use an appropriate {option}")
        self.sets = {var: self._sets(var) for var in (self.xvar, self.fix) if var is not None}

    @staticmethod
    def _setup(var, title, unit):
        var.SetTitle(title)
        var.setPlotLabel(title)
        var.setUnit(unit)
        return var

    def _sets(self, var):
        return {
            "arglist": ROOT.RooArgList(var),
            "argset": ROOT.RooArgSet(var),
            "wargset": ROOT.RooArgSet(var, self.weight),
        }

    def var(self, cat: str):
        return self.fix if self.fix is not None and cat in HH_FIX_CATS else self.xvar

    def argset(self, cat: str, kind: str = "argset"):
        return self.sets[self.var(cat)][kind]


def category_weights(workspace, cats: list[str], mass: _MassVariables, options: SplusBOptions) -> dict:
    """S/(S+B) in the +-effSigma window around MH per category, normalised to keep the total signal yield."""
    sb_model, b_model = workspace.pdf("model_s"), workspace.pdf("model_b")
    weights, s_tot, s_weighted = {}, 0, 0
    for c in cats:
        var, argset = mass.var(c), mass.argset(c)
        sbpdf, bpdf = sb_model.getPdf(c), b_model.getPdf(c)
        h_sb = sbpdf.createHistogram(f"h_sb_tmp_pdfNBins_{c}", var, ROOT.RooFit.Binning(options.pdf_n_bins))
        h_b = bpdf.createHistogram(f"h_b_tmp_pdfNBins_{c}", var, ROOT.RooFit.Binning(options.pdf_n_bins))
        sb, b = sbpdf.expectedEvents(argset), bpdf.expectedEvents(argset)
        s = sb - b
        if options.do_bkg_renormalization:
            # renormalise the B pdf to S+B - S
            h_sb.Scale(b / sb)
            h_b.Scale((b - s) / b)
            b = b - s
        sigma = eff_sigma(h_sb - h_b)
        range_name = f"effSigma_{c}"
        mh = workspace.var("MH").getVal()
        # the same in all categories, MH is shared between them
        logger.info("In category %s, the peak position is fitted to %.2f." % (c, mh))
        mass.xvar.setRange(range_name, mh - sigma, mh + sigma)
        b_eff = bpdf.createIntegral(argset, argset, range_name).getVal() * b
        s_eff = math.erf(1.0 / math.sqrt(2)) * s
        weights[c] = s_eff / (s_eff + b_eff)
        logger.info(
            "   * %s: S = %.2f, B = %.2f --> effSigma = %.2f, S_eff = %.2f, B_eff = %.2f"
            % (c, s, b, sigma, s_eff, b_eff)
        )
        s_tot += s
        s_weighted += s * weights[c]
    for c in cats:
        weights[c] *= s_tot / s_weighted
    return weights


def toy_yields(
    toy_files: list[str], cats: list[str], chan, mass: _MassVariables, weights: dict, options
) -> pd.DataFrame:
    """Yields per bin (and of the (weighted) sum of the categories) of every toy, one row per toy."""
    n_bins = range(1, options.n_bins + 1)
    columns = [f"{cat}_{ibin:g}" for cat in cats for ibin in n_bins]
    if options.do_sum_categories:
        columns += [f"sum_{ibin:g}" for ibin in n_bins]
        if options.do_cat_weights:
            columns += [f"wsum_{ibin:g}" for ibin in n_bins]
    df = pd.DataFrame(columns=columns)
    xvar = mass.xvar
    for itoy, toy_file in enumerate(toy_files):
        logger.info(" --> Processing toy (%g/%g) ::: %s" % (itoy, len(toy_files), toy_file))
        f = ROOT.TFile(toy_file)
        toy = f.Get("toys/toy_asimov")
        veto = False
        values = dict.fromkeys(columns, 0)
        for cidx in range(chan.numTypes()):
            chan.setIndex(cidx)
            c = chan.getLabel()
            if not options.selected(c):
                continue
            dtoy = toy.reduce("CMS_channel==%g" % cidx)
            htoy = mass.var(c).createHistogram(
                f"h_{c}", ROOT.RooFit.Binning(options.n_bins, xvar.getMin(), xvar.getMax())
            )
            dtoy.fillHistogram(htoy, mass.argset(c, "arglist"))
            for ibin in range(1, htoy.GetNbinsX() + 1):
                v = htoy.GetBinContent(ibin)
                if v != v:  # NaN
                    veto = True
                values[f"{c}_{ibin:g}"] = v
                if options.do_sum_categories:
                    values[f"sum_{ibin:g}"] += v
                    if options.do_cat_weights:
                        values[f"wsum_{ibin:g}"] += v * weights[c]
            if options.do_toy_veto and values[f"{c}_1"] == 0:
                veto = True
            htoy.Delete()
            dtoy.Delete()
        toy.Delete()
        f.Close()
        if veto:
            logger.info("   --> Toy veto: zero entries in first bin")
        else:
            df.loc[len(df)] = values
    return df


def _sum(total, hists):
    """Add the histograms (a dict of them, or one) to the sum, starting it with clones."""
    if isinstance(hists, dict):
        return {b: _sum(None if total is None else total[b], h) for b, h in hists.items()}
    if total is None:
        return hists.Clone()
    total.Add(hists)
    return total


def _data_histogram(name, data, var, arglist, binning, options: SplusBOptions):
    """Histogram of the data (outside the blinded region), scaled by data_scaler."""
    hist = var.createHistogram(name, ROOT.RooFit.Binning(*binning))
    hist.SetBinErrorOption(ROOT.TH1.kPoisson)
    if not options.unblind:
        low, high = options.blinding
        data = data.reduce("%s<%f|%s>%f" % (var.GetName(), low, var.GetName(), high))
    data.fillHistogram(hist, arglist)
    hist.Scale(options.data_scaler)
    return hist


def _background_subtracted(h_data, h_bpdf, options: SplusBOptions):
    ratio = h_data.Clone()
    ratio.Reset()
    for ibin in range(1, h_data.GetNbinsX() + 1):
        center = h_data.GetBinCenter(ibin)
        if not options.unblind and options.blinding[0] < center < options.blinding[1]:
            continue
        ratio.SetBinContent(ibin, h_data.GetBinContent(ibin) - h_bpdf.GetBinContent(ibin))
        ratio.SetBinError(ibin, h_data.GetBinError(ibin))
    return ratio


def make_splusb_plots(options: SplusBOptions) -> None:
    """S+B plots of every selected category and (with do_sum_categories) of their (weighted) sum."""
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    logger.info(f"Opening workspace: {options.input_ws_file}")
    f = ROOT.TFile(options.input_ws_file)
    w = f.Get("w")
    if options.load_snapshot is not None:
        logger.info(f"Loading snapshot: {options.load_snapshot}")
        w.loadSnapshot(options.load_snapshot)
    if options.parameter_map is not None:
        if options.load_snapshot is not None:
            logger.warning("Already loaded snapshot. Also setting parameters from the parameter map")
        for kv in options.parameter_map.split(","):
            w.var(kv.split(":")[0]).setVal(float(kv.split(":")[1]))
    blinding = options.blinding
    if options.mass is not None:
        w.var("MH").setVal(float(options.mass))
    mass = _MassVariables(w, options)
    xvar = mass.xvar
    chan = w.cat("CMS_channel")
    sb_model, b_model = w.pdf("model_s"), w.pdf("model_b")

    d_obs = w.data("data_obs")
    data_cats, wdata_cats = {}, {}
    for cidx in range(chan.numTypes()):
        chan.setIndex(cidx)
        c = chan.getLabel()
        if options.selected(c):
            data_cats[c] = ROOT.RooDataSet(f"d_{c}", f"d_{c}", mass.argset(c))
            wdata_cats[c] = ROOT.RooDataSet(f"wd_{c}", f"wd_{c}", mass.argset(c, "wargset"), "weight")
    cats = list(data_cats)

    weights = {}
    if options.load_weights != "":
        logger.info(f"Loading category S/S+B weights from json file: {options.load_weights}")
        with open(options.load_weights) as jf:
            weights = json.load(jf)
        missing = [c for c in cats if c not in weights]
        if missing:
            raise KeyError(f"Categories {missing} are not in the S/S+B weights json {options.load_weights}")
    elif options.do_cat_weights:
        logger.info("Extracting S/S+B weights for categories")
        weights = category_weights(w, cats, mass, options)
        if options.save_weights:
            path = f"./jsons/catsWeights_sospb{options.ext}_{options.xvar_name}.json"
            logger.info(f"Saving S/S+B weights to json file: {path}")
            os.makedirs("./jsons", exist_ok=True)
            with open(path, "w") as jf:
                json.dump(weights, jf)

    logger.info("Extracting datasets")
    # one entry per event of the (binned) data
    for i in range(d_obs.numEntries()):
        p = d_obs.get(i)
        c = p.getCatLabel("CMS_channel")
        if not options.selected(c):
            continue
        n_entries = int(d_obs.weight())
        for _ in range(n_entries):
            data_cats[c].add(p)
        if options.do_cat_weights:
            for _ in range(n_entries):
                wdata_cats[c].add(p, weights[c])

    toys = None
    if options.do_bands:
        if options.load_toy_yields != "":
            logger.info(f"Loading toy yields for +-1/2sigma bands from {options.load_toy_yields}")
            with open(options.load_toy_yields, "rb") as fD:
                toys = pickle.load(fD)
        else:
            logger.info("Extracting toy yields for +-1/2sigma bands")
            toy_files = glob.glob(f"./SplusBModels{options.ext}/toys/toy_*.root")
            if len(toy_files) == 0:
                logger.error(f"No toys files of form ./SplusBModels{options.ext}/toys/toy_*.root. Skipping bands")
            else:
                toys = toy_yields(toy_files, cats, chan, mass, weights, options)
                if options.save_toy_yields:
                    path = f"SplusBModels{options.ext}/toyYields_{options.xvar_name}.pkl"
                    logger.info(f"Saving toy yields to: {path}")
                    with open(path, "wb") as fD:
                        pickle.dump(toys, fD)

    binning = (options.n_bins, xvar.getMin(), xvar.getMax())
    sums = None
    for c in cats:
        var, argset, arglist = mass.var(c), mass.argset(c), mass.argset(c, "arglist")
        reduce_range = [mass.fix.getMin(), mass.fix.getMax()] if var is not xvar else None
        logger.info(f"Processing category: {c}")
        h_data = _data_histogram(f"h_data_{c}", data_cats[c], var, arglist, binning, options)
        if options.do_cat_weights:
            h_wdata = _data_histogram(f"h_wdata_{c}", wdata_cats[c], var, arglist, binning, options)
        if options.do_zeroes:
            # error of +-1 event on the empty bins outside the blinded region, to show them
            for ibin in range(1, h_data.GetNbinsX() + 1):
                center = h_data.GetBinCenter(ibin)
                if not options.unblind and blinding[0] < center < blinding[1]:
                    continue
                if h_data.GetBinContent(ibin) == 0.0:
                    h_data.SetBinError(ibin, scipy.stats.gamma.interval(0.68, 1)[1])
                    if options.do_cat_weights:
                        h_wdata.SetBinError(ibin, weights[c])

        sbpdf, bpdf = sb_model.getPdf(c), b_model.getPdf(c)
        pdf_binnings = {key: ROOT.RooFit.Binning(n, xvar.getMin(), xvar.getMax()) for key, n in
                        (("pdfNBins", options.pdf_n_bins), ("nBins", options.n_bins))}  # fmt: skip
        h_sbpdf, h_bpdf = (
            {
                key: pdf.createHistogram(f"h_{name}_{key}_{c}", var, pdf_binning)
                for key, pdf_binning in pdf_binnings.items()
            }
            for name, pdf in (("sb", sbpdf), ("b", bpdf))
        )
        sb, b = sbpdf.expectedEvents(argset), bpdf.expectedEvents(argset)
        s = sb - b
        if options.do_bkg_renormalization:
            # renormalise the B pdf to S+B - S
            for h in h_sbpdf.values():
                h.Scale(b / sb)
            for h in h_bpdf.values():
                h.Scale((b - s) / b)
            b = b - s
        logger.info("    * Yield for category: S = %.2f, B=%.2f" % (s, b))
        h_spdf = {key: h_sbpdf[key] - h_bpdf[key] for key in h_sbpdf}
        # pdf histograms in events per GeV, as the data
        xvar_range = int(xvar.getBinning().highBound() - xvar.getBinning().lowBound())
        if options.n_bins != xvar_range:
            for h in [*h_sbpdf.values(), *h_bpdf.values(), *h_spdf.values()]:
                h.Scale(float(xvar_range) / options.n_bins)

        hists = {
            "data": h_data,
            "sb": h_sbpdf,
            "b": h_bpdf,
            "s": h_spdf,
            "data_ratio": _background_subtracted(h_data, h_bpdf["nBins"], options),
            "b_ratio": h_bpdf["pdfNBins"] - h_bpdf["pdfNBins"],
            "s_ratio": h_spdf["pdfNBins"].Clone(),
        }
        if options.do_cat_weights:
            # S/(S+B) weighted
            weighted = {key: {b: h.Clone() for b, h in hists[key].items()} for key in ("sb", "b", "s")}
            for hs in weighted.values():
                for h in hs.values():
                    h.Scale(weights[c])
            hists["wdata"] = h_wdata
            hists.update({f"w{key}": value for key, value in weighted.items()})
            hists["wb_ratio"] = weighted["b"]["pdfNBins"] - weighted["b"]["pdfNBins"]
            hists["ws_ratio"] = weighted["s"]["pdfNBins"].Clone()
            hists["wdata_ratio"] = _background_subtracted(h_wdata, weighted["b"]["nBins"], options)

        if options.several_cats and options.do_sum_categories:
            sums = {key: _sum(None if sums is None else sums[key], h) for key, h in hists.items()}

        if not options.skip_individual_cat_plots:
            os.makedirs(f"./SplusBModels{options.ext}", exist_ok=True)
            plot_splusb(w, *(hists[k] for k in ("data", "sb", "b", "s", "data_ratio", "b_ratio", "s_ratio")), c,
                        options, toys, reduce_range)  # fmt: skip

        for h in hists.values():
            for hist in h.values() if isinstance(h, dict) else [h]:
                hist.Delete()

    if options.several_cats and options.do_sum_categories:
        reduce_range = [mass.fix.getMin(), mass.fix.getMax()] if mass.fix is not None else None
        os.makedirs(f"./SplusBModels{options.ext}", exist_ok=True)
        logger.info("Making plot for sum of categories")
        keys = ("data", "sb", "b", "s", "data_ratio", "b_ratio", "s_ratio")
        plot_splusb(w, *(sums[k] for k in keys), "all", options, toys, reduce_range)
        if options.do_cat_weights:
            logger.info("Making weighted plot for sum of categories")
            plot_splusb(w, *(sums[f"w{k}"] for k in keys), "wall", options, toys, reduce_range)
