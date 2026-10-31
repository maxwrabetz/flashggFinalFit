"""Plots of the background fTest (ROOT): single fits, truth functions, RooMultiPdf envelope and toys."""

from __future__ import annotations

import logging
import math

import ROOT

from finalfits import cms_style

logger = logging.getLogger(__name__)
RF = ROOT.RooFit

MGG_LOW, MGG_HIGH = 100, 180
BLIND_LOW, BLIND_HIGH = 115, 135
FAMILY_LABELS = {
    "bern": "Bernstein polynomial",
    "exp": "Sum of exponentials",
    "pow": "Power-law sum",
    "lau": "Laurent series",
}


def configure_style(cms_label: str, int_lumi: float) -> None:
    """TDR style with one italic "<cms_label> (CMS data/simulation)" label and the luminosity."""
    ROOT.gROOT.SetBatch(True)
    cms_style.setTDRStyle()
    cms_style.cmsText = f"{cms_label} (CMS data/simulation)"
    cms_style.cmsTextFont = cms_style.extraTextFont
    cms_style.cmsTextSize = cms_style.toFloat(cms_style.cmsTextSize * cms_style.extraOverCmsTextSize)
    cms_style.writeExtraText = False
    cms_style.lumi_sqrtS = "13 TeV"
    cms_style.lumi_13p6TeV = f"{int_lumi:.0f} fb^{{-1}}"


def _owned(obj):
    ROOT.SetOwnership(obj, True)
    return obj


def _plot_data(mass, data, blind: bool, fit_range: str):
    """Frame with the data in 1 GeV bins, only the sidebands if blinded."""
    frame = _owned(mass.frame())
    binning = RF.Binning(MGG_HIGH - MGG_LOW)
    # the ranges end up in the multipdf workspace with the mass variable
    mass.setRange("unblindReg_1", MGG_LOW, BLIND_LOW)
    mass.setRange("unblindReg_2", BLIND_HIGH, MGG_HIGH)
    if blind:
        data.plotOn(frame, binning, RF.CutRange("unblindReg_1"))
        data.plotOn(frame, binning, RF.CutRange("unblindReg_2"))
        data.plotOn(frame, binning, RF.Invisible())
    else:
        data.plotOn(frame, binning, RF.Range(fit_range))
    return frame


def _plot_pdf(frame, pdf, entries: float, fit_range: str, *options):
    pdf.plotOn(
        frame,
        RF.Range(float(MGG_LOW), float(MGG_HIGH)),
        RF.NormRange(fit_range),
        RF.Normalization(entries, ROOT.RooAbsReal.NumEvent),
        *options,
    )
    return frame.getObject(int(frame.numItems() - 1))


def _save(canvas, path_without_extension: str) -> None:
    canvas.SaveAs(f"{path_without_extension}.pdf")
    canvas.SaveAs(f"{path_without_extension}.png")


def plot_fit(mass, pdf, data, path: str, gof, prob: float, status: int, blind: bool, fit_range: str) -> None:
    """One fitted pdf with its parameters and goodness of fit (path ends with .pdf, a .png is added)."""
    frame = _plot_data(mass, data, blind, fit_range)
    canvas = ROOT.TCanvas()
    _plot_pdf(frame, pdf, data.sumEntries(), fit_range)
    pdf.paramOn(frame, RF.Layout(0.34, 0.96, 0.89), RF.Format("NEA", RF.AutoPrecision(1)))
    if blind:
        frame.SetMinimum(0.0001)
    frame.SetTitle("")
    frame.Draw()
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.03)
    latex.DrawLatex(
        0.1,
        0.92,
        f"#chi^{{2}}/dof = {gof.chi2_reduced:.3f} ({gof.chi2:.3f} / {gof.ndof}), Prob = {prob:.2f}, "
        f"Fit Status = {status} ",
    )
    _save(canvas, path[:-4] if path.endswith(".pdf") else path)


def plot_truths(mass, pdfs: dict, data, path: str, cat: str, blind: bool, fit_range: str) -> None:
    """The truth function of every family."""
    colors = [ROOT.kBlue, ROOT.kRed, ROOT.kMagenta, ROOT.kGreen + 1, ROOT.kOrange + 7, ROOT.kAzure + 10, ROOT.kBlack]
    canvas = ROOT.TCanvas()
    legend = ROOT.TLegend(0.6, 0.65, 0.88, 0.88)
    legend.SetFillColor(0)
    legend.SetLineColor(0)
    frame = _plot_data(mass, data, blind, fit_range)
    legend.AddEntry(frame.getObject(int(frame.numItems() - 1)), f"Data - {cat}", "LEP")
    style = 1
    for i, name in enumerate(sorted(pdfs)):
        color = colors[i] if i <= 6 else ROOT.kBlack
        style += i > 6
        curve = _plot_pdf(frame, pdfs[name], data.sumEntries(), fit_range, RF.LineColor(color), RF.LineStyle(style))
        legend.AddEntry(curve, name, "L")
    frame.SetTitle(f" {cat}")
    if blind:
        frame.SetMinimum(0.0001)
    frame.Draw()
    legend.Draw("same")
    cms_style.CMS_lumi(canvas, 2022, 0)
    _save(canvas, path)


def plot_multipdf(mass, multipdf, index, data, path: str, cat: str, cat_index: int, best: int, blind: bool, fit_range):
    """All pdfs of the RooMultiPdf (each refitted), with the residuals of the best fit in a lower pad."""
    colors = [ROOT.kBlue + 1, ROOT.kRed + 1, ROOT.kMagenta + 1, ROOT.kGreen + 2, ROOT.kOrange + 7, ROOT.kCyan + 2]
    colors.append(ROOT.kGray + 2)
    legend = ROOT.TLegend(0.52, 0.45, 0.94, 0.84)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.028)
    frame = _plot_data(mass, data, blind, fit_range)

    canvas = ROOT.TCanvas()
    pad1 = ROOT.TPad(f"pad1_{cat_index}", "", 0, 0.25, 1, 1)
    pad2 = ROOT.TPad(f"pad2_{cat_index}", "", 0, 0, 1, 0.35)
    pad1.SetBottomMargin(0.18)
    pad1.SetTicks(1, 1)
    pad2.SetTopMargin(0.04)
    pad2.SetBottomMargin(0.3)
    pad2.SetTicks(1, 1)
    pad1.Draw()
    pad2.Draw()
    pad1.cd()

    # Data histogram with the blinded bins zeroed, for the residuals
    data_hist = data.createHistogram(f"h_data_{cat}", mass, RF.Binning(MGG_HIGH - MGG_LOW, MGG_LOW, MGG_HIGH))
    if data_hist:
        _owned(data_hist)

    current = index.getIndex()
    label = f"Category {cat[3:]}" if cat.startswith("cat") else cat
    legend.AddEntry(frame.getObject(int(frame.numItems() - 1)), f"Data ({label})", "LEP")
    style, best_pdf, best_color = 1, None, -1
    for i in range(index.numTypes()):
        color = colors[i] if i <= 6 else ROOT.kBlack
        style += i > 6
        index.setIndex(i)
        pdf = multipdf.getCurrentPdf()
        pdf.fitTo(data, RF.Minos(False), RF.Minimizer("Minuit2", "minimize"), RF.SumW2Error(True), RF.Range(fit_range))
        width = 4 if i == best else 2
        curve = _plot_pdf(
            frame, pdf, data.sumEntries(), fit_range, RF.LineColor(color), RF.LineStyle(style), RF.LineWidth(width)
        )
        if i == best:
            best_pdf, best_color = pdf, color
        name = pdf.GetName()
        family = next((label for short, label in FAMILY_LABELS.items() if short in name), "Candidate function")
        order = name[-1] if name else "?"
        legend.AddEntry(curve, f"{family}, order {order}{' (best fit)' if i == best else ''}", "L")
    frame.SetTitle("")
    if blind:
        frame.SetMinimum(0.0001)
    frame.GetXaxis().SetLabelSize(0.0)
    frame.GetXaxis().SetTitleSize(0.0)
    frame.GetYaxis().SetTitle("Events / (1 GeV)")
    frame.GetYaxis().SetTitleOffset(1.55)
    frame.Draw()
    legend.Draw("same")
    cms_style.CMS_lumi(pad1, 2022, 0)

    # Best fit pdf in the binning of the data, scaled to the observed sideband yield
    pdf_hist = None
    if best_pdf:
        pdf_hist = best_pdf.createHistogram(f"h_pdf_{cat}", mass, RF.Binning(MGG_HIGH - MGG_LOW, MGG_LOW, MGG_HIGH))
        if pdf_hist:
            _owned(pdf_hist)
    residuals = ROOT.TGraphAsymmErrors()
    diff_max, diff_min = -1e9, 1e9
    if data_hist and pdf_hist:
        data_yield = pdf_yield = 0.0
        for b in range(1, data_hist.GetNbinsX() + 1):
            x = data_hist.GetBinCenter(b)
            if BLIND_LOW < x < BLIND_HIGH:
                for hist in (data_hist, pdf_hist):
                    hist.SetBinContent(b, 0.0)
                    hist.SetBinError(b, 0.0)
                continue
            data_yield += data_hist.GetBinContent(b)
            pdf_yield += pdf_hist.GetBinContent(b)
        if pdf_yield > 0.0:
            pdf_hist.Scale(data_yield / pdf_yield)
        for b in range(1, data_hist.GetNbinsX() + 1):
            x = data_hist.GetBinCenter(b)
            if BLIND_LOW < x < BLIND_HIGH:
                continue
            diff = data_hist.GetBinContent(b) - pdf_hist.GetBinContent(b)
            error = data_hist.GetBinError(b)
            half_width = data_hist.GetBinWidth(b) / 2.0
            point = residuals.GetN()
            residuals.SetPoint(point, x, diff)
            residuals.SetPointError(point, half_width, half_width, error, error)
            diff_max = max(diff_max, diff + error)
            diff_min = min(diff_min, diff - error)
    else:
        logger.warning("Could not build the histograms for the residuals of %s", cat)

    pad2.cd()
    axes = ROOT.TH1D("hdummyweight", "", MGG_HIGH - MGG_LOW, MGG_LOW, MGG_HIGH)
    if diff_max <= diff_min:
        diff_max, diff_min = 1.0, -1.0
    residual_max = 1.15 * max(abs(diff_max), abs(diff_min))
    axes.SetMaximum(residual_max)
    axes.SetMinimum(-residual_max)
    # pad2 is 0.35/0.75 as high as pad1: this keeps the text and ticks at the size of pad1
    text_size = 0.04 * 0.75 / 0.35
    axes.GetYaxis().SetTitle("Data - fit")
    axes.GetYaxis().SetTitleSize(text_size)
    axes.GetYaxis().SetLabelSize(text_size)
    axes.GetYaxis().SetTitleOffset(0.70)
    axes.GetYaxis().CenterTitle(True)
    axes.GetXaxis().SetTitle("m_{#gamma#gamma} (GeV)")
    axes.GetXaxis().SetTitleSize(text_size)
    axes.GetXaxis().SetLabelSize(text_size)
    axes.GetXaxis().SetTickLength(frame.GetXaxis().GetTickLength() * 0.75 / 0.35)
    axes.GetYaxis().SetTickLength(frame.GetYaxis().GetTickLength())
    axes.Draw("HIST")
    axes.GetYaxis().SetNdivisions(808)
    zero = ROOT.TLine(MGG_LOW, 0.0, MGG_HIGH, 0.0)
    zero.SetLineColor(best_color)
    zero.SetLineWidth(2)
    zero.Draw()
    residuals.Draw("PESAME")
    _save(canvas, path)
    index.setIndex(current)


def plot_gof_toys(toy_chi2: list[float], observed: float, name: str, path: str) -> None:
    """Chi2 of the goodness-of-fit toys with the observed value."""
    canvas = ROOT.TCanvas()
    center = toy_chi2[len(toy_chi2) // 2]
    width = math.sqrt(center)
    hist = ROOT.TH1F(f"gofTest_{name}.pdf", ";Chi2;", 50, center - 5 * width, center + 5 * width)
    for chi2 in toy_chi2:
        hist.Fill(chi2)
    hist.Draw()
    arrow = ROOT.TArrow(observed, hist.GetMaximum(), observed, 0)
    arrow.SetLineWidth(2)
    arrow.Draw()
    canvas.SaveAs(path)


def plot_ftest_toys(
    toy_chi2, observed, ndof, prob, prob_asymptotic, status_null, status_test, pdf_null, pdf_test, path
):
    """Chi2 of the F-test toys compared to the chi2 distribution, and the toy fit status."""
    canvas = ROOT.TCanvas()
    canvas.SetLogy()
    hist = ROOT.TH1F(f"toys_fTest_{pdf_null.GetName()}.pdf", ";Chi2;", 60, -2, 10)
    for chi2 in toy_chi2:
        hist.Fill(chi2)
    curve = ROOT.TGraph()
    curve.SetLineColor(ROOT.kGreen + 2)
    for b in range(1, hist.GetNbinsX() + 1):
        x = hist.GetBinCenter(b)
        if x > 0:
            curve.SetPoint(curve.GetN(), x, ROOT.Math.chisquared_pdf(x, ndof))
    hist.Scale(1.0 / (hist.GetBinWidth(1) * hist.Integral()))
    hist.Draw()
    arrow = ROOT.TArrow(observed, hist.GetMaximum(), observed, 0)
    arrow.SetLineWidth(2)
    arrow.Draw()
    curve.Draw("L")
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.DrawLatex(0.1, 0.91, f"Prob (asymptotic) = {prob:.4f} ({prob_asymptotic:.4f})")
    canvas.SaveAs(path)

    status_canvas = ROOT.TCanvas()
    legend = ROOT.TLegend(0.2, 0.6, 0.4, 0.87)
    legend.SetFillColor(0)
    legend.SetTextFont(42)
    hists = []
    for pdf, statuses, color, label in ((pdf_null, status_null, 2, "Null Hyp"), (pdf_test, status_test, 1, "Test Hyp")):
        status_hist = ROOT.TH1I(f"Status_{pdf.GetName()}.pdf", ";FitStatus;", 8, -4, 4)
        for status in statuses:
            status_hist.Fill(status)
        status_hist.SetLineColor(color)
        legend.AddEntry(status_hist, label, "L")
        status_hist.Draw("same" if hists else "")
        hists.append(status_hist)
    legend.Draw()
    status_canvas.SaveAs(f"{path}_fitstatus.pdf")
