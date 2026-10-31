"""Plots of the signal modelling (ROOT)."""

from __future__ import annotations

import os

import ROOT

LINE_COLORS = {1: ROOT.kAzure + 1, 2: ROOT.kRed - 4, 3: ROOT.kGreen + 2, 4: ROOT.kMagenta - 9, 5: ROOT.kOrange}


def _draw_texts(label: str) -> None:
    latex = ROOT.TLatex()
    latex.SetTextFont(42)
    latex.SetTextAlign(31)
    latex.SetNDC()
    latex.SetTextSize(0.03)
    latex.DrawLatex(0.9, 0.92, label)
    cms = ROOT.TLatex()
    cms.SetNDC()
    cms.SetTextFont(42)
    cms.SetTextSize(0.03)
    cms.DrawLatex(ROOT.gPad.GetLeftMargin(), 0.91, "#it{Private Work (CMS data/simulation)}")


def _save(canvas, path_without_extension: str) -> None:
    os.makedirs(os.path.dirname(path_without_extension), exist_ok=True)
    canvas.Update()
    canvas.SaveAs(f"{path_without_extension}.png")
    canvas.SaveAs(f"{path_without_extension}.pdf")


def plot_ftest(fits: dict, best: int, out_dir: str, scenario: str, proc: str, cat: str, mass: str, label: str = ""):
    """The fitted shapes with 1..N Gaussians on top of the simulation."""
    canvas = ROOT.TCanvas()
    canvas.SetLeftMargin(0.15)
    hists, h_max, h_min = {}, 0, 0
    for n, fit in fits.items():
        fit.mh.setVal(int(mass))
        hist = fit.pdfs["final"].createHistogram(f"h_nGauss_{n}_{scenario}", fit.xvar, ROOT.RooFit.Binning(1600))
        hist.SetLineWidth(3 if n == best else 1)
        hist.SetLineColor(LINE_COLORS[n])
        hist.SetTitle("")
        hist.GetXaxis().SetTitle("m_{#gamma#gamma} (GeV)")
        hist.SetMinimum(0)
        h_max, h_min = max(h_max, hist.GetMaximum()), min(h_min, hist.GetMinimum())
        hist.GetXaxis().SetRangeUser(115, 140)
        hists[n] = hist
    data = fit.xvar.createHistogram(f"h_data{scenario}", ROOT.RooFit.Binning(fit.n_bins))
    fit.data_hists[mass].fillHistogram(data, ROOT.RooArgList(fit.xvar))
    data.Scale(float(fit.n_bins) / 1600)
    data.SetMarkerStyle(20)
    data.SetMarkerColor(1)
    data.SetLineColor(1)
    data.SetTitle("")
    data.GetXaxis().SetTitle("m_{#gamma#gamma} (GeV)")
    data.SetMinimum(0)
    data.GetXaxis().SetRangeUser(115, 140)
    h_max, h_min = max(h_max, data.GetMaximum()), min(h_min, data.GetMinimum())
    data.SetMaximum(1.2 * h_max)
    data.SetMinimum(1.2 * h_min)
    data.Draw("PE")
    for hist in hists.values():
        hist.Draw("HIST SAME")

    legend = ROOT.TLegend(0.55, 0.3, 0.86, 0.8)
    legend.SetFillStyle(0)
    legend.SetLineColor(0)
    legend.SetTextSize(0.03)
    legend.AddEntry(data, "Simulation", "ep")
    for n, fit in fits.items():
        head = f"#bf{{N_{{gauss}} = {n}}}" if n == best else f"N_{{gauss}} = {n}"
        legend.AddEntry(hists[n], f"{head}: #chi^{{2}}/n(dof) = {fit.reduced_chi2():.4f}", "L")
    legend.Draw("Same")
    _draw_texts(label or f"( {scenario} , {proc} , {cat} )")
    _save(canvas, f"{out_dir}/fTest_{cat}_{proc}_{scenario}")


def plot_ftest_results(fits: dict, best: int, out_dir: str, scenario: str, proc: str, cat: str, mass: str, label=""):
    """chi2/ndof against the number of Gaussians."""
    canvas = ROOT.TCanvas()
    graph = ROOT.TGraph()
    x_max, y_max = 1, -1
    for n, fit in fits.items():
        fit.mh.setVal(int(mass))
        y = fit.reduced_chi2()
        x_max, y_max = max(x_max, n), max(y_max, y)
        graph.SetPoint(graph.GetN(), n, y)
    axes = ROOT.TH1F(f"h_axes_{proc}_{scenario}", f"h_axes_{proc}_{scenario}", x_max + 1, 0, x_max + 1)
    axes.SetTitle("")
    axes.GetXaxis().SetTitle("N_{gauss}")
    axes.GetXaxis().SetTitleSize(0.05)
    axes.GetXaxis().SetTitleOffset(0.85)
    axes.GetXaxis().SetLabelSize(0.035)
    axes.GetYaxis().SetTitle("#chi^{2} / n(dof)")
    axes.GetYaxis().SetTitleOffset(0.85)
    axes.GetYaxis().SetTitleSize(0.05)
    axes.SetMaximum(1.2 * y_max)
    axes.SetMinimum(0)
    axes.Draw()
    graph.SetLineColor({"RV": ROOT.kRed - 4, "WV": ROOT.kAzure + 1}.get(scenario, 1))
    graph.SetMarkerColor(1)
    graph.SetMarkerStyle(20)
    graph.Draw("Same PL")
    _draw_texts(label or f"( {scenario} , {proc} , {cat} )")
    latex = ROOT.TLatex()
    latex.SetTextFont(42)
    latex.SetTextAlign(31)
    latex.SetNDC()
    latex.SetTextSize(0.03)
    latex.DrawLatex(0.6, 0.75, f"Optimum N_{{gauss}} = {best}")
    _save(canvas, f"{out_dir}/fTest_{cat}_{proc}_{scenario}_chi2_vs_nGauss")


def plot_pdf_components(fit, out_dir: str, prefix: str, proc: str, cat: str, label: str = "") -> None:
    """Fitted shape at MH = 125 GeV with its Gaussian components on top of the simulation."""
    colors = {0: ROOT.kAzure + 1, 1: ROOT.kRed - 4, 2: ROOT.kOrange, 3: ROOT.kGreen + 2, 4: ROOT.kMagenta - 9}
    canvas = ROOT.TCanvas()
    canvas.SetLeftMargin(0.1)
    fit.mh.setVal(125)
    final = fit.pdfs["final"].createHistogram(f"h_final{prefix}", fit.xvar, ROOT.RooFit.Binning(1600))
    final.SetLineWidth(2)
    final.SetLineColor(1)
    final.SetTitle("")
    final.GetXaxis().SetTitle("m_{#gamma#gamma} (GeV)")
    final.SetMinimum(0)
    h_max, h_min = max(0, final.GetMaximum()), min(0, final.GetMinimum())
    final.GetXaxis().SetRangeUser(100, 150)
    data = fit.xvar.createHistogram(f"h_data{prefix}", ROOT.RooFit.Binning(fit.n_bins))
    fit.data_hists["125"].fillHistogram(data, ROOT.RooArgList(fit.xvar))
    data.SetTitle("")
    data.GetXaxis().SetTitle("m_{#gamma#gamma} (GeV)")
    data.SetMinimum(0)
    data.GetXaxis().SetRangeUser(100, 150)
    data.Scale(float(fit.n_bins) / 1600)
    data.SetMarkerStyle(20)
    data.SetMarkerColor(1)
    data.SetLineColor(1)
    h_max, h_min = max(h_max, data.GetMaximum()), min(h_min, data.GetMinimum())
    data.SetMaximum(1.2 * h_max)
    data.SetMinimum(1.2 * h_min)
    data.Draw("PE")
    final.Draw("Same HIST")

    components = {k: v for k, v in fit.pdfs.items() if k != "final"}
    hists = {}
    if len(components) != 1:
        values = fit.pdfs["final"].getComponents()
        for i, (name, pdf) in enumerate(components.items()):
            if i == 0:
                fraction = values.getRealValue("frac_g0_constrained" if "gaus" in name else "frac_constrained")
            else:
                fraction = values.getRealValue(f"{fit.proc}_{fit.cat}_recursive_fraction_{name}_{i + 1}")
            hist = pdf.createHistogram(f"h_{name}{prefix}", fit.xvar, ROOT.RooFit.Binning(1600))
            hist.Scale(fraction)
            hist.SetLineColor(colors[i])
            hist.SetLineWidth(2)
            hist.SetLineStyle(2)
            hist.Draw("HIST SAME")
            hists[name] = hist

    legend = ROOT.TLegend(0.58, 0.6, 0.86, 0.8)
    legend.SetFillStyle(0)
    legend.SetLineColor(0)
    legend.SetTextSize(0.04)
    legend.AddEntry(data, "Simulation", "ep")
    legend.AddEntry(final, "Parametric Model", "L")
    legend.Draw("Same")
    if hists:
        components_legend = ROOT.TLegend(0.6, 0.4, 0.86, 0.6)
        components_legend.SetFillStyle(0)
        components_legend.SetLineColor(0)
        components_legend.SetTextSize(0.035)
        for name, hist in hists.items():
            components_legend.AddEntry(hist, name, "L")
        components_legend.Draw("Same")
    _draw_texts(label or f"( {fit.name} , {proc} , {cat} )")
    latex = ROOT.TLatex()
    latex.SetTextFont(42)
    latex.SetTextAlign(11)
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.65, 0.3, f"#chi^{{2}}/n(dof) = {fit.chi2() / fit.ndof:.4f}")
    _save(canvas, f"{out_dir}/{prefix}shape_pdf_components_{proc}_{cat}")


INTERPOLATION_MASSES = tuple(str(m) for m in range(120, 131))


def plot_interpolation(model, out_dir: str, label: str = "", mass_points=INTERPOLATION_MASSES):
    """Final model normalised to the luminosity for every MH, with the simulated mass points."""
    from finalfits import analysis

    colors = [
        ROOT.kRed,
        ROOT.kCyan,
        ROOT.kBlue + 1,
        ROOT.kOrange - 3,
        ROOT.kMagenta - 7,
        ROOT.kGreen + 1,
        ROOT.kYellow - 7,
        ROOT.kViolet + 6,
        ROOT.kTeal + 1,
        ROOT.kPink + 1,
        ROOT.kAzure + 1,
    ]
    canvas = ROOT.TCanvas()
    model.int_lumi.setVal(analysis.lumiScaleFactor * float(analysis.lumiMap[model.year]))
    hists, keep, h_max = {}, [], 0.0001
    n_bins = model.xvar.getBins()
    for color, mass in zip(colors, mass_points):
        model.mh.setVal(int(mass))
        hist = model.pdfs["final"].createHistogram(f"h_{mass}", model.xvar, ROOT.RooFit.Binning(3200))
        norm = model.functions["final_normThisLumi"].getVal()
        hist.Scale(0.0 if norm == 0.0 else (norm * 3200) / (hist.Integral() * n_bins))
        simulated = mass in model.out_datasets
        hist.SetLineWidth(2 if simulated else 1)
        if not simulated:
            hist.SetLineStyle(2)
        hist.SetLineColor(color)
        hist.SetTitle("")
        h_max = max(h_max, hist.GetMaximum())
        hists[mass] = hist
        if simulated:
            binned = ROOT.RooDataHist(f"dh_{mass}", f"dh_{mass}", ROOT.RooArgSet(model.xvar), model.out_datasets[mass])
            data = model.xvar.createHistogram(f"h_data_{mass}", ROOT.RooFit.Binning(n_bins))
            binned.fillHistogram(data, ROOT.RooArgList(model.xvar))
            data.Scale(0 if norm == 0.0 else norm / data.Integral())
            data.SetMarkerStyle(20)
            data.SetMarkerColor(color)
            data.SetLineColor(color)
            hists[f"data_{mass}"] = data
            keep.append(binned)
    axes = next(iter(hists.values())).Clone()
    axes.GetXaxis().SetTitle("m_{#gamma#gamma} (GeV)")
    axes.GetYaxis().SetTitle("Events / %.2f GeV" % ((model.xvar.getMax() - model.xvar.getMin()) / n_bins))
    axes.GetYaxis().SetTitleOffset(0.9)
    axes.SetMinimum(0)
    axes.SetMaximum(h_max * 1.2)
    axes.GetXaxis().SetRangeUser(100, 150)
    axes.Draw("AXIS")
    for name, hist in hists.items():
        hist.Draw("Same EP" if "data" in name else "Same HIST")
    _draw_texts(label or f"({model.name})")
    _save(canvas, f"{out_dir}/{model.name}_model_vs_mH")


def plot_splines(
    model, out_dir: str, label: str = "", nominal_mass: str = "125", splines=("xs", "br", "ea", "fracRV")
) -> None:
    """XS, BR, eff x acc, RV fraction and the expected yield against MH, relative to 125 GeV."""
    from finalfits import analysis
    from finalfits.signal_fit import MH_HIGH, MH_LOW

    colors = {
        "xs": ROOT.kRed - 4,
        "br": ROOT.kAzure + 1,
        "ea": ROOT.kGreen + 1,
        "fracRV": ROOT.kMagenta - 7,
        "norm": ROOT.kBlack,
    }
    canvas = ROOT.TCanvas()
    graphs = {"norm": ROOT.TGraph(), **{s: ROOT.TGraph() for s in splines}}
    model.mh.setVal(float(nominal_mass))
    nominal = {s: model.splines[s].getVal() for s in splines}
    model.int_lumi.setVal(analysis.lumiScaleFactor * float(analysis.lumiMap[model.year]))
    nominal["norm"] = model.functions["final_normThisLumi"].getVal()
    y_max, y_min = 0, 0.5
    for point, mh in enumerate(range(int(MH_LOW), int(MH_HIGH) + 1)):
        model.mh.setVal(float(mh))
        values = {s: model.splines[s].getVal() for s in splines}
        values["norm"] = model.functions["final_normThisLumi"].getVal()
        for name in [*splines, "norm"]:
            ratio = 1.0 if nominal[name] == 0.0 else values[name] / nominal[name]
            graphs[name].SetPoint(point, mh, ratio)
            y_max, y_min = max(y_max, ratio), min(y_min, ratio)
    axes = ROOT.TH1F("h_axes_spl", "h_axes_spl", int(MH_HIGH) - int(MH_LOW), int(MH_LOW), int(MH_HIGH))
    axes.SetTitle("")
    axes.GetXaxis().SetTitle("m_{H} (GeV)")
    axes.GetXaxis().SetTitleSize(0.05)
    axes.GetXaxis().SetTitleOffset(0.85)
    axes.GetXaxis().SetLabelSize(0.035)
    axes.GetYaxis().SetTitle("X/X(m_{H}=125)")
    axes.GetYaxis().SetTitleOffset(0.85)
    axes.GetYaxis().SetTitleSize(0.05)
    axes.SetMaximum(1.2 * y_max)
    axes.SetMinimum(y_min)
    axes.Draw()
    legend = ROOT.TLegend(0.15, 0.15, 0.4, 0.4)
    legend.SetFillStyle(0)
    legend.SetLineColor(0)
    legend.SetTextSize(0.04)
    texts = {
        "norm": f"N_{{exp}}: @{nominal_mass} = {nominal['norm']:.2f}/pb",
        "xs": f"#sigma: @{nominal_mass} = {nominal.get('xs', 0):.2f} pb",
        "br": f"#bf{{#it{{#Beta}}}}: @{nominal_mass} = {100 * nominal.get('br', 0):.2f}%",
        "ea": f"#epsilon x #it{{#Alpha}}: @{nominal_mass} = {100 * nominal.get('ea', 0):.2f}%",
        "fracRV": f"RV fraction: @{nominal_mass} = {100 * nominal.get('fracRV', 0):.2f}%",
    }
    for name, graph in graphs.items():
        graph.SetLineColor(colors[name])
        graph.SetMarkerColor(colors[name])
        graph.SetMarkerStyle(20)
        graph.Draw("Same PL")
        legend.AddEntry(graph, texts[name])
    legend.Draw("Same")
    graphs["norm"].Draw("Same PL")
    _draw_texts(label or f"({model.name})")
    _save(canvas, f"{out_dir}/{model.name}_splines")
