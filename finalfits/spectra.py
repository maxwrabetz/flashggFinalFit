"""Differential cross-section spectra (matplotlib).

The measured cross sections are the signal strengths of the likelihood scans times the MadGraph5_aMC@NLO + NNLOPS
prediction. They are compared to the predictions of MadGraph5_aMC@NLO with and without NNLOPS and of POWHEG, in the
fiducial phase space (``fidXS_*`` theory modules) or extrapolated to the total phase space (``totXS_*``).
The binning and the predictions come from the theory modules named in the ``spectra`` block of the analysis config,
the plot layout from ``Plots/Spectra/config/<year>_<variable>.yml``.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
from functools import partial

import CombineHarvester.CombineTools.plotting as chplot
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import ROOT
from matplotlib.patches import Rectangle

from finalfits.analysis import lumiMap

logger = logging.getLogger(__name__)

HIGGS_MASS = "125.07"
SCAN_Y_CUT = 7.0  # scan points with 2*deltaNLL above are dropped
ROUNDING_DIGITS = 3
BR_UNCERTAINTY = 0.02  # relative uncertainty on BR(H->gg), added in quadrature to the theory uncertainty
CMS_LABEL = "Private work (CMS data/simulation)"
LEGEND = {
    "madgraph": "ggH (MadGraph5_aMC@NLO + NNLOPS + Pythia) + xH",
    "no_nnlops": "ggH (MadGraph5_aMC@NLO + Pythia) + xH",
    "powheg": "ggH (POWHEG + Pythia) + xH",
    "xh": "xH = ttH + VH + VBF (MadGraph5_aMC@NLO + Pythia)",
}
COLORS = {"madgraph": "tab:blue", "no_nnlops": "tab:purple", "powheg": "brown"}


@dataclass(frozen=True)
class PhaseSpace:
    """Cross-section family of a spectrum and the names of its outputs."""

    name: str = "fiducial"
    prefix: str = "fidXS"
    label: str = r"\text{fid}"
    output_stem: str = "spectra"
    upper_panel_scale: float = 1.0  # e.g. 1e-3 to show fb as pb


FIDUCIAL = PhaseSpace()
TOTAL = PhaseSpace("total", "totXS", r"\text{tot}", "spectra_total", 1e-3)


@dataclass
class Spectrum:
    """Everything drawn: per-bin arrays of the predictions, their uncertainties and the measurement."""

    bins_plot: np.ndarray  # bin edges as drawn
    bins_c: np.ndarray  # marker positions
    bin_w_plot: np.ndarray  # bin widths as drawn (place the theory uncertainty boxes)
    madgraph: np.ndarray  # ggH (MadGraph + NNLOPS) + xH, per unit of the variable
    no_nnlops: np.ndarray
    powheg: np.ndarray
    xh: np.ndarray
    unc: dict  # (up, down) theory uncertainty of madgraph, no_nnlops and powheg
    exp_xs: np.ndarray
    err_up: np.ndarray
    err_down: np.ndarray
    sys_up: np.ndarray
    sys_down: np.ndarray


def output_names(stem: str, unblinded: bool, log_x: bool) -> list[str]:
    stem += ("_logx" if log_x else "") + ("" if unblinded else "_blinded")
    return [f"{stem}.pdf", f"{stem}.png"]


def spectra_config(config: dict, phase_space: PhaseSpace) -> dict:
    """The ``spectra`` block of the analysis config with the overrides of the phase space."""
    spectra = config["spectra"]
    current = {key: value for key, value in spectra.items() if key not in ("fiducial", "total")}
    current.update(spectra.get(phase_space.name, {}))
    return current


def luminosity(plotting_config: dict, year: str) -> float:
    """From the plot config, else the sum of the years (e.g. 2022_2023)."""
    if "lumi" in plotting_config:
        return plotting_config["lumi"]
    return sum(lumiMap[y] for y in str(year).split("_"))


def phase_space_text(value, phase_space: PhaseSpace):
    """Adapt the fiducial labels of the plot config to the phase space."""
    if phase_space.prefix == "fidXS" or not isinstance(value, str):
        return value
    text = value.replace(r"\sigma_{\text{fid}}", r"\sigma_{\text{tot}}").replace(r"\sigma_{fid}", r"\sigma_{tot}")
    if phase_space.upper_panel_scale == 1e-3:
        text = text.replace("fb", "pb")
    return text


def scan_interval(param: str, path: str, y_cut: float = SCAN_Y_CUT) -> tuple[float, float, float]:
    """(best fit, up, down) of the 1 sigma interval of a likelihood scan, from a spline through the scan points."""
    files = [f for f in [path] if chplot.TFileIsGood(f)]
    graph = chplot.TGraphFromTree(chplot.MakeTChain(files, "limit"), param, "2*deltaNLL", "quantileExpected > -1.5")
    graph.Sort()
    chplot.RemoveGraphXDuplicates(graph)
    chplot.RemoveGraphYAbove(graph, y_cut)
    if graph.GetN() <= 1:
        graph.Print()
        raise RuntimeError(f"Attempting to build {param} scan from TGraph with zero or one point (see above)")
    bestfit = None
    for i in range(graph.GetN()):
        if graph.GetY()[i] == 0.0:
            bestfit = graph.GetX()[i]
    if bestfit is None:
        raise RuntimeError(f"The scan of {param} in {path} has no best-fit point (2*deltaNLL == 0)")
    spline = ROOT.TSpline3("spline3", graph)

    def evaluate(x, _):
        return spline.Eval(x[0])

    func = ROOT.TF1(f"splinefn_{param}", evaluate, graph.GetX()[0], graph.GetX()[graph.GetN() - 1], 1)
    for crossing in chplot.FindCrossingsWithSpline(graph, func, 1.0):
        if crossing["lo"] <= bestfit <= crossing["hi"]:
            return bestfit, crossing["hi"] - bestfit, crossing["lo"] - bestfit
    raise RuntimeError(f"No 1 sigma interval of {param} contains the best fit {bestfit}")


def measured_signal_strengths(fit_dir: str, pois: list[str], unblinded: bool) -> dict:
    """Best fit, total (up, down) and statistical (up, down) uncertainty of every POI, rounded."""
    measured = {key: [] for key in ("mu", "err_up", "err_down", "stat_up", "stat_down")}
    for poi in pois:
        if unblinded:
            name = os.path.join(
                fit_dir, "dataFit", f"higgsCombineDataPostFitScan{{}}_{poi}.MultiDimFit.mH{HIGGS_MASS}.root"
            )
        else:
            name = os.path.join(fit_dir, "asimov", f"higgsCombineAsimovPostFitScan{{}}_{poi}.root")
        syst = scan_interval(poi, name.format("Fit"))
        stat = scan_interval(poi, name.format("Stat"))
        measured["stat_up"].append(abs(round(stat[1], ROUNDING_DIGITS)))
        measured["stat_down"].append(abs(round(stat[2], ROUNDING_DIGITS)))
        measured["mu"].append(round(syst[0], ROUNDING_DIGITS))
        measured["err_up"].append(abs(round(syst[1], ROUNDING_DIGITS)))
        measured["err_down"].append(abs(round(syst[2], ROUNDING_DIGITS)))
    return measured


def read_pvalue(fit_dir: str, unblinded: bool):
    """p-value of the measurement against MadGraph + NNLOPS (written by PValueCalculation), 1 when blinded."""
    if not unblinded:
        return 1
    with open(os.path.join(fit_dir, "pvalue.txt")) as f:
        return f.readline()


def cross_section(module, prefix: str, source: str | None = None) -> np.ndarray:
    attribute = prefix if source is None else f"{prefix}_{source}"
    try:
        return np.asarray(getattr(module, attribute), dtype=float)
    except AttributeError as exc:
        raise AttributeError(f"Theory input module '{module.__name__}' does not define '{attribute}'") from exc


def generator_dependence_uncertainty(current_config: dict, phase_space: PhaseSpace, n_bins: int) -> np.ndarray:
    """Relative uncertainty of the extrapolation to the total phase space, zero for fiducial spectra."""
    if phase_space.name == "fiducial":
        return np.zeros(n_bins, dtype=float)
    uncertainty_config = current_config.get("generator_dependence_uncertainty")
    if uncertainty_config is None:
        raise KeyError("The total spectrum requires 'spectra.total.generator_dependence_uncertainty' in the config.")
    module_name, attribute = uncertainty_config["module"], uncertainty_config["attribute"]
    module = importlib.import_module(module_name)
    try:
        relative = np.asarray(getattr(module, attribute), dtype=float)
    except AttributeError as exc:
        raise AttributeError(f"Generator-dependence module '{module_name}' does not define '{attribute}'.") from exc
    if relative.ndim != 1 or len(relative) != n_bins:
        raise ValueError(
            f"Generator-dependence uncertainty '{module_name}.{attribute}' has {len(relative)} bins, "
            f"but the spectrum has {n_bins}."
        )
    if not np.all(np.isfinite(relative)) or np.any(relative < 0):
        raise ValueError(f"Generator-dependence uncertainty '{module_name}.{attribute}' must be finite and >= 0.")
    return relative


def bin_layout(boundaries, current_config: dict):
    """(edges, centres, widths, drawn widths) of the bins.

    With ``plot_2d_as_1d`` the (nj_low, nj_high, pt_low, pt_high) bins of a 2D measurement are drawn side by side,
    one panel per jet bin; otherwise the overflow bin ends at ``overflow`` and an underflow bin with
    ``first_bin_center <= 0`` starts at the left edge of the plot.
    """
    if not current_config.get("plot_2d_as_1d", False):
        bins_plot = np.array(boundaries)
        if "overflow" in current_config:
            bins_plot[-1] = current_config["overflow"]
        if current_config.get("first_bin_center", 1) <= 0:
            bins_plot[0] = current_config.get("x_lim", [0, 1])[0]
        bins_c = (bins_plot[1:] + bins_plot[:-1]) * 0.5
        bin_w = np.array([bins_plot[k + 1] - bins_plot[k] for k in range(len(boundaries) - 1)])
        return bins_plot, bins_c, bin_w, bin_w.copy()

    bins_plot, bins_c, bin_w, bin_w_plot = [0.0], [], [], []
    overflow_edges = current_config.get("pthvsnj_overflow_edges", [])
    overflow_threshold = current_config.get("pthvsnj_overflow_threshold", 9999.0)
    panel_width = current_config.get("pthvsnj_panel_width", None)
    group_key, group_index, group_offset, group_scale = None, -1, 0.0, 1.0
    for nj_low, nj_high, pt_low, pt_high in boundaries:
        if (nj_low, nj_high) != group_key:
            group_key = (nj_low, nj_high)
            group_index += 1
            if panel_width is not None:
                group_offset = group_index * panel_width
                group_scale = panel_width / overflow_edges[group_index]
            else:
                group_offset = bins_plot[-1] - pt_low
                group_scale = 1.0
        pt_high_plot = pt_high
        if pt_high >= overflow_threshold and group_index < len(overflow_edges):
            pt_high_plot = overflow_edges[group_index]
        bins_plot.append(group_offset + pt_high_plot * group_scale)
        bins_c.append(group_offset + 0.5 * (pt_low + pt_high_plot) * group_scale)
        bin_w.append(pt_high_plot - pt_low)
        bin_w_plot.append((pt_high_plot - pt_low) * group_scale)
    return tuple(np.array(x, dtype=float) for x in (bins_plot, bins_c, bin_w, bin_w_plot))


def _marker_positions(bins_c, current_config: dict) -> None:
    """Move the first and last markers to the configured positions (in place)."""
    if current_config.get("plot_2d_as_1d", False):
        return
    if current_config["last_bin_center"] > 0:
        bins_c[-1] = current_config["last_bin_center"]
    if current_config.get("show_underflow_bin", False) or current_config["first_bin_center"] > 0:
        # e.g. the categorical N_jets < 2 bin of m_jj or the 0-jet bin of the leading jet pT
        bins_c[0] = current_config["first_bin_center"]
    else:
        # the underflow bin is not shown: push its marker far off-screen
        bins_c[0] = current_config.get("x_lim", [0, 1])[0] - 1e6


def _theory_uncertainty(per_source: np.ndarray, total: np.ndarray, bin_w: np.ndarray) -> np.ndarray:
    # sources (scale, PDF, alpha_s) in quadrature, plus the BR uncertainty
    return np.sqrt((np.sqrt(np.sum(np.square(per_source), axis=0)) / total) ** 2 + BR_UNCERTAINTY**2) * total / bin_w


def compute_spectrum(measured: dict, current_config: dict, phase_space: PhaseSpace) -> Spectrum:
    """Predictions and measured cross sections per unit of the variable."""
    prefix = phase_space.prefix
    modules = {
        key: importlib.import_module(current_config[f"{key}_xs"])
        for key in ("ggh", "vbf", "vh", "tth", "ggh_powheg", "ggh_no_nnlops")
    }
    bins_plot, bins_c, bin_w, bin_w_plot = bin_layout(modules["ggh"].Boundaries, current_config)
    xs = {key: cross_section(module, prefix) for key, module in modules.items()}
    norm = {key: value / bin_w for key, value in xs.items()}
    if current_config["underflow_bin_normalized"]:
        # the underflow bin shows its integrated cross section
        for value in norm.values():
            value[0] = value[0] * bin_w[0]
    xh = norm["vbf"] + norm["vh"] + norm["tth"]

    # |variation - nominal| per source and production mode, summed linearly over the production modes as each
    # source is fully correlated between them
    def variations(direction):
        sources = [f"{source}_{direction}" for source in ("scale", "pdf", "alpha")]
        diff = {key: [abs(cross_section(modules[key], prefix, s) - xs[key]) for s in sources] for key in modules}
        xh_diff = [diff["vbf"], diff["vh"], diff["tth"]]
        return {
            "madgraph": np.sum([diff["ggh"], *xh_diff], axis=0),
            "powheg": np.sum([diff["ggh_powheg"], *xh_diff], axis=0),
            "no_nnlops": np.sum([diff["ggh_no_nnlops"], *xh_diff], axis=0),
        }

    total = xs["ggh"] + xs["vbf"] + xs["vh"] + xs["tth"]
    up, down = variations("up"), variations("dn")
    unc = {
        key: (_theory_uncertainty(up[key], total, bin_w), _theory_uncertainty(down[key], total, bin_w)) for key in up
    }
    if current_config["underflow_bin_normalized"]:
        for key in ("madgraph", "powheg", "no_nnlops"):
            for value in unc[key]:
                value[0] = value[0] * bin_w[0]
        # the underflow/overflow bins are drawn with the configured widths (e.g. the -130..0 GeV bin of m_jj)
        bin_w_plot[0] = current_config.get("underflow_bin_width", 15)
        bin_w_plot[-1] = current_config.get("overflow_bin_width", 100)
    _marker_positions(bins_c, current_config)

    madgraph = norm["ggh"] + xh
    exp_xs = np.array(measured["mu"]) * madgraph
    err_up = np.array(measured["err_up"]) * madgraph
    err_down = np.array(measured["err_down"]) * madgraph
    stat_up = np.array(measured["stat_up"]) * madgraph
    stat_down = np.array(measured["stat_down"]) * madgraph
    # the generator dependence of the extrapolation to the total phase space is a symmetric systematic
    # uncertainty on the measurement (zero for fiducial spectra)
    gen_unc = np.abs(exp_xs) * generator_dependence_uncertainty(current_config, phase_space, len(exp_xs))
    err_up = np.sqrt(err_up**2 + gen_unc**2)
    err_down = np.sqrt(err_down**2 + gen_unc**2)

    def systematic(err, stat, name):
        sys_unc = []
        for diff in err**2 - stat**2:
            if diff >= 0:
                sys_unc.append(np.sqrt(diff))
            else:
                logger.warning(f"(err_{name}**2 - stat_{name}**2) == {diff}: Setting sys_{name} == 0")
                sys_unc.append(0)
        return np.asarray(sys_unc, dtype=float)

    return Spectrum(
        bins_plot=bins_plot,
        bins_c=bins_c,
        bin_w_plot=bin_w_plot,
        madgraph=madgraph,
        no_nnlops=norm["ggh_no_nnlops"] + xh,
        powheg=norm["ggh_powheg"] + xh,
        xh=xh,
        unc={key: unc[key] for key in ("madgraph", "no_nnlops", "powheg")},
        exp_xs=exp_xs,
        err_up=err_up,
        err_down=err_down,
        sys_up=systematic(err_up, stat_up, "up"),
        sys_down=systematic(err_down, stat_down, "down"),
    )


def _draw_theory_boxes(ax, centres, values, err_low, err_high, widths, key: str, hatch: str) -> None:
    """Hatched boxes of the theory uncertainty, side by side for the three predictions."""
    for center, value, low, high, width in zip(centres, values, err_low, err_high, widths):
        if key == "madgraph":
            x = center - width / 4
        else:
            x = center + width / {"powheg": 10, "no_nnlops": 3.6}[key]
        ax.add_patch(
            Rectangle((x, value - low), width / 8, low + high, fill=False, lw=0, color=COLORS[key], hatch=hatch)
        )


def _draw_axvlines(ax, settings) -> None:
    for params in settings if isinstance(settings, list) else [settings]:
        ax.axvline(**params)


def _draw_inward_ticks(ax, position: float) -> None:
    """Major-sized inward ticks at both edges of the panel, without touching the tick locators."""
    for axis_height, tick_offset in ((0, plt.rcParams["xtick.major.size"]), (1, -plt.rcParams["xtick.major.size"])):
        ax.annotate(
            "",
            xy=(position, axis_height),
            xycoords=ax.get_xaxis_transform(),
            xytext=(0, tick_offset),
            textcoords="offset points",
            arrowprops={
                "arrowstyle": "-",
                "color": "black",
                "linewidth": plt.rcParams["xtick.major.width"],
                "shrinkA": 0,
                "shrinkB": 0,
            },
            annotation_clip=False,
        )


def _suppress_minor_xticks(ax, low: float, high: float) -> None:
    """Remove the automatic minor ticks inside (low, high), e.g. inside a categorical underflow bin."""
    xlim = ax.get_xlim()
    ax.set_xticks([tick for tick in ax.xaxis.get_minorticklocs() if not low < tick < high], minor=True)
    # set_xticks expands the axis when the locator returned an outer tick: restore the range
    ax.set_xlim(xlim)


def _upper_panel(fig, spectrum: Spectrum, current_config, plotting_config, phase_space, log_x, x_lim, lumi, pvalue):
    s, scale = spectrum, phase_space.upper_panel_scale
    text = partial(phase_space_text, phase_space=phase_space)
    ax = fig.add_axes((0.1, 0.35, 0.8, 0.6))
    hep.cms.label(CMS_LABEL, exp="", data=True, lumi=lumi, com=13.6, fontsize=20)

    ax.stairs(s.madgraph * scale, s.bins_plot, linewidth=2, label=LEGEND["madgraph"], color=COLORS["madgraph"])
    ax.stairs(s.no_nnlops * scale, s.bins_plot, linewidth=2, label=LEGEND["no_nnlops"], color=COLORS["no_nnlops"])
    ax.stairs(s.powheg * scale, s.bins_plot, linewidth=2, label=LEGEND["powheg"], color=COLORS["powheg"])
    ax.stairs(s.xh * scale, s.bins_plot, linewidth=2, color="green")
    ax.stairs(s.xh * scale, s.bins_plot, linewidth=2, label=LEGEND["xh"], alpha=0.2, color="green", fill=True)

    plt.rcParams["hatch.linewidth"] = 2
    for key, hatch in (("madgraph", "///"), ("powheg", "////"), ("no_nnlops", "////")):
        up, down = s.unc[key]
        _draw_theory_boxes(ax, s.bins_c, getattr(s, key) * scale, down * scale, up * scale, s.bin_w_plot, key, hatch)

    fontsize, title_fontsize = 14, 14
    settings = plotting_config.get("frame1", {})
    custom_ylabel = False
    if "frame1" in plotting_config:
        if "custom_xtick_labels" in settings:
            ax.set_xticks(settings["custom_xticks"])
        if "axvline" in settings:
            _draw_axvlines(ax, settings["axvline"])
        # a log-x plot can point at the overflow bin with an arrow instead of the fixed overflow text
        figtext_key = None if log_x and "overflow_annotation_log_x" in settings else "figtext"
        if figtext_key is not None and figtext_key in settings:
            params = settings[figtext_key]
            fig.text(
                params["x"],
                params["y"],
                text(params["text"]),
                horizontalalignment=params["horizontalalignment"],
                rotation=params["rotation"],
                fontsize=params["fontsize"],
            )
        for label in settings.get("additional_labels", []):
            ax.text(
                label["position"][0],
                label["position"][1] * scale,
                text(label["text"]),
                rotation=label["rotation"],
                fontsize=label["fontsize"],
            )
        if "ylabel" in settings:
            ax.set_ylabel(text(settings["ylabel"]["text"]), fontsize=settings["ylabel"]["fontsize"])
            custom_ylabel = True
        if "ylim" in settings:
            ax.set_ylim(**{k: v * scale if k in ("bottom", "top") else v for k, v in settings["ylim"].items()})
        fontsize = settings.get("fontsize", fontsize)
        title_fontsize = settings.get("title_fontsize", title_fontsize)

    exp, err, sys_err = s.exp_xs * scale, [s.err_down * scale, s.err_up * scale], [s.sys_down * scale, s.sys_up * scale]
    marker = {"linestyle": "None", "ms": 5, "capsize": 4}
    ax.errorbar(
        s.bins_c, exp, yerr=err, marker="o", color="k", linewidth=2, label=r"Data (stat $\oplus$ sys unc.)", **marker
    )
    ax.errorbar(
        s.bins_c, exp, yerr=sys_err, marker="None", color="red", linewidth=6, label="Systematic uncertainty", **marker
    )

    if current_config["plot_log"]:
        ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")
    if not custom_ylabel:
        ylabel = rf"$\Delta\sigma_{{{phase_space.label}}} / \Delta {current_config['variable']}$ "
        ax.set_ylabel(ylabel + text(current_config["y_unit"]), fontsize=20)
    y_lim_top_key = "y_lim_top_log_x" if log_x and "y_lim_top_log_x" in current_config else "y_lim_top"
    if y_lim_top_key in current_config:
        ax.set_ylim(top=current_config[y_lim_top_key])
    ax.set_xlim(x_lim)

    if log_x and "overflow_annotation_log_x" in settings:
        annotation = settings["overflow_annotation_log_x"]
        ax.annotate(
            text(annotation["text"]),
            # points at the MadGraph + NNLOPS prediction at the left edge of the last bin
            xy=(s.bins_plot[-2], s.madgraph[-1] * scale),
            xycoords="data",
            xytext=annotation["text_position"],
            textcoords="axes fraction",
            horizontalalignment=annotation.get("horizontalalignment", "center"),
            verticalalignment=annotation.get("verticalalignment", "center"),
            rotation=annotation.get("rotation", 0),
            fontsize=annotation.get("fontsize", 22),
            arrowprops=annotation.get("arrowprops", {"arrowstyle": "->", "color": "black", "linewidth": 1.5}),
            annotation_clip=False,
        )
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.legend(
        fontsize=fontsize,
        title=f"p-value (MadGraph NNLOPS) = {pvalue}",
        alignment="left",
        loc=current_config.get("legend_location", "upper right"),
        title_fontsize=title_fontsize,
    )
    ax.set_xticklabels([])
    if settings.get("suppress_minor_xticks") is not None:
        _suppress_minor_xticks(ax, *settings["suppress_minor_xticks"])
    for tick in settings.get("additional_xaxis_ticks", []):
        _draw_inward_ticks(ax, tick["position"])
    return ax


def _ratio_panel(fig, upper, spectrum: Spectrum, current_config, plotting_config, phase_space, unblinded, log_x):
    s = spectrum
    text = partial(phase_space_text, phase_space=phase_space)
    ax = fig.add_axes((0.1, 0.05, 0.8, 0.25))
    ratio = {key: getattr(s, key) / s.madgraph for key in ("madgraph", "powheg", "no_nnlops")}
    ratio_unc = {key: (up / s.madgraph, down / s.madgraph) for key, (up, down) in s.unc.items()}
    marker = {"linestyle": "None", "ms": 5, "capsize": 4}
    ratio_xs = s.exp_xs / s.madgraph
    err = [s.err_down / s.madgraph, s.err_up / s.madgraph]
    sys_err = [s.sys_down / s.madgraph, s.sys_up / s.madgraph]
    ax.errorbar(s.bins_c, ratio_xs, yerr=err, marker="o", color="k", linewidth=2, label="Data (Stat + Syst)", **marker)
    ax.errorbar(
        s.bins_c, ratio_xs, yerr=sys_err, marker="None", color="red", linewidth=6, label="Systematic error", **marker
    )
    ax.stairs(ratio["madgraph"], s.bins_plot, linewidth=2, color=COLORS["madgraph"])
    for key, hatch in (("madgraph", "/////"), ("powheg", "/////"), ("no_nnlops", "////")):
        up, down = ratio_unc[key]
        _draw_theory_boxes(ax, s.bins_c, ratio[key], down, up, s.bin_w_plot, key, hatch)
    ax.stairs(ratio["powheg"], s.bins_plot, linewidth=2, color=COLORS["powheg"])
    ax.stairs(ratio["no_nnlops"], s.bins_plot, linewidth=2, color=COLORS["no_nnlops"])

    settings = plotting_config.get("frame2", {})
    data_key = "is_data" if unblinded else "not_data"
    if "xticks" in settings:
        ax.set_xticks(settings["xticks"])
    if "xtick_labels" in settings:
        ax.set_xticklabels(settings["xtick_labels"])
    if "yticks" in settings and "ytick_labels" in settings:
        ax.set_yticks(settings["yticks"].get(data_key, []))
        ax.set_yticklabels(settings["ytick_labels"].get(data_key, []))
    if "additional_labels" in settings:
        for label in settings["additional_labels"][data_key]:
            if phase_space.name == "total" and label.get("fiducial_only", False):
                continue
            ax.text(
                label["position"][0],
                label["position"][1],
                text(label["text"]),
                rotation=label["rotation"],
                fontsize=label["fontsize"],
            )
    if "axvline" in settings:
        _draw_axvlines(ax, settings["axvline"])
    xtick_fontsize = settings.get("xtick_fontsize", 20)

    for edge in s.bins_plot:
        ax.axvline(x=edge, color="gray", ls="dashed", lw=1, alpha=0.5)
    # not "MC": NNLOPS is based on a calculation
    ax.set_ylabel(r"Data / Prediction", fontsize=settings.get("ylabel_fontsize", 20))
    ax.set_xlabel(
        rf"${current_config['variable']}$ {current_config['x_unit']}", fontsize=settings.get("xlabel_fontsize", 20)
    )
    plt.xticks(fontsize=xtick_fontsize, rotation=settings.get("xtick_rotation", 0))
    plt.yticks(fontsize=settings.get("ytick_fontsize", 20))
    if log_x:
        ax.set_xscale("log")
    # exactly the range of the upper panel, also for a categorical underflow bin
    ax.set_xlim(upper.get_xlim())
    ax.set_ylim(current_config["y_lim"])
    if settings.get("suppress_minor_xticks") is not None:
        _suppress_minor_xticks(ax, *settings["suppress_minor_xticks"])
    # categorical underflow labels, placed independently of the (automatic) numerical ticks
    for label in settings.get("additional_xaxis_labels", []):
        _draw_inward_ticks(ax, label["position"])
        ax.annotate(
            text(label["text"]),
            xy=(label["position"], 0),
            xycoords=ax.get_xaxis_transform(),
            xytext=(label.get("x_offset", 0), label.get("y_offset", -30)),
            textcoords="offset points",
            horizontalalignment=label.get("horizontalalignment", "center"),
            verticalalignment="top",
            fontsize=label.get("fontsize", xtick_fontsize),
            annotation_clip=False,
        )
    return ax


def plot_spectrum(
    spectrum: Spectrum,
    current_config: dict,
    plotting_config: dict,
    paths: list[str],
    phase_space: PhaseSpace = FIDUCIAL,
    unblinded: bool = False,
    log_x: bool = False,
    lumi: float = 0.0,
    pvalue=1,
) -> None:
    """Spectrum with the predictions in the upper panel, data / prediction below; saved to every path."""
    x_lim = current_config.get("x_lim_log", current_config["x_lim"]) if log_x else current_config["x_lim"]
    if log_x and x_lim[0] <= 0:
        raise ValueError("A logarithmic x axis requires a positive lower x limit.")
    with plt.style.context(hep.style.CMS):
        fig = plt.figure(figsize=(10, 8), dpi=120)
        upper = _upper_panel(fig, spectrum, current_config, plotting_config, phase_space, log_x, x_lim, lumi, pvalue)
        _ratio_panel(fig, upper, spectrum, current_config, plotting_config, phase_space, unblinded, log_x)
        for path in paths:
            fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)


def make_spectrum(
    fit_dir: str,
    pois: list[str],
    config: dict,
    plotting_config: dict,
    year: str,
    phase_space: PhaseSpace = FIDUCIAL,
    unblinded: bool = False,
    log_x: bool = False,
) -> Spectrum:
    """Spectrum of the scans in <fit_dir>/asimov (or dataFit when unblinded), saved in fit_dir.

    pois: the signal strengths of the bins, in the order of the bins of the theory modules.
    """
    measured = measured_signal_strengths(fit_dir, pois, unblinded)
    pvalue = read_pvalue(fit_dir, unblinded)
    current_config = spectra_config(config, phase_space)
    spectrum = compute_spectrum(measured, current_config, phase_space)
    paths = [os.path.join(fit_dir, name) for name in output_names(phase_space.output_stem, unblinded, log_x)]
    plot_spectrum(
        spectrum,
        current_config,
        plotting_config,
        paths,
        phase_space,
        unblinded,
        log_x,
        luminosity(plotting_config, year),
        pvalue,
    )
    logger.info(f"Saved {', '.join(paths)}")
    return spectrum
