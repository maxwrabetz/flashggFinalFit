#!/usr/bin/env python3
"""Signal model of one process in one category (the law task SignalFit runs this).

    ./scripts/signal_fit.py <ws_signal dir> <output dir> --ext <ext> -p ggh_NJ_0p0_1p0_in -c RECO_NJ_0p0_1p0_cat0 \\
        --year 2023preBPix --analysis-xsbr Run3FidXSAnalysis --analysis-rm Run3FidXSAnalysisNJ --plots
writes <output dir>/outdir_<ext>/signalFit/output/CMS-HGG_sigfit_<ext>_<proc>_<year>_<cat>.root, the plots to
signalFit/Plots and the replacement report to signalFit/logs. The fTest and photon systematics are read
from <ingredients dir>/outdir_<ext> (default: the output dir).
"""

import argparse
import logging
import os
import sys

import ROOT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from finalfits import signal_fit, signal_plots  # noqa: E402
from finalfits.utils import add_verbosity_args, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_verbosity_args(parser)
    parser.add_argument("in_dir", help="Directory with the signal workspaces of Trees2WS (ws_signal)")
    parser.add_argument("out_dir", help="Output directory (outdir_<ext>/signalFit is created inside)")
    parser.add_argument("--ext", required=True, help="Extension of the output directory")
    parser.add_argument("--proc", "-p", required=True, help="Signal process")
    parser.add_argument("--cat", "-c", required=True, help="Analysis category")
    parser.add_argument("--year", required=True, help="Year (with era), e.g. 2023preBPix")
    parser.add_argument("--ingredients-dir", default="", help="Directory with outdir_<ext>/fTest and calcPhotonSyst")
    parser.add_argument("--analysis-xsbr", default="STXS", help="Entry of globalXSBRMap (commonTools/XSBRMap.py)")
    parser.add_argument("--analysis-rm", default="STXS", help="Entry of globalReplacementMap")
    parser.add_argument("--mass-points", default="120,125,130", help="Mass points to fit")
    parser.add_argument("--scales", default="", help="Scale systematics, separate nuisance per year")
    parser.add_argument("--scales-corr", default="", help="Scale systematics correlated across years")
    parser.add_argument("--scales-global", default="", help="Scale systematics affecting all processes equally")
    parser.add_argument("--smears", default="", help="Smearing systematics")
    parser.add_argument("--replacement-threshold", type=int, default=0, help="Minimum events, else replacement shape")
    parser.add_argument("--beamspot-width-data", type=float, default=3.5, help="Beamspot width in data [cm]")
    parser.add_argument("--beamspot-width-mc", type=float, default=3.7, help="Beamspot width in simulation [cm]")
    parser.add_argument("--skip-beamspot-reweigh", action="store_true", help="No beamspot reweighting")
    parser.add_argument("--skip-vertex-scenario-split", action="store_true", help="One fit for RV and WV together")
    parser.add_argument("--skip-systematics", action="store_true", help="No shape systematics")
    parser.add_argument("--skip-zeroes", action="store_true", help="No model for a process without events")
    parser.add_argument("--use-dcb", action="store_true", help="Double crystal ball + Gaussian instead of Gaussians")
    parser.add_argument("--mh-poly-order", type=int, default=1, help="Order of the polynomials in MH")
    parser.add_argument("--n-bins", type=int, default=80, help="Number of bins in 100-180 GeV")
    parser.add_argument("--minimizer-method", default="TNC", help="scipy minimizer")
    parser.add_argument("--replacement-report", default="", help="Replacement report file")
    parser.add_argument("--plots", action="store_true", help="Make the plots")
    parser.add_argument("--plot-label", default="", help="Top-right label of the plots")
    return parser.parse_args()


def main():
    args = get_args()
    setup_logging(args.verbose, args.roofit_verbose)
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    base = os.path.join(args.out_dir, f"outdir_{args.ext}", "signalFit")
    name = f"{args.proc}_{args.year}_{args.cat}"
    opts = signal_fit.SignalFitOptions(
        mass_points=tuple(args.mass_points.split(",")),
        analysis_xsbr=args.analysis_xsbr,
        analysis_rm=args.analysis_rm,
        systematics={
            "scales": args.scales,
            "scalesCorr": args.scales_corr,
            "scalesGlobal": args.scales_global,
            "smears": args.smears,
        },
        replacement_threshold=args.replacement_threshold,
        beamspot_width_data=args.beamspot_width_data,
        beamspot_width_mc=args.beamspot_width_mc,
        skip_beamspot_reweigh=args.skip_beamspot_reweigh,
        skip_vertex_scenario_split=args.skip_vertex_scenario_split,
        skip_systematics=args.skip_systematics,
        skip_zeroes=args.skip_zeroes,
        use_dcb=args.use_dcb,
        mh_poly_order=args.mh_poly_order,
        n_bins=args.n_bins,
        minimizer_method=args.minimizer_method,
    )
    result = signal_fit.run_signal_fit(
        args.in_dir,
        args.ingredients_dir or args.out_dir,
        os.path.join(base, "output", f"CMS-HGG_sigfit_{args.ext}_{name}.root"),
        args.proc,
        args.cat,
        args.year,
        args.ext,
        opts,
        report_file=args.replacement_report or os.path.join(base, "logs", f"replacements_{name}.txt"),
    )
    if result is None or not args.plots:
        return
    model, fits, sources = result
    plot_dir = os.path.join(base, "Plots")
    try:
        rv = sources["RV"]
        if "Total" in fits:
            signal_plots.plot_pdf_components(fits["Total"], plot_dir, "total_", rv.proc, rv.cat, args.plot_label)
        else:
            signal_plots.plot_pdf_components(fits["RV"], plot_dir, "RV_", rv.proc, rv.cat, args.plot_label)
            signal_plots.plot_pdf_components(fits["WV"], plot_dir, "WV_", sources["WV"].proc, rv.cat)
        signal_plots.plot_interpolation(model, plot_dir, args.plot_label)
        signal_plots.plot_splines(model, plot_dir, args.plot_label)
    except Exception:  # plots must not fail the model (e.g. empty bins)
        logger.exception("Making the signal fit plots failed")


if __name__ == "__main__":
    main()
