#!/usr/bin/env python3
"""Signal fTest: number of Gaussians per process for one category (the law task FTest runs this).

    ./scripts/signal_ftest.py <ws_signal dir> <outdir>/fTest -c RECO_NJ_0p0_1p0_cat0 \\
        -p ggh_NJ_0p0_1p0_in,vbf_NJ_0p0_1p0_in --plots RV,RVResults,WV,WVResults
writes <outdir>/fTest/json/nGauss_<cat>.json and the plots to <outdir>/fTest/Plots.
"""

import argparse
import os
import sys

import ROOT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from finalfits import signal_ftest  # noqa: E402
from finalfits.utils import add_verbosity_args, setup_logging  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_verbosity_args(parser)
    parser.add_argument("in_dir", help="Directory with the signal workspaces of Trees2WS (ws_signal)")
    parser.add_argument("out_dir", help="Output directory (json/ and Plots/ are created inside)")
    parser.add_argument("--cat", "-c", required=True, help="Analysis category")
    parser.add_argument("--procs", "-p", required=True, help="Comma separated signal processes")
    parser.add_argument("--mass", default="125", help="Mass point to fit")
    parser.add_argument("--n-bins", type=int, default=80, help="Number of bins in 100-180 GeV")
    parser.add_argument("--threshold", type=int, default=30, help="Minimum number of events to run the fTest")
    parser.add_argument("--n-gauss-max", type=int, default=5, help="Maximum number of Gaussians")
    parser.add_argument("--n-procs-to-ftest", type=int, default=-1, help="Only the processes with the largest yields")
    parser.add_argument("--skip-wv", action="store_true", help="Keep one Gaussian for the wrong vertex scenario")
    parser.add_argument("--minimizer-method", default="TNC", help="scipy minimizer")
    parser.add_argument("--plots", default="", help="Comma separated selection of RV,RVResults,WV,WVResults")
    parser.add_argument("--plot-label-rv", default="", help="Top-right label of the RV plots")
    parser.add_argument("--plot-label-wv", default="", help="Top-right label of the WV plots")
    parser.add_argument("--skip-json", action="store_true", help="Do not write the nGauss json")
    return parser.parse_args()


def main():
    args = get_args()
    setup_logging(args.verbose, args.roofit_verbose)
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    signal_ftest.run_ftest(
        args.in_dir,
        args.out_dir,
        args.cat,
        args.procs.split(","),
        mass=args.mass,
        n_bins=args.n_bins,
        threshold=args.threshold,
        n_gauss_max=args.n_gauss_max,
        n_procs_to_ftest=args.n_procs_to_ftest,
        skip_wv=args.skip_wv,
        minimizer_method=args.minimizer_method,
        plots={p for p in args.plots.split(",") if p},
        plot_labels={"RV": args.plot_label_rv, "WV": args.plot_label_wv},
        write_json=not args.skip_json,
    )


if __name__ == "__main__":
    main()
