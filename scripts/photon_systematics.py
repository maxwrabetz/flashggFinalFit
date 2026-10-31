#!/usr/bin/env python3
"""Photon scale/smearing systematics of the signal shapes in one category (the law task CalcPhotonSyst runs this).

    ./scripts/photon_systematics.py <ws_signal dir> outdir_<ext>/calcPhotonSyst/pkl/<cat>.pkl -c <cat> \\
        -p <proc1>,<proc2> --scales ScaleEEZee,ScaleEBZee --scales-corr ScaleEEZmmg,ScaleEBZmmg --smears Smearing
"""

import argparse
import os
import sys

import ROOT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from finalfits import photon_systematics  # noqa: E402
from finalfits.utils import add_verbosity_args, setup_logging  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_verbosity_args(parser)
    parser.add_argument("in_dir", help="Directory with the signal workspaces of Trees2WS (ws_signal)")
    parser.add_argument("out_file", help="Output pickle file")
    parser.add_argument("--cat", "-c", required=True, help="Analysis category")
    parser.add_argument("--procs", "-p", required=True, help="Comma separated signal processes")
    parser.add_argument("--scales", default="", help="Scale systematics, separate nuisance per year")
    parser.add_argument("--scales-corr", default="", help="Scale systematics correlated across years")
    parser.add_argument("--smears", default="", help="Smearing systematics")
    parser.add_argument("--n-bins", type=int, default=80, help="Number of bins in 100-180 GeV")
    parser.add_argument("--threshold-mean", type=float, default=0.05, help="Cap on the mean variation")
    parser.add_argument("--threshold-sigma", type=float, default=0.5, help="Cap on the sigma variation")
    parser.add_argument("--threshold-rate", type=float, default=0.05, help="Cap on the rate variation")
    return parser.parse_args()


def main():
    args = get_args()
    setup_logging(args.verbose, args.roofit_verbose)
    ROOT.gROOT.SetBatch(True)
    data = photon_systematics.calc_photon_systematics(
        args.in_dir,
        args.cat,
        args.procs.split(","),
        {"scales": args.scales.split(","), "scalesCorr": args.scales_corr.split(","), "smears": args.smears.split(",")},
        n_bins=args.n_bins,
        thresholds={"mean": args.threshold_mean, "sigma": args.threshold_sigma, "rate": args.threshold_rate},
    )
    photon_systematics.write_pickle(data, args.out_file)


if __name__ == "__main__":
    main()
