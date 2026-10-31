#!/usr/bin/env python3
"""Package the signal models of one category (the law task SignalPackaging runs this).

    ./scripts/signal_package.py <output dir> -c <cat> --exts <ext1>,<ext2>
writes <output dir>/outdir_packaged<output ext>/CMS-HGG_sigfit_packaged<output ext>_<cat>.root (with --year:
..._<cat>_<year>.root, for models of a single year).
"""

import argparse
import os
import sys

import ROOT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from finalfits import signal_package  # noqa: E402
from finalfits.utils import add_verbosity_args, setup_logging  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_verbosity_args(parser)
    parser.add_argument("out_dir", help="Directory with outdir_<ext>/signalFit of every extension")
    parser.add_argument("--cat", "-c", required=True, help="Analysis category")
    parser.add_argument("--exts", required=True, help="Comma separated extensions (e.g. one per era)")
    parser.add_argument("--output-ext", default="", help="Extension of the packaged output")
    parser.add_argument("--mass-points", default="120,125,130", help="Mass points of the merged datasets")
    parser.add_argument("--year", default="", help="Add the year to the file name (models of a single year)")
    return parser.parse_args()


def main():
    args = get_args()
    setup_logging(args.verbose, args.roofit_verbose)
    ROOT.gROOT.SetBatch(True)
    ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit")
    name = f"CMS-HGG_sigfit_packaged{args.output_ext}_{args.cat}" + (f"_{args.year}" if args.year else "")
    signal_package.package_signal(
        args.out_dir,
        args.cat,
        args.exts.split(","),
        args.mass_points.split(","),
        os.path.join(args.out_dir, f"outdir_packaged{args.output_ext}", f"{name}.root"),
    )


if __name__ == "__main__":
    main()
