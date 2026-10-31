#!/usr/bin/env python3
"""Background model: fTest and RooMultiPdf envelope per category (the law task Background runs this).

Run one category per process, so that the random numbers of a category do not depend on the others:
    ./scripts/background.py allData.root out/bkgfTest-Data -c cat0 --year 2023 --int-lumi 27.2 \\
        --multipdf out/CMS-HGG_multipdf_cat0.root --dat-file out/fTest_cat0.dat
"""

import argparse
import os
import sys

import ROOT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from finalfits import background, background_plots  # noqa: E402
from finalfits.utils import add_verbosity_args, setup_logging  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_verbosity_args(parser)
    parser.add_argument("in_file", help="Data workspace from Trees2WS (allData.root)")
    parser.add_argument("out_dir", help="Directory for the plots and fTestResults.txt")
    parser.add_argument("--categories", "-c", required=True, help="Comma separated analysis categories")
    parser.add_argument("--year", required=True, help="Year in the pdf names, 'all' for combined years")
    parser.add_argument("--int-lumi", type=float, required=True, help="Integrated luminosity in fb^-1 for the plots")
    parser.add_argument("--cat-offset", type=int, default=0, help="Number of the first category in the plot names")
    parser.add_argument("--multipdf", help="Write the RooMultiPdf envelopes to this file")
    parser.add_argument("--dat-file", default="fTest.dat", help="Output file with the chosen orders")
    parser.add_argument("--mc", action="store_true", help="Input is MC (workspace cms_hgg_workspace)")
    parser.add_argument("--unblind", action="store_true", help="Show the data in 115-135 GeV in the plots")
    parser.add_argument("--ftest-toys", action="store_true", help="Check the F-test p-values with toys")
    parser.add_argument("--cms-label", default="Preliminary", help="Label in the plots")
    return parser.parse_args()


def main():
    args = get_args()
    setup_logging(args.verbose, args.roofit_verbose)
    if args.roofit_verbose <= 0:
        ROOT.RooMsgService.instance().setSilentMode(True)
        ROOT.gErrorIgnoreLevel = ROOT.kWarning
    background_plots.configure_style(args.cms_label, args.int_lumi)
    background.run_ftest(
        args.in_file,
        args.categories.split(","),
        args.out_dir,
        args.dat_file,
        args.year,
        cat_offset=args.cat_offset,
        multipdf_file=args.multipdf,
        is_data=not args.mc,
        blind=not args.unblind,
        ftest_toys=args.ftest_toys,
    )


if __name__ == "__main__":
    main()
