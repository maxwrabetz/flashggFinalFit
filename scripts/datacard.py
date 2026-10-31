#!/usr/bin/env python3
"""Datacard: yields per category, the datacard, and its cleaning (the law tasks MakeYields and MakeDatacard run this).

./scripts/datacard.py yields Datacards/yields_<ext>/<cat>.pkl -c <cat> --input-ws-dirs 2023preBPix=<dir>,... ...
./scripts/datacard.py make Datacards/yields_<ext> Datacards/Datacard.txt --years 2023preBPix,2023postBPix ...
./scripts/datacard.py clean Datacards/Datacard.txt --factor 10000
"""

import argparse
import os
import sys

import ROOT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from finalfits import datacard, datacard_yields  # noqa: E402
from finalfits.utils import add_verbosity_args, setup_logging  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_verbosity_args(parser)
    sub = parser.add_subparsers(dest="mode", required=True)

    y = sub.add_parser("yields", help="Yields table of one category (pickle)")
    y.add_argument("out_file", help="Output pickle file")
    y.add_argument("--cat", "-c", required=True, help="Analysis category")
    y.add_argument("--input-ws-dirs", required=True, help="Signal workspaces per year: year=dir,year=dir")
    y.add_argument("--procs", default="auto", help="Comma separated signal processes, auto: from the workspaces")
    y.add_argument("--variable", default="", help="Differential variable (jet systematics only for jet variables)")
    y.add_argument("--mass", default="125", help="Mass point of the workspaces")
    y.add_argument("--merge-years", action="store_true", help="One category for all years")
    y.add_argument("--skip-bkg", action="store_true", help="Only signal processes")
    y.add_argument("--bkg-scaler", type=float, default=1.0, help="Scale factor of the background rate")
    y.add_argument("--sig-model-dir", default="./Models/signal", help="Signal model directory in the datacard")
    y.add_argument("--sig-model-ext", default="packaged", help="Extension of the signal model files")
    y.add_argument("--bkg-model-dir", default="./Models/background", help="Background model directory in the datacard")
    y.add_argument("--bkg-model-ext", default="multipdf", help="Extension of the background model files")
    y.add_argument("--skip-zeroes", action="store_true", help="Leave out processes with zero yield")
    y.add_argument("--skip-cow-corr", action="store_true", help="No centralObjectWeight correction")
    y.add_argument("--do-systematics", action="store_true", help="Also the systematic yield variations")
    y.add_argument("--ignore-warnings", action="store_true", help="Nominal yield for missing systematics")
    y.add_argument(
        "--weight-scheme",
        default="accEff",
        choices=["accEff", "legacyHiggsDNA"],
        help="accEff: systematic weights include the genWeight; legacyHiggsDNA: normalise to weight_central",
    )

    m = sub.add_parser("make", help="Datacard from the yields tables")
    m.add_argument("yields_dir", help="Directory with the yields pickles of all categories")
    m.add_argument("out_file", help="Output datacard")
    m.add_argument("--years", required=True, help="Comma separated years with eras")
    m.add_argument("--variable", default="", help="Differential variable (jet systematics only for jet variables)")
    m.add_argument("--prune", action="store_true", help="Prune processes below --prune-threshold of the category")
    m.add_argument("--prune-threshold", type=float, default=0.001, help="Fraction of the category yield")
    m.add_argument("--true-yield", action="store_true", help="Prune on XS * BR * eff x acc * lumi")
    m.add_argument("--mass", default="125", help="MH for --true-yield")
    m.add_argument("--analysis", default="STXS", help="Entry of globalXSBRMap for --true-yield")
    m.add_argument("--skip-cow-corr", action="store_true", help="No centralObjectWeight correction")
    m.add_argument("--do-systematics", action="store_true", help="Add the systematics")
    m.add_argument("--mc-stat", action="store_true", help="Add the uncertainty of the simulated yields")
    m.add_argument("--save-dataframe", default="", help="Save the final table to this pickle file")

    c = sub.add_parser("clean", help="Remove unphysical lnN values")
    c.add_argument("datacard", help="Input datacard")
    c.add_argument("--output", default=None, help="Output datacard (default: <datacard>_cleaned.txt)")
    c.add_argument("--factor", type=float, default=5.0, help="Remove values beyond factor or below 1/factor")
    c.add_argument("--remove-double-sided", action="store_true", help="One value for up/down in the same direction")
    c.add_argument("--remove-non-diagonal", action="store_true", help="Only lumi effects for off-diagonal processes")
    c.add_argument("--symmetrize", default="", help="Comma separated nuisances to symmetrise when one-sided")
    return parser.parse_args()


def main():
    args = get_args()
    setup_logging(args.verbose, args.roofit_verbose)
    ROOT.gROOT.SetBatch(True)
    if args.mode == "yields":
        dirs = dict(item.split("=") for item in args.input_ws_dirs.split(","))
        for path in dirs.values():
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Directory {path} does not exist")
        data = datacard_yields.make_yields(
            dirs,
            args.cat,
            procs=args.procs if args.procs == "auto" else args.procs.split(","),
            variable=args.variable,
            mass=args.mass,
            merge_years=args.merge_years,
            skip_bkg=args.skip_bkg,
            bkg_scaler=args.bkg_scaler,
            sig_model_dir=args.sig_model_dir,
            sig_model_ext=args.sig_model_ext,
            bkg_model_dir=args.bkg_model_dir,
            bkg_model_ext=args.bkg_model_ext,
            skip_zeroes=args.skip_zeroes,
            skip_cow_corr=args.skip_cow_corr,
            do_systematics=args.do_systematics,
            ignore_warnings=args.ignore_warnings,
            weight_scheme=args.weight_scheme,
        )
        datacard_yields.write_yields(data, args.out_file)
    elif args.mode == "make":
        opts = datacard.DatacardOptions(
            years=args.years,
            variable=args.variable,
            skip_cow_corr=args.skip_cow_corr,
            do_systematics=args.do_systematics,
            do_mc_stat=args.mc_stat,
            prune=args.prune,
            prune_threshold=args.prune_threshold,
            true_yield=args.true_yield,
            mass=args.mass,
            analysis=args.analysis,
        )
        datacard.make_datacard(args.yields_dir, args.out_file, opts, args.save_dataframe or None)
    else:
        datacard.clean_datacard(
            args.datacard, args.output, args.factor, args.remove_double_sided, args.remove_non_diagonal, args.symmetrize
        )


if __name__ == "__main__":
    main()
