#!/usr/bin/env python3
"""Statistical interpretation with combine (the law tasks of Combine/law_combine.py build their commands with
finalfits.combine and run text2workspace through this script).

./scripts/combine.py text2workspace Combine/Datacard_NJ_2023.txt Combine/Datacard_NJ_2023.root --model NJ
./scripts/combine.py nuisances Combine/Datacard_NJ_2023.root --pois r_NJ_0p0_1p0,r_NJ_1p0_2p0 --exclude 'rgx{.*_bkg}'
./scripts/combine.py pvalue Combine/runFits_NJ/dataFit/higgsCombine.pvalue.MultiDimFit.mH125.07.root --n-pois 5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from finalfits import combine  # noqa: E402
from finalfits.utils import add_verbosity_args, setup_logging  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_verbosity_args(parser)
    sub = parser.add_subparsers(dest="mode", required=True)
    t2w = sub.add_parser("text2workspace", help="Datacard to workspace")
    t2w.add_argument("datacard", help="Input datacard (.txt)")
    t2w.add_argument("workspace", help="Output workspace (.root)")
    t2w.add_argument("--model", required=True, help="Physics model of Combine/models.py, e.g. mu_fiducial or NJ")
    t2w.add_argument("--common-options", default=combine.COMMON_T2W_OPTIONS, help="Further text2workspace options")
    t2w.add_argument("--dry-run", action="store_true", help="Only print the command")

    nuisances = sub.add_parser("nuisances", help="Nuisances of a workspace for the impacts (comma separated)")
    nuisances.add_argument("workspace", help="Workspace (.root) of text2workspace")
    nuisances.add_argument("--pois", default="r", help="Comma separated POIs")
    nuisances.add_argument("--exclude", default="", help="Names or regular expressions (rgx{...}) to exclude")
    nuisances.add_argument(
        "--all-free", action="store_true", help="All free parameters, not only the ModelConfig nuisances"
    )

    pvalue = sub.add_parser("pvalue", help="p-value of the fit at the SM point of all POIs")
    pvalue.add_argument("fit", help="Output of the fit with --algo fixed")
    pvalue.add_argument("--n-pois", type=int, required=True, help="Number of POIs (degrees of freedom)")
    return parser.parse_args()


def main():
    args = get_args()
    setup_logging(args.verbose, args.roofit_verbose)
    if args.mode == "text2workspace":
        command = combine.text2workspace_command(args.datacard, args.workspace, args.model, args.common_options)
        if args.dry_run:
            print(" ".join(command))
        else:
            combine.run(command)
    elif args.mode == "nuisances":
        pois = args.pois.split(",")
        if args.all_free:
            names = combine.free_parameters(args.workspace, pois)
        else:
            names = combine.modelconfig_nuisances(args.workspace, pois, args.exclude)
        print(",".join(names))
    else:
        print(combine.round_significant(combine.pvalue(args.fit, args.n_pois)))


if __name__ == "__main__":
    main()
