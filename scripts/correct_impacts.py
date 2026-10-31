#!/usr/bin/env python3
"""Correct the POI values of an impacts json with a frozen parameter (the law impact tasks run this through
Plots/correctImpacts.py, which takes the same options).

    ./scripts/correct_impacts.py --impacts-json impacts/impacts.json --frozen-param MH --drop-bkg-model-params
writes impacts/impacts_corrected_dropBkgModelParams.json (impacts_corrected.json without --drop-bkg-model-params).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from finalfits import impacts  # noqa: E402
from finalfits.utils import add_verbosity_args, setup_logging  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_verbosity_args(parser)
    parser.add_argument("--impacts-json", "--impactsJson", dest="impacts_json", default="impacts.json",
                        help="Impacts json of combineTool.py -M Impacts")  # fmt: skip
    parser.add_argument("--frozen-param", "--frozenParam", dest="frozen_param", default="STXS_constrain_THU_qqH_JET01",
                        help="Frozen parameter from which to extract the true values of the POIs")  # fmt: skip
    parser.add_argument("--drop-bkg-model-params", "--dropBkgModelParams", dest="drop_bkg_model_params",
                        action="store_true", help="Drop the parameters of the background model")  # fmt: skip
    return parser.parse_args()


def main():
    args = get_args()
    setup_logging(args.verbose, args.roofit_verbose)
    impacts.write_corrected_impacts(args.impacts_json, args.frozen_param, args.drop_bkg_model_params)


if __name__ == "__main__":
    main()
