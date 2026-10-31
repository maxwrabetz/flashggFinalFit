#!/usr/bin/env python3
"""Correlation (or covariance) matrix of the POIs of a robustHesse fit (the law tasks AsimovCovCorr and
UnblindedCovCorr run this through Plots/makeCorrMatrix.py, which takes the same options).

    ./scripts/plot_corr_matrix.py --input hesse/robustHessefirstStep.root --mode NJ --output hesse/Plots \\
        --input-json Plots/inputs_robustHesse.json --translate Plots/poi_differential_hesse_noLabels.json [--do-cov]
writes <output>/corrMatrix_<mode>_<fit type>[_obs]<ext>.png/.pdf (covMatrix_... with --do-cov).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from finalfits import correlation_plots  # noqa: E402
from finalfits.utils import add_verbosity_args, setup_logging  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_verbosity_args(parser)
    parser.add_argument("--input-json", "--inputJson", dest="input_json", default="inputs.json",
                        help="Json with the POIs and the fit of every mode")  # fmt: skip
    parser.add_argument("--mode", default="", help="Fit mode (key of the input json), e.g. NJ")
    parser.add_argument("--ext", default="", help="Extension of the output names")
    parser.add_argument("--translate", default=None, help="Json with the display names of the POIs and modes")
    parser.add_argument("--do-cov", "--doCov", dest="do_cov", action="store_true", help="Covariance matrix")
    parser.add_argument("--input", default="./robustHessefirstStep.root", help="robustHesse output of combine")
    parser.add_argument("--output", default="", help="Output directory")
    parser.add_argument("--drop-thq", "--dropTHQ", dest="drop_thq", action="store_true", help="Drop r_tHq")
    parser.add_argument("--do-observed", "--doObserved", dest="do_observed", action="store_true",
                        help="Observed (data) fit: CMS label without Simulation, _obs in the names")  # fmt: skip
    parser.add_argument("--no-preliminary", "--noPreliminary", dest="no_preliminary", action="store_true",
                        help="CMS label without Preliminary")  # fmt: skip
    parser.add_argument("--year", default="2022", help="Year for the integrated luminosity")
    return parser.parse_args()


def main():
    args = get_args()
    setup_logging(args.verbose, args.roofit_verbose)
    translate = None
    if args.translate is not None:
        with open(args.translate) as f:
            translate = json.load(f)
    correlation_plots.plot_matrix(
        args.input,
        args.input_json,
        args.mode,
        args.output,
        translate,
        cov=args.do_cov,
        observed=args.do_observed,
        preliminary=not args.no_preliminary,
        year=args.year,
        ext=args.ext,
        drop_thq=args.drop_thq,
    )


if __name__ == "__main__":
    main()
