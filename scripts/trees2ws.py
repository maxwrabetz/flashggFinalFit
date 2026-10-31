#!/usr/bin/env python3
"""Convert HiggsDNA trees to FinalFits workspaces (the law task Trees2WS runs the same code).

Examples:
    ./scripts/trees2ws.py signal config/2023_NJ.yml output_ggh_M125.root out/ -p ggh -m 125 -y 2023preBPix
    ./scripts/trees2ws.py data config/2023_NJ.yml allData_2023.root out/allData.root
"""

import argparse
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from finalfits import trees2ws  # noqa: E402
from finalfits.utils import add_verbosity_args, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_verbosity_args(parser)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    signal = subparsers.add_parser("signal", help="Signal MC: one workspace per signal process")
    signal.add_argument("config", help="Analysis config, its trees2wsCfg block is used")
    signal.add_argument("in_file", help="HiggsDNA output of one production mode and mass point")
    signal.add_argument("out_dir", help="The workspaces are written to <out_dir>/ws_<process>/")
    signal.add_argument("--production-mode", "-p", required=True, help="e.g. ggh, vbf, vh, tth, bbh")
    signal.add_argument("--mass", "-m", required=True, help="Higgs boson mass of the sample, e.g. 125")
    signal.add_argument("--year", "-y", required=True, help="Replaces YEAR in systematic names, e.g. 2023preBPix")
    signal.add_argument("--variable", default="", help="Differential variable for doDiffSplitting, e.g. NJ")

    data = subparsers.add_parser("data", help="Data: one workspace with all categories")
    data.add_argument("config", help="Analysis config, its trees2wsCfg block is used")
    data.add_argument("in_file", help="HiggsDNA output for data (all categories)")
    data.add_argument("out_file", help="Output workspace file, e.g. allData.root")
    return parser.parse_args()


def main():
    args = get_args()
    setup_logging(args.verbose, args.roofit_verbose)
    with open(args.config) as f:
        cfg = trees2ws.Trees2WSConfig.from_dict(yaml.safe_load(f)["trees2wsCfg"])

    if args.mode == "signal":
        paths = trees2ws.convert_signal_file(
            args.in_file, args.out_dir, args.production_mode, args.mass, args.year, cfg, args.variable
        )
        logger.info("Wrote %d workspaces", len(paths))
    else:
        trees2ws.convert_data_file(args.in_file, args.out_file, cfg)


if __name__ == "__main__":
    main()
