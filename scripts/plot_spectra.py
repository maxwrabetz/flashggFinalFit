#!/usr/bin/env python3
"""Differential cross-section spectrum from the likelihood scans (the law tasks CreateDiffSpectra and
CreateTotalDiffSpectra run finalfits.spectra the same way).

    ./scripts/plot_spectra.py --year 2023 --variable NJ [--output-dir <dir>] [--total] [--unblinded] [--log-x]
reads <output dir>/Combine/runFits_<variable>/asimov (dataFit with --unblinded) and writes
<output dir>/Combine/runFits_<variable>/spectra[_total][_logx][_blinded].pdf/.png. The analysis config is
config/<year>_<variable>.yml (its outputFolder is the default output dir), the plot layout
Plots/Spectra/config/<year>_<variable>.yml.
"""

import argparse
import os
import sys

import yaml

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
# the theory input modules (fidXS_*, totXS_*)
sys.path.append(os.path.join(REPO, "Plots", "Spectra", "fidXS"))

from finalfits import spectra  # noqa: E402
from finalfits.analysis import combineVariableDict  # noqa: E402
from finalfits.utils import add_verbosity_args, setup_logging  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_verbosity_args(parser)
    parser.add_argument("--year", required=True, help="Year, e.g. 2023 or 2022_2023_2024")
    parser.add_argument("--variable", required=True, help="Differential variable, e.g. NJ")
    parser.add_argument("--output-dir", default="", help="Output directory of the analysis (default: outputFolder)")
    parser.add_argument("--config", default="", help="Analysis config (default: config/<year>_<variable>.yml)")
    parser.add_argument("--plot-config", default="", help="Plot layout (default: Plots/Spectra/config/...)")
    parser.add_argument("--total", action="store_true", help="Extrapolated to the total phase space (totXS)")
    parser.add_argument("--unblinded", action="store_true", help="Scans of the fit to data")
    parser.add_argument("--log-x", action="store_true", help="Logarithmic x axis")
    return parser.parse_args()


def main():
    args = get_args()
    setup_logging(args.verbose, args.roofit_verbose)
    name = f"{args.year}_{args.variable}.yml"
    with open(args.config or os.path.join(REPO, "config", name)) as f:
        config = yaml.safe_load(f)
    with open(args.plot_config or os.path.join(REPO, "Plots", "Spectra", "config", name)) as f:
        plotting_config = yaml.safe_load(f)
    output_dir = args.output_dir or config["outputFolder"]
    spectra.make_spectrum(
        os.path.join(output_dir, "Combine", f"runFits_{args.variable}"),
        combineVariableDict(args.variable, args.year)["paramStrNoOne"],
        config,
        plotting_config,
        args.year,
        spectra.TOTAL if args.total else spectra.FIDUCIAL,
        unblinded=args.unblinded,
        log_x=args.log_x,
    )


if __name__ == "__main__":
    main()
