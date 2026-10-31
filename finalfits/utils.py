"""Helpers shared by the finalfits modules and scripts."""

import argparse
import logging

import ROOT

LOG_FORMAT = "%(name)-20s: %(levelname)-8s %(message)s"
LOG_LEVELS = {-2: logging.CRITICAL, -1: logging.ERROR, 0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
ROOFIT_LEVELS = {
    -2: ROOT.RooFit.FATAL,
    -1: ROOT.RooFit.ERROR,
    0: ROOT.RooFit.WARNING,
    1: ROOT.RooFit.PROGRESS,
    2: ROOT.RooFit.INFO,
    3: ROOT.RooFit.DEBUG,
}


def add_verbosity_args(parser: argparse.ArgumentParser) -> None:
    """Add the --verbose and --roofit-verbose options common to all finalfits scripts."""
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=1,
        choices=sorted(LOG_LEVELS),
        help="Set verbosity level for finalFits scripts: -2=CRITICAL, -1=ERROR, 0=WARNING, 1=INFO, 2=DEBUG",
    )
    parser.add_argument(
        "--roofit-verbose",
        type=int,
        default=0,
        choices=sorted(ROOFIT_LEVELS),
        help="Set verbosity level for RooFit: -2=FATAL, -1=ERROR, 0=WARNING, 1=PROGRESS, 2=INFO, 3=DEBUG",
    )


def setup_logging(verbose: int = 1, roofit_verbose: int = 0) -> None:
    """Configure the python logging and the RooFit message service."""
    logging.basicConfig(format=LOG_FORMAT)
    logging.getLogger().setLevel(LOG_LEVELS[verbose])
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOFIT_LEVELS[roofit_verbose])
