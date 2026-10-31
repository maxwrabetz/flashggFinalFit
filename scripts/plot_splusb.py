#!/usr/bin/env python3
"""Signal + background model plots of the diphoton mass (the law task MggDistribution runs this through
Plots/makeSplusBModelPlot.py, which takes the same options).

    ./scripts/plot_splusb.py --input-ws-file Datacard.root --cats cat0,cat1,cat2 --do-sum-categories --do-cat-weights \\
        --translate-cats Plots/cats.json --ext _r [--do-bands --load-snapshot MultiDimFit --unblind ...]
writes ./SplusBModels<ext>/<ext>_<cat>_<xvar>.png/.pdf (cat all/wall: the sum and the S/(S+B) weighted sum).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from finalfits import splusb_plots  # noqa: E402
from finalfits.utils import add_verbosity_args, setup_logging  # noqa: E402

# (name, old name, keyword arguments of add_argument)
OPTIONS = [
    ("input-ws-file", "inputWSFile", {"required": True, "help": "Workspace (post-fit: saved with --saveWorkspace)"}),
    ("load-snapshot", "loadSnapshot", {"default": None, "help": "Load the best-fit snapshot of this name"}),
    ("cats", None, {"required": True, "help": "Comma separated analysis categories, or all"}),
    ("unblind", None, {"action": "store_true", "help": "Unblind the signal region"}),
    ("blinding-region", "blindingRegion", {"default": "116,134", "help": "Blinded mass region"}),
    ("data-scaler", "dataScaler", {"default": 1.0, "type": float, "help": "Scale of the data histograms"}),
    ("do-bkg-renormalization", "doBkgRenormalization", {"action": "store_true", "help": "Renormalise B to S+B - S"}),
    ("do-bands", "doBands", {"action": "store_true", "help": "+-1/2 sigma bands of the bkg model from toys"}),
    ("do-toy-veto", "doToyVeto", {"action": "store_true", "help": "Veto toys with no entries in the first bin"}),
    ("load-toy-yields", "loadToyYields", {"default": "", "help": "Pkl with the toy yields (instead of the toys)"}),
    ("save-toy-yields", "saveToyYields", {"action": "store_true", "help": "Save the toy yields to a pkl"}),
    ("do-zeroes", "doZeroes", {"action": "store_true", "help": "Error of one event on empty bins, to show them"}),
    ("skip-individual-cat-plots", "skipIndividualCatPlots", {"action": "store_true", "help": "Only the sum plots"}),
    ("do-sum-categories", "doSumCategories", {"action": "store_true", "help": "Plot of the sum of the categories"}),
    ("do-cat-weights", "doCatWeights", {"action": "store_true", "help": "S/(S+B) weighted sum plot"}),
    ("load-weights", "loadWeights", {"default": "", "help": "Json with the category weights"}),
    ("save-weights", "saveWeights", {"action": "store_true", "help": "Save the category weights to ./jsons"}),
    ("parameter-map", "parameterMap", {"default": None, "help": "Set parameters: name:value,..."}),
    ("ext", None, {"default": "", "help": "Extension of the output directory and names"}),
    ("mass", None, {"default": None, "help": "Higgs mass"}),
    ("xvar", None, {"default": "CMS_hgg_mass,m_{#gamma#gamma},GeV", "help": "Mass variable: name,title,unit"}),
    ("n-bins", "nBins", {"default": 80, "type": int, "help": "Number of bins of the data"}),
    ("pdf-n-bins", "pdfNBins", {"default": 3200, "type": int, "help": "Number of bins of the drawn pdfs"}),
    ("translate-cats", "translateCats", {"default": None, "help": "Json with the display names of the categories"}),
    ("translate-pois", "translatePOIs", {"default": None, "help": "Json with the display names of the POIs (unused)"}),
    ("problematic-cats", "problematicCats", {"default": "", "help": "Categories to skip"}),
    ("do-hh-mjj-fix", "doHHMjjFix", {"action": "store_true", "help": "Mjj_90GeV variable of the HH categories"}),
    ("poi", "POI", {"default": "", "help": "POI whose best fit is shown (with --load-snapshot)"}),
    ("lumi-label", "lumiLabel", {"default": "61.9 fb^{-1} (13.6 TeV)", "help": "Luminosity label"}),
]


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_verbosity_args(parser)
    for name, old_name, kwargs in OPTIONS:
        flags = [f"--{name}"] + ([f"--{old_name}"] if old_name else [])
        parser.add_argument(*flags, dest=name.replace("-", "_"), **kwargs)
    return parser.parse_args()


def main():
    args = vars(get_args())
    setup_logging(args.pop("verbose"), args.pop("roofit_verbose"))
    args.pop("translate_pois")
    splusb_plots.make_splusb_plots(splusb_plots.SplusBOptions(**args))


if __name__ == "__main__":
    main()
