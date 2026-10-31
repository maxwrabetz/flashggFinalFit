"""Package the signal models of all processes (and eras) of one category into one workspace.

The workspace ``wsig_13TeV`` holds every pdf, function and variable of the signal fits (shared
objects such as MH are taken from the first file) and their datasets, plus the merged datasets
``sig_mass_m<mass>_<cat>`` with the simulation of all processes and eras.
"""

from __future__ import annotations

import glob
import logging
import os

import ROOT

from finalfits import analysis
from finalfits.trees2ws import SQRTS

logger = logging.getLogger(__name__)


def signal_fit_files(output_dir: str, ext: str, cat: str) -> list[str]:
    # glob order (directory order), as the packaged workspaces were always built
    return glob.glob(f"{output_dir}/outdir_{ext}/signalFit/output/CMS-HGG_sigfit_{ext}_*_{cat}.root")


def package_signal(output_dir: str, cat: str, exts: list[str], mass_points: list[str], output_file: str) -> str:
    ws_name = f"{analysis.outputWSName__}_{SQRTS}"
    files = [path for ext in exts for path in signal_fit_files(output_dir, ext, cat)]
    if not files:
        raise FileNotFoundError(f"No signal fits for {cat} in {output_dir}/outdir_{{{','.join(exts)}}}")
    logger.info("Packaging %d signal models of %s", len(files), cat)
    packaged = ROOT.RooWorkspace(ws_name, ws_name)
    keep_open = [ROOT.TFile(path) for path in files]
    workspaces = [f.Get(ws_name) for f in keep_open]

    merged = {}
    for mass in mass_points:
        name = f"sig_mass_m{mass}_{cat}"
        merged[mass] = workspaces[0].data(name).emptyClone(name)
        for ws in workspaces:
            merged[mass].append(ws.data(name))
        packaged.Import(merged[mass])
    merged_names = {d.GetName() for d in merged.values()}

    options = (ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())
    for ws in workspaces:
        if not ws:
            continue
        for collection in (ws.allVars(), ws.allFunctions(), ws.allPdfs()):
            for obj in collection:
                packaged.Import(obj, *options)
        for data in ws.allData():
            if data.GetName() not in merged_names:
                packaged.Import(data)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    f = ROOT.TFile(output_file, "RECREATE")
    packaged.Write()
    f.Close()
    for f in keep_open:
        f.Close()
    return output_file
