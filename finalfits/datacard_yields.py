"""Yields of all processes in one category, with the yield variations of every systematic.

The result is a pandas DataFrame (one row per process, as read by :mod:`finalfits.datacard`):
signal rows per year with their nominal yield (sum of weights of the Trees2WS dataset), sum of
weights squared and, for every factory systematic, the varied yields; plus background and data rows.

Systematics are of three kinds (see :func:`factory_type`):

* ``a_w``: up/down weights in the nominal dataset (e.g. ``weight_PileupUp``)
* ``s_w``: a single weight in the nominal dataset (e.g. ``weight_LHEScal_0``)
* ``a_h``: up/down RooDataHists from Trees2WS (e.g. ``..._JecSystTotalUp01sigma``)
"""

from __future__ import annotations

import glob
import logging
import os
import pickle

import numpy as np
import pandas as pd
import ROOT

from finalfits import analysis
from finalfits.signal_ftest import dataset_name
from finalfits.trees2ws import SQRTS, WS_DIR, WS_NAME

logger = logging.getLogger(__name__)

JET_SYSTEMATICS = ("JecSystTotal", "JerSyst")
COLUMNS = [
    "year",
    "type",
    "procOriginal",
    "proc",
    "proc_s0",
    "cat",
    "inputWSFile",
    "nominalDataName",
    "modelWSFile",
    "model",
    "rate",
]


def experimental_systematics_for(variable: str, systematics: list[dict]) -> list[dict]:
    """The jet energy systematics only apply to jet variables of a differential analysis."""
    if variable != "" and variable in analysis.jetVariables:
        return list(systematics)
    return [s for s in systematics if s["name"] not in JET_SYSTEMATICS]


def _sequential_sum(values) -> float:
    # summed event by event, in the same order as the datasets were always summed
    return float(np.cumsum(values)[-1]) if len(values) else 0


def factory_type(rows: pd.DataFrame, syst: dict) -> str:
    """Kind of a factory systematic (a_w, s_w, a_h), from the first signal workspace that has it."""
    if "weight_LHEPd" in syst["name"]:
        return "s_w"
    for _, row in rows[rows["type"] == "sig"].iterrows():
        f = ROOT.TFile(row.inputWSFile)
        ws = f.Get(f"{WS_DIR}/{WS_NAME}")
        n_weights = ws.allVars().selectByName(f"{syst['name']}*").getSize()
        has_hists = bool(ws.data(f"{row.nominalDataName}_{syst['name']}Up01sigma")) and bool(
            ws.data(f"{row.nominalDataName}_{syst['name']}Down01sigma")
        )
        ws.Delete()
        f.Close()
        if n_weights:
            if n_weights > 2:
                raise ValueError(f"Systematic {syst['name']}: {n_weights} > 2 weights in the workspace")
            return "a_w" if n_weights == 2 else "s_w"
        if has_hists:
            return "a_h"
    raise ValueError(f"Systematic {syst['name']}: no (anti-)symmetric weights or histograms found")


def _central_weight(columns: dict, contents: str, name: str) -> np.ndarray:
    if name not in contents:
        raise KeyError(f"Central weight {name} is not in the nominal dataset")
    return columns[name]


def systematic_yields(
    dataset,
    contents,
    ws,
    types: dict,
    skip_cow_corr=True,
    proc="",
    year="",
    weight_scheme="accEff",
    ignore_warnings=False,
) -> dict:
    """Varied yields of the systematics ``types`` (name -> factory type) for one nominal dataset.

    ``contents`` lists the variables of the dataset (RooArgSet::contentsString); weights that are
    missing are replaced by the nominal weight if ``ignore_warnings``, else they are an error.
    """
    columns = dataset.to_numpy()
    w = columns[dataset.weightVar().GetName()] if dataset.numEntries() else np.zeros(0)
    n = len(w)
    ones = np.ones(n)
    f_cow = columns["centralObjectWeight"] if "centralObjectWeight" in contents else ones
    f_nnlops = np.abs(columns["NNLOPSweight"]) if "NNLOPSweight" in contents else ones
    cow = f_cow != 0
    cow_factor = np.divide(f_nnlops, f_cow, out=np.zeros(n), where=cow)

    def missing(name):
        message = f"Weight for systematic {name} does not exist in the nominal dataset of ({proc}, {year})"
        if not ignore_warnings:
            raise KeyError(message)
        logger.warning("%s: using the nominal yield", message)

    if n < 100:
        logger.warning("Less than 100 events in %s: its variations will be excluded", dataset.GetName())
    yields = {"numEvents": dataset.numEntries()}

    def add(key, contributions, cow_contributions=None):
        yields[key] = _sequential_sum(contributions)
        if not skip_cow_corr:
            cow_contributions = contributions * cow_factor if cow_contributions is None else cow_contributions
            yields[f"{key}_COWCorr"] = _sequential_sum(np.where(cow, cow_contributions, 0.0))

    for name, kind in types.items():
        if kind == "a_w":
            if f"{name}Up" not in contents or f"{name}Down" not in contents:
                missing(name)
                for direction in ("up", "down"):
                    add(f"{name}_{direction}", w)
                continue
            f_central = _central_weight(columns, contents, "weight_central")
            f_up, f_down = columns[f"{name}Up"], columns[f"{name}Down"]
            if weight_scheme == "legacyHiggsDNA":
                w_up, w_down = w * (f_up / f_central), w * (f_down / f_central)
            elif weight_scheme == "accEff":
                w_up, w_down = f_up, f_down
            else:
                raise ValueError(f"Unknown systematic weight scheme {weight_scheme}")
            # events with a zero central weight are skipped, equal up/down weights keep the nominal one
            same = f_up == f_down
            keep = f_central != 0
            for direction, varied in (("up", w_up), ("down", w_down)):
                add(f"{name}_{direction}", np.where(keep, np.where(same, w, varied), 0.0))
        elif kind == "s_w":
            if name not in contents:
                missing(name)
                add(name, w)
                continue
            central = (
                "weight_LHEScal_4"
                if "LHEScal" in name
                else "weight_LHEPd_0"
                if "LHEPd" in name
                else "centralObjectWeight"
            )
            f_central = columns[central] if central in contents else ones
            f = columns[name]
            both_zero = (f_central == 0) & (f == 0)
            ratio = np.divide(f, f_central, out=np.zeros(n), where=f_central != 0)
            if weight_scheme == "legacyHiggsDNA" or (
                weight_scheme == "accEff" and ("LHEScal" in name or "LHEPd" in name)
            ):
                varied = w * ratio
            elif weight_scheme == "accEff":
                varied = w
            else:
                raise ValueError(f"Unknown systematic weight scheme {weight_scheme}")
            contributions = np.where(both_zero, w, np.where(f_central == 0, 0.0, varied))
            cow_contributions = np.where(both_zero, w * cow_factor, w * cow_factor * ratio)
            add(
                name,
                contributions,
                np.where(f_central == 0, np.where(both_zero, cow_contributions, 0.0), cow_contributions),
            )

    for name, kind in types.items():
        if kind != "a_h":
            continue
        up = ws.data(f"{dataset.GetName()}_{name}Up01sigma")
        down = ws.data(f"{dataset.GetName()}_{name}Down01sigma")
        if not up or not down:
            message = f"RooDataHist for systematic {name} does not exist for ({proc}, {year})"
            if not ignore_warnings:
                raise KeyError(message)
            logger.warning("%s: using the nominal yield", message)
            yields[f"{name}_up"] = yields[f"{name}_down"] = dataset.sumEntries()
        else:
            yields[f"{name}_up"], yields[f"{name}_down"] = up.sumEntries(), down.sumEntries()
    return yields


def make_yields(
    input_ws_dirs: dict[str, str],
    cat: str,
    procs: list[str] | str = "auto",
    variable: str = "",
    mass: str = "125",
    merge_years: bool = False,
    skip_bkg: bool = False,
    bkg_scaler: float = 1.0,
    sig_model_dir: str = "./Models/signal",
    sig_model_ext: str = "packaged",
    bkg_model_dir: str = "./Models/background",
    bkg_model_ext: str = "multipdf",
    skip_zeroes: bool = False,
    skip_cow_corr: bool = False,
    do_systematics: bool = False,
    ignore_warnings: bool = False,
    weight_scheme: str = "accEff",
) -> pd.DataFrame:
    """Yields table of one category; ``input_ws_dirs`` maps the years to their Trees2WS signal workspaces."""
    years = list(input_ws_dirs)
    if procs == "auto":
        procs_by_year = {
            y: analysis.extractListOfProcs(analysis.extractWSFileNames(d)).split(",") for y, d in input_ws_dirs.items()
        }
        if any(set(procs_by_year[y]) != set(procs_by_year[years[0]]) for y in years):
            raise ValueError(f"Different processes in the years {years}")
        procs = procs_by_year[years[0]]
    procs = sorted(procs)

    data = pd.DataFrame(columns=COLUMNS)
    ws_name = f"{analysis.outputWSName__}_{SQRTS}"
    for year in years:
        for proc in procs:
            name = dataset_name(proc, mass, cat)
            path = glob.glob(f"{input_ws_dirs[year]}/*M{mass}*_{proc}.root")[0]
            if skip_zeroes:
                f = ROOT.TFile(path)
                ws = f.Get(f"{WS_DIR}/{WS_NAME}")
                zero = ws.data(name).sumEntries() == 0.0
                ws.Delete()
                f.Close()
                if zero:
                    continue
            row_cat = cat if merge_years else f"{cat}_{year}"
            if cat == "NOTAG":
                model_file = model = "-"
            else:
                model_file = f"{sig_model_dir}/CMS-HGG_sigfit_{sig_model_ext}_{row_cat}.root"
                model = f"{ws_name}:{analysis.outputWSObjectTitle__}_{proc}_{year}_{cat}_{SQRTS}"
            datacard_proc = f"{analysis.procToDatacardName(proc)}_{year}_{analysis.decayMode}"
            data.loc[len(data)] = [
                year,
                "sig",
                proc,
                datacard_proc,
                analysis.procToData(proc.split("_")[0]),
                row_cat,
                path,
                name,
                model_file,
                model,
                float(analysis.lumiMap[year]) * 1000,
            ]

    if not skip_bkg and cat != "NOTAG":
        bkg_ws = analysis.bkgWSName__
        for year, row_cat in [("merged", cat)] if merge_years else [("year", f"{cat}_{y}") for y in years]:
            model_file = f"{bkg_model_dir}/CMS-HGG_{bkg_model_ext}_{row_cat}.root"
            bkg_model = f"{bkg_ws}:CMS_{analysis.decayMode}_{row_cat}_{SQRTS}_bkgshape"
            data_model = f"{bkg_ws}:roohist_data_mass_{cat}"
            data.loc[len(data)] = [
                year,
                "bkg",
                "bkg_mass",
                "bkg_mass",
                "-",
                row_cat,
                "-",
                "-",
                model_file,
                bkg_model,
                bkg_scaler,
            ]
            data.loc[len(data)] = [
                year,
                "data",
                "data_obs",
                "data_obs",
                "-",
                row_cat,
                "-",
                "-",
                model_file,
                data_model,
                -1,
            ]

    data["nominal_yield"] = "-"
    data["sumw2"] = "-"
    if not skip_cow_corr:
        data["nominal_yield_COWCorr"] = "-"

    experimental_types, theory_types = {}, {}
    if do_systematics:
        experimental, theory, _ = analysis.datacard_systematics()
        if cat != "NOTAG":
            for s in experimental_systematics_for(variable, experimental):
                if s["type"] == "factory":
                    experimental_types[s["name"]] = "a_h" if s["name"] == "JetHEM" else factory_type(data, s)
                    _add_yield_columns(data, s["name"], experimental_types[s["name"]], cow=False)
        for s in theory:
            if s["type"] == "factory":
                theory_types[s["name"]] = factory_type(data, s)
                _add_yield_columns(data, s["name"], theory_types[s["name"]], cow=not skip_cow_corr)

    for i, row in data[data["type"] == "sig"].iterrows():
        logger.info("Yields of (%s, %s)", row["proc"], row["cat"])
        f = ROOT.TFile(row.inputWSFile)
        ws = f.Get(f"{WS_DIR}/{WS_NAME}")
        dataset = ws.data(row.nominalDataName)
        contents = dataset.get(0).contentsString() if dataset.numEntries() else ""
        w = dataset.to_numpy()[dataset.weightVar().GetName()] if dataset.numEntries() else np.zeros(0)
        data.at[i, "nominal_yield"] = _sequential_sum(w)
        data.at[i, "sumw2"] = _sequential_sum(w * w)
        if not skip_cow_corr:
            data.at[i, "nominal_yield_COWCorr"] = _cow_corrected_yield(dataset, contents, w)
        if do_systematics:
            options = {
                "proc": row["proc"],
                "year": row["year"],
                "weight_scheme": weight_scheme,
                "ignore_warnings": ignore_warnings,
            }
            if "NOTAG" not in row["cat"]:
                varied = systematic_yields(dataset, contents, ws, experimental_types, skip_cow_corr=True, **options)
                _store(data, i, varied, experimental_types, cow=False)
            varied = systematic_yields(dataset, contents, ws, theory_types, skip_cow_corr=skip_cow_corr, **options)
            _store(data, i, varied, theory_types, cow=not skip_cow_corr)
        ws.Delete()
        f.Close()
    return data


def _cow_corrected_yield(dataset, contents, w):
    columns = dataset.to_numpy() if len(w) else {}
    f_cow = columns["centralObjectWeight"] if "centralObjectWeight" in contents else np.ones(len(w))
    f_nnlops = np.abs(columns["NNLOPSweight"]) if "NNLOPSweight" in contents else np.ones(len(w))
    keep = f_cow != 0
    return _sequential_sum(np.where(keep, w * np.divide(f_nnlops, f_cow, out=np.zeros(len(w)), where=keep), 0.0))


def _add_yield_columns(data, name, kind, cow):
    keys = [f"{name}_up_yield", f"{name}_down_yield"] if kind in ("a_w", "a_h") else [f"{name}_yield"]
    for key in keys:
        data[key] = "-"
        if cow:
            data[f"{key}_COWCorr"] = "-"


def _store(data, i, varied, types, cow):
    for name, kind in types.items():
        data.at[i, "numEvents"] = varied["numEvents"]
        for key in [f"{name}_up", f"{name}_down"] if kind in ("a_w", "a_h") else [name]:
            data.at[i, f"{key}_yield"] = varied[key]
            if cow:
                data.at[i, f"{key}_yield_COWCorr"] = varied[f"{key}_COWCorr"]


def write_yields(data: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
