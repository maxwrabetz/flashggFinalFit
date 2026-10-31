"""Convert HiggsDNA ROOT trees into the RooWorkspaces read by the signal and background modelling.

This is the only module that knows the HiggsDNA output format, so changes upstream of FinalFits
should only need changes here.

Signal MC
    One input file per production mode and mass point. The directory ``inputTreeDir`` holds a tree
    ``<proc>_<mass>_13TeV_<cat>`` per analysis category and a tree
    ``<proc>_<mass>_13TeV_<cat>_<syst>{Up,Down}01sigma`` per systematic variation. The events are
    split into signal processes (fiducial in/out or the bins of a differential variable), and every
    process is written to its own workspace ``ws_<process>/<input>_<process>.root`` with

    * a weighted RooDataSet ``<proc>_<mass>_13TeV_<cat>`` per category (all ``mainVars``), and
    * a RooDataHist ``<proc>_<mass>_13TeV_<cat>_<syst>{Up,Down}01sigma`` per systematic variation
      (the ``systematicsVars``, without the weight and the branches used for the splitting).

Data
    One tree ``Data_13TeV_<cat>`` per category, written as unit-weight RooDataSets
    ``Data_13TeV_<cat>`` into a single workspace.

The workspaces are stored as ``tagsDumper/cms_hgg_13TeV``, where the later steps expect them.
Events are read with uproot as dicts of numpy arrays ("columns", one array per branch).
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import awkward as ak
import numpy as np
import ROOT
import uproot

from finalfits import analysis

logger = logging.getLogger(__name__)

SQRTS = "13TeV"
WS_DIR = "tagsDumper"
WS_NAME = "cms_hgg_13TeV"
MASS = "CMS_hgg_mass"
WEIGHT = "weight"
FIDUCIAL_FLAG = "fiducialGeometricFlag"

# Production modes whose trees have no theory weights: they are filled with ones
MODES_WITHOUT_THEORY_WEIGHTS = ("thq", "thw")

# Particle-level branch that defines the bins of each differential variable
GEN_VARIABLES = {
    "PTH": "GenPTH",
    "rapidity": "GenYH",
    "NJ": "GenNJ",
    "PTJ0": "GenPTJ0",
    "CosThetaStarCS": "GenCosThetaStarCS",
    "PhiEtaStar": "GenPhiEtaStar",
    "NBJet": "GenNBJet",
    "YJ0": "GenYJ0",
    "DPhiHJ0": "GenDPhiHJ0",
    "DYHJ0": "GenDYHJ0",
    "TauJC": "GenTauJC",
    "PTJ1": "GenPTJ1",
    "YJ1": "GenYJ1",
    "DPhiJ0J1": "GenDPhiJ0J1",
    "DPhiHJ0J1": "GenDPhiHJ0J1",
    "DEtaJ0J1H": "GenDEtaJ0J1H",
    "MassJ0J1": "GenMassJ0J1",
    "EtaJ0J1": "GenEtaJ0J1",
}
# Variables binned in their signed value (they have a dedicated negative bin, e.g. events without
# jets); all other variables are binned in their absolute value
SIGNED_VARIABLES = {
    "PTJ0",
    "YJ0",
    "DPhiHJ0",
    "DYHJ0",
    "TauJC",
    "PTJ1",
    "YJ1",
    "DPhiJ0J1",
    "DPhiHJ0J1",
    "DEtaJ0J1H",
    "MassJ0J1",
    "EtaJ0J1",
}
# Two-dimensional variables: their bins are selected by the HiggsDNA bin index only
TWO_D_VARIABLES = {"PTHvDPhiJ0J1", "PTHvYH"}

Columns = dict[str, np.ndarray]


def _as_bool(value) -> bool:
    # law parameters given on the command line arrive as strings
    return value in (True, "True", "true")


@dataclass
class Trees2WSConfig:
    """The ``trees2wsCfg`` block of an analysis config (``config/<year>_<variable>.yml``)."""

    input_tree_dir: str
    main_vars: list[str]
    data_vars: list[str]
    systematics_vars: list[str]
    systematics: list[str] = field(default_factory=list)
    theory_weight_containers: dict[str, int] = field(default_factory=dict)
    diff_var: str = ""
    cats: str | list[str] = "auto"
    do_systematics: bool = False
    do_diff_splitting: bool = False
    do_inout_splitting: bool = False
    do_stxs_splitting: bool = False
    apply_mass_cut: bool = False
    mass_cut_range: tuple[float, float] = (100.0, 180.0)

    @classmethod
    def from_dict(cls, cfg: dict) -> Trees2WSConfig:
        low, high = (float(x) for x in str(cfg.get("mass_cut_range", "100,180")).split(","))
        return cls(
            input_tree_dir=cfg["inputTreeDir"],
            main_vars=list(cfg["mainVars"]),
            data_vars=list(cfg["dataVars"]),
            systematics_vars=list(cfg["systematicsVars"]),
            systematics=list(cfg.get("systematics") or []),
            theory_weight_containers=dict(cfg.get("theoryWeightContainers") or {}),
            diff_var=cfg.get("diffVar") or "",
            cats=cfg.get("cats", "auto"),
            do_systematics=_as_bool(cfg.get("doSystematics", False)),
            do_diff_splitting=_as_bool(cfg.get("doDiffSplitting", False)),
            do_inout_splitting=_as_bool(cfg.get("doInOutSplitting", False)),
            do_stxs_splitting=_as_bool(cfg.get("doSTXSSplitting", False)),
            apply_mass_cut=_as_bool(cfg.get("apply_mass_cut", False)),
            mass_cut_range=(low, high),
        )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Reading the trees


def theory_weight_names(container: str, n_weights: int) -> list[str]:
    """Columns for the entries of an array branch, e.g. weight_LHEPdf -> weight_LHEPd_0, ...

    The last character of the container name is dropped (it used to turn ``pdfWeights`` into
    ``pdfWeight_0``); the datacard step relies on these names.
    """
    return [f"{container[:-1]}_{i}" for i in range(n_weights)]


def list_categories(directory) -> list[str]:
    """Analysis categories of the nominal trees in an uproot directory, in file order."""
    cats = []
    for name in directory.keys(recursive=False, cycle=False, filter_classname="TTree"):
        if name.endswith("01sigma"):
            continue
        cat = name.split(f"_{SQRTS}_")[-1]
        if cat not in cats:
            cats.append(cat)
    return cats


def n_events(columns: Columns) -> int:
    return len(next(iter(columns.values()), ()))


def _select(columns: Columns, mask: np.ndarray) -> Columns:
    return {name: values[mask] for name, values in columns.items()}


def _apply_mass_cut(columns: Columns, cfg: Trees2WSConfig) -> Columns:
    if not cfg.apply_mass_cut:
        return columns
    low, high = cfg.mass_cut_range
    return _select(columns, (columns[MASS] >= low) & (columns[MASS] <= high))


def _is_split_column(name: str) -> bool:
    # Branches only used to split the signal into processes are not stored in the workspace
    return "fiducial" in name or "diff" in name


def _read_branches(tree, names: list[str], extra_columns: tuple[str, ...]) -> Columns:
    """Read ``names`` and those ``extra_columns`` that exist in the tree."""
    branches = set(tree.keys())
    names = names + [c for c in extra_columns if c in branches and c not in names]
    return tree.arrays(names, library="np") if names else {}


def _read_nominal_tree(tree, cfg: Trees2WSConfig, production_mode: str) -> Columns:
    columns = {}
    for container, n_weights in cfg.theory_weight_containers.items():
        if production_mode in MODES_WITHOUT_THEORY_WEIGHTS:
            weights = np.ones((tree.num_entries, n_weights))
        else:
            weights = ak.to_numpy(tree[container].array()).reshape(tree.num_entries, n_weights)
        columns.update(zip(theory_weight_names(container, n_weights), weights.T))
    columns.update(tree.arrays([v for v in cfg.main_vars if "*" not in v], library="np"))
    for pattern in (v for v in cfg.main_vars if "*" in v):
        columns.update(tree.arrays(filter_name=pattern, library="np"))
    return columns


@dataclass
class SignalEvents:
    """The events of one signal MC file."""

    categories: list[str]
    # Columns stored in the workspace (mainVars without the branches used for the splitting)
    variables: list[str]
    nominal: dict[str, Columns]
    # (category, "<syst>Up"/"<syst>Down") -> events of the systematic variation
    systematics: dict[tuple[str, str], Columns]


def read_signal_events(
    path: str | Path,
    cfg: Trees2WSConfig,
    production_mode: str,
    mass: int | str,
    year: str,
    split_columns: tuple[str, ...] = (),
) -> SignalEvents:
    """Read the nominal and systematic trees of a signal MC file.

    The ``split_columns`` (needed to split the events into processes) are read from all trees
    where they exist, in addition to the ``mainVars`` and ``systematicsVars``.
    ``YEAR`` in a systematic's name is replaced by ``year`` when looking up its trees.
    """
    variables, nominal, systematics = [], {}, {}
    with uproot.open(path) as f:
        directory = f[cfg.input_tree_dir] if cfg.input_tree_dir else f
        cats = list_categories(directory) if cfg.cats == "auto" else list(cfg.cats)
        for cat in cats:
            tree_name = f"{production_mode}_{mass}_{SQRTS}_{cat}"
            tree = directory[tree_name]
            logger.info("Reading %s (%d events)", tree_name, tree.num_entries)
            columns = _read_nominal_tree(tree, cfg, production_mode)
            variables += [name for name in columns if name not in variables and not _is_split_column(name)]
            columns.update(_read_branches(tree, [], tuple(c for c in split_columns if c not in columns)))
            nominal[cat] = _apply_mass_cut(columns, cfg)
            # Systematic variations are only kept for categories with nominal events
            if not cfg.do_systematics or tree.num_entries == 0:
                continue
            for syst in cfg.systematics:
                for direction in ("Up", "Down"):
                    syst_tree = directory[f"{tree_name}_{syst}{direction}01sigma".replace("YEAR", year)]
                    if syst_tree.num_entries == 0:
                        continue
                    columns = _read_branches(syst_tree, list(cfg.systematics_vars), split_columns)
                    systematics[(cat, f"{syst}{direction}")] = _apply_mass_cut(columns, cfg)
    return SignalEvents(cats, variables, nominal, systematics)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Splitting the signal into processes

# Events of a tree that belong to a process, or None if the tree lacks the branches to decide
Selector = Callable[[Columns], Optional[np.ndarray]]


@dataclass(frozen=True)
class Split:
    """One signal process, written to its own workspace."""

    label: str  # bin name or fiducial tag
    ws_proc: str  # output ws_<ws_proc>/<input>_<ws_proc>.root
    dataset_proc: str  # datasets <dataset_proc>_<mass>_13TeV_<cat>
    select: Selector


def _select_fiducial(columns: Columns, fiducial: bool) -> np.ndarray | None:
    return columns[FIDUCIAL_FLAG] == fiducial if FIDUCIAL_FLAG in columns else None


def inout_splits(production_mode: str) -> list[Split]:
    """Split into events inside and outside the fiducial phase space."""
    proc = analysis.dataToProc(production_mode)
    return [
        Split(
            tag, f"{proc}_{tag}", f"{production_mode}_{tag}", functools.partial(_select_fiducial, fiducial=tag == "in")
        )
        for tag in ("in", "out")
    ]


@dataclass(frozen=True)
class DiffBin:
    """A bin of a differential variable, e.g. ``NJ_1p0_2p0_in`` with HiggsDNA index 31."""

    name: str
    higgsdna_index: int
    fiducial: bool
    bounds: tuple[float, float] | None  # None for two-dimensional bins

    @classmethod
    def from_name(cls, higgsdna_index: int, name: str, two_dimensional: bool = False) -> DiffBin:
        if name.endswith("_in"):
            fiducial = True
        elif name.endswith("_out"):
            fiducial = False
        else:
            raise ValueError(f"Bin name {name} does not end with '_in' or '_out'")
        bounds = None
        if not two_dimensional:
            # e.g. PTJ0_m10000p0_30p0_in -> (-10000, 30)
            low, high = (float(s.replace("m", "-", 1).replace("p", ".")) for s in name.split("_")[1:3])
            bounds = (low, high)
        return cls(name, higgsdna_index, fiducial, bounds)


def differential_bins(variable: str) -> list[DiffBin]:
    """Bins of a differential variable as defined in differentialProcTable_ (commonObjects.py)."""
    if variable not in analysis.differentialProcTable_:
        raise KeyError(f"No bins defined for variable '{variable}' in differentialProcTable_")
    two_dimensional = variable in TWO_D_VARIABLES
    return [
        DiffBin.from_name(index, name, two_dimensional) for index, name in analysis.differentialProcTable_[variable]
    ]


def _select_bin(columns, diff_bin, gen_var, diff_var, use_abs, by_index):
    if diff_bin.bounds is not None and gen_var in columns and FIDUCIAL_FLAG in columns:
        x = np.abs(columns[gen_var]) if use_abs else columns[gen_var]
        low, high = diff_bin.bounds
        return (x >= low) & (x < high) & (columns[FIDUCIAL_FLAG] == diff_bin.fiducial)
    if by_index and diff_var in columns:
        return columns[diff_var] == diff_bin.higgsdna_index
    return None


def differential_splits(
    production_mode: str,
    variable: str,
    bins: list[DiffBin],
    diff_var: str,
    nominal: dict[str, Columns],
) -> list[Split]:
    """Split into the bins of a differential variable.

    Events are assigned by their particle-level value (``GEN_VARIABLES``) and fiducial flag. The
    systematic trees of HiggsDNA do not store the particle-level variables, only the HiggsDNA bin
    index ``diff_var``: their events are assigned by that index where it reproduces the bin on the
    nominal events (checked here), otherwise the bin cannot be selected in the systematic trees.
    """
    two_dimensional = variable in TWO_D_VARIABLES
    if not two_dimensional and variable not in GEN_VARIABLES:
        raise KeyError(f"No particle-level variable defined for '{variable}' in GEN_VARIABLES")
    gen_var = GEN_VARIABLES.get(variable, "")
    use_abs = variable not in SIGNED_VARIABLES
    filled = [columns for columns in nominal.values() if n_events(columns)]

    splits = []
    for diff_bin in bins:
        by_index = two_dimensional
        if not two_dimensional and filled and all(diff_var in columns for columns in filled):
            by_index = all(
                np.array_equal(
                    _select_bin(columns, diff_bin, gen_var, diff_var, use_abs, by_index=False),
                    columns[diff_var] == diff_bin.higgsdna_index,
                )
                for columns in filled
            )
        logger.debug("%s: HiggsDNA index usable for the systematic trees: %s", diff_bin.name, by_index)
        select = functools.partial(
            _select_bin, diff_bin=diff_bin, gen_var=gen_var, diff_var=diff_var, use_abs=use_abs, by_index=by_index
        )
        splits.append(Split(diff_bin.name, f"{production_mode}_{diff_bin.name}", production_mode, select))
    return splits


def signal_splits(
    cfg: Trees2WSConfig, production_mode: str, variable: str = "", nominal: dict[str, Columns] | None = None
) -> list[Split]:
    """The processes a signal MC file is split into, as configured by doInOutSplitting/doDiffSplitting.

    Without the ``nominal`` events, the systematic trees of a differential split cannot be selected
    by the HiggsDNA bin index (enough to know the output files).
    """
    if cfg.do_stxs_splitting:
        raise NotImplementedError("STXS splitting is not implemented")
    if cfg.do_inout_splitting == cfg.do_diff_splitting:
        raise ValueError("Exactly one of doInOutSplitting and doDiffSplitting must be set in trees2wsCfg")
    if cfg.do_inout_splitting:
        return inout_splits(production_mode)
    return differential_splits(production_mode, variable, differential_bins(variable), cfg.diff_var, nominal or {})


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Writing the workspaces


def make_variable(name: str) -> ROOT.RooRealVar:
    """RooRealVar for a tree column; the diphoton mass and dZ get their analysis range and binning."""
    if name == MASS:
        var = ROOT.RooRealVar(name, name, 125.0, 100.0, 180.0)
        var.setBins(160)
    elif name == "dZ":
        var = ROOT.RooRealVar(name, name, 0.0, -20.0, 20.0)
        var.setBins(40)
    elif name == WEIGHT:
        var = ROOT.RooRealVar(name, name, 0.0)
    else:
        var = ROOT.RooRealVar(name, name, 1.0, -999999, 999999)
        var.setBins(1)
    return var


def _new_workspace(variables: list[str]) -> ROOT.RooWorkspace:
    ws = ROOT.RooWorkspace(WS_NAME, WS_NAME)
    int_lumi = ROOT.RooRealVar("intLumi", "intLumi", 1000.0, 0.0, 999999999.0)
    int_lumi.setConstant(True)
    ws.Import(int_lumi)
    for name in variables:
        ws.Import(make_variable(name), ROOT.RooFit.Silence())
    return ws


def _write_workspace(ws: ROOT.RooWorkspace, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = ROOT.TFile(str(path), "RECREATE")
    f.mkdir(WS_DIR).cd()
    ws.Write()
    f.Close()


def _ws_vars(ws: ROOT.RooWorkspace, names: list[str]) -> list[ROOT.RooRealVar]:
    variables = [ws.var(name) for name in names]
    missing = [name for name, var in zip(names, variables) if not var]
    if missing:
        raise KeyError(f"Variables {missing} are not in the workspace")
    return variables


def _arg_set(variables) -> ROOT.RooArgSet:
    arg_set = ROOT.RooArgSet()
    for var in variables:
        arg_set.add(var)
    return arg_set


def _roofit_values(name: str, values: np.ndarray, var: ROOT.RooRealVar) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n_bad = np.count_nonzero(~np.isfinite(values))
    if n_bad:
        raise ValueError(f"{name} has {n_bad} non-finite values")
    # Clip into the variable range like RooRealVar::setVal (e.g. dZ outside +-20 cm is stored at
    # the boundary): RooDataSet.from_numpy would drop these events instead
    return np.clip(values, var.getMin(), var.getMax())


def make_dataset(name: str, variables: list[ROOT.RooRealVar], columns: Columns, weights) -> ROOT.RooDataSet:
    """Weighted RooDataSet of the events in ``columns``, the same as adding them with add(row, weight).

    ``variables`` are the RooRealVars of the dataset including the weight variable.
    """
    values = {}
    for var in variables:
        column = weights if var.GetName() == WEIGHT else columns[var.GetName()]
        values[var.GetName()] = _roofit_values(var.GetName(), column, var)
    return ROOT.RooDataSet.from_numpy(values, _arg_set(variables), name=name, title=name, weight_name=WEIGHT)


def write_signal_workspace(
    path: Path, events: SignalEvents, split: Split, mass: int | str, cfg: Trees2WSConfig
) -> None:
    ws = _new_workspace(events.variables)
    dataset_vars = _ws_vars(ws, events.variables)
    hist_vars = _ws_vars(ws, [v for v in cfg.systematics_vars if v != WEIGHT and not _is_split_column(v)])
    hist_args = _arg_set(hist_vars)

    unselectable = []
    for cat in events.categories:
        name = f"{split.dataset_proc}_{mass}_{SQRTS}_{cat}"
        mask = split.select(events.nominal[cat])
        if mask is None:
            raise ValueError(f"Cannot select {split.label} events: missing branches in the nominal trees")
        selected = _select(events.nominal[cat], mask)
        ws.Import(make_dataset(name, dataset_vars, selected, selected[WEIGHT]))
        if not cfg.do_systematics:
            continue
        for syst in cfg.systematics:
            for direction in ("Up", "Down"):
                hist_name = f"{name}_{syst}{direction}01sigma"
                hist = ROOT.RooDataHist(hist_name, hist_name, hist_args)
                columns = events.systematics.get((cat, f"{syst}{direction}"))
                if columns is not None:
                    mask = split.select(columns)
                    if mask is None:
                        unselectable.append(hist_name)
                    else:
                        selected = _select(columns, mask)
                        hist.add(make_dataset("", [*hist_vars, ws.var(WEIGHT)], selected, selected[WEIGHT]))
                ws.Import(hist)
    if unselectable:
        logger.warning(
            "%s: the systematic trees cannot be split into this process (no particle-level variable and no matching"
            " HiggsDNA index): %d systematic histograms are left EMPTY, e.g. %s",
            split.label,
            len(unselectable),
            unselectable[0],
        )
    _write_workspace(ws, path)


def signal_output_path(output_dir: str | Path, input_path: str | Path, split: Split) -> Path:
    return Path(output_dir) / f"ws_{split.ws_proc}" / f"{Path(input_path).stem}_{split.ws_proc}.root"


def convert_signal_file(
    input_path: str | Path,
    output_dir: str | Path,
    production_mode: str,
    mass: int | str,
    year: str,
    cfg: Trees2WSConfig,
    variable: str = "",
) -> dict[str, Path]:
    """Convert one signal MC file; returns the written workspace of every process (by split label)."""
    signal_splits(cfg, production_mode, variable)  # check the configuration before reading the file
    split_columns = (GEN_VARIABLES.get(variable, ""), cfg.diff_var, FIDUCIAL_FLAG)
    events = read_signal_events(input_path, cfg, production_mode, mass, year, split_columns)
    paths = {}
    for split in signal_splits(cfg, production_mode, variable, events.nominal):
        path = signal_output_path(output_dir, input_path, split)
        logger.info("Writing %s", path)
        write_signal_workspace(path, events, split, mass, cfg)
        paths[split.label] = path
    return paths


def convert_data_file(input_path: str | Path, output_path: str | Path, cfg: Trees2WSConfig) -> Path:
    """Convert the data trees of all categories into one workspace with unit-weight RooDataSets."""
    ws = _new_workspace(cfg.data_vars)
    dataset_vars = _ws_vars(ws, cfg.data_vars)
    with uproot.open(input_path) as f:
        directory = f[cfg.input_tree_dir] if cfg.input_tree_dir else f
        cats = list_categories(directory) if cfg.cats == "auto" else list(cfg.cats)
        for cat in cats:
            name = f"Data_{SQRTS}_{cat}"
            tree = directory[name]
            logger.info("Reading %s (%d events)", name, tree.num_entries)
            columns = _apply_mass_cut(tree.arrays([v for v in cfg.data_vars if v != WEIGHT], library="np"), cfg)
            ws.Import(make_dataset(name, dataset_vars, columns, np.ones(n_events(columns))))
    output_path = Path(output_path)
    logger.info("Writing %s", output_path)
    _write_workspace(ws, output_path)
    return output_path
