import numpy as np
import pytest
import ROOT
import uproot

ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

N_THEORY = 3
CATS = ("cat0", "cat1")
SYSTS = ("Scale",)


def signal_columns(rng, n):
    """Nominal HiggsDNA-like branches; the HiggsDNA index of the NJ bins is 30+NJ (in) or 35 (out)."""
    gen_nj = rng.integers(0, 7, n)
    fiducial = rng.random(n) < 0.8
    return {
        "CMS_hgg_mass": np.concatenate([[99.0, 181.0], rng.uniform(100, 180, n - 2)]),
        "weight": rng.normal(1e-3, 2e-4, n),
        "weight_central": rng.normal(1e-3, 2e-4, n),
        "dZ": np.concatenate([[-25.0, 30.0], rng.normal(0, 5, n - 2)]).astype(np.float32),
        "weight_PileupUp": rng.normal(1e-3, 2e-4, n),
        "weight_PileupDown": rng.normal(1e-3, 2e-4, n),
        "GenNJ": gen_nj,
        "fiducialGeometricFlag": fiducial,
        "diffVariable_GenNJ": np.where(fiducial, 30 + np.minimum(gen_nj, 4), 35),
        "weight_LHEScale": rng.normal(1, 0.1, (n, N_THEORY)).astype(np.float32),
    }


def systematic_columns(rng, n):
    """Systematic trees only have a few branches, in particular no particle-level variables."""
    columns = signal_columns(rng, n)
    return {k: columns[k] for k in ("CMS_hgg_mass", "weight", "fiducialGeometricFlag", "diffVariable_GenNJ")}


@pytest.fixture
def signal_file(tmp_path):
    """HiggsDNA-like output of one production mode (tth) and mass point (125)."""
    rng = np.random.default_rng(1)
    path = tmp_path / "output_TTHToGG_M125.root"
    with uproot.recreate(path) as f:
        for cat, n in zip(CATS, (500, 300)):
            name = f"DiphotonTree/tth_125_13TeV_{cat}"
            f[name] = signal_columns(rng, n)
            for syst in SYSTS:
                for direction in ("Up", "Down"):
                    f[f"{name}_{syst}{direction}01sigma"] = systematic_columns(rng, n + 7)
    return path


@pytest.fixture
def data_file(tmp_path):
    rng = np.random.default_rng(2)
    path = tmp_path / "allData_2023.root"
    with uproot.recreate(path) as f:
        for cat, n in zip(CATS, (400, 200)):
            f[f"DiphotonTree/Data_13TeV_{cat}"] = {
                "CMS_hgg_mass": np.concatenate([[181.0], rng.uniform(100, 180, n - 1)]),
                "weight": np.full(n, 3.0),
            }
    return path


@pytest.fixture
def config():
    """trees2wsCfg as in config/<year>_<variable>.yml"""
    return {
        "inputTreeDir": "DiphotonTree",
        "doSTXSSplitting": False,
        "doDiffSplitting": False,
        "doInOutSplitting": True,
        "doSystematics": True,
        "mainVars": [
            "CMS_hgg_mass",
            "weight",
            "weight_central",
            "dZ",
            "*Up",
            "*Down",
            "fiducialGeometricFlag",
            "GenNJ",
        ],
        "dataVars": ["CMS_hgg_mass", "weight"],
        "stxsVar": "",
        "diffVar": "",
        "systematicsVars": ["CMS_hgg_mass", "weight", "fiducialGeometricFlag"],
        "theoryWeightContainers": {"weight_LHEScale": N_THEORY},
        "systematics": list(SYSTS),
        "cats": "auto",
    }
