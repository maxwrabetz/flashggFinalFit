import json

import numpy as np
import pytest
import ROOT
import uproot

from finalfits import signal_ftest, trees2ws
from finalfits.signal_model import SignalFit


@pytest.fixture
def signal_ws_dir(tmp_path, config):
    """Trees2WS output of a ttH sample at 125 GeV: a narrow and a wide Gaussian, 10% wrong vertex."""
    rng = np.random.default_rng(5)
    n = 20000
    narrow = rng.random(n) < 0.7
    mass = np.where(narrow, rng.normal(125.3, 1.4, n), rng.normal(124.6, 3.0, n))
    dz = np.where(rng.random(n) < 0.9, rng.normal(0, 0.2, n), rng.normal(0, 6, n))
    columns = {
        "CMS_hgg_mass": mass,
        "weight": np.full(n, 1e-3),
        "dZ": dz.astype(np.float32),
        "fiducialGeometricFlag": np.ones(n, dtype=bool),
    }
    trees = tmp_path / "output_TTHToGG_M125.root"
    with uproot.recreate(trees) as f:
        f["DiphotonTree/tth_125_13TeV_cat0"] = columns
    config.update(mainVars=["CMS_hgg_mass", "weight", "dZ", "fiducialGeometricFlag"], doSystematics=False)
    config["theoryWeightContainers"] = {}
    cfg = trees2ws.Trees2WSConfig.from_dict(config)
    trees2ws.convert_signal_file(trees, tmp_path / "ws", "tth", 125, "2023", cfg)
    return tmp_path / "ws" / "ws_tth_in"


def rv_dataset(ws_dir):
    f = ROOT.TFile(str(next(ws_dir.glob("*.root"))))
    ws = f.Get("tagsDumper/cms_hgg_13TeV")
    observables = ROOT.RooArgSet(ws.var("CMS_hgg_mass"), ws.var("dZ"))
    data = ws.data("tth_in_125_13TeV_cat0").reduce(observables, "abs(dZ)<=1.")
    return f, ws, data


def test_signal_fit_recovers_gaussian(signal_ws_dir):
    f, ws, data = rv_dataset(signal_ws_dir)
    mh = ROOT.RooRealVar("MH", "m_{H}", 120, 130)
    one = SignalFit("t1", "tth", "cat0", {"125": data}, ws.var("CMS_hgg_mass").Clone(), mh, 80)
    one.build_gaussians(1)
    one.fit()
    two = SignalFit("t2", "tth", "cat0", {"125": data}, ws.var("CMS_hgg_mass").Clone(), mh, 80)
    two.build_gaussians(2)
    two.fit()
    assert two.reduced_chi2() < one.reduced_chi2()
    # the narrow component dominates: the single Gaussian sits near its mean
    assert one.vars["dm_g0_p0"].getVal() == pytest.approx(0.2, abs=0.3)
    # non-empty 1 GeV bins in 110-140 GeV minus the two shape parameters
    assert 15 < len(one.chi2_bins["125"].x) <= 30
    assert one.ndof == len(one.chi2_bins["125"].x) - 2
    f.Close()


def test_signal_ftest(signal_ws_dir, tmp_path):
    out_dir = tmp_path / "fTest"
    (result,) = signal_ftest.run_ftest(str(signal_ws_dir), str(out_dir), "cat0", ["tth_in"], plots={"RV", "RVResults"})
    assert result.n_rv >= 2
    with open(out_dir / "json" / "nGauss_cat0.json") as f:
        assert json.load(f) == {"tth_in__cat0": {"nRV": result.n_rv, "nWV": result.n_wv}}
    assert (out_dir / "Plots" / "fTest_cat0_tth_in_RV.png").exists()
    assert (out_dir / "Plots" / "fTest_cat0_tth_in_RV_chi2_vs_nGauss.pdf").exists()
