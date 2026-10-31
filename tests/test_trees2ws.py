from pathlib import Path

import numpy as np
import pytest
import ROOT
import uproot
import yaml

from finalfits import trees2ws

REPO = Path(__file__).resolve().parents[1]


def open_ws(path):
    f = ROOT.TFile(str(path))
    ws = f.Get(f"{trees2ws.WS_DIR}/{trees2ws.WS_NAME}")
    return f, ws


def nominal(path, cat):
    with uproot.open(path) as f:
        return f[f"DiphotonTree/tth_125_13TeV_{cat}"].arrays(library="np")


def systematic(path, cat, variation):
    with uproot.open(path) as f:
        return f[f"DiphotonTree/tth_125_13TeV_{cat}_{variation}01sigma"].arrays(library="np")


def test_config_from_analysis_yaml():
    with open(REPO / "config" / "2023_NJ.yml") as f:
        cfg = trees2ws.Trees2WSConfig.from_dict(yaml.safe_load(f)["trees2wsCfg"])
    assert cfg.do_diff_splitting and not cfg.do_inout_splitting
    assert cfg.diff_var == "diffVariable_GenNJ"
    assert cfg.theory_weight_containers == {"weight_LHEPdf": 101, "weight_LHEScale": 9}
    assert "JecSystTotal" in cfg.systematics


def test_diff_bin_from_name():
    b = trees2ws.DiffBin.from_name(40, "PTJ0_m10000p0_30p0_in")
    assert b.bounds == (-10000.0, 30.0) and b.fiducial
    assert not trees2ws.DiffBin.from_name(934, "NJ_4p0_100p0_out").fiducial
    assert trees2ws.DiffBin.from_name(4601, "PTHvYH_0p0_50p0_0p0_0p2_in", two_dimensional=True).bounds is None
    with pytest.raises(ValueError):
        trees2ws.DiffBin.from_name(1, "NJ_0p0_1p0")


def test_theory_weight_names():
    # the datacard step expects the container name without its last character
    assert trees2ws.theory_weight_names("weight_LHEPdf", 2) == ["weight_LHEPd_0", "weight_LHEPd_1"]


def test_make_dataset_matches_row_by_row_filling():
    """make_dataset() stores exactly what RooDataSet/RooDataHist.add(row, weight) per event stores."""
    rng = np.random.default_rng(3)
    names = ["CMS_hgg_mass", "dZ", "x", "weight"]
    ws = trees2ws._new_workspace(names)
    row_vars = [ws.var(n) for n in names]
    columns = {
        "CMS_hgg_mass": np.array([95.0, 185.0, 125.3, 100.0, 180.0]),
        "dZ": np.array([-21.0, 22.5, 0.1, 3.0, -3.0]),
        "x": rng.normal(0, 1e6, 5),
    }
    weights = rng.normal(1, 1, 5)

    dataset = trees2ws.make_dataset("d", row_vars, columns, weights)
    by_row = ROOT.RooDataSet("by_row", "by_row", trees2ws._arg_set(row_vars), "weight")
    row = trees2ws._arg_set(row_vars)
    for i, w in enumerate(weights):
        for var in row_vars[:3]:
            var.setVal(columns[var.GetName()][i])
        by_row.add(row, w)

    assert dataset.GetName() == "d"
    assert dataset.numEntries() == by_row.numEntries() == 5
    for i in range(5):
        a, b = dataset.get(i), by_row.get(i)
        assert [a.getRealValue(n) for n in names[:3]] == [b.getRealValue(n) for n in names[:3]]
        assert dataset.weight() == by_row.weight()

    hist_vars = [ws.var("CMS_hgg_mass")]
    hist = ROOT.RooDataHist("h1", "h1", trees2ws._arg_set(hist_vars))
    hist.add(trees2ws.make_dataset("", [*hist_vars, ws.var("weight")], columns, weights))
    hist_by_row = ROOT.RooDataHist("h2", "h2", trees2ws._arg_set(hist_vars))
    for i, w in enumerate(weights):
        hist_vars[0].setVal(columns["CMS_hgg_mass"][i])
        hist_by_row.add(trees2ws._arg_set(hist_vars), w)
    for i in range(hist.numEntries()):
        hist.get(i)
        hist_by_row.get(i)
        assert hist.weight() == hist_by_row.weight()
        assert hist.weightSquared() == hist_by_row.weightSquared()


def test_make_dataset_rejects_non_finite_values():
    ws = trees2ws._new_workspace(["CMS_hgg_mass", "weight"])
    variables = [ws.var("CMS_hgg_mass"), ws.var("weight")]
    with pytest.raises(ValueError, match="weight"):
        trees2ws.make_dataset("d", variables, {"CMS_hgg_mass": np.array([125.0])}, np.array([np.nan]))


def test_inout_splitting(signal_file, config, tmp_path):
    cfg = trees2ws.Trees2WSConfig.from_dict(config)
    paths = trees2ws.convert_signal_file(signal_file, tmp_path / "out", "tth", 125, "2023preBPix", cfg)
    assert paths["in"] == tmp_path / "out" / "ws_tth_in" / "output_TTHToGG_M125_tth_in.root"

    for tag, fiducial in (("in", True), ("out", False)):
        f, ws = open_ws(paths[tag])
        assert ws.var("weight_LHEScal_2") and ws.var("weight_PileupUp") and ws.var("GenNJ")
        assert not ws.var("fiducialGeometricFlag")
        for cat in ("cat0", "cat1"):
            events = nominal(signal_file, cat)
            selected = events["fiducialGeometricFlag"] == fiducial
            dataset = ws.data(f"tth_{tag}_125_13TeV_{cat}")
            assert dataset.numEntries() == selected.sum()
            assert dataset.sumEntries() == pytest.approx(events["weight"][selected].sum(), rel=1e-12)
            # out-of-range values are kept at the boundary of the variable range
            assert dataset.get(0).getRealValue("dZ") in (-20.0, 20.0) or not selected[0]

            for variation in ("ScaleUp", "ScaleDown"):
                syst = systematic(signal_file, cat, variation)
                hist = ws.data(f"tth_{tag}_125_13TeV_{cat}_{variation}01sigma")
                expected = syst["weight"][syst["fiducialGeometricFlag"] == fiducial].sum()
                assert hist.sumEntries() == pytest.approx(expected, rel=1e-12)
        f.Close()


def test_differential_splitting(signal_file, config, tmp_path, caplog):
    config.update(doDiffSplitting=True, doInOutSplitting=False, diffVar="diffVariable_GenNJ")
    config["systematicsVars"] += ["diffVariable_GenNJ"]
    cfg = trees2ws.Trees2WSConfig.from_dict(config)
    paths = trees2ws.convert_signal_file(signal_file, tmp_path, "tth", 125, "2023preBPix", cfg, variable="NJ")
    assert len(paths) == 10

    # In-fiducial bins: systematic events selected by the HiggsDNA index, which matches the bins
    f, ws = open_ws(paths["NJ_1p0_2p0_in"])
    for cat in ("cat0", "cat1"):
        events = nominal(signal_file, cat)
        selected = (events["GenNJ"] == 1) & events["fiducialGeometricFlag"]
        assert ws.data(f"tth_125_13TeV_{cat}").sumEntries() == pytest.approx(events["weight"][selected].sum())
        syst = systematic(signal_file, cat, "ScaleUp")
        expected = syst["weight"][syst["diffVariable_GenNJ"] == 31].sum()
        assert ws.data(f"tth_125_13TeV_{cat}_ScaleUp01sigma").sumEntries() == pytest.approx(expected)
    f.Close()

    # Out-of-fiducial bins: HiggsDNA has one index for all of them, the systematics stay empty
    f, ws = open_ws(paths["NJ_4p0_100p0_out"])
    events = nominal(signal_file, "cat0")
    selected = (events["GenNJ"] >= 4) & ~events["fiducialGeometricFlag"]
    assert ws.data("tth_125_13TeV_cat0").numEntries() == selected.sum() > 0
    assert ws.data("tth_125_13TeV_cat0_ScaleUp01sigma").sumEntries() == 0
    f.Close()
    assert "NJ_4p0_100p0_out: the systematic trees cannot be split" in caplog.text


def test_data(data_file, config, tmp_path):
    cfg = trees2ws.Trees2WSConfig.from_dict(config)
    path = trees2ws.convert_data_file(data_file, tmp_path / "ws" / "allData.root", cfg)
    f, ws = open_ws(path)
    for cat, n in (("cat0", 400), ("cat1", 200)):
        dataset = ws.data(f"Data_13TeV_{cat}")
        assert dataset.numEntries() == n
        assert dataset.sumEntries() == n  # unit weights
    assert ws.data("Data_13TeV_cat0").get(0).getRealValue("CMS_hgg_mass") == 180.0
    f.Close()


def test_mass_cut(data_file, config, tmp_path):
    config.update(apply_mass_cut=True, mass_cut_range="100,180")
    path = trees2ws.convert_data_file(data_file, tmp_path / "allData.root", trees2ws.Trees2WSConfig.from_dict(config))
    f, ws = open_ws(path)
    assert ws.data("Data_13TeV_cat0").numEntries() == 399
    f.Close()


def test_invalid_splitting(signal_file, config, tmp_path):
    config.update(doDiffSplitting=True)
    with pytest.raises(ValueError, match="Exactly one"):
        trees2ws.convert_signal_file(
            signal_file, tmp_path, "tth", 125, "2023", trees2ws.Trees2WSConfig.from_dict(config), "NJ"
        )
