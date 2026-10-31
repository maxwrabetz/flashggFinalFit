from pathlib import Path

import numpy as np
import pytest
import ROOT
import uproot
from scipy.stats import chi2

from finalfits import analysis, combine

REPO = Path(__file__).resolve().parents[1]
XRTD = [
    "--X-rtd", "MINIMIZER_freezeDisassociatedParams", "--X-rtd", "MINIMIZER_multiMin_hideConstants",
    "--X-rtd", "MINIMIZER_multiMin_maskConstraints", "--X-rtd", "MINIMIZER_multiMin_maskChannels=2",
]  # fmt: skip
MIN = ["--cminDefaultMinimizerStrategy=0", "--cminFallbackAlgo", "Minuit2,Migrad,1:10"]


@pytest.mark.parametrize(
    "expression, overrides, expected",
    [
        ("r=0,3", None, "r_NJ_0p0_1p0=0,3"),
        ("r=0,3:MH=124,126", None, "r_NJ_0p0_1p0=0,3:MH=124,126"),
        (" r = 0,3 : ", None, "r_NJ_0p0_1p0=0,3"),
        ("r=0,3", {"r_NJ_0p0_1p0": "0.5,1.5"}, "r_NJ_0p0_1p0=0.5,1.5"),
        ("r=0,3", {"r_NJ_0p0_1p0": "r=0.2,2:MH=124,126"}, "r_NJ_0p0_1p0=0.2,2:MH=124,126"),
        ("r=0,3", {"r_NJ_1p0_2p0": "0.5,1.5"}, "r_NJ_0p0_1p0=0,3"),
    ],
)
def test_parameter_range_for_poi(expression, overrides, expected):
    assert combine.parameter_range_for_poi(expression, "r_NJ_0p0_1p0", overrides) == expected


@pytest.mark.parametrize("expression", ["r", "", ":"])
def test_invalid_parameter_range(expression):
    with pytest.raises(ValueError):
        combine.parameter_range_for_poi(expression, "r")


def test_scan_grid_configuration():
    assert combine.scan_grid_configuration(30, 0, -1, -1) == (30, list(range(30)))
    # refinement: points 10..14 of a 200 point grid
    assert combine.scan_grid_configuration("30", "200", "10", "14") == (200, [10, 11, 12, 13, 14])
    assert combine.scan_grid_configuration(30, 0, 5, 5) == (30, [5])
    for first, last in ((-1, 3), (5, 4), (0, 30)):
        with pytest.raises(ValueError):
            combine.scan_grid_configuration(30, 0, first, last)


def test_filter_excluded():
    names = ["CMS_hgg_scale", "CMS_hgg_smear", "lumi_13p6TeV", "BR_hgg"]
    assert combine.filter_excluded(names, "") == names
    assert combine.filter_excluded(names, "BR_hgg, lumi_13p6TeV") == names[:2]
    assert combine.filter_excluded(names, "rgx{CMS_hgg_.*}") == names[2:]
    assert combine.filter_excluded(names, "/scale$/,BR_hgg") == ["CMS_hgg_smear", "lumi_13p6TeV"]
    assert combine.filter_excluded(names, "rgx{(}") == names  # invalid expression: a literal name


def test_asimov_commands():
    """Commands as the original law tasks wrote them."""
    assert combine.asimov_first_step_command("ws.root", "r_X", "pdfindex_a", poi="r_X", set_parameters="r_X=1") == [
        "combine", "-M", "MultiDimFit", "ws.root", "--freezeParameters", "MH", "-m", "125.07", "-n", "firstStep_r_X",
        *MIN, "--expectSignal", "1", "--saveWorkspace", *XRTD, "-t", "-1", "-P", "r_X", "--saveFitResult",
        "--floatOtherPOIs", "1", "--saveSpecifiedIndex", "pdfindex_a", "--setParameters", "r_X=1",
    ]  # fmt: skip
    inclusive = combine.asimov_scan_command("fs.root", "r", 3, 30, "r=0,2", "r=1", stat_only=True, inclusive=True)
    assert inclusive == [
        "combine", "-M", "MultiDimFit", "-d", "fs.root", "--snapshotName", "MultiDimFit",
        "--freezeParameters", "allConstrainedNuisances,MH", "-m", "125.07", "-n", "AsimovPostFitScanStat_r.POINTS.3.3",
        *MIN, "--algo", "grid", "--points", "30", "--expectSignal", "1", *XRTD, "-t", "-1", "-P", "r",
        "--firstPoint", "3", "--lastPoint", "3", "--floatOtherPOIs", "1", "--alignEdges", "1",
        "--setParameterRanges", "r=0,2", "--saveSpecifiedNuis", "all", "--setParameters", "r=1",
    ]  # fmt: skip
    differential = combine.asimov_scan_command("fs.root", "r_X", 0, 200, "r_X=0,3", "r_X=1,pdfindex_a=1")
    assert differential[:3] == ["combineTool.py", "-M", "MultiDimFit"]
    assert differential[-6:] == ["--snapshotName", "MultiDimFit", "--setParameters", "r_X=1,pdfindex_a=1",
                                 "--setParameterRanges", "r_X=0,3"]  # fmt: skip


def test_observed_and_impact_commands():
    scan = combine.observed_scan_command("bf.root", "r_X", 7, 100, save_index="pdfindex_a", set_parameters="r_X=1")
    assert scan[scan.index("-n") + 1] == "DataPostFitScanFit_r_X.POINTS.7.7"
    assert scan[-6:] == ["--saveSpecifiedIndex", "pdfindex_a", "--setParameters", "r_X=1", "-w", "w"]
    stat = combine.observed_scan_command("bf.root", "r", 7, 100, stat_only=True, inclusive=True, r_range=(0.0, 3.0))
    assert stat[stat.index("--algo") : stat.index("--algo") + 8] == [
        "--algo", "grid", "--rMin", "0.0", "--rMax", "3.0", "--points", "100",
    ]  # fmt: skip
    fit = combine.observed_impacts_parameter_fit_command("ws.root", "lumi", ["r"], "r=1.02", inclusive_options=[])
    assert fit[fit.index("-P") : fit.index("-P") + 4] == ["-P", "lumi", "--setParameters", "r=1.02"]
    assert combine.impacts_collect_command("ws.root", freeze_mh=False) == [
        "combineTool.py", "-M", "Impacts", "-d", "ws.root", "-m", "125.07", "-o", "impacts/impacts.json",
    ]  # fmt: skip
    assert combine.plot_impacts_command("i.json", "out", poi="r_X", translate="t.json", blind=True)[-5:] == [
        "--POI", "r_X", "--translate", "t.json", "--blind",
    ]  # fmt: skip


def test_toy_commands():
    postfit = combine.toy_fit_command(3, "r", "MultiDimFit", "r=1.1", postfit=True)
    prefit = combine.toy_fit_command(3, "r", "MultiDimFit", "r=1")
    assert postfit[postfit.index("-t") + 2 : postfit.index("-t") + 4] == ["--setParameters", "r=1.1"]
    assert prefit[-2:] == ["--setParameters", "r=1"]
    assert sorted(postfit) == sorted([*prefit[:-1], "r=1.1"])
    assert combine.toy_throw_command(3, "MultiDimFit", "r=0")[1] == "fit_3.root"


def test_output_file():
    assert combine.output_file("firstStep_r") == "higgsCombinefirstStep_r.MultiDimFit.mH125.07.root"


@pytest.fixture
def model_workspace(tmp_path):
    """Workspace with a ModelConfig: POI r, nuisances (one constant), a free background parameter."""
    w = ROOT.RooWorkspace("w")
    w.factory("Gaussian::sig(x[100,180], MH[125.07], sigma[2])")
    w.factory("Exponential::bkg(x, env_pdf_p1[-0.02,-1,0])")
    w.factory("expr::nsig('r*(1+0.01*lumi)*(1+0.02*BR_hgg)', r[1,0,3], lumi[0,-5,5], BR_hgg[0,-5,5])")
    w.factory("SUM::model(nsig*sig, nbkg[1000,0,1e6]*bkg)")
    w.var("BR_hgg").setConstant(True)
    config = ROOT.RooStats.ModelConfig("ModelConfig", w)
    config.SetPdf(w.pdf("model"))
    config.SetParametersOfInterest("r")
    config.SetNuisanceParameters("lumi,BR_hgg")
    config.SetObservables("x")
    w.Import(config)
    path = tmp_path / "workspace.root"
    w.writeToFile(str(path))
    return str(path)


def test_workspace_parameters(model_workspace):
    assert combine.modelconfig_nuisances(model_workspace, ["r"]) == ["lumi"]
    assert combine.modelconfig_nuisances(model_workspace, ["r"], exclude="lumi") == []
    assert sorted(combine.free_parameters(model_workspace, ["r"])) == ["env_pdf_p1", "lumi", "nbkg"]
    with pytest.raises(OSError):
        combine.modelconfig_nuisances(model_workspace + ".missing", ["r"])


@pytest.fixture
def limit_tree(tmp_path):
    path = tmp_path / "higgsCombine.pvalue.MultiDimFit.mH125.07.root"
    with uproot.recreate(path) as f:
        f["limit"] = {
            "r_X": np.array([1.25, 1.0], dtype=np.float32),
            "deltaNLL": np.array([0.0, 2.3]),
            "pdfindex_cat0": np.array([2, 2], dtype=np.int32),
            "pdfindex_cat1": np.array([0, 0], dtype=np.int32),
        }
    return str(path)


def test_fit_results(limit_tree):
    assert combine.best_fit_values(limit_tree, ["r_X"]) == "r_X=1.25"
    with pytest.raises(RuntimeError, match="Branch 'r_Y'"):
        combine.best_fit_values(limit_tree, ["r_Y"])
    assert combine.pvalue(limit_tree, 5) == pytest.approx(chi2(5).sf(4.6))
    with pytest.raises(FileNotFoundError, match="fit at the SM point"):
        combine.pvalue(limit_tree + ".missing", 5)
    assert combine.round_significant(0.0123456) == 0.012
    assert combine.round_significant(0.56789) == 0.57


def test_fitted_pdf_indices(limit_tree):
    macro = REPO / "Combine" / "checkPdfIdx.C"
    assert combine.fitted_pdf_indices(limit_tree, macro) == "pdfindex_cat0=2,pdfindex_cat1=0"
    # ROOT only prints "Processing ..." (and the error) for a missing file
    assert combine.fitted_pdf_indices(limit_tree + ".missing", macro) is None


def test_snapshot_value(tmp_path):
    w = ROOT.RooWorkspace("w")
    w.factory("r[1,0,3]")
    w.var("r").setVal(0.8)
    w.saveSnapshot("MultiDimFit", ROOT.RooArgSet(w.var("r")))
    w.var("r").setVal(1.0)
    w.writeToFile(str(tmp_path / "fit.root"))
    assert combine.snapshot_value(str(tmp_path / "fit.root"), "r") == pytest.approx(0.8)


def test_pvalue_failed_fit(tmp_path):
    # a failed fixed point fit leaves only the best fit entry
    path = str(tmp_path / "higgsCombine.pvalue.MultiDimFit.mH125.07.root")
    with uproot.recreate(path) as f:
        f["limit"] = {"deltaNLL": np.array([0.0])}
    with pytest.raises(RuntimeError, match="no fixed point entry"):
        combine.pvalue(path, 5)


def test_corr_matrix_luminosity():
    command = combine.corr_matrix_command(
        "m.py", "in.json", "NJ", "hesse.root", "out", "t.json", "2023", covariance=True
    )
    assert command[command.index("--year") + 1] == "2023"
    assert command[-1] == "--doCov"
    assert analysis.luminosity("2023") == analysis.lumiMap["2023"]
    assert analysis.luminosity("2022_2023") == pytest.approx(analysis.lumiMap["2022"] + analysis.lumiMap["2023"])
