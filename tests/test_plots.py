import array
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest
import ROOT

from finalfits import correlation_plots, impacts, spectra, splusb_plots


def impacts_json(frozen="MH"):
    params = [
        {"name": frozen, "fit": [0.0, 0.0, 0.0], "prefit": [-1.0, 0.0, 1.0], "groups": [], "type": "Gaussian",
         "r": [0.9, 1.3, 1.1], "impact_r": 0.2},
        {"name": "lumi", "fit": [-1.0, 0.0, 1.0], "prefit": [-1.0, 0.0, 1.0], "groups": ["theory"], "type": "Gaussian",
         "r": [0.95, 1.3, 1.05], "impact_r": 0.25},
        {"name": "env_pdf_0_p1", "fit": [-1.0, 0.2, 1.0], "prefit": [-1.0, 0.0, 1.0], "groups": [],
         "type": "Unconstrained", "r": [0.99, 1.3, 1.3], "impact_r": 0.3},
    ]  # fmt: skip
    return {"POIs": [{"name": "r", "fit": [0.8, 1.3, 1.2]}], "method": "default", "params": params}


def test_correct_impacts():
    corrected = impacts.correct_impacts(impacts_json(), "MH")
    # midpoint of the variations of the frozen MH
    assert corrected["POIs"][0]["fit"] == [0.8, 1.0, 1.2]
    lumi, env = corrected["params"][1], corrected["params"][2]
    assert lumi["r"] == [0.95, 1.0, 1.05] and lumi["impact_r"] == pytest.approx(0.05)
    assert env["r"] == [0.99, 1.0, 1.3]  # bkg model parameters keep the corrected value
    assert [p["name"] for p in impacts.correct_impacts(impacts_json(), "MH", True)["params"]] == ["MH", "lumi"]
    # without the frozen parameter: the POI central values
    assert impacts.correct_impacts(impacts_json(frozen="other"), "MH")["POIs"][0]["fit"] == [0.8, 1.3, 1.2]


def test_write_corrected_impacts(tmp_path):
    path = tmp_path / "impacts.json"
    path.write_text(json.dumps(impacts_json()))
    out = impacts.write_corrected_impacts(str(path), "MH", drop_bkg_model_params=True)
    assert out == str(tmp_path / "impacts_corrected_dropBkgModelParams.json")
    data = json.loads(open(out).read())  # valid json, also with non-empty groups
    assert data["params"][1]["groups"] == ["theory"]
    assert "        1.0000000000000000," in open(out).read()


def test_output_names_and_labels():
    assert spectra.output_names("spectra", False, False) == ["spectra_blinded.pdf", "spectra_blinded.png"]
    assert spectra.output_names("spectra_total", True, True) == ["spectra_total_logx.pdf", "spectra_total_logx.png"]
    text = r"$\sigma_{\text{fid}}$ (fb)"
    assert spectra.phase_space_text(text, spectra.FIDUCIAL) == text
    assert spectra.phase_space_text(text, spectra.TOTAL) == r"$\sigma_{\text{tot}}$ (pb)"
    config = {"spectra": {"x_lim": [0, 5], "y_unit": "(fb)", "total": {"y_unit": "(pb)"}}}
    assert spectra.spectra_config(config, spectra.TOTAL) == {"x_lim": [0, 5], "y_unit": "(pb)"}
    assert spectra.spectra_config(config, spectra.FIDUCIAL) == {"x_lim": [0, 5], "y_unit": "(fb)"}


def test_bin_layout():
    edges, centres, widths, drawn = spectra.bin_layout([0, 1, 2, 100], {"overflow": 3, "first_bin_center": 0.5})
    assert list(edges) == [0, 1, 2, 3] and list(centres) == [0.5, 1.5, 2.5] and list(widths) == list(drawn) == [1] * 3
    edges, centres, _, _ = spectra.bin_layout([-100, 0, 10], {"first_bin_center": -1, "x_lim": [-20, 10]})
    assert list(edges) == [-20, 0, 10]
    # 2D bins (NJ low, NJ high, pT low, pT high) side by side
    boundaries = [(0, 1, 0, 10), (0, 1, 10, 10000), (1, 2, 0, 50)]
    edges, centres, widths, drawn = spectra.bin_layout(
        boundaries, {"plot_2d_as_1d": True, "pthvsnj_overflow_edges": [20, 100]}
    )
    assert list(edges) == [0, 10, 20, 70] and list(centres) == [5, 15, 45] and list(widths) == [10, 10, 50]


@pytest.fixture
def theory_modules(monkeypatch):
    """Theory input modules with three bins, the variations 10% (up) and 5% (down) off."""
    for name, scale in (
        ("ggh", 10.0),
        ("vbf", 1.0),
        ("vh", 0.5),
        ("tth", 0.2),
        ("ggh_powheg", 9.0),
        ("ggh_nonnlops", 11.0),
    ):
        module = types.ModuleType(f"theory_{name}")
        module.Boundaries = [0, 1, 2, 3]
        module.fidXS = list(scale * np.array([3.0, 2.0, 1.0]))
        for source in ("scale", "pdf", "alpha"):
            setattr(module, f"fidXS_{source}_up", list(1.1 * np.array(module.fidXS)))
            setattr(module, f"fidXS_{source}_dn", list(0.95 * np.array(module.fidXS)))
        monkeypatch.setitem(sys.modules, module.__name__, module)
    keys = {"ggh": "ggh", "vbf": "vbf", "vh": "vh", "tth": "tth", "ggh_powheg": "ggh_powheg"}
    config = {f"{k}_xs": f"theory_{v}" for k, v in keys.items()}
    config["ggh_no_nnlops_xs"] = "theory_ggh_nonnlops"
    config.update(underflow_bin_normalized=False, last_bin_center=0, first_bin_center=0.5, plot_log=0, variable="N",
                  y_unit="(fb)", x_unit="", x_lim=[0, 3], y_lim=[0, 2])  # fmt: skip
    return config


def test_compute_spectrum(theory_modules):
    measured = {"mu": [1.0, 0.5, 2.0], "err_up": [0.2, 0.3, 0.4], "err_down": [0.1, 0.3, 0.4],
                "stat_up": [0.1, 0.4, 0.2], "stat_down": [0.05, 0.2, 0.2]}  # fmt: skip
    s = spectra.compute_spectrum(measured, theory_modules, spectra.FIDUCIAL)
    madgraph = 11.7 * np.array([3.0, 2.0, 1.0])
    assert s.madgraph == pytest.approx(madgraph)
    assert s.xh == pytest.approx(1.7 * np.array([3.0, 2.0, 1.0]))
    assert s.exp_xs == pytest.approx(np.array([1.0, 0.5, 2.0]) * madgraph)
    # stat > total: no systematic
    assert s.sys_up[1] == 0 and s.sys_up[0] == pytest.approx(np.sqrt(0.2**2 - 0.1**2) * madgraph[0])
    # three fully correlated 10% sources in quadrature, plus 2% BR
    up, down = s.unc["madgraph"]
    assert up == pytest.approx(np.sqrt(3 * 0.1**2 + 0.02**2) * madgraph)
    assert down == pytest.approx(np.sqrt(3 * 0.05**2 + 0.02**2) * madgraph)
    assert list(s.bins_c) == [0.5, 1.5, 2.5]


def write_scan(path, poi, bestfit, up, down):
    f = ROOT.TFile(str(path), "RECREATE")
    tree = ROOT.TTree("limit", "limit")
    r, nll, quantile = (array.array("f", [0]) for _ in range(3))
    tree.Branch(poi, r, f"{poi}/F")
    tree.Branch("deltaNLL", nll, "deltaNLL/F")
    tree.Branch("quantileExpected", quantile, "quantileExpected/F")
    for x in [bestfit, *np.linspace(0, 3, 61)]:
        r[0], nll[0], quantile[0] = x, 0.5 * ((x - bestfit) / (up if x > bestfit else down)) ** 2, -0.5
        tree.Fill()
    tree.Write()
    f.Close()


def test_make_spectrum(tmp_path, theory_modules):
    fit_dir = tmp_path / "runFits_N"
    (fit_dir / "asimov").mkdir(parents=True)
    pois = ["r_N_0", "r_N_1", "r_N_2"]
    for poi in pois:
        write_scan(fit_dir / "asimov" / f"higgsCombineAsimovPostFitScanFit_{poi}.root", poi, 1.0, 0.21, 0.26)
        write_scan(fit_dir / "asimov" / f"higgsCombineAsimovPostFitScanStat_{poi}.root", poi, 1.0, 0.11, 0.16)
    assert spectra.scan_interval(pois[0], str(fit_dir / "asimov" / "higgsCombineAsimovPostFitScanFit_r_N_0.root")) == (
        pytest.approx(1.0),
        pytest.approx(0.21, abs=2e-3),
        pytest.approx(-0.26, abs=2e-3),
    )
    s = spectra.make_spectrum(str(fit_dir), pois, {"spectra": theory_modules}, {"lumi": 27.2}, "2023")
    assert (fit_dir / "spectra_blinded.png").exists() and (fit_dir / "spectra_blinded.pdf").exists()
    assert s.exp_xs == pytest.approx(s.madgraph)


def test_correlation_matrix(tmp_path):
    params = ["lumi", "r_a", "r_b", "shapeBkg_norm"]
    path = str(tmp_path / "robustHesse.root")
    f = ROOT.TFile(path, "RECREATE")
    hist = ROOT.TH2F("h_correlation", "", 4, 0, 4, 4, 0, 4)
    values = {}
    for i in range(4):
        for j in range(4):
            values[i, j] = 1.0 if i == j else 0.3 * (i + j) / 6
            hist.SetBinContent(i + 1, j + 1, values[i, j])
    hist.Write()
    variables = [ROOT.RooRealVar(p, p, 1.0) for p in params]
    arglist = ROOT.RooArgList()
    for v in variables:
        arglist.add(v)
    arglist.Write("floatParsFinal", ROOT.TObject.kSingleKey)
    f.Close()
    matrix, n = correlation_plots.read_matrix(path)
    assert n == 2 and matrix["r_a", "r_b"] == pytest.approx(values[1, 2])
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"test": {"pois": "r_a,r_b", "fits": "robustHesse:syst:r_a"}}))
    paths = correlation_plots.plot_matrix(path, str(inputs), "test", str(tmp_path), {"r_a": "a"})
    assert [p.split("/")[-1] for p in paths] == ["corrMatrix_test_syst.png", "corrMatrix_test_syst.pdf"]
    assert all((tmp_path / p.split("/")[-1]).exists() for p in paths)


def test_eff_sigma_and_bands():
    hist = ROOT.TH1F("h_gauss", "", 400, 100, 140)
    for i in range(1, 401):
        x = hist.GetBinCenter(i)
        hist.SetBinContent(i, np.exp(-0.5 * ((x - 120) / 2.0) ** 2))
    assert splusb_plots.eff_sigma(hist) == pytest.approx(2.0, rel=0.02)
    toys = pd.DataFrame({"sum_1": np.arange(1001.0), "cat0_1": np.zeros(1001)})
    props = splusb_plots.band_properties(toys, "all", 1)
    assert props["median"] == 500 and props["up1sigma"] == pytest.approx(841.3, abs=0.1)
    assert splusb_plots.band_properties(toys, "cat0", 1)["up2sigma"] == 0
