import numpy as np
import pandas as pd
import pytest
import ROOT

from finalfits import datacard, datacard_yields, photon_systematics, trees2ws


def gaussian_hist(name, sigma, n=200000, scale=1.0, seed=1):
    hist = ROOT.TH1F(name, name, 80, 100, 180)
    for x in np.random.default_rng(seed).normal(125, sigma, n):
        hist.Fill(x, scale)
    return hist


def test_eff_sigma_of_gaussian():
    assert photon_systematics.eff_sigma(gaussian_hist("g", 3.0)) == pytest.approx(3.0, rel=0.05)


def test_photon_variations():
    thresholds = {"mean": 0.05, "sigma": 0.5, "rate": 0.05}
    nominal = gaussian_hist("n", 2.0)
    same = {"nominal": nominal, "up": gaussian_hist("u", 2.0), "down": gaussian_hist("d", 2.0)}
    assert photon_systematics.variations(same, thresholds) == (0.0, 0.0, 0.0)
    wider = {"nominal": nominal, "up": gaussian_hist("u2", 3.0), "down": gaussian_hist("d2", 3.0, scale=1.02)}
    mean, sigma, rate = photon_systematics.variations(wider, thresholds)
    assert mean == pytest.approx(0.0, abs=1e-3)
    assert sigma == pytest.approx(0.5, rel=0.1)  # +50% width, at the cap
    assert rate == pytest.approx(0.01 / 1.01, rel=0.02)  # up and down differ by 2% around their midpoint


def row(**values):
    return pd.Series(values)


def test_compare_yield():
    r = row(nominal_yield=2.0, sys_up_yield=2.2, sys_down_yield=1.9, w_yield=2.4)
    assert datacard.compare_yield(r, "a_w", "sys") == pytest.approx([0.95, 1.1])
    assert datacard.compare_yield(r, "s_w", "w") == pytest.approx([1.2])
    # histogram variations are compared to their midpoint
    assert datacard.compare_yield(r, "a_h", "sys") == pytest.approx([1.9 / 2.05, 2.2 / 2.05])
    assert datacard.compare_yield(row(nominal_yield=0, w_yield=1.0), "s_w", "w") == [1.0]


@pytest.mark.parametrize(
    "value,n_events,title,expected",
    [
        ("-", 1000, "x", "-"),
        ([1.0002], 1000, "x", "-"),
        ([1.05], 1000, "x", "1.050"),
        ([1.05], 50, "x", "-"),
        ([1.05], 50, "lumi_1", "1.050"),
        ([0.95, 1.05], 1000, "x", "0.950/1.050"),
        ([0.8, 1.25], 1000, "x", "1.250"),  # symmetric
        ([-0.1, 1.05], 1000, "x", "-"),
    ],
)
def test_format_value(value, n_events, title, expected):
    assert datacard._format_value("", value, title, "proc", "cat", n_events).strip() == expected


def test_clean_datacard(tmp_path):
    card = tmp_path / "card.txt"
    card.write_text(
        "bin  c0  c0\nprocess  ggh_2023_hgg  bkg_mass\nprocess  0  1\n"
        "lumi  lnN  1.010  -\nsys  lnN  0.00005/1.050  20000.0\none  lnN  0.900/1.001  -\n"
        "two  lnN  1.000/1.050  0.999/0.950\nthree  lnN  0.900/1.100  -\n"
    )
    lines = open(datacard.clean_datacard(str(card), factor=10000)).read().splitlines()
    assert lines[3].split()[2:] == ["1.010", "-"]
    assert lines[4].split()[2:] == ["1.000/1.050", "-"]  # beyond the factor
    assert lines[5].split()[2:] == ["0.900/1.001", "-"]
    # the factor applies also with nuisances to symmetrise
    lines = open(datacard.clean_datacard(str(card), factor=10000, symmetrize="one,two,three")).read().splitlines()
    assert lines[4].split()[2:] == ["1.000/1.050", "-"]
    # one-sided: the deviating side as symmetric kappa, i.e. 1/kappa_down if only the down variation deviates
    assert lines[5].split()[2:] == ["1.111", "-"]
    assert lines[6].split()[2:] == ["1.050", "0.950"]
    assert lines[7].split()[2:] == ["0.900/1.100", "-"]  # two-sided: unchanged


def test_systematic_yields(tmp_path):
    ws = trees2ws._new_workspace(["CMS_hgg_mass", "weight", "weight_central", "weight_PileupUp", "weight_PileupDown"])
    variables = [
        ws.var(n) for n in ["CMS_hgg_mass", "weight", "weight_central", "weight_PileupUp", "weight_PileupDown"]
    ]
    columns = {
        "CMS_hgg_mass": np.array([120.0, 125.0, 130.0]),
        "weight_central": np.array([1.0, 0.0, 1.0]),
        "weight_PileupUp": np.array([1.1, 5.0, 2.2]),
        "weight_PileupDown": np.array([0.9, 5.0, 2.2]),
    }
    dataset = trees2ws.make_dataset("d", variables, columns, np.array([1.0, 1.0, 2.0]))
    contents = dataset.get(0).contentsString()
    yields = datacard_yields.systematic_yields(dataset, contents, ws, {"weight_Pileup": "a_w"})
    # event 2 has no central weight, event 3 has equal up/down weights: nominal weight
    assert yields["weight_Pileup_up"] == pytest.approx(1.1 + 2.0)
    assert yields["weight_Pileup_down"] == pytest.approx(0.9 + 2.0)
    assert yields["numEvents"] == 3
