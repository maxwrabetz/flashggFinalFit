import numpy as np
import pytest
import ROOT
import uproot

from finalfits import background, background_plots, trees2ws
from finalfits.pdfs import BackgroundPdfs


@pytest.fixture
def data_workspace(tmp_path, config):
    """allData.root with one category of falling (exponential) diphoton masses."""
    rng = np.random.default_rng(4)
    mass = 100 + rng.exponential(30, 40000)
    trees = tmp_path / "allData_trees.root"
    with uproot.recreate(trees) as f:
        f["DiphotonTree/Data_13TeV_cat0"] = {"CMS_hgg_mass": mass[mass < 180], "weight": np.ones(np.sum(mass < 180))}
    return trees2ws.convert_data_file(trees, tmp_path / "allData.root", trees2ws.Trees2WSConfig.from_dict(config))


@pytest.mark.parametrize(
    "family,order,n_params",
    [("Bernstein", 3, 3), ("Bernstein", 6, None), ("Exponential", 3, 3), ("Exponential", 2, None), ("Laurent", 2, 2)],
)
def test_pdf_families(family, order, n_params):
    mass = ROOT.RooRealVar("CMS_hgg_mass", "m", 125, 100, 180)
    pdfs = BackgroundPdfs(mass)
    pdf = pdfs.make(family, order, "test")
    if n_params is None:
        assert pdf is None
        return
    params = pdf.getParameters(ROOT.RooArgSet(mass))
    assert pdf.GetName().startswith("test_") and params.getSize() == n_params
    mass.setVal(130)
    assert pdf.getVal(ROOT.RooArgSet(mass)) > 0


def test_ftest(data_workspace, tmp_path):
    background_plots.configure_style("Test", 10)
    out_dir = tmp_path / "plots"
    multipdf_file = tmp_path / "CMS-HGG_multipdf_cat0.root"
    dat_file = tmp_path / "fTest.dat"
    (result,) = background.run_ftest(
        str(data_workspace), ["cat0"], str(out_dir), str(dat_file), "2023", multipdf_file=str(multipdf_file)
    )
    # the data are a single exponential: no higher order is needed
    assert result.truth_orders["Exponential"] == 1
    assert all(result.envelope_orders[family] for family in ("Bernstein", "Exponential"))

    f = ROOT.TFile(str(multipdf_file))
    ws = f.Get("multipdf")
    multipdf = ws.pdf("CMS_hgg_cat0_2023_13TeV_bkgshape")
    index = ws.cat("pdfindex_cat0_2023_13TeV")
    n_pdfs = sum(len(orders) for orders in result.envelope_orders.values())
    assert multipdf and index.numTypes() == n_pdfs and 0 <= index.getIndex() < n_pdfs
    # the sideband yield scaled to 100-180 GeV is the total number of events
    n_events = ws.data("roohist_data_mass_cat0").sumEntries()
    assert ws.var("CMS_hgg_cat0_2023_13TeV_bkgshape_norm").getVal() == pytest.approx(n_events, rel=0.02)
    f.Close()

    assert dat_file.read_text().startswith("cat=0\ntruth=Bernstein:")
    assert (out_dir / "fTestResults.txt").exists()
    for name in ("multipdf_cat0.pdf", "multipdf_cat0.png", "truths_cat0.pdf", "Exponential1_cat0.png"):
        assert (out_dir / name).exists()
