"""Signal model of one process in one category: a sum of Gaussians (or DCB + Gaussian) in m_gg.

The shape parameters are polynomials in dMH = MH - 125 of order ``mh_poly_order``. They are fitted
simultaneously to the normalised simulation of all mass points by minimising (scipy) the chi2 of
the binned distributions in 110-140 GeV, with Poisson errors derived from the sum of weights.
"""

from __future__ import annotations

import ctypes
import logging
from array import array
from dataclasses import dataclass

import numpy as np
import ROOT
import scipy.stats
from scipy.optimize import minimize

ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit")

logger = logging.getLogger(__name__)

CHI2_RANGE = (110.0, 140.0)

# Initial value and range of the polynomial coefficients p0, p1, p2 (in dMH) of every parameter;
# "func" marks initial values that depend on the component (see build_gaussians)
PARAMETERS = {
    "DCB": {
        "dm": ([0.1, -2.5, 2.5], [0.0, -0.1, 0.1], [0.0, -0.001, 0.001]),
        "sigma": ([2.0, 1.0, 20.0], [0.0, -0.1, 0.1], [0.0, -0.001, 0.001]),
        "n1": ([20.0, 1.00001, 500], [0.0, -0.1, 0.1], [0.0, -0.001, 0.001]),
        "n2": ([20.0, 1.00001, 500], [0.0, -0.1, 0.1], [0.0, -0.001, 0.001]),
        "a1": ([1.0, 1.0, 10.0], [0.0, -0.1, 0.1], [0.0, -0.001, 0.001]),
        "a2": ([1.0, 1.0, 20.0], [0.0, -0.1, 0.1], [0.0, -0.001, 0.001]),
    },
    "Gaussian_wdcb": {"sigma": ([1.5, 1.0, 4.0], [0.0, -0.1, 0.1], [0.0, -0.001, 0.001])},
    "Frac": {"frac": ([0.25, 0.01, 0.99], [0.0, -0.05, 0.05], [0.0, -0.0001, 0.0001])},
    "Gaussian": {
        "dm": ([0.1, -5.0, 5.0], [0.0, -0.01, 0.01], [0.0, -0.01, 0.01]),
        "sigma": (["func", 0.5, 10.0], [0.0, -0.01, 0.01], [0.0, -0.01, 0.01]),
    },
    "FracGaussian": {"frac": (["func", 0.01, 0.99], [0.01, -0.005, 0.005], [0.00001, -0.00001, 0.00001])},
}
# Fractions are kept within [0, 1)
FRACTION_CONSTRAINT = "(@0>0)*(@0<1)*@0+(@0>1.0)*0.9999"


def poisson_interval(x: np.ndarray, error_sumw2: np.ndarray, level: float = 0.68):
    """Lower and upper errors of weighted bin contents from a gamma distribution with the same variance."""
    n_eff = x**2 / (error_sumw2**2)
    scale = abs(x) / n_eff
    low = scipy.stats.gamma.interval(level, n_eff, scale=scale)[0]
    high = scipy.stats.gamma.interval(level, n_eff + 1, scale=scale)[1]
    low[n_eff == 0] = 0.0
    low[error_sumw2 == 0.0] = 0.0
    high[error_sumw2 == 0.0] = np.inf
    return abs(low - x), abs(high - x)


@dataclass
class Chi2Bins:
    """The non-empty bins of a RooDataHist in the chi2 range, with their Poisson errors."""

    x: np.ndarray
    content: np.ndarray
    error: np.ndarray
    volume: np.ndarray
    norm: float

    @classmethod
    def from_hist(cls, hist, xvar, fit_range=CHI2_RANGE) -> Chi2Bins:
        x, content, error_sumw2, volume = [], [], [], []
        for i in range(hist.numEntries()):
            value = hist.get(i).getRealValue(xvar.GetName())
            weight = hist.weight()
            if value < fit_range[0] or value > fit_range[1] or weight * weight == 0:
                continue
            low, high = ctypes.c_double(), ctypes.c_double()
            hist.weightError(low, high, ROOT.RooAbsData.SumW2)
            x.append(value)
            content.append(weight)
            error_sumw2.append(high.value)  # the sum-of-weights errors are symmetric
            volume.append(hist.binVolume())
        content, error_sumw2 = np.asarray(content), np.asarray(error_sumw2, dtype=float)
        low, high = poisson_interval(content, error_sumw2)
        return cls(np.asarray(x), content, np.maximum(high, low), np.asarray(volume), hist.sumEntries())


def normalised_hist(dataset, xvar) -> ROOT.RooDataHist:
    """RooDataHist of ``dataset`` in ``xvar`` (with its binning), normalised to unit sum of weights."""
    values = dataset.to_numpy()
    weights = values[dataset.weightVar().GetName()]
    weight_var = ROOT.RooRealVar("weight", "weight", -10000, 10000)
    columns = {
        # values are clipped into the ranges as RooRealVar::setVal does
        xvar.GetName(): np.clip(values[xvar.GetName()], xvar.getMin(), xvar.getMax()),
        "weight": np.clip((1 / dataset.sumEntries()) * weights, -10000, 10000) if len(weights) else weights,
    }
    normalised = ROOT.RooDataSet.from_numpy(columns, ROOT.RooArgSet(xvar, weight_var), weight_name="weight")
    name = f"{dataset.GetName()}_hist"
    return ROOT.RooDataHist(name, name, ROOT.RooArgSet(xvar), normalised)


class SignalFit:
    """Simultaneous fit of the signal shape to the simulation of several mass points.

    ``datasets`` maps the mass points (strings) to the RooDataSets of the simulation; ``xvar`` must
    be a copy of the observable owned by this fit, ``mh`` is shared between fits.
    """

    def __init__(
        self,
        name: str,
        proc: str,
        cat: str,
        datasets: dict,
        xvar,
        mh,
        n_bins: int,
        mh_poly_order: int = 0,
        minimizer_method: str = "TNC",
    ):
        self.name, self.proc, self.cat = name, proc, cat
        self.xvar, self.mh = xvar, mh
        self.n_bins, self.mh_poly_order, self.minimizer_method = n_bins, mh_poly_order, minimizer_method
        mh.setConstant(False)
        mh.setVal(125)
        mh.setBins(10)
        self.dmh = ROOT.RooFormulaVar("dMH", "dMH", "@0-125.0", ROOT.RooArgList(mh))
        xvar.setVal(125)
        xvar.setBins(n_bins)
        self.vars, self.polynomials, self.pdfs, self.splines = {}, {}, {}, {}
        self._keep_alive = []
        self.data_hists = {mass: normalised_hist(d, xvar) for mass, d in datasets.items()}
        self.chi2_bins = {mass: Chi2Bins.from_hist(h, xvar) for mass, h in self.data_hists.items()}
        self.fit_parameters = None
        self.ndof = None
        self.result = None
        self.n_gaussians = 1

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Model

    def _polynomial(self, name: str, coefficients) -> ROOT.RooPolyVar:
        """Polynomial in dMH; ``coefficients`` holds (initial value, min, max) of p0, p1, ..."""
        coeff_list = ROOT.RooArgList(f"{name}_coeffs")
        for order in range(self.mh_poly_order + 1):
            var = f"{name}_p{order}"
            self.vars[var] = ROOT.RooRealVar(var, var, *coefficients[order])
            coeff_list.add(self.vars[var])
        self._keep_alive.append(coeff_list)
        self.polynomials[name] = ROOT.RooPolyVar(name, name, self.dmh, coeff_list)
        return self.polynomials[name]

    def _fraction(self, name: str, coefficients):
        self._polynomial(name, coefficients)
        constrained = ROOT.RooFormulaVar(
            f"{name}_constrained", f"{name}_constrained", FRACTION_CONSTRAINT, ROOT.RooArgList(self.polynomials[name])
        )
        self.polynomials[f"{name}_constrained"] = constrained
        return constrained

    def _mean(self, name: str, dm):
        self.polynomials[name] = ROOT.RooFormulaVar(name, name, "(@0+@1)", ROOT.RooArgList(self.mh, dm))
        return self.polynomials[name]

    def build_gaussians(self, n_gaussians: int) -> None:
        """Sum of n Gaussians with recursive fractions; the initial widths increase with the index."""
        self.n_gaussians = n_gaussians
        pdfs, fractions = ROOT.RooArgList(), ROOT.RooArgList()
        table = PARAMETERS["Gaussian"]
        for g in range(n_gaussians):
            dm = self._polynomial(f"dm_g{g}", table["dm"])
            sigma_table = ((g + 1) * 1.0, *table["sigma"][0][1:]), *table["sigma"][1:]
            sigma = self._polynomial(f"sigma_g{g}", sigma_table)
            mean = self._mean(f"mean_g{g}", dm)
            self.pdfs[f"gaus_g{g}"] = ROOT.RooGaussian(f"gaus_g{g}", f"gaus_g{g}", self.xvar, mean, sigma)
            pdfs.add(self.pdfs[f"gaus_g{g}"])
            if g < n_gaussians - 1:
                frac_table = PARAMETERS["FracGaussian"]["frac"]
                frac_table = ((0.5 - 0.05 * g, *frac_table[0][1:]), *frac_table[1:])
                fractions.add(self._fraction(f"frac_g{g}", frac_table))
        self._keep_alive += [pdfs, fractions]
        name = f"{self.proc}_{self.cat}"
        self.pdfs["final"] = ROOT.RooAddPdf(name, name, pdfs, fractions, True)

    def build_dcb_plus_gaussian(self) -> None:
        """Double crystal ball plus a Gaussian with the same mean."""
        table = PARAMETERS["DCB"]
        shape = {p: self._polynomial(f"{p}_dcb", table[p]) for p in ("dm", "sigma", "n1", "n2", "a1", "a2")}
        mean = self._mean("mean_dcb", shape["dm"])
        self.pdfs["dcb"] = ROOT.RooDoubleCBFast(
            "dcb", "dcb", self.xvar, mean, shape["sigma"], shape["a1"], shape["n1"], shape["a2"], shape["n2"]
        )
        sigma = self._polynomial("sigma_gaus", PARAMETERS["Gaussian_wdcb"]["sigma"])
        self.pdfs["gaus"] = ROOT.RooGaussian("gaus", "gaus", self.xvar, mean, sigma)
        fraction = self._fraction("frac", PARAMETERS["Frac"]["frac"])
        pdfs, fractions = ROOT.RooArgList(self.pdfs["dcb"], self.pdfs["gaus"]), ROOT.RooArgList(fraction)
        self._keep_alive += [pdfs, fractions]
        name = f"{self.proc}_{self.cat}"
        self.pdfs["final"] = ROOT.RooAddPdf(name, name, pdfs, fractions, True)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fit

    def _parameter_values(self) -> np.ndarray:
        return np.asarray([self.fit_parameters[i].getVal() for i in range(len(self.fit_parameters))])

    def chi2_at(self, values) -> float:
        """Chi2 summed over the mass points with the fit parameters (MH included, unused) set to values."""
        for i, value in enumerate(values):
            self.fit_parameters[i].setVal(value)
        norm_set = ROOT.RooArgSet(self.xvar)
        pdf = self.pdfs["final"]
        chi2, n_bins = 0, 0
        for mass, bins in self.chi2_bins.items():
            self.mh.setVal(int(mass))
            n_pdf = np.empty(len(bins.x))
            for i, x in enumerate(bins.x):
                self.xvar.setVal(x)
                n_pdf[i] = pdf.getVal(norm_set) * bins.norm * bins.volume[i]
            chi2 += ((n_pdf - bins.content) ** 2 / (bins.error**2)).sum()
            n_bins += len(bins.x)
        self.ndof = n_bins - (len(values) - 1)
        return chi2

    def fit(self) -> None:
        """Minimise the chi2; the parameters are left at the last point evaluated by the minimiser."""
        parameters = self.pdfs["final"].getVariables().Clone()
        parameters.remove(self.xvar)
        self._keep_alive.append(parameters)
        self.fit_parameters = ROOT.RooArgList(parameters)
        x0 = self._parameter_values()
        bounds = np.asarray([(p.getMin(), p.getMax()) for p in self.fit_parameters])
        self.chi2_at(x0)
        self.result = minimize(self.chi2_at, x0, bounds=bounds, method=self.minimizer_method)

    def chi2(self) -> float:
        return self.chi2_at(self._parameter_values())

    def reduced_chi2(self) -> float:
        return self.chi2() / int(self.ndof)

    def build_splines(self) -> None:
        """RooSpline1D in MH of every polynomial, sampled from 100 to 180 GeV in steps of 0.1 GeV."""
        for name, polynomial in self.polynomials.items():
            x, y = [], []
            mh = 100.0
            while mh < 180.1:
                self.mh.setVal(mh)
                x.append(mh)
                y.append(polynomial.getVal())
                mh += 0.1
            x, y = array("f", x), array("f", y)
            self.splines[name] = ROOT.RooSpline1D(polynomial.GetName(), polynomial.GetName(), self.mh, len(x), x, y)
