"""Background function families of the discrete profiling method.

Bernstein polynomials (with squared coefficients), sums of exponentials, sums of power laws and
Laurent series in the diphoton mass, as in Background/src/PdfModelBuilder.cc. The Bernstein
polynomials and power laws are the fast implementations of combine (RooBernsteinFast, RooPower).
"""

from __future__ import annotations

import math

import ROOT

ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit")

# Family name -> short name used in the pdf names (e.g. <prefix>_bern3) and the .dat file
FAMILIES = {"Bernstein": "bern", "Exponential": "exp", "PowerLaw": "pow", "Laurent": "lau"}
DAT_NAMES = {"Bernstein": "pol", "Exponential": "exp", "PowerLaw": "pow", "Laurent": "lau"}


class BackgroundPdfs:
    """Builds the pdfs of all families for one observable.

    RooFit objects only hold raw pointers to the objects they depend on, so every parameter and
    component is kept alive here. Parameters and components are shared by name: building a pdf
    whose parameter name already exists reuses the existing parameter.
    """

    def __init__(self, observable: ROOT.RooRealVar):
        self.observable = observable
        self._objects = {}
        self._keep_alive = []

    def _shared(self, obj):
        # The first object with a given name is used, later ones are only kept alive
        self._keep_alive.append(obj)
        return self._objects.setdefault(obj.GetName(), obj)

    def make(self, family: str, order: int, prefix: str) -> ROOT.RooAbsPdf | None:
        """The pdf ``<prefix>_<short family name><order>``, or None if the order is not allowed."""
        build = {
            "Bernstein": self.bernstein,
            "Exponential": self.exponentials,
            "PowerLaw": self.power_laws,
            "Laurent": self.laurent,
        }[family]
        pdf = build(f"{prefix}_{FAMILIES[family]}{order}", order)
        if pdf is not None:
            self._keep_alive.append(pdf)
        return pdf

    def bernstein(self, name: str, order: int) -> ROOT.RooAbsPdf | None:
        """Bernstein polynomial of degree ``order`` (1-5) with coefficients p_i^2 > 0."""
        if order < 1 or order > 5:
            return None
        coefficients = ROOT.RooArgList()
        for i in range(order):
            param = self._shared(ROOT.RooRealVar(f"{name}_p{i}", f"{name}_p{i}", 0.1 * (i + 1), -15.0, 15.0))
            square = ROOT.RooFormulaVar(f"{name}_p{i}_sq", f"{name}_p{i}_sq", "@0*@0", ROOT.RooArgList(param))
            # the squares are stored under the parameter name, as in PdfModelBuilder.cc
            self._keep_alive.append(square)
            coefficients.add(self._objects.setdefault(f"{name}_p{i}#sq", square))
        return ROOT.RooBernsteinFast[order](name, name, self.observable, coefficients)

    def exponentials(self, name: str, order: int) -> ROOT.RooAbsPdf | None:
        """Sum of (order+1)/2 exponentials with recursive fractions; ``order`` must be odd."""
        if order % 2 == 0:
            return None
        n_fractions = (order - 1) // 2
        fractions, exponentials = ROOT.RooArgList(), ROOT.RooArgList()
        for i in range(1, n_fractions + 1):
            start = 0.9 - float(i - 1) * 1.0 / n_fractions
            fractions.add(self._shared(ROOT.RooRealVar(f"{name}_f{i}", f"{name}_f{i}", start, 0.0001, 0.9999)))
        for i in range(1, order - n_fractions + 1):
            slope = self._shared(ROOT.RooRealVar(f"{name}_p{i}", f"{name}_p{i}", max(-1.0, -0.04 * (i + 1)), -1.0, 0.0))
            exponentials.add(self._shared(ROOT.RooExponential(f"{name}_e{i}", f"{name}_e{i}", self.observable, slope)))
        return ROOT.RooAddPdf(name, name, exponentials, fractions, True)

    def power_laws(self, name: str, order: int) -> ROOT.RooAbsPdf | None:
        """Sum of (order+1)/2 power laws with recursive fractions; ``order`` must be odd."""
        if order % 2 == 0:
            return None
        n_fractions = (order - 1) // 2
        fractions, powers = ROOT.RooArgList(), ROOT.RooArgList()
        for i in range(1, n_fractions + 1):
            start = 0.9 - float(i - 1) * 1.0 / n_fractions
            fractions.add(self._shared(ROOT.RooRealVar(f"{name}_f{i}", f"{name}_f{i}", start, 0.0, 1.0)))
        # The exponents are kept away from zero, otherwise very hard spectra (e.g. PTJ0 2024) drive
        # them to NaN in the fTest
        min_abs, max_abs = 0.05, 10.0
        for i in range(1, order - n_fractions + 1):
            start = -min(max_abs, max(min_abs, float(i + 1)))
            exponent = self._shared(ROOT.RooRealVar(f"{name}_p{i}", f"{name}_p{i}", start, -max_abs, -min_abs))
            powers.add(self._shared(ROOT.RooPower(f"{name}_e{i}", f"{name}_e{i}", self.observable, exponent)))
        return ROOT.RooAddPdf(name, name, powers, fractions, True)

    def laurent(self, name: str, order: int) -> ROOT.RooAbsPdf:
        """Laurent series: m^-4 plus ``order`` terms m^(-4-i) and m^(-4+i) with recursive fractions."""
        n_lower = int(math.ceil(order / 2.0))
        powers, fractions = ROOT.RooArgList(), ROOT.RooArgList()
        powers.add(
            self._shared(ROOT.RooPower(f"{name}_pow0", f"{name}_pow0", self.observable, ROOT.RooFit.RooConst(-4.0)))
        )
        for kind, n_terms, sign in (("l", n_lower, -1), ("h", order - n_lower, 1)):
            for i in range(1, n_terms + 1):
                fraction = ROOT.RooRealVar(f"{name}_{kind}{i}", f"{name}_{kind}{i}", 0.25 / order, 0.000001, 0.999999)
                fractions.add(self._shared(fraction))
                exponent = ROOT.RooFit.RooConst(-4.0 + sign * i)
                power = ROOT.RooPower(f"{name}_pow{kind}{i}", f"{name}_pow{kind}{i}", self.observable, exponent)
                powers.add(self._shared(power))
        return ROOT.RooAddPdf(name, name, powers, fractions, True)
