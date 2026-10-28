# Python port of the background function families from ../src/PdfModelBuilder.cc used in the fTest
# Requires the combine library (RooBernsteinFast, RooPower)
import math
import sys
import ROOT

ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit")

class PdfModelBuilder:

  def __init__(self):
    self.obs_var = None
    # Every created RooFit object is kept alive here: RooFit only holds raw pointers to its servers
    self.params = {}
    self.prods = {}
    self.utilities = {}
    self.keepAlive = []

  def setObsVar(self, var):
    self.obs_var = var

  # Mirrors std::map::insert in the C++: an existing entry with the same name is kept
  def _insert(self, store, name, obj):
    if name not in store: store[name] = obj
    else: self.keepAlive.append(obj)
    return store[name]

  def _keep(self, pdf):
    self.keepAlive.append(pdf)
    return pdf

  def getBernstein(self, prefix, order):
    if order < 1 or order > 5: return None
    coeffList = ROOT.RooArgList()
    for i in range(order):
      name = "%s_p%d"%(prefix,i)
      param = ROOT.RooRealVar(name, name, 0.1*(i+1), -15., 15.)
      form = ROOT.RooFormulaVar("%s_sq"%name, "%s_sq"%name, "@0*@0", ROOT.RooArgList(param))
      self._insert(self.params, name, param)
      self._insert(self.prods, name, form)
      coeffList.add(self.prods[name])
    return self._keep(ROOT.RooBernsteinFast[order](prefix, prefix, self.obs_var, coeffList))

  def getExponentialSingle(self, prefix, order):
    if order%2 == 0:
      print("ERROR -- addExponential -- only odd number of params allowed", file=sys.stderr)
      return None
    nfracs = (order-1)//2
    nexps = order-nfracs
    fracs = ROOT.RooArgList()
    exps = ROOT.RooArgList()
    for i in range(1, nfracs+1):
      name = "%s_f%d"%(prefix,i)
      self._insert(self.params, name, ROOT.RooRealVar(name, name, 0.9-float(i-1)*1./nfracs, 0.0001, 0.9999))
      fracs.add(self.params[name])
    for i in range(1, nexps+1):
      name = "%s_p%d"%(prefix,i)
      ename = "%s_e%d"%(prefix,i)
      self._insert(self.params, name, ROOT.RooRealVar(name, name, max(-1., -0.04*(i+1)), -1., 0.))
      self._insert(self.utilities, ename, ROOT.RooExponential(ename, ename, self.obs_var, self.params[name]))
      exps.add(self.utilities[ename])
    exp = ROOT.RooAddPdf(prefix, prefix, exps, fracs, True)
    print("--------------------------")
    return self._keep(exp)

  def getPowerLawSingle(self, prefix, order):
    if order%2 == 0:
      print("ERROR -- addPowerLaw -- only odd number of params allowed", file=sys.stderr)
      return None
    nfracs = (order-1)//2
    npows = order-nfracs
    fracs = ROOT.RooArgList()
    pows = ROOT.RooArgList()
    for i in range(1, nfracs+1):
      name = "%s_f%d"%(prefix,i)
      self._insert(self.params, name, ROOT.RooRealVar(name, name, 0.9-float(i-1)*1./nfracs, 0., 1.))
      fracs.add(self.params[name])
    # keeping the single power-law terms bounded away from zero is crucial,
    # otherwise PowerLaw datasets with very hard spectra (e.g. PTJ0 2024 run)
    # drive the unconstrained coefficients to NaN during the F-test step
    coeffMinAbs = 0.05
    coeffMaxAbs = 10.0
    for i in range(1, npows+1):
      name = "%s_p%d"%(prefix,i)
      ename = "%s_e%d"%(prefix,i)
      initAbs = min(coeffMaxAbs, max(coeffMinAbs, float(i+1)))
      self._insert(self.params, name, ROOT.RooRealVar(name, name, -initAbs, -coeffMaxAbs, -coeffMinAbs))
      self._insert(self.utilities, ename, ROOT.RooPower(ename, ename, self.obs_var, self.params[name]))
      pows.add(self.utilities[ename])
    return self._keep(ROOT.RooAddPdf(prefix, prefix, pows, fracs, True))

  def getLaurentSeries(self, prefix, order):
    nlower = int(math.ceil(order/2.))
    nhigher = order-nlower
    # first do 0th order
    pows = ROOT.RooArgList()
    plist = ROOT.RooArgList()
    pname = "%s_pow0"%prefix
    self._insert(self.utilities, pname, ROOT.RooPower(pname, pname, self.obs_var, ROOT.RooFit.RooConst(-4.)))
    pows.add(self.utilities[pname])
    # even terms
    for i in range(1, nlower+1):
      name = "%s_l%d"%(prefix,i)
      self._insert(self.params, name, ROOT.RooRealVar(name, name, 0.25/order, 0.000001, 0.999999))
      plist.add(self.params[name])
      pname = "%s_powl%d"%(prefix,i)
      self._insert(self.utilities, pname, ROOT.RooPower(pname, pname, self.obs_var, ROOT.RooFit.RooConst(-4.-i)))
      pows.add(self.utilities[pname])
    # odd terms
    for i in range(1, nhigher+1):
      name = "%s_h%d"%(prefix,i)
      self._insert(self.params, name, ROOT.RooRealVar(name, name, 0.25/order, 0.000001, 0.999999))
      plist.add(self.params[name])
      pname = "%s_powh%d"%(prefix,i)
      self._insert(self.utilities, pname, ROOT.RooPower(pname, pname, self.obs_var, ROOT.RooFit.RooConst(-4.+i)))
      pows.add(self.utilities[pname])
    return self._keep(ROOT.RooAddPdf(prefix, prefix, pows, plist, True))
