# Python port of ../test/fTest.cpp: background model fTest and RooMultiPdf envelope construction
# * run per analysis category on the data workspace (allData.root)
# * no compilation needed: the pdfs are built from RooFit and combine classes only
import os, sys
import math
from optparse import OptionParser
import ROOT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../tools"))
import cmsStyle
from cmsStyle import setTDRStyle, CMS_lumi
from pdfModelBuilder import PdfModelBuilder

def get_options():
  parser = OptionParser()
  parser.add_option("-i", "--infilename", dest="fileName", default="", help="In file name")
  parser.add_option("-d", "--datfile", dest="datfile", default="dat/fTest.dat", help="Right results to datfile for BiasStudy")
  parser.add_option("-D", "--outDir", dest="outDir", default="plots/fTest", help="Out directory for plots")
  parser.add_option("--saveMultiPdf", dest="saveMultiPdf", default=None, help="Save a MultiPdf model with the appropriate pdfs")
  parser.add_option("--runFtestCheckWithToys", dest="runFtestCheckWithToys", default=False, action="store_true", help="When running the F-test, use toys to calculate pvals (and make plots)")
  parser.add_option("--unblind", dest="unblind", default=False, action="store_true", help="Dont blind plots")
  parser.add_option("--isData", dest="isData", default=0, type="int", help="Use Data not MC")
  parser.add_option("-f", "--flashggCats", dest="flashggCats", default="UntaggedTag_0,UntaggedTag_1,UntaggedTag_2,UntaggedTag_3,UntaggedTag_4,VBFTag_0,VBFTag_1,VBFTag_2,TTHHadronicTag,TTHLeptonicTag,VHHadronicTag,VHTightTag,VHLooseTag,VHEtTag", help="Flashgg category names to consider")
  parser.add_option("--year", dest="year", default="2016", help="Dataset year")
  parser.add_option("--intLumi", dest="intLumi", default=61.9, type="float", help="Integrated luminosity shown on plots in fb^-1")
  parser.add_option("--cmsLabel", dest="cmsLabel", default="Preliminary", help="Left-hand plot label")
  parser.add_option("--catOffset", dest="catOffset", default=0, type="int", help="Category numbering scheme offset")
  parser.add_option("-v", "--verbose", dest="verbose", default=False, action="store_true", help="Run with more output")
  return parser.parse_args()

BLIND = True
runFtestCheckWithToys = False
mgg_low = 100
mgg_high = 180
blind_low = 115
blind_high = 135
nBinsForMass = 4*(mgg_high-mgg_low)
nBinsSidebands = 4*(mgg_high-mgg_low - (blind_high - blind_low)) # 240 instead of 320
MASS_FIT_RANGE = "low,high"

intLumi_ = ROOT.RooRealVar("IntLumi", "hacked int lumi", 1000.)

RandomGen = ROOT.TRandom3()

# Objects returned by pointer are not owned by python: take ownership where the C++ deletes them
def owned(obj):
  ROOT.SetOwnership(obj, True)
  return obj

class GoFResult:
  def __init__(self):
    self.chi2Red = 0.
    self.chi2Abs = 0.
    self.prob = 0.
    self.nPoints = 0
    self.nFitParams = 0
    self.ndof = 0

def isSidebandPoint(x):
  return ((x >= mgg_low) and (x < blind_low)) or ((x >= blind_high) and (x <= mgg_high))

def computePlotGoF(plot, pdfName, dataName, nFitParams):
  result = GoFResult()
  # RooPlot::chiSquare(...) returns the reduced chi2 = chi2 / ndof.
  result.chi2Red = plot.chiSquare(pdfName, dataName, nFitParams)
  # Count only points whose x-values lie in the sideband windows. In practice
  # the RooHist attached to the plot can still carry graph points across the
  # full mass range even when the visible fit is restricted to "low,high".
  # Using the sideband x positions avoids hard-coding 240 while keeping the
  # ndof tied to the actual sideband points entering the plotted GOF.
  hData = plot.findObject(dataName)
  result.nPoints = 0
  if hData:
    for iPoint in range(hData.GetN()):
      if isSidebandPoint(hData.GetPointX(iPoint)): result.nPoints += 1
  if result.nPoints <= 0: result.nPoints = nBinsSidebands
  result.nFitParams = nFitParams
  result.ndof = result.nPoints - result.nFitParams
  if result.ndof > 0:
    result.chi2Abs = result.chi2Red * result.ndof
    result.prob = ROOT.TMath.Prob(result.chi2Abs, result.ndof)
  else:
    result.chi2Abs = 0.
    result.prob = 0.
    print("[WARNING] computePlotGoF: non-positive ndof = %d (nPoints = %d, nFitParams = %d)"%(result.ndof, result.nPoints, result.nFitParams), file=sys.stderr)
  return result

def getPdf(pdfsModel, _type, order, ext=""):
  if _type == "Bernstein": return pdfsModel.getBernstein("%s_bern%d"%(ext,order), order)
  elif _type == "Exponential": return pdfsModel.getExponentialSingle("%s_exp%d"%(ext,order), order)
  elif _type == "PowerLaw": return pdfsModel.getPowerLawSingle("%s_pow%d"%(ext,order), order)
  elif _type == "Laurent": return pdfsModel.getLaurentSeries("%s_lau%d"%(ext,order), order)
  else:
    print("[ERROR] -- getPdf() -- type %s not recognised."%_type, file=sys.stderr)
    return None

# Returns (NLL, fit status)
def runFit(pdf, data, MaxTries):
  ntries = 0
  params_test = pdf.getParameters(ROOT.RooArgSet())
  stat = 1
  minnll = 10e8
  while stat != 0:
    if ntries >= MaxTries: break
    fitTest = pdf.fitTo(data, ROOT.RooFit.Save(True), ROOT.RooFit.Minimizer("Minuit2","minimize"), ROOT.RooFit.SumW2Error(True), ROOT.RooFit.Range(MASS_FIT_RANGE))
    stat = fitTest.status()
    minnll = fitTest.minNll()
    if stat != 0: params_test.assignValueOnly(fitTest.randomizePars())
    ntries += 1
  return minnll, stat

# Needed to scale the sideband yield to the full range yield
# Python version of Jon's code -> https://github.com/jonathon-langford/flashggFinalFit/blob/dev_lowmass/Background/tools/modelBuilder_v2.py#L266-L306
def computeSidebandScaledYield(pdf, mass, sidebandYield):
  if (not pdf) or (not mass):
    print("[ERROR] computeSidebandScaledYield: pdf or mass is null. Returning uncorrected sidebandYield (%g)."%sidebandYield, file=sys.stderr)
    return sidebandYield
  observables = ROOT.RooArgSet(mass)
  sidebandIntegral = owned(pdf.createIntegral(observables, ROOT.RooFit.NormSet(observables), ROOT.RooFit.Range(MASS_FIT_RANGE)))
  fullIntegral = owned(pdf.createIntegral(observables, ROOT.RooFit.NormSet(observables), ROOT.RooFit.Range("fullRange")))

  pdfSidebandFrac = sidebandIntegral.getVal() if sidebandIntegral else 0.
  pdfFullFrac = fullIntegral.getVal() if fullIntegral else 0.

  if pdfSidebandFrac <= 0.: return sidebandYield
  if pdfFullFrac <= 0.: pdfFullFrac = 1.

  return sidebandYield * (pdfFullFrac / pdfSidebandFrac)

def getProbabilityFtest(chi2, ndof, pdfNull, pdfTest, mass, data, name):

  prob_asym = ROOT.TMath.Prob(chi2, ndof)
  if not runFtestCheckWithToys: return prob_asym

  ndata = int(data.sumEntries())

  # fit the pdfs to the data and keep this fit Result (for randomizing)
  fitNullData = pdfNull.fitTo(data, ROOT.RooFit.Save(True), ROOT.RooFit.Strategy(1), ROOT.RooFit.Minimizer("Minuit2","minimize"), ROOT.RooFit.SumW2Error(True), ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Range(MASS_FIT_RANGE))
  fitTestData = pdfTest.fitTo(data, ROOT.RooFit.Save(True), ROOT.RooFit.Strategy(1), ROOT.RooFit.Minimizer("Minuit2","minimize"), ROOT.RooFit.SumW2Error(True), ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Range(MASS_FIT_RANGE))

  # Ok we want to check the distribution in toys then
  # Step 1, cache the parameters of each pdf so as not to upset anything
  params_null = pdfNull.getParameters(ROOT.RooArgSet())
  preParams_null = ROOT.RooArgSet()
  params_null.snapshot(preParams_null)
  params_test = pdfTest.getParameters(ROOT.RooArgSet())
  preParams_test = ROOT.RooArgSet()
  params_test.snapshot(preParams_test)

  ntoys = 5000
  can = ROOT.TCanvas()
  can.SetLogy()
  toyhist = ROOT.TH1F("toys_fTest_%s.pdf"%pdfNull.GetName(), ";Chi2;", 60, -2, 10)
  toyhistStatN = ROOT.TH1I("Status_%s.pdf"%pdfNull.GetName(), ";FitStatus;", 8, -4, 4)
  toyhistStatT = ROOT.TH1I("Status_%s.pdf"%pdfTest.GetName(), ";FitStatus;", 8, -4, 4)

  gChi2 = ROOT.TGraph()
  gChi2.SetLineColor(ROOT.kGreen+2)
  w = toyhist.GetBinWidth(1)

  ipoint = 0
  for b in range(toyhist.GetNbinsX()):
    x = toyhist.GetBinCenter(b+1)
    if x > 0:
      gChi2.SetPoint(ipoint, x, ROOT.Math.chisquared_pdf(x, ndof))
      ipoint += 1

  npass = 0
  nsuccesst = 0
  mass.setBins(nBinsForMass)
  for itoy in range(ntoys):

    params_null.assignValueOnly(preParams_null)
    params_test.assignValueOnly(preParams_test)
    # Now the toys should be thrown only in the sidebands
    binnedtoy = owned(pdfNull.generateBinned(ROOT.RooArgSet(mass), ROOT.RooFit.NumEvents(ndata), ROOT.RooFit.Range(MASS_FIT_RANGE), ROOT.RooFit.ExpectedData(False), ROOT.RooFit.Extended(False)))

    stat_n = 1
    stat_t = 1
    ntries = 0
    nllNull, nllTest = 0., 0.
    # Iterate on the fit
    MaxTries = 2
    while stat_n != 0:
      if ntries >= MaxTries: break
      fitNull = pdfNull.fitTo(binnedtoy, ROOT.RooFit.Save(True), ROOT.RooFit.Strategy(1), ROOT.RooFit.SumW2Error(True), ROOT.RooFit.Minimizer("Minuit2","minimize"), ROOT.RooFit.Minos(False), ROOT.RooFit.Hesse(False), ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Range(MASS_FIT_RANGE))
      nllNull = fitNull.minNll()
      stat_n = fitNull.status()
      if stat_n != 0: params_null.assignValueOnly(fitNullData.randomizePars())
      ntries += 1

    ntries = 0
    while stat_t != 0:
      if ntries >= MaxTries: break
      fitTest = pdfTest.fitTo(binnedtoy, ROOT.RooFit.Save(True), ROOT.RooFit.Strategy(1), ROOT.RooFit.SumW2Error(True), ROOT.RooFit.Minimizer("Minuit2","minimize"), ROOT.RooFit.Minos(False), ROOT.RooFit.Hesse(False), ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Range(MASS_FIT_RANGE))
      nllTest = fitTest.minNll()
      stat_t = fitTest.status()
      if stat_t != 0: params_test.assignValueOnly(fitTestData.randomizePars())
      ntries += 1

    toyhistStatN.Fill(stat_n)
    toyhistStatT.Fill(stat_t)

    if stat_t != 0 or stat_n != 0: continue
    nsuccesst += 1
    chi2_t = 2*(nllNull-nllTest)
    if chi2_t >= chi2: npass += 1
    toyhist.Fill(chi2_t)

  prob = 0
  if nsuccesst != 0: prob = float(npass) / nsuccesst
  toyhist.Scale(1./(w*toyhist.Integral()))
  toyhist.Draw()
  lData = ROOT.TArrow(chi2, toyhist.GetMaximum(), chi2, 0)
  lData.SetLineWidth(2)
  lData.Draw()
  gChi2.Draw("L")
  lat = ROOT.TLatex()
  lat.SetNDC()
  lat.SetTextFont(42)
  lat.DrawLatex(0.1, 0.91, "Prob (asymptotic) = %.4f (%.4f)"%(prob, prob_asym))
  can.SaveAs(name)

  stas = ROOT.TCanvas()
  toyhistStatN.SetLineColor(2)
  toyhistStatT.SetLineColor(1)
  leg = ROOT.TLegend(0.2, 0.6, 0.4, 0.87)
  leg.SetFillColor(0)
  leg.SetTextFont(42)
  leg.AddEntry(toyhistStatN, "Null Hyp", "L")
  leg.AddEntry(toyhistStatT, "Test Hyp", "L")
  toyhistStatN.Draw()
  toyhistStatT.Draw("same")
  leg.Draw()
  stas.SaveAs("%s_fitstatus.pdf"%name)
  # reassign params
  params_null.assignValueOnly(preParams_null)
  params_test.assignValueOnly(preParams_test)

  # Still return the asymptotic prob (usually its close to the toys one)
  return prob_asym

def getGoodnessOfFit(mass, mpdf, data, name):

  ntoys = 500
  # Routine to calculate the goodness of fit.
  name += "_gofTest.pdf"
  norm = ROOT.RooRealVar("norm", "norm", data.sumEntries(), 0, 10E6)

  pdf = ROOT.RooExtendPdf("ext", "ext", mpdf, norm)
  sidebandEntries = data.sumEntries()

  # get The Chi2 value from the data
  plot_chi2 = owned(mass.frame())
  data.plotOn(plot_chi2, ROOT.RooFit.Binning(nBinsForMass), ROOT.RooFit.Name("data"), ROOT.RooFit.Range(MASS_FIT_RANGE))

  pdf.plotOn(plot_chi2, ROOT.RooFit.Name("pdf"), ROOT.RooFit.Range(MASS_FIT_RANGE), ROOT.RooFit.NormRange(MASS_FIT_RANGE), ROOT.RooFit.Normalization(sidebandEntries, ROOT.RooAbsReal.NumEvent))
  np = pdf.getParameters(data).getSize()

  gof = computePlotGoF(plot_chi2, "pdf", "data", np)
  print("[INFO] Calculating GOF for pdf %s, using %d fitted parameters"%(pdf.GetName(), np))

  # The first thing is to check if the number of entries in any bin is < 5
  # if so, we don't rely on asymptotic approximations
  if data.sumEntries()/nBinsSidebands < 5:

    print("[INFO] Running toys for GOF test ")
    # store pre-fit params
    params = pdf.getParameters(data)
    preParams = ROOT.RooArgSet()
    params.snapshot(preParams)
    ndata = int(data.sumEntries())

    npass = 0
    toy_chi2 = []
    for itoy in range(ntoys):
      params.assignValueOnly(preParams)
      nToyEvents = RandomGen.Poisson(ndata)
      binnedtoy = owned(pdf.generateBinned(ROOT.RooArgSet(mass), ROOT.RooFit.NumEvents(nToyEvents), ROOT.RooFit.Range(MASS_FIT_RANGE), ROOT.RooFit.ExpectedData(False), ROOT.RooFit.Extended(False)))
      pdf.fitTo(binnedtoy, ROOT.RooFit.Minimizer("Minuit2","minimize"), ROOT.RooFit.Minos(False), ROOT.RooFit.Hesse(False), ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Strategy(0), ROOT.RooFit.SumW2Error(True), ROOT.RooFit.Range(MASS_FIT_RANGE))

      plot_t = owned(mass.frame())
      toyEntries = binnedtoy.sumEntries()
      binnedtoy.plotOn(plot_t, ROOT.RooFit.Binning(nBinsForMass), ROOT.RooFit.Name("toy_data"), ROOT.RooFit.Range(MASS_FIT_RANGE))
      pdf.plotOn(plot_t, ROOT.RooFit.Name("toy_pdf"), ROOT.RooFit.Range(MASS_FIT_RANGE), ROOT.RooFit.NormRange(MASS_FIT_RANGE), ROOT.RooFit.Normalization(toyEntries, ROOT.RooAbsReal.NumEvent))
      toyGof = computePlotGoF(plot_t, "toy_pdf", "toy_data", np)
      if toyGof.chi2Abs >= gof.chi2Abs: npass += 1
      toy_chi2.append(toyGof.chi2Abs)
      del plot_t
      del binnedtoy
    print("[INFO] complete")
    prob = float(npass) / ntoys

    can = ROOT.TCanvas()
    medianChi2 = toy_chi2[int(float(ntoys)/2)]
    rms = math.sqrt(medianChi2)

    toyhist = ROOT.TH1F("gofTest_%s.pdf"%pdf.GetName(), ";Chi2;", 50, medianChi2-5*rms, medianChi2+5*rms)
    for itx in toy_chi2: toyhist.Fill(itx)
    toyhist.Draw()

    lData = ROOT.TArrow(gof.chi2Abs, toyhist.GetMaximum(), gof.chi2Abs, 0)
    lData.SetLineWidth(2)
    lData.Draw()
    can.SaveAs(name)

    # back to best fit
    params.assignValueOnly(preParams)
  else:
    prob = gof.prob

  print("[INFO] nPoints = %d"%gof.nPoints)
  print("[INFO] nFitParams = %d"%gof.nFitParams)
  print("[INFO] ndof = %d"%gof.ndof)
  print("[INFO] chi2/ndof = %g"%gof.chi2Red)
  print("[INFO] Chi2 in Observed = %g"%gof.chi2Abs)
  print("[INFO] p-value = %g"%prob)
  del plot_chi2
  del pdf
  return prob, gof

# Plot a single fitted pdf with its GOF, returns the GOF probability
def plot(mass, pdf, data, name, flashggCats_, status):

  prob, gof = getGoodnessOfFit(mass, pdf, data, name)
  sidebandEntries = data.sumEntries()
  plot = owned(mass.frame())
  mass.setRange("unblindReg_1", mgg_low, 115)
  mass.setRange("unblindReg_2", 135, mgg_high)
  if BLIND:
    data.plotOn(plot, ROOT.RooFit.Binning(mgg_high-mgg_low), ROOT.RooFit.CutRange("unblindReg_1"))
    data.plotOn(plot, ROOT.RooFit.Binning(mgg_high-mgg_low), ROOT.RooFit.CutRange("unblindReg_2"))
    data.plotOn(plot, ROOT.RooFit.Binning(mgg_high-mgg_low), ROOT.RooFit.Invisible())
  else: data.plotOn(plot, ROOT.RooFit.Binning(mgg_high-mgg_low), ROOT.RooFit.Range(MASS_FIT_RANGE))

  canv = ROOT.TCanvas()
  pdf.plotOn(plot, ROOT.RooFit.Range(float(mgg_low), float(mgg_high)), ROOT.RooFit.NormRange(MASS_FIT_RANGE), ROOT.RooFit.Normalization(sidebandEntries, ROOT.RooAbsReal.NumEvent))
  pdf.paramOn(plot, ROOT.RooFit.Layout(0.34, 0.96, 0.89), ROOT.RooFit.Format("NEA", ROOT.RooFit.AutoPrecision(1)))
  if BLIND: plot.SetMinimum(0.0001)
  plot.SetTitle("")
  plot.Draw()
  lat = ROOT.TLatex()
  lat.SetNDC()
  lat.SetTextFont(42)
  lat.SetTextSize(0.03)
  lat.DrawLatex(0.1, 0.92, "#chi^{2}/dof = %.3f (%.3f / %d), Prob = %.2f, Fit Status = %d "%(gof.chi2Red, gof.chi2Abs, gof.ndof, prob, status))
  canv.SaveAs(name)
  if len(name) > 4 and name.endswith(".pdf"):
    canv.SaveAs(name[:-4]+".png") # also keep a png copy for quick inspections / web plots

  del canv
  del lat
  del plot
  return prob

def plotMultiPdf(mass, pdfs, catIndex, data, name, flashggCats_, cat, bestFitPdf=-1):

  color = [ROOT.kBlue+1, ROOT.kRed+1, ROOT.kMagenta+1, ROOT.kGreen+2, ROOT.kOrange+7, ROOT.kCyan+2, ROOT.kGray+2]
  leg = ROOT.TLegend(0.52, 0.45, 0.94, 0.84)
  leg.SetFillStyle(0)
  leg.SetBorderSize(0)
  leg.SetTextFont(42)
  leg.SetTextSize(0.028)
  plot = owned(mass.frame())

  mass.setRange("unblindReg_1", mgg_low, 115)
  mass.setRange("unblindReg_2", 135, mgg_high)
  if BLIND:
    data.plotOn(plot, ROOT.RooFit.Binning(mgg_high-mgg_low), ROOT.RooFit.CutRange("unblindReg_1"))
    data.plotOn(plot, ROOT.RooFit.Binning(mgg_high-mgg_low), ROOT.RooFit.CutRange("unblindReg_2"))
    data.plotOn(plot, ROOT.RooFit.Binning(mgg_high-mgg_low), ROOT.RooFit.Invisible())
  else: data.plotOn(plot, ROOT.RooFit.Binning(mgg_high-mgg_low), ROOT.RooFit.Range(MASS_FIT_RANGE))
  canv = ROOT.TCanvas()
  ###start extra bit for ratio plot###
  pad1 = ROOT.TPad("pad1_%d"%cat, "", 0, 0.25, 1, 1)
  pad2 = ROOT.TPad("pad2_%d"%cat, "", 0, 0, 1, 0.35)
  pad1.SetBottomMargin(0.18)
  pad1.SetTicks(1, 1)
  pad2.SetTopMargin(0.04)
  pad2.SetBottomMargin(0.3)
  pad2.SetTicks(1, 1)
  pad1.Draw()
  pad2.Draw()
  pad1.cd()
  # end extra bit for ratio plot###
  # Choose a suffix for saved objects, falling back to a generic catN label if no names are provided.
  histSuffix = "cat%d"%cat if len(flashggCats_) == 0 else flashggCats_[cat]
  # Build an explicit TH1 copy of the data so we can zero the blinded bins before computing residuals.
  dataHist = data.createHistogram("h_data_%s"%histSuffix, mass, ROOT.RooFit.Binning(mgg_high-mgg_low, mgg_low, mgg_high))
  if dataHist: owned(dataHist)
  else: print("[WARNING] Failed to build data histogram for ratio plot in %s"%histSuffix, file=sys.stderr)

  currentIndex = catIndex.getIndex()
  datLeg = plot.getObject(int(plot.numItems()-1))
  categoryLabel = flashggCats_[cat]
  if categoryLabel.startswith("cat"): categoryLabel = "Category " + categoryLabel[3:]
  leg.AddEntry(datLeg, "Data (%s)"%categoryLabel, "LEP")
  style = 1
  pdf = None
  nomBkgCurve = None
  bestcol = -1
  sidebandEntries = data.sumEntries()
  for icat in range(catIndex.numTypes()):
    if icat <= 6: col = color[icat]
    else:
      col = ROOT.kBlack
      style += 1
    catIndex.setIndex(icat)
    pdfs.getCurrentPdf().fitTo(data, ROOT.RooFit.Minos(False), ROOT.RooFit.Minimizer("Minuit2","minimize"), ROOT.RooFit.SumW2Error(True), ROOT.RooFit.Range(MASS_FIT_RANGE))
    lineWidth = 4 if bestFitPdf == icat else 2
    pdfs.getCurrentPdf().plotOn(plot, ROOT.RooFit.Range(float(mgg_low), float(mgg_high)), ROOT.RooFit.NormRange(MASS_FIT_RANGE), ROOT.RooFit.Normalization(sidebandEntries, ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineColor(col), ROOT.RooFit.LineStyle(style), ROOT.RooFit.LineWidth(lineWidth))
    pdfLeg = plot.getObject(int(plot.numItems()-1))
    if bestFitPdf == icat:
      pdf = pdfs.getCurrentPdf()
      nomBkgCurve = plot.getObject(int(plot.numItems()-1))
      bestcol = col
    pdfName = pdfs.getCurrentPdf().GetName()
    family = "Candidate function"
    if "bern" in pdfName: family = "Bernstein polynomial"
    elif "exp" in pdfName: family = "Sum of exponentials"
    elif "pow" in pdfName: family = "Power-law sum"
    elif "lau" in pdfName: family = "Laurent series"
    order = "?" if pdfName == "" else pdfName[-1]
    leg.AddEntry(pdfLeg, "%s, order %s%s"%(family, order, " (best fit)" if bestFitPdf == icat else ""), "L")
  plot.SetTitle("")
  if BLIND: plot.SetMinimum(0.0001)
  plot.GetXaxis().SetLabelSize(0.)
  plot.GetXaxis().SetTitleSize(0.)
  plot.GetYaxis().SetTitle("Events / (1 GeV)")
  plot.GetYaxis().SetTitleOffset(1.55)
  plot.Draw()
  leg.Draw("same")
  CMS_lumi(pad1, 2022, 0)
  ###start extra bit for ratio plot###
  # Mirror the (best-fit) pdf into a histogram with the same binning as the data histogram.
  pdfHist = None
  if pdf:
    pdfHist = pdf.createHistogram("h_pdf_%s"%histSuffix, mass, ROOT.RooFit.Binning(mgg_high-mgg_low, mgg_low, mgg_high))
    if pdfHist: owned(pdfHist)
  if not pdfHist:
    print("[WARNING] Failed to build pdf histogram for ratio plot in %s"%histSuffix, file=sys.stderr)
  dataSidebandYield = 0.
  pdfSidebandYield = 0.
  if dataHist:
    for ibin in range(1, dataHist.GetNbinsX()+1):
      x = dataHist.GetBinCenter(ibin)
      if x > blind_low and x < blind_high:
        dataHist.SetBinContent(ibin, 0.)
        dataHist.SetBinError(ibin, 0.)
        if pdfHist:
          pdfHist.SetBinContent(ibin, 0.)
          pdfHist.SetBinError(ibin, 0.)
        continue
      dataSidebandYield += dataHist.GetBinContent(ibin)
      if pdfHist:
        pdfSidebandYield += pdfHist.GetBinContent(ibin)
  if pdfHist and pdfSidebandYield > 0.:
    # Rescale so the pdf sideband integral matches the observed sideband yield.
    pdfHist.Scale(dataSidebandYield/pdfSidebandYield)
  if pdfHist:
    pdfHist.SetLineColor(bestcol)
    pdfHist.SetLineWidth(2)
  # Build the (data - pdf) graph that we draw in the ratio pad.
  hdatasub = ROOT.TGraphAsymmErrors()
  diffMax = -1e9
  diffMin = 1e9
  point = 0
  if dataHist and pdfHist:
    for ibin in range(1, dataHist.GetNbinsX()+1):
      x = dataHist.GetBinCenter(ibin)
      if x > blind_low and x < blind_high: continue
      dataVal = dataHist.GetBinContent(ibin)
      dataErr = dataHist.GetBinError(ibin)
      pdfVal = pdfHist.GetBinContent(ibin)
      diff = dataVal - pdfVal
      halfWidth = dataHist.GetBinWidth(ibin)/2.0
      hdatasub.SetPoint(point, x, diff)
      hdatasub.SetPointError(point, halfWidth, halfWidth, dataErr, dataErr)
      diffMax = max(diffMax, diff + dataErr)
      diffMin = min(diffMin, diff - dataErr)
      point += 1
  pad2.cd()
  hdummy = ROOT.TH1D("hdummyweight", "", mgg_high-mgg_low, mgg_low, mgg_high)
  if diffMax <= diffMin:
    diffMax = 1.
    diffMin = -1.
  # Expand the plotting window slightly so residual markers and errors are fully visible.
  residualMax = 1.15 * max(abs(diffMax), abs(diffMin))
  hdummy.SetMaximum(residualMax)
  hdummy.SetMinimum(-residualMax)
  hdummy.GetYaxis().SetTitle("Data - fit")
  # pad2 is 0.35/0.75 as high as pad1.  This normalized size therefore
  # gives its text the same visible size as the default 0.04 in pad1.
  lowerAxisTextSize = 0.04 * 0.75 / 0.35
  hdummy.GetYaxis().SetTitleSize(lowerAxisTextSize)
  hdummy.GetYaxis().SetLabelSize(lowerAxisTextSize)
  hdummy.GetYaxis().SetTitleOffset(0.70)
  hdummy.GetYaxis().CenterTitle(True)
  hdummy.GetXaxis().SetTitle("m_{#gamma#gamma} (GeV)")
  hdummy.GetXaxis().SetTitleSize(lowerAxisTextSize)
  hdummy.GetXaxis().SetLabelSize(lowerAxisTextSize)
  # ROOT scales x tick lengths with the pad height. Compensate for the
  # shorter lower pad so that its ticks have the same visible length.
  hdummy.GetXaxis().SetTickLength(plot.GetXaxis().GetTickLength() * 0.75 / 0.35)
  hdummy.GetYaxis().SetTickLength(plot.GetYaxis().GetTickLength())
  hdummy.Draw("HIST")
  hdummy.GetYaxis().SetNdivisions(808)

  line3 = ROOT.TLine(mgg_low, 0., mgg_high, 0.)
  line3.SetLineColor(bestcol)
  line3.SetLineWidth(2)
  line3.Draw()
  hdatasub.Draw("PESAME")
  # end extra bit for ratio plot###
  canv.SaveAs("%s.pdf"%name)
  canv.SaveAs("%s.png"%name) # keep png twin alongside the pdf
  catIndex.setIndex(currentIndex)
  del hdatasub
  del hdummy
  del line3
  del leg
  del canv

def plotTruths(mass, pdfs, data, name, flashggCats_, cat, bestFitPdf=-1):

  color = [ROOT.kBlue, ROOT.kRed, ROOT.kMagenta, ROOT.kGreen+1, ROOT.kOrange+7, ROOT.kAzure+10, ROOT.kBlack]
  canv = ROOT.TCanvas()
  leg = ROOT.TLegend(0.6, 0.65, 0.88, 0.88)
  leg.SetFillColor(0)
  leg.SetLineColor(0)
  plot = owned(mass.frame())

  mass.setRange("unblindReg_1", mgg_low, 115)
  mass.setRange("unblindReg_2", 135, mgg_high)
  if BLIND:
    data.plotOn(plot, ROOT.RooFit.Binning(mgg_high-mgg_low), ROOT.RooFit.CutRange("unblindReg_1"))
    data.plotOn(plot, ROOT.RooFit.Binning(mgg_high-mgg_low), ROOT.RooFit.CutRange("unblindReg_2"))
    data.plotOn(plot, ROOT.RooFit.Binning(mgg_high-mgg_low), ROOT.RooFit.Invisible())
  else: data.plotOn(plot, ROOT.RooFit.Binning(mgg_high-mgg_low), ROOT.RooFit.Range(MASS_FIT_RANGE))

  datLeg = plot.getObject(int(plot.numItems()-1))
  if len(flashggCats_) > 0:
    leg.AddEntry(datLeg, "Data - %s"%flashggCats_[cat], "LEP")
  else:
    leg.AddEntry(datLeg, "Data - %d"%cat, "LEP")
  i = 0
  style = 1
  # Same ordering as the std::map in the C++
  for pdfName in sorted(pdfs.keys()):
    if i <= 6: col = color[i]
    else:
      col = ROOT.kBlack
      style += 1
    pdfs[pdfName].plotOn(plot, ROOT.RooFit.Range(float(mgg_low), float(mgg_high)), ROOT.RooFit.NormRange(MASS_FIT_RANGE), ROOT.RooFit.Normalization(data.sumEntries(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineColor(col), ROOT.RooFit.LineStyle(style))
    pdfLeg = plot.getObject(int(plot.numItems()-1))
    ext = " (Best Fit Pdf) " if bestFitPdf == i else ""
    leg.AddEntry(pdfLeg, "%s%s"%(pdfName, ext), "L")
    i += 1
  plot.SetTitle(" %s"%flashggCats_[cat])
  if BLIND: plot.SetMinimum(0.0001)
  plot.Draw()
  leg.Draw("same")
  CMS_lumi(canv, 2022, 0)
  canv.SaveAs("%s.pdf"%name)
  canv.SaveAs("%s.png"%name) # keep png twin alongside the pdf
  del canv

def transferMacros(inFile, outFile):
  for key in inFile.GetListOfKeys():
    if key.ReadObj().ClassName() == "TMacro":
      macro = inFile.Get(key.GetName())
      outFile.cd()
      macro.Write()

def getBestFitFunction(bkg, data, cat, silent=False):

  global_minNll = 1E10
  best_index = 0
  number_of_indeces = cat.numTypes()

  snap = ROOT.RooArgSet()
  clean = ROOT.RooArgSet()
  params = bkg.getParameters(ROOT.RooArgSet())
  params.remove(cat)
  params.snapshot(snap)
  params.snapshot(clean)

  for _id in range(number_of_indeces):
    params.assignValueOnly(clean)
    cat.setIndex(_id)

    minNll, fitStatus = runFit(bkg.getCurrentPdf(), data, 3)
    # Add the penalty
    minNll = minNll + bkg.getCorrection()

    if not silent:
      print("[INFO] AFTER FITTING")
      print("[INFO] Function was %s"%bkg.getCurrentPdf().GetName())
      print("[INFO] Correction Applied is %g"%bkg.getCorrection())
      print("[INFO] NLL + c = %g"%minNll)
      print("-----------------------")

    if minNll < global_minNll:
      global_minNll = minNll
      snap.assignValueOnly(params)
      best_index = _id

  cat.setIndex(best_index)
  params.assignValueOnly(snap)

  if not silent:
    print("[INFO] Best fit Function -- %s %d"%(bkg.getCurrentPdf().GetName(), cat.getIndex()))
  return best_index

def main():
  global BLIND, runFtestCheckWithToys
  (opt, args) = get_options()

  ROOT.gROOT.SetBatch(True)
  setTDRStyle()
  cmsStyle.writeExtraText = True       # if extra text
  cmsStyle.lumi_8TeV = "19.1 fb^{-1}"  # default is "19.7 fb^{-1}"
  cmsStyle.lumi_7TeV = "4.9 fb^{-1}"   # default is "5.1 fb^{-1}"
  cmsStyle.lumi_sqrtS = "13 TeV"       # used with iPeriod = 0, e.g. for simulation-only plots (default is an empty string)
  # Draw one compact, italic label instead of separate "CMS" and extra text.
  cmsStyle.cmsText = "%s (CMS data/simulation)"%opt.cmsLabel
  cmsStyle.cmsTextFont = cmsStyle.extraTextFont
  cmsStyle.cmsTextSize = cmsStyle.toFloat(cmsStyle.cmsTextSize*cmsStyle.extraOverCmsTextSize)
  cmsStyle.writeExtraText = False
  cmsStyle.lumi_13p6TeV = "%.0f fb^{-1}"%opt.intLumi
  if opt.unblind: BLIND = False
  saveMultiPdf = opt.saveMultiPdf is not None
  verbose = opt.verbose
  if opt.runFtestCheckWithToys: runFtestCheckWithToys = True
  isData_ = bool(opt.isData)
  year_ = opt.year
  catOffset = opt.catOffset
  outDir = opt.outDir

  if not verbose:
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    ROOT.RooMsgService.instance().setSilentMode(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
  flashggCats_ = opt.flashggCats.split(",")
  ncats = len(flashggCats_)

  if verbose: print("[INFO] SaveMultiPdf? %d"%saveMultiPdf)

  if saveMultiPdf:
    outputfile = ROOT.TFile(opt.saveMultiPdf, "RECREATE")
    outputws = ROOT.RooWorkspace()
    outputws.SetName("multipdf")
    wsImport = getattr(outputws, "import")

  os.makedirs(outDir, exist_ok=True)
  inFile = ROOT.TFile.Open(opt.fileName)
  if isData_:
    inWS = inFile.Get("tagsDumper/cms_hgg_13TeV")
  else:
    inWS = inFile.Get("cms_hgg_workspace")
  if verbose: print("[INFO]  inWS open %s"%inWS.GetName())
  if saveMultiPdf:
    transferMacros(inFile, outputfile)

    intL = intLumi_
    sqrts = inWS.var("SqrtS")
    if not sqrts: sqrts = ROOT.RooRealVar("SqrtS", "SqrtS", 13)
    wsImport(intL)
    wsImport(sqrts)
    print("[INFO] got intL and sqrts %s, %s"%(intL.GetName(), sqrts.GetName()))

  functionClasses = ["Bernstein", "Exponential", "PowerLaw", "Laurent"]
  namingMap = {"Bernstein":"pol", "Exponential":"exp", "PowerLaw":"pow", "Laurent":"lau"}

  # store results here
  resFile = open("%s/fTestResults.txt"%outDir, "w")
  choices_vec = []
  choices_envelope_vec = []
  pdfs_vec = []

  pdfsModel = PdfModelBuilder()
  mass = inWS.var("CMS_hgg_mass")
  mass.setRange("low", mgg_low, blind_low)
  mass.setRange("high", blind_high, mgg_high)
  mass.setRange("fullRange", mgg_low, mgg_high)
  print("[INFO] Got mass from ws %s"%mass.GetName())
  pdfsModel.setObsVar(mass)
  upperEnvThreshold = 0.1 # upper threshold on delta(chi2) to include function in envelope (looser than truth function)

  resFile.write("Truth Model & d.o.f & $\\Delta NLL_{N+1}$ & $p(\\chi^{2}>\\chi^{2}_{(N\\rightarrow N+1)})$ \\\\\n")
  resFile.write("\\hline\n")

  ext = "13TeV" if year_ == "all" else "%s_13TeV"%year_
  for cat in range(ncats):

    choices = {}
    choices_envelope = {}
    pdfs = {}
    allPdfs = {}
    catname = flashggCats_[cat]
    if isData_:
      dataFull = inWS.data("Data_13TeV_%s"%catname)
      if verbose: print("[INFO] opened data for  Data_%s - %s"%(catname, dataFull.GetName()))
    else:
      dataFull = inWS.data("data_mass_%s"%catname)
      if verbose: print("[INFO] opened data for  data_mass_%s - %s"%(catname, dataFull.GetName()))

    mass.setBins(nBinsForMass)
    thisdataBinned_name = "roohist_data_mass_%s"%flashggCats_[cat]
    # Build the sideband-only dataset used for fitting, keeping both total and sideband yields.
    dataFullSideband = dataFull.reduce(ROOT.RooFit.Cut("CMS_hgg_mass<%d || CMS_hgg_mass>%d"%(blind_low, blind_high)))
    if dataFullSideband: owned(dataFullSideband)
    sidebandData = dataFullSideband if dataFullSideband else dataFull
    data = ROOT.RooDataHist(thisdataBinned_name, "data", ROOT.RooArgSet(mass), sidebandData)
    fullDataEntries = dataFull.sumEntries()
    sidebandYield = data.sumEntries()

    storedPdfs = ROOT.RooArgList("store")

    resFile.write("\\multicolumn{4}{|c|}{\\textbf{Category %d}} \\\\\n"%cat)
    resFile.write("\\hline\n")

    MinimimNLLSoFar = 1e10
    simplebestFitPdfIndex = 0

    # Standard F-Test to find the truth functions
    for funcType in functionClasses:

      thisNll = 0.; prevNll = 0.; chi2 = 0.; prob = 0.
      order = 1; prev_order = 0; cache_order = 0

      prev_pdf = None
      cache_pdf = None
      pdforders = []

      counter = 0
      while prob < 0.05 and order < 7:
        bkgPdf = getPdf(pdfsModel, funcType, order, "ftest_pdf_%d_%s"%((cat+catOffset), ext))
        if bkgPdf is None:
          # assume this order is not allowed
          order += 1
        else:

          bkgPdf.Print()
          thisNll, fitStatus = runFit(bkgPdf, data, 3)
          if fitStatus != 0: print("[WARNING] Warning -- Fit status for %s at %d"%(bkgPdf.GetName(), fitStatus))

          chi2 = 2.*(prevNll-thisNll)
          if chi2 < 0. and order > 1: chi2 = 0.
          if prev_pdf is not None:
            prob = getProbabilityFtest(chi2, order-prev_order, prev_pdf, bkgPdf, mass, data, "%s/Ftest_from_%s%d_cat%d.pdf"%(outDir, funcType, order, (cat+catOffset)))
            print("[INFO]  F-test Prob(chi2>chi2(data)) == %g"%prob)
          else:
            prob = 0
          gofProb = plot(mass, bkgPdf, data, "%s/%s%d_cat%d.pdf"%(outDir, funcType, order, (cat+catOffset)), flashggCats_, fitStatus)
          print("[INFO]\t %s %d %g %g %g %g"%(funcType, order, prevNll, thisNll, chi2, prob))
          prevNll = thisNll
          cache_order = prev_order
          cache_pdf = prev_pdf
          prev_order = order
          prev_pdf = bkgPdf
          order += 1
        counter += 1

      resFile.write("%15s & %d & %5.2f & %5.2f \\\\\n"%(funcType, cache_order+1, chi2, prob))
      choices.setdefault(funcType, cache_order)
      pdfs.setdefault("%s%d"%(funcType, cache_order), cache_pdf)

      truthOrder = cache_order

      # Now run loop to determine functions inside envelope
      if saveMultiPdf:
        chi2 = 0.
        thisNll = 0.
        prevNll = 0.
        prob = 0.
        order = 1
        prev_order = 0
        cache_order = 0
        print("[INFO] Determining Envelope Functions for Family %s, cat %d"%(funcType, cat))
        print("[INFO] Upper end Threshold for highest order function %g"%upperEnvThreshold)

        while prob < upperEnvThreshold:
          bkgPdf = getPdf(pdfsModel, funcType, order, "env_pdf_%d_%s"%((cat+catOffset), ext))
          if bkgPdf is None:
            # assume this order is not allowed
            if order > 6:
              print(" [WARNING] could not add ] ")
              break
            order += 1
          else:
            thisNll, fitStatus = runFit(bkgPdf, data, 3)
            if fitStatus != 0: print("[WARNING] Warning -- Fit status for %s at %d"%(bkgPdf.GetName(), fitStatus))
            myNll = 2.*thisNll
            chi2 = 2.*(prevNll-thisNll)
            if chi2 < 0. and order > 1: chi2 = 0.
            prob = ROOT.TMath.Prob(chi2, order-prev_order)

            print("[INFO] \t %s %d %g %g %g %g"%(funcType, order, prevNll, thisNll, chi2, prob))
            prevNll = thisNll
            cache_order = prev_order
            cache_pdf = prev_pdf

            # Calculate goodness of fit for the thing to be included (will use toys for lowstats)!
            # Also produce the GOF figure (computed in the sidebands) for this candidate.
            gofProb = plot(mass, bkgPdf, data, "%s/%s%d_cat%d.pdf"%(outDir, funcType, order, (cat+catOffset)), flashggCats_, fitStatus)

            if prob < upperEnvThreshold: # Looser requirements for the envelope

              if gofProb > 0.01 or order == truthOrder: # Good looking fit or one of our regular truth functions

                print("[INFO] Adding to Envelope %s %g 2xNLL + c is %g"%(bkgPdf.GetName(), gofProb, myNll + bkgPdf.getVariables().getSize()))
                allPdfs.setdefault("%s%d"%(funcType, order), bkgPdf)
                storedPdfs.add(bkgPdf)
                pdforders.append(order)

                # Keep track but we shall redo this later
                if (myNll + bkgPdf.getVariables().getSize()) < MinimimNLLSoFar:
                  simplebestFitPdfIndex = storedPdfs.getSize()-1
                  MinimimNLLSoFar = myNll + bkgPdf.getVariables().getSize()

            prev_order = order
            prev_pdf = bkgPdf
            order += 1

        resFile.write("%15s & %d & %5.2f & %5.2f \\\\\n"%(funcType, cache_order+1, chi2, prob))
        choices_envelope.setdefault(funcType, pdforders)

    resFile.write("\\hline\n")
    choices_vec.append(choices)
    choices_envelope_vec.append(choices_envelope)
    pdfs_vec.append(pdfs)

    plotTruths(mass, pdfs, data, "%s/truths_cat%d"%(outDir, (cat+catOffset)), flashggCats_, cat)

    if saveMultiPdf:

      # Put selectedModels into a MultiPdf
      catindexname = "pdfindex_%s_%s"%(flashggCats_[cat], ext)
      catname = flashggCats_[cat]
      catIndex = ROOT.RooCategory(catindexname, "c")
      pdf = ROOT.RooMultiPdf("CMS_hgg_%s_%s_bkgshape"%(catname, ext), "all pdfs", catIndex, storedPdfs)

      # double check the best pdf!
      bestFitPdfIndex = getBestFitFunction(pdf, data, catIndex, not verbose)
      catIndex.setIndex(bestFitPdfIndex)
      bestPdfForNorm = storedPdfs.at(bestFitPdfIndex)
      scaledYield = computeSidebandScaledYield(bestPdfForNorm, mass, sidebandYield)
      if scaledYield <= 0.:
        raise RuntimeError("Error: computed scaled yield is non-positive.")

      upperBound = max(3.0 * scaledYield, scaledYield + 10.0) # generous roof for the background yield
      nBackground = ROOT.RooRealVar("CMS_hgg_%s_%s_bkgshape_norm"%(catname, ext), "nbkg", scaledYield, 0, upperBound)
      print("// ------------------------------------------------------------------------- //")
      print("[INFO] Created MultiPdf %s, in Category %d with a total of %d pdfs"%(pdf.GetName(), cat, catIndex.numTypes()))
      storedPdfs.Print()
      print("[INFO] Best Fit Pdf = %d, %s"%(bestFitPdfIndex, storedPdfs.at(bestFitPdfIndex).GetName()))
      print("// ------------------------------------------------------------------------- //")
      print("[INFO] Simple check of index %d"%simplebestFitPdfIndex)

      mass.setBins(nBinsForMass)
      dataBinned = ROOT.RooDataHist("roohist_data_mass_%s"%catname, "data", ROOT.RooArgSet(mass), dataFull)

      # Save it (also a binned version of the dataset
      wsImport(pdf)
      wsImport(nBackground)
      wsImport(catIndex)
      wsImport(dataBinned)
      wsImport(data)
      plotMultiPdf(mass, pdf, catIndex, data, "%s/multipdf_%s"%(outDir, catname), flashggCats_, cat, bestFitPdfIndex)

  if saveMultiPdf:
    outputfile.cd()
    outputws.Write()
    outputfile.Close()

  resFile.close()

  dfile = open(opt.datfile, "w")
  print("[RESULT] Recommended options")

  for cat in range(ncats):
    print("Cat %d"%cat)
    dfile.write("cat=%d\n"%(cat+catOffset))
    for funcType, order in sorted(choices_vec[cat].items()):
      print("\t%s - %d"%(funcType, order))
      dfile.write("truth=%s:%d:%s%d\n"%(funcType, order, namingMap[funcType], order))
    for funcType, ords in sorted(choices_envelope_vec[cat].items()):
      for order in ords:
        dfile.write("paul=%s:%d:%s%d\n"%(funcType, order, namingMap[funcType], order))
    dfile.write("\n")
  dfile.close()
  inFile.Close()

if __name__ == "__main__":
  main()
