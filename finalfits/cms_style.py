"""CMS plot style: python port of tdrStyle/tdrstyle.C and tdrStyle/CMS_lumi.C.

Module level variables play the role of the global variables in CMS_lumi.h.
"""

import ctypes

import ROOT


# The C++ globals are single precision floats: round to float wherever the C++ does float arithmetic
def toFloat(x):
    return ctypes.c_float(x).value


cmsText = "CMS"
cmsTextFont = toFloat(61)  # default is helvetic-bold

writeExtraText = False
extraText = "Simulation Preliminary"
extraTextFont = toFloat(52)  # default is helvetica-italics

# text sizes and text offsets with respect to the top frame
# in unit of the top margin size
lumiTextSize = toFloat(0.65)
lumiTextOffset = toFloat(0.35)  # Increase to push lumi and sqrts string more to the top
cmsTextSize = toFloat(0.8)
cmsTextOffset = toFloat(0.18)  # only used in outOfFrame version: Make larger to push CMS Label more to the top

relPosX = toFloat(0.045)
relPosY = toFloat(0.045)
relExtraDY = toFloat(1.2)

# ratio of "CMS" and extra text size
extraOverCmsTextSize = toFloat(0.76)

lumi_13p6TeV = "61.9 fb^{-1}"
lumi_13TeV = "20.1 fb^{-1}"
lumi_8TeV = "19.7 fb^{-1}"
lumi_7TeV = "5.1 fb^{-1}"
lumi_sqrtS = ""

drawLogo = False


def setTDRStyle():
    tdrStyle = ROOT.TStyle("tdrStyle", "Style for P-TDR")
    ROOT.SetOwnership(tdrStyle, False)

    # For the canvas:
    tdrStyle.SetCanvasBorderMode(0)
    tdrStyle.SetCanvasColor(ROOT.kWhite)
    tdrStyle.SetCanvasDefH(600)
    tdrStyle.SetCanvasDefW(600)
    tdrStyle.SetCanvasDefX(0)
    tdrStyle.SetCanvasDefY(0)

    # For the Pad:
    tdrStyle.SetPadBorderMode(0)
    tdrStyle.SetPadColor(ROOT.kWhite)
    tdrStyle.SetPadGridX(False)
    tdrStyle.SetPadGridY(False)
    tdrStyle.SetGridColor(0)
    tdrStyle.SetGridStyle(3)
    tdrStyle.SetGridWidth(1)

    # For the frame:
    tdrStyle.SetFrameBorderMode(0)
    tdrStyle.SetFrameBorderSize(1)
    tdrStyle.SetFrameFillColor(0)
    tdrStyle.SetFrameFillStyle(0)
    tdrStyle.SetFrameLineColor(1)
    tdrStyle.SetFrameLineStyle(1)
    tdrStyle.SetFrameLineWidth(1)

    # For the histo:
    tdrStyle.SetHistLineColor(1)
    tdrStyle.SetHistLineStyle(0)
    tdrStyle.SetHistLineWidth(1)

    tdrStyle.SetEndErrorSize(2)
    tdrStyle.SetMarkerStyle(20)

    # For the fit/function:
    tdrStyle.SetOptFit(1)
    tdrStyle.SetFitFormat("5.4g")
    tdrStyle.SetFuncColor(2)
    tdrStyle.SetFuncStyle(1)
    tdrStyle.SetFuncWidth(1)

    # For the date:
    tdrStyle.SetOptDate(0)

    # For the statistics box:
    tdrStyle.SetOptFile(0)
    tdrStyle.SetOptStat(0)
    tdrStyle.SetStatColor(ROOT.kWhite)
    tdrStyle.SetStatFont(42)
    tdrStyle.SetStatFontSize(0.025)
    tdrStyle.SetStatTextColor(1)
    tdrStyle.SetStatFormat("6.4g")
    tdrStyle.SetStatBorderSize(0)
    tdrStyle.SetStatH(0.1)
    tdrStyle.SetStatW(0.15)

    # Margins:
    tdrStyle.SetPadTopMargin(0.08)
    tdrStyle.SetPadBottomMargin(0.13)
    tdrStyle.SetPadLeftMargin(0.12)
    tdrStyle.SetPadRightMargin(0.03)

    # For the Global title:
    tdrStyle.SetOptTitle(0)
    tdrStyle.SetTitleFont(42)
    tdrStyle.SetTitleColor(1)
    tdrStyle.SetTitleTextColor(1)
    tdrStyle.SetTitleFillColor(10)
    tdrStyle.SetTitleFontSize(0.05)

    # For the axis titles:
    tdrStyle.SetTitleColor(1, "XYZ")
    tdrStyle.SetTitleFont(42, "XYZ")
    tdrStyle.SetTitleSize(0.04, "XYZ")
    tdrStyle.SetTitleXOffset(1.1)
    tdrStyle.SetTitleYOffset(1.4)

    # For the axis labels:
    tdrStyle.SetLabelColor(1, "XYZ")
    tdrStyle.SetLabelFont(42, "XYZ")
    tdrStyle.SetLabelOffset(0.01, "XYZ")
    tdrStyle.SetLabelSize(0.04, "XYZ")

    # For the axis:
    tdrStyle.SetAxisColor(1, "XYZ")
    tdrStyle.SetStripDecimals(True)
    tdrStyle.SetTickLength(0.03, "XYZ")
    tdrStyle.SetNdivisions(510, "XYZ")

    # Change for log plots:
    tdrStyle.SetOptLogx(0)
    tdrStyle.SetOptLogy(0)
    tdrStyle.SetOptLogz(0)

    # Postscript options:
    tdrStyle.SetPaperSize(20.0, 20.0)

    tdrStyle.SetHatchesLineWidth(5)
    tdrStyle.SetHatchesSpacing(0.05)

    tdrStyle.cd()
    return tdrStyle


def CMS_lumi(pad, iPeriod=3, iPosX=10, extraExtraText=""):
    f = toFloat

    outOfFrame = iPosX // 10 == 0
    alignY_ = 3
    alignX_ = 2
    if iPosX // 10 == 0:
        alignX_ = 1
    if iPosX == 0:
        alignX_ = 1
    if iPosX == 0:
        alignY_ = 1
    if iPosX // 10 == 1:
        alignX_ = 1
    if iPosX // 10 == 2:
        alignX_ = 2
    if iPosX // 10 == 3:
        alignX_ = 3
    align_ = 10 * alignX_ + alignY_

    H = f(pad.GetWh())
    W = f(pad.GetWw())
    l = pad.GetLeftMargin()  # noqa: E741 (names as in CMS_lumi.C)
    t = pad.GetTopMargin()
    r = pad.GetRightMargin()
    b = pad.GetBottomMargin()

    pad.cd()

    lumiText = ""
    if iPeriod == 1:
        lumiText += lumi_7TeV + " (7 TeV)"
    elif iPeriod == 2:
        lumiText += lumi_8TeV + " (8 TeV)"
    elif iPeriod == 3:
        lumiText = lumi_8TeV + " (8 TeV)" + " + " + lumi_7TeV + " (7 TeV)"
    elif iPeriod == 4:
        lumiText += lumi_13TeV + " (13 TeV)"
    elif iPeriod == 7:
        if outOfFrame:
            lumiText += "#scale[0.85]{"
        lumiText += lumi_13TeV + " (13 TeV)" + " + " + lumi_8TeV + " (8 TeV)" + " + " + lumi_7TeV + " (7 TeV)"
        if outOfFrame:
            lumiText += "}"
    elif iPeriod == 12:
        lumiText += "8 TeV"
    elif iPeriod == 2022:
        lumiText += lumi_13p6TeV + " (13.6 TeV)"
    elif iPeriod == 0:
        lumiText += lumi_sqrtS

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextAngle(0)
    latex.SetTextColor(ROOT.kBlack)

    extraTextSize = f(extraOverCmsTextSize * cmsTextSize)
    topY = f(f(1 - t) + f(lumiTextOffset * t))

    latex.SetTextFont(42)
    latex.SetTextAlign(31)
    latex.SetTextSize(0.9 * lumiTextSize * t)
    latex.DrawLatex(f(1 - r), topY, lumiText)

    if outOfFrame:
        latex.SetTextFont(int(cmsTextFont))
        latex.SetTextAlign(11)
        latex.SetTextSize(0.9 * cmsTextSize * t)
        latex.DrawLatex(l, topY, cmsText)

    pad.cd()

    posX_ = 0.0
    if iPosX % 10 <= 1:
        posX_ = f(l + f(relPosX * f(f(1 - l) - r)))
    elif iPosX % 10 == 2:
        posX_ = f(l + 0.5 * f(f(1 - l) - r))
    elif iPosX % 10 == 3:
        posX_ = f(f(1 - r) - f(relPosX * f(f(1 - l) - r)))
    posY_ = f(f(1 - t) - f(relPosY * f(f(1 - t) - b)))
    if not outOfFrame:
        if drawLogo:
            posX_ = f(l + 0.045 * f(f(1 - l) - r) * W / H)
            posY_ = f(f(1 - t) - 0.045 * f(f(1 - t) - b))
            xl_0 = posX_
            yl_0 = f(posY_ - 0.15)
            xl_1 = f(posX_ + 0.15 * H / W)
            yl_1 = posY_
            pad_logo = ROOT.TPad("logo", "logo", xl_0, yl_0, xl_1, yl_1)
            pad_logo.Draw()
            pad_logo.cd()
            pad_logo.Modified()
            pad.cd()
        else:
            latex.SetTextFont(int(cmsTextFont))
            latex.SetTextSize(0.9 * cmsTextSize * t)
            latex.SetTextAlign(align_)
            latex.DrawLatex(posX_, posY_, cmsText)
            if writeExtraText:
                latex.SetTextFont(int(extraTextFont))
                latex.SetTextAlign(align_)
                latex.SetTextSize(0.9 * extraTextSize * t)
                latex.DrawLatex(posX_, posY_ - f(f(relExtraDY * cmsTextSize) * t), extraText + " " + extraExtraText)
    elif writeExtraText:
        if iPosX == 0:
            posX_ = f(l + f(relPosX * f(f(1 - l) - r)))
            posY_ = topY
        latex.SetTextFont(int(extraTextFont))
        latex.SetTextSize(0.9 * extraTextSize * t)
        latex.SetTextAlign(align_)
        latex.DrawLatex(posX_ + 0.1, posY_, extraText + " " + extraExtraText)
