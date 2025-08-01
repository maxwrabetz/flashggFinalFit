from collections import OrderedDict as od

globalReplacementMap = od()
globalReplacementMap["earlyAnalysisDiffNjets2p5"] = od()

# ---- WV replacement (nur ein Beispiel)
globalReplacementMap["earlyAnalysisDiffNjets2p5"]['procWV'] = "ggh_Njets2p5_1p0_2p0_in"
globalReplacementMap["earlyAnalysisDiffNjets2p5"]['catWV'] = "RECO_Njets2p5_1p0_2p0_cat2"

# ---- Process replacement map (RV)
globalReplacementMap["earlyAnalysisDiffNjets2p5"]['procRVMap'] = od()
for nj in ["0p0_1p0","1p0_2p0","2p0_3p0","3p0_100p0"]:
    for cat in range(3):
        globalReplacementMap["earlyAnalysisDiffNjets2p5"]["procRVMap"][f"RECO_Njets2p5_{nj}_cat{cat}"] = f"ggh_Njets2p5_{nj}_in"

# ---- Category replacement map (gleiche Kategorien wie Key)
globalReplacementMap["earlyAnalysisDiffNjets2p5"]["catRVMap"] = od()
for nj in ["0p0_1p0","1p0_2p0","2p0_3p0","3p0_100p0"]:
    for cat in range(3):
        globalReplacementMap["earlyAnalysisDiffNjets2p5"]["catRVMap"][f"RECO_Njets2p5_{nj}_cat{cat}"] = f"RECO_Njets2p5_{nj}_cat{cat}"
