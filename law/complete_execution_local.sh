# law run Background --year 2223 --variable PTH --batch-flavor htcondor --version v1 --workflow htcondor
# law run SignalPackaging --year 2223 --variable PTH --batch-flavor htcondor
# law run MakeYields --year 2223 --variable PTH --batch-flavor htcondor
# law run MakeDatacard --year 2223 --variable PTH --workers 24 --batch-flavor local --version v1 --workflow local
# law run RunText2Workspace --year 2223 --variable PTH --batch-flavor htcondor --version v1 --workflow htcondor
# law run CreateDiffSpectra --year 2223 --variable PTH --batch-flavor htcondor
# law run CreateAsimovFit --year 2223 --variable PTH --batch-flavor htcondor --version v1 --workflow htcondor
# law run AsimovImpactSecondStep --year 2223 --variable PTH --batch-flavor htcondor --version v1 --workflow htcondor
# law run AsimovImpactThirdStep --year 2223 --variable PTH --batch-flavor htcondor --version v1 --workflow htcondor
# law run AsimovCovCorrHesse --year 2223 --variable PTH --batch-flavor htcondor --version v1 --workflow htcondor
# law run AsimovCovCorr --year 2223 --variable PTH --batch-flavor htcondor --version v1 --workflow htcondor
law run MggDistribution --year 2223 --variable PTH --batch-flavor htcondor --version v1 --workflow htcondor
