YEAR="$1"
VAR="$2"

VERSION="${VAR}_v1"

law run Background --year "$YEAR" --variable "$VAR" --batch-flavor htcondor --version "$VERSION" --workflow htcondor --workers 36
law run SignalPackaging --year "$YEAR" --variable "$VAR" --batch-flavor htcondor --workers 36
law run MakeYields --year "$YEAR" --variable "$VAR" --batch-flavor htcondor --workers 36
law run MakeDatacard --year "$YEAR" --variable "$VAR" --workers 36 --batch-flavor htcondor --version "$VERSION" --workflow htcondor
law run RunText2Workspace --year "$YEAR" --variable "$VAR" --batch-flavor htcondor --version "$VERSION" --workflow htcondor --workers 36
law run CreateDiffSpectra --year "$YEAR" --variable "$VAR" --batch-flavor htcondor --workers 36
law run CreateAsimovFit --year "$YEAR" --variable "$VAR" --batch-flavor htcondor --version "$VERSION" --workflow htcondor --workers 36
# law run AsimovImpactFirstStep --year "$YEAR" --variable "$VAR" --batch-flavor htcondor --version "$VERSION" --workflow htcondor --workers 36

# sleep 10
# rm -rf "/net/data_cms3a-1/wrabetz/CMSSW_14_1_0_pre4/src/flashggFinalFit/output_${YEAR}_${VAR}/Combine/runFits_${VAR}/impact/higgsCombine_initialFit_Test.MultiDimFit.mH125.38.root"

# law run AsimovImpactFirstStep --year "$YEAR" --variable "$VAR" --batch-flavor htcondor --version "$VERSION" --workflow htcondor --workers 36
# law run AsimovImpactSecondStep --year "$YEAR" --variable "$VAR" --batch-flavor htcondor --version "$VERSION" --workflow htcondor --workers 36
# law run AsimovImpactThirdStep --year "$YEAR" --variable "$VAR" --batch-flavor htcondor --version "$VERSION" --workflow htcondor --workers 36
# law run AsimovCovCorrHesse --year "$YEAR" --variable "$VAR" --batch-flavor htcondor --version "$VERSION" --workflow htcondor --workers 36
# law run AsimovCovCorr --year "$YEAR" --variable "$VAR" --batch-flavor htcondor --version "$VERSION" --workflow htcondor --workers 36