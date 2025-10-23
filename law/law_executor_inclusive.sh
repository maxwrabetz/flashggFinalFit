YEAR="$1"

law run Background --year "$YEAR" --batch-flavor local --version v1 --workflow local --workers 36
law run SignalPackaging --year "$YEAR" --batch-flavor local --workers 36
law run MakeYields --year "$YEAR" --batch-flavor local --workers 36
law run MakeDatacard --year "$YEAR" --workers 36 --batch-flavor local --version v1 --workflow local
law run RunText2Workspace --year "$YEAR" --batch-flavor local --version v1 --workflow local --workers 36
law run CreateAsimovFit --year "$YEAR" --batch-flavor local --version v1 --workflow local --workers 36
# law run AsimovImpactFirstStep --year "$YEAR" --batch-flavor local --version v1 --workflow local --workers 36

# sleep 10
# rm -rf "/net/data_cms3a-1/wrabetz/CMSSW_14_1_0_pre4/src/flashggFinalFit/output_${YEAR}_inclusive/Combine/runFits_mu_fiducial/impact/higgsCombine_initialFit_Test.MultiDimFit.mH125.38.root"

# law run AsimovImpactFirstStep --year "$YEAR" --batch-flavor local --version v1 --workflow local --workers 36
# law run AsimovImpactSecondStep --year "$YEAR" --batch-flavor local --version v1 --workflow local --workers 36
# law run AsimovImpactThirdStep --year "$YEAR" --batch-flavor local --version v1 --workflow local --workers 36
law run MggDistribution --year "$YEAR" --batch-flavor local --version v1 --workflow local --workers 36