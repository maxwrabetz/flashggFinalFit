law run Background --year 2223 --variable Njets2p5 --batch-flavor local --version v1 --workflow local --workers 48
law run SignalPackaging --year 2223 --variable Njets2p5 --batch-flavor local --workers 48
law run MakeYields --year 2223 --variable Njets2p5 --batch-flavor local --workers 48
law run MakeDatacard --year 2223 --variable Njets2p5 --workers 48 --batch-flavor local --version v1 --workflow local
law run RunText2Workspace --year 2223 --variable Njets2p5 --batch-flavor local --version v1 --workflow local --workers 48
law run CreateDiffSpectra --year 2223 --variable Njets2p5 --batch-flavor local --workers 48
law run CreateAsimovFit --year 2223 --variable Njets2p5 --batch-flavor local --version v1 --workflow local --workers 48
law run AsimovImpactSecondStep --year 2223 --variable Njets2p5 --batch-flavor local --version v1 --workflow local --workers 48
law run AsimovImpactThirdStep --year 2223 --variable Njets2p5 --batch-flavor local --version v1 --workflow local --workers 48
law run AsimovCovCorrHesse --year 2223 --variable Njets2p5 --batch-flavor local --version v1 --workflow local --workers 48
law run AsimovCovCorr --year 2223 --variable Njets2p5 --batch-flavor local --version v1 --workflow local --workers 48
