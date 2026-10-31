# Final Fits (higgsdnafinalfit)

This is the branch for using final fits with the output of HiggsDNA.

Welcome to the new Final Fits package. Here lies a a series of scripts which are used to run the final stages of CMS Hgg analyses: signal modelling, background modelling, datacard creation, final statistical interpretation and final result plots.

You can follow the latest Final Fits tutorial [here](https://gitlab.cern.ch/jspah/higgsdna_finalfits_tutorial_24/-/tree/master). Slides from an older flashgg-based tutorial series can be found [here](https://indico.cern.ch/event/963619/contributions/4112177/attachments/2151275/3627204/finalfits_tutorial_201126.pdf)

## Download and setup instructions

```
export SCRAM_ARCH=el9_amd64_gcc12
cmsrel CMSSW_14_1_0_pre4
cd CMSSW_14_1_0_pre4/src
cmsenv

COMBINE_TAG=07b56c67ba6e4304b42c3a6cdba710d59c719192
COMBINEHARVESTER_TAG=94017ba5a3a657f7b88669b1a525b19d34ea41a2
FINALFIT_TAG=higgsdnafinalfit

# Install Combine with the latest EL9 compatible branch
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit && git fetch origin ${COMBINE_TAG} && git checkout ${COMBINE_TAG}

# Install CombineTools in CombineHarvester
cd ${CMSSW_BASE}/src
bash <(curl -s https://raw.githubusercontent.com/cms-analysis/CombineHarvester/${COMBINEHARVESTER_TAG}/CombineTools/scripts/sparse-checkout-https.sh)
cd CombineHarvester && git fetch origin ${COMBINEHARVESTER_TAG} && git checkout ${COMBINEHARVESTER_TAG}

# Compile libraries
cd ${CMSSW_BASE}/src
cmsenv
scram b clean
scram b -j 8

# Install Final Fit package and install the background functions (only run once)
git clone -b $FINALFIT_TAG https://github.com/JaLuka98/flashggFinalFit.git
cd flashggFinalFit/Background
make clean; make
# Move to Law folder and source the setup file (to be activated every new shell)
cd ../law
source setup.sh
law index --verbose
```

## Contents
The Finals Fits package contains several subfolders which are used for the following steps:

* Create the Signal Model (see `Signal` dir)
* Create the Background Model (see `Background` dir)
* Generate a Datacard (see `Datacard` dir)
* Running fits with combine (see `Combine` dir)
* Scripts to produce plots (see `Plots` dir)

The signal modelling, background modelling and datacard creation can be ran in parallel. Of course the final fits (`Combine`) requires the output of these three steps. In addition, the scripts in the `Trees2WS` dir are a series of lightweight scripts for converting standard ROOT trees into a RooWorkspace that can be read by the Final Fits package.

Finally, the objects and tools which are common to all subfolders are defined in the `tools` directory. If your input workspaces differ from the flashgg output workspace structure, then you may need to change the options here.

Each of the relevant folders are documented with specific `README.md` files. 

## The `finalfits` python package

The steps are rewritten as a python package, following the [FinalFits developer notes](https://finalfits.readthedocs.io/en/latest/dev.html):

```
finalfits/      # the package: one module per step, plus shared helpers (utils.py, analysis.py)
scripts/        # command line entry points, e.g. scripts/trees2ws.py
tests/          # pytest tests on small generated inputs
pyproject.toml  # package metadata, pytest and ruff configuration
```

Rewritten steps, from the trees to the plots of the results:

| Step | Module | Script | law task |
|---|---|---|---|
| Trees2WS | `trees2ws.py` | `trees2ws.py` | `Trees2WS`, `Trees2WSData` |
| Background fTest | `background.py`, `pdfs.py` | `background.py` | `Background` |
| Signal fTest | `signal_ftest.py`, `signal_model.py` | `signal_ftest.py` | `FTest` |
| Photon systematics | `photon_systematics.py` | `photon_systematics.py` | `CalcPhotonSyst` |
| Signal fit | `signal_fit.py` | `signal_fit.py` | `SignalFit` |
| Signal packaging | `signal_package.py` | `signal_package.py` | `SignalPackaging` |
| Yields and datacard | `datacard_yields.py`, `datacard.py` | `datacard.py yields/make/clean` | `MakeYields`, `MakeDatacard` |
| Combine | `combine.py` | `combine.py text2workspace/nuisances/pvalue` | all tasks of `Combine/law_combine.py` |
| Spectra | `spectra.py` | `plot_spectra.py` | `CreateDiffSpectra`, `CreateTotalDiffSpectra` |
| S+B model plots | `splusb_plots.py` | `plot_splusb.py` | via `Plots/makeSplusBModelPlot.py` |
| Correlation matrices | `correlation_plots.py` | `plot_corr_matrix.py` | via `Plots/makeCorrMatrix.py` |
| Impacts | `impacts.py` | `correct_impacts.py` | via `Plots/correctImpacts.py` |

Each step was validated against the previous scripts on real inputs, mainly 2023 NJ (identical outputs, see the step READMEs); the intended changes are bug fixes, all listed in [DIFFERENCES.md](DIFFERENCES.md), e.g. the Trees2WS fix of the systematic histograms of the differential splitting. The law tasks call the package directly, so nothing changes in how the analysis is run. Scripts the pipeline does not use (e.g. most of `Plots/`) are unchanged, and `finalfits/analysis.py` is the only place where the package uses the shared definitions in `commonTools/`.

In the CMSSW environment (ROOT comes from CMSSW, nothing needs to be installed):
```
cd flashggFinalFit
python3 -m pytest              # run the tests
./scripts/trees2ws.py --help   # all scripts share --verbose/-v and --roofit-verbose
```
For linting and formatting, install `ruff` (e.g. `pip install --user ruff`) and run `ruff check` and `ruff format` in `flashggFinalFit`.