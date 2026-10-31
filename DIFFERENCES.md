# Differences between the `finalfits` rewrite and the previous scripts

This page lists every change in what the analysis does, step by step. Changes in how the code is written
(package layout, logging, argparse, tests) are not listed.

Each step was run on the same real inputs with the old code (git `HEAD` before the rewrite) and with the new
code, mainly on 2023 NJ (`output_2023_NJ_new_bkg/`), and the outputs were compared bit by bit. For Combine the full
command sequence of every task and branch was compared. Everything not listed here gives identical results.

A detailed explanation in German, with examples and numbers from the 2023 NJ outputs, is in
[UNTERSCHIEDE_ERKLAERT.md](UNTERSCHIEDE_ERKLAERT.md).

Legend: **Fix** changes results, **Error handling** only changes when and how a task fails, **Not supported** is a
feature of the old code that the new code refuses with an error.

## Trees2WS

- **Fix: systematic histograms of the differential splitting.** The old code selected the events of the systematic
  RooDataHists with masks of the nominal tree applied to the systematic trees, whose events are in a different
  order. Most of these histograms were empty and the rest contained the wrong events. The new code assigns the
  events of the systematic trees by the HiggsDNA bin index (`diffVar`) wherever that index reproduces the bin of
  the nominal events. All results that use these histograms change: photon systematics, signal models, datacard.
  The old photon systematics were almost all at the clipping limits (0.05 and 0.5) because of the empty
  histograms.
- **Limitation (input, not code): out-of-fiducial bins.** HiggsDNA does not split the out-of-fiducial events by
  bin and the systematic trees contain no particle-level variable (e.g. `GenNJ`). The systematic histograms of
  these bins stay empty, with a warning. As before, the photon systematics of these processes are then at
  their caps (5 % mean, 50 % width) and their JEC/JER lnN values are `-`; their yields are small (about 0.4 % of the
  in-fiducial process in the same category). Storing the particle-level variable in the HiggsDNA systematic trees fixes
  this without changes here.
- **Error handling:** non-finite values (NaN, inf) in a variable and variables missing from the workspace stop the
  task with an error. Exactly one of `doInOutSplitting` and `doDiffSplitting` must be set.
- **Not supported:** STXS splitting (`doSTXSSplitting`), no config uses it.

## Background

- The chosen function orders are written per category to `bkgfTest-Data/fTest_<cat>.dat`. Before, all categories
  wrote to the same `dat/fTest.dat` and overwrote each other. The envelopes themselves are identical.

## Signal

- fTest, photon systematics, signal fit and packaging give identical results.
- The current value of `CMS_hgg_mass` stored in the signal workspace differs: the old code left the value of the
  last event of its loop. Combine sets the observable itself, so this has no effect on any fit.

## Datacard

- **Fix: cleaning with `--symmetrize` (`symmetrizeNuisance`).**
  - Values outside `[1/factor, factor]` are now removed also when nuisances to symmetrise are given. Before, the
    factor was only applied without a symmetrisation list.
  - One-sided asymmetric values `kappa_down/kappa_up` of the listed nuisances (one side within 0.003 of 1, the
    two sides at least 0.010 apart) become the symmetric kappa of the side that deviates: `kappa_up` if the up
    side deviates, `1/kappa_down` if the down side deviates. The old code had a branch that could never be
    reached and gave, e.g., `1.000` for `1.000/1.050` (the uncertainty was lost) and `0.900` for `0.900/1.001`
    (the uncertainty pointed the wrong way).
  - A warning is printed for nuisances in the list that have no lnN line in the datacard.
  - Asymmetric `lumi` values were written twice per entry by the old cleaner, which broke the line. They are now
    written once. All current datacards have symmetric `lumi` values, so their output is unchanged.
  - For the 2023 NJ datacard nothing changes: the names in `symmetrizeNuisance` of `config/2023_NJ.yml`
    (`CMS_hgg_SigmaEOverEShift`, `CMS_hgg_scale_*_shape`) do not appear in the datacard (it has
    `CMS_HIG26007_SigmaEOverEShift`, two-sided), so the list has no effect. The new warning reports this.
- The order of the years in the `shapes` and `pdfindex` lines is sorted. Before, it was random when there were
  several years. combine does not depend on this order.
- **Not supported:** STXS merging (`mnorm`) and the STXS scale correlation scheme. No config uses them; the task
  stops with an error if they are set.

## Combine

- **Fix: `fitted_pdf_indices`.** The pdf indices of the Asimov first step are read from the line that
  `checkPdfIdx.C` marks. If the first-step file was missing, the old code passed ROOT's `Processing ...` line to
  `--setParameters` as if it were the indices; now the indices are left out. A path containing an uppercase `X`
  also no longer breaks the parsing.
- **Fix: `PrepareTheDirectory`** no longer fails with a `NameError` when the signal and background models are in
  different folders.
- **Fix: correlation matrices get the luminosity of the year.** `AsimovCovCorr` and `UnblindedCovCorr` now pass
  `--year` to `Plots/makeCorrMatrix.py`. Before, every matrix was labelled with the 2022 luminosity
  (34.7 fb^-1). Combined years such as `2022_2023` get the sum of their luminosities.
- **Fix: post-fit toys (`MggToyGeneration --is-postfit True`)** always read `SLURM_JOB_ID` and failed with a
  `KeyError` outside slurm (htcondor, local). The job scratch is now only used on slurm/psi, as for the pre-fit
  toys.
- **Fix: `UnblindedFitSystSingle` on slurm/psi** now creates the output directory and copies its output back like
  all other tasks. Before, it skipped the directory and always copied with `xrdcp`, which fails for outputs on
  `/work`.
- **Error handling: p-value (`PValueCalculation`).** If the fit at the SM point did not produce its output, or
  the output has no fixed point entry, the task stops with a message saying so. Before, it failed with an
  unrelated `TypeError` from rounding `None`, and a failure to write `pvalue.txt` was only printed.
- **Error handling:** listing the nuisances of a workspace fails with different exception types and messages.

## Plots

- **Fix: differential spectra.** In the ratio panel, the up and down theory uncertainties of MadGraph + NNLOPS were
  swapped. This is the only change visible in the spectra.
- **Fix: S+B model plots with `--problematicCats`.**
  - The problematic categories are now left out everywhere: plots, sums, S/(S+B) weights and the toy bands. Before,
    the toy bands of the summed plot still included them, while the summed data did not.
  - Together with `--doCatWeights` the old code always failed with a `KeyError`; the plots are now made.
  - The sum plots no longer fail when the first category is problematic.
- **Fix: corrected impacts json.** `"groups"` is written as valid json; before it was a python repr. The current
  inputs have empty groups, so their output is unchanged.
- **Fix: correlation matrices** without `--translate`, or with a POI that has no translation, show the plain name
  instead of failing.
- **Error handling: S+B model plots** stop with an error (instead of exit code 0) on an invalid HH-fix binning or on
  categories missing in `--loadWeights`.
- **Unchanged, but worth knowing:** the CMS label of the spectra is fixed to
  `Private work (CMS data/simulation)`. The `no_preliminary` and `private_work` keys of the `spectra` config have
  no effect, in the old code as in the new.

## Not validated

These code paths could not be compared for lack of inputs:

- S+B model plots with `--doHHMjjFix` (no HH workspace).
- Spectra of 2D variables (`PTHvsNJ`, no config).
- Spectra and fits on unblinded data, and real combine fits: for Combine only the commands were compared.
- Combine for combined years (e.g. `2022_2023`).
