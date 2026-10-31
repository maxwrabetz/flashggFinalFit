# Plotting

Apologies for the lack of documentation for all the different plotting scripts!

## Final signal fit

## Rewrite

The plots used by the law workflow are now part of the `finalfits` package; the scripts here keep their paths and
options (the law tasks in `Combine/` call them) and only run the new entry points in `scripts/`:

| Plot | Module | Script | Called through |
|---|---|---|---|
| Differential spectra | `finalfits/spectra.py` | `scripts/plot_spectra.py` | law tasks `CreateDiffSpectra`, `CreateTotalDiffSpectra` (`Spectra/law_spectra.py`) |
| S+B model plots | `finalfits/splusb_plots.py` | `scripts/plot_splusb.py` | `makeSplusBModelPlot.py` |
| Correlation/covariance matrix | `finalfits/correlation_plots.py` | `scripts/plot_corr_matrix.py` | `makeCorrMatrix.py` |
| Corrected impacts json | `finalfits/impacts.py` | `scripts/correct_impacts.py` | `correctImpacts.py` |

```
./scripts/plot_spectra.py --year 2022_2023_2024 --variable PTH [--total] [--unblinded] [--log-x] [--output-dir <dir>]
./scripts/plot_splusb.py --input-ws-file Datacard.root --cats cat0,cat1,cat2 --do-sum-categories --do-cat-weights --ext _r
./scripts/plot_corr_matrix.py --input robustHessefirstStep.root --mode NJ --input-json inputs_robustHesse.json --output .
./scripts/correct_impacts.py --impacts-json impacts.json --frozen-param MH --drop-bkg-model-params
```
The new scripts accept the old option names too (e.g. `--inputWSFile`, `--doCov`). The spectra keep reading the
`spectra` block of `config/<year>_<variable>.yml`, the layout from `Spectra/config/<year>_<variable>.yml` and the theory
inputs from `Spectra/fidXS/`. The outputs are identical to the previous scripts (validated on the 2023 NJ and other
real inputs, and on synthetic scans for every spectra config), except for the fixes: the theory uncertainty boxes of
MadGraph + NNLOPS in the ratio panel of the spectra had up and down swapped; the corrected impacts json is valid json
also with non-empty `groups`; `makeCorrMatrix.py` no longer fails without `--translate` or for a POI without translation
(the name is shown); `makeSplusBModelPlot.py` fails with an error (instead of exit code 0) on invalid HH-fix binnings or
categories missing in `--loadWeights`, and the categories in `--problematicCats` are left out everywhere (also of the weights and toy bands, which
made it fail together with `--doCatWeights`). All differences are listed in [DIFFERENCES.md](../DIFFERENCES.md). The other scripts in this directory are not used by the workflow and unchanged.
