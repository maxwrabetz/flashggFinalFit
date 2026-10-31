"""FinalFits as a python package.

The analysis steps are rewritten here one at a time, following
https://finalfits.readthedocs.io/en/latest/dev.html. Steps that are not rewritten yet still live
in their own directories (Signal/, Background/, Datacard/, ...) and are driven by the law tasks.

Rewritten so far:

* :mod:`finalfits.trees2ws`: HiggsDNA trees to RooWorkspaces (signal MC and data)
* :mod:`finalfits.background`: fTest and RooMultiPdf envelope (pdf families in :mod:`finalfits.pdfs`,
  plots in :mod:`finalfits.background_plots`)
* :mod:`finalfits.signal_ftest`: number of Gaussians of the signal models (model and fit in
  :mod:`finalfits.signal_model`, plots in :mod:`finalfits.signal_plots`)
* :mod:`finalfits.photon_systematics`: photon scale/smearing systematics of the signal shapes
* :mod:`finalfits.signal_fit`: signal models interpolated in MH; :mod:`finalfits.signal_package`
* :mod:`finalfits.datacard_yields` and :mod:`finalfits.datacard`: yields, systematics and datacard
* :mod:`finalfits.combine`: the combine commands of the fits, scans, impacts, Hesse, toys and p-value
* plots of the results: :mod:`finalfits.spectra` (differential spectra), :mod:`finalfits.splusb_plots` (S+B model
  plots), :mod:`finalfits.correlation_plots` (correlation/covariance matrices), :mod:`finalfits.impacts` (corrected
  impacts json)
"""
