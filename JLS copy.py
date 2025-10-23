import uproot
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.interpolate import make_interp_spline

# Use CMS style
hep.style.use("CMS")


def process_scan(file_path):
    """Extract POI and -2ΔNLL values from a ROOT scan file."""

    # Open the ROOT file
    file = uproot.open(file_path)

    # Access the tree containing the scan results
    tree = file["limit"]

    # Extract parameter of interest (e.g., r) and likelihood values (-2ΔNLL)
    poi_values = tree["r"].array()
    nll_values = tree["deltaNLL"].array()

    # Convert to numpy arrays
    # Slice away the first entry (best fit), we don’t want an extra line there
    poi_values = np.array(poi_values)[1:]
    nll_values = np.array(nll_values)[1:]

    # Convert to -2ΔNLL
    nll_values = 2 * nll_values

    # Sort by POI
    sorted_indices = np.argsort(poi_values)
    poi_values_sorted = poi_values[sorted_indices]
    nll_values_sorted = nll_values[sorted_indices]

    # Remove duplicates
    unique_poi, unique_indices = np.unique(poi_values_sorted, return_index=True)
    poi_values_sorted = unique_poi
    nll_values_sorted = nll_values_sorted[unique_indices]

    return poi_values_sorted, nll_values_sorted


def get_interval(poi_values, nll_values):
    """Find 1σ interval crossing points."""
    crossing_indices = np.where(np.diff(np.sign(nll_values - 1)))[0]
    crossing_points = []
    for idx in crossing_indices:
        x0, x1 = poi_values[idx], poi_values[idx + 1]
        y0, y1 = nll_values[idx], nll_values[idx + 1]
        cp = x0 + (1 - y0) * (x1 - x0) / (y1 - y0)
        crossing_points.append(cp)
    if len(crossing_points) >= 2:
        return min(crossing_points), max(crossing_points)
    else:
        return None, None


# Process scans
poi_values_statonly, nll_values_statonly = process_scan(
    "higgsCombineAsimovPostFitScanStat_r.root"
)
poi_values_with_syst, nll_values_with_syst = process_scan(
    "higgsCombineAsimovPostFitScanFit_r.root"
)

# Theory input
theory_uncertainty = 3.80
do_xsec = True

if do_xsec:
    theory_value = 67.80
    poi_values_statonly *= theory_value
    poi_values_with_syst *= theory_value
else:
    theory_value = 1.0
    theory_uncertainty = theory_uncertainty / 67.80

# Best-fit values (minimum of curve)
bestfit_idx_syst = np.argmin(nll_values_with_syst)
bestfit_val_syst = poi_values_with_syst[bestfit_idx_syst]

bestfit_idx_stat = np.argmin(nll_values_statonly)
bestfit_val_stat = poi_values_statonly[bestfit_idx_stat]

# 1σ intervals
low_syst, high_syst = get_interval(poi_values_with_syst, nll_values_with_syst)
low_stat, high_stat = get_interval(poi_values_statonly, nll_values_statonly)

err_down_syst = bestfit_val_syst - low_syst if low_syst else None
err_up_syst = high_syst - bestfit_val_syst if high_syst else None

err_down_stat = bestfit_val_stat - low_stat if low_stat else None
err_up_stat = high_stat - bestfit_val_stat if high_stat else None

# --- Smooth spline for stat-only ---
x_stat_smooth = np.linspace(
    poi_values_statonly.min(), poi_values_statonly.max(), 500
)
spline_stat = make_interp_spline(poi_values_statonly, nll_values_statonly, k=3)
y_stat_smooth = spline_stat(x_stat_smooth)

plt.plot(
    x_stat_smooth,
    y_stat_smooth,
    color="blue",
    ls="dashdot",
    label=fr"Expected - Stat. only: {theory_value:.2f}$^{{+{err_up_stat:.2f}}}_{{-{err_down_stat:.2f}}}$ fb",
    #label=fr"Expected - Stat. only: {bestfit_val_stat:.2f}$^{{+{err_up_stat:.2f}}}_{{-{err_down_stat:.2f}}}$ fb",
)

# --- Smooth spline for with-syst ---
x_syst_smooth = np.linspace(
    poi_values_with_syst.min(), poi_values_with_syst.max(), 500
)
spline_syst = make_interp_spline(poi_values_with_syst, nll_values_with_syst, k=3)
y_syst_smooth = spline_syst(x_syst_smooth)

plt.plot(
    x_syst_smooth,
    y_syst_smooth,
    color="black",
    lw=2,
    label=fr"Expected: {theory_value:.2f}$^{{+{err_up_syst:.2f}}}_{{-{err_down_syst:.2f}}}$ fb",
    #label=fr"Expected: {bestfit_val_syst:.2f}$^{{+{err_up_syst:.2f}}}_{{-{err_down_syst:.2f}}}$ fb",
)

# Add the theoretical prediction with uncertainty
plt.plot(
    [theory_value, theory_value],
    [0, 1.8],
    color="red",
    linestyle="-",
    linewidth=3,
    label=fr"MG5_aMC@NLO, NNLOPS: {theory_value:.2f}$^{{+{theory_uncertainty:.2f}}}_{{-{theory_uncertainty:.2f}}}$ fb",
)
plt.fill_betweenx(
    y=[0, 1.8],
    x1=theory_value - theory_uncertainty,
    x2=theory_value + theory_uncertainty,
    color="none",
    edgecolor="red",
    facecolor="none",
    hatch="//",
)

# Grey line at -2ΔlnL = 1
plt.axhline(1.0, color="grey", linestyle="--", linewidth=2)

# Mark crossing points for syst curve
for cp in (low_syst, high_syst):
    if cp:
        plt.plot([cp, cp], [0, 1], color="grey", linestyle="--")

# Debug prints
print("=== Stat only ===")
print(f"Best fit: {bestfit_val_stat:.3f}, -1σ: {low_stat:.3f}, +1σ: {high_stat:.3f}")
print("=== With syst ===")
print(f"Best fit: {bestfit_val_syst:.3f}, -1σ: {low_syst:.3f}, +1σ: {high_syst:.3f}")

# Customize the plot
plt.xlabel(r"$\sigma_{\mathrm{fid}}$ (fb)")
plt.ylabel(r"$-2 \Delta \ln L$")
plt.xlim(47.5, 88.5)
plt.ylim(0, 2.25)
plt.legend(loc="upper center", fontsize=18)

hep.cms.label(
    "Private work (Simulation/Data)",
    data=True,
    lumi=34.7,
    com=13.6,
    fontsize=22,
)

plt.tight_layout()

# Save the plot
plt.savefig("likelihood_scan_with_theory_prediction.pdf")
plt.savefig("likelihood_scan_with_theory_prediction.png")
