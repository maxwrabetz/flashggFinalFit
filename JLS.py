import uproot
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.interpolate import make_interp_spline

hep.style.use("CMS")

def process_scan(file_path):
    # Open the ROOT file
    file = uproot.open(file_path)

    # Access the tree containing the scan results
    tree = file["limit"]

    # Extract the parameter of interest (e.g., r) and the likelihood values (-2 Delta NLL)
    poi_values = tree["r"].array()
    nll_values = tree["deltaNLL"].array()

    # Convert the uproot arrays to numpy arrays for easier handling
    # Also slice away the first value because it is best fit and we do not want an extra line
    poi_values = np.array(poi_values)[1:]
    nll_values = np.array(nll_values)[1:]

    nll_values = 2 * nll_values  # convert to -2 Delta NLL

    # Sort the values based on poi_values
    sorted_indices = np.argsort(poi_values)
    poi_values_sorted = poi_values[sorted_indices]
    nll_values_sorted = nll_values[sorted_indices]

    # Remove duplicates
    unique_poi, unique_indices = np.unique(poi_values_sorted, return_index=True)
    poi_values_sorted = unique_poi
    nll_values_sorted = nll_values_sorted[unique_indices]

    return poi_values_sorted, nll_values_sorted

# Process the scans
poi_values_statonly, nll_values_statonly = process_scan("higgsCombineAsimovPostFitScanStat_r.root")
poi_values_with_syst, nll_values_with_syst = process_scan("higgsCombineAsimovPostFitScanFit_r.root")

theory_uncertainty = 3.80
do_xsec = True
if do_xsec:
    theory_value = 67.80
    poi_values_statonly *= theory_value
    poi_values_with_syst *= theory_value
else:
    theory_value = 1.0
    theory_uncertainty = theory_uncertainty / 67.80

# --- Smooth spline for stat-only ---
x_stat_smooth = np.linspace(poi_values_statonly.min(), poi_values_statonly.max(), 500)
spline_stat = make_interp_spline(poi_values_statonly, nll_values_statonly, k=3)
y_stat_smooth = spline_stat(x_stat_smooth)
plt.plot(x_stat_smooth, y_stat_smooth, color='blue', ls='dashdot', label='Expected - Stat. only (spline)')
plt.scatter(poi_values_statonly, nll_values_statonly, color='blue', s=15, marker='o', label='Stat-only (raw points)')

# --- Smooth spline for with-syst ---
x_syst_smooth = np.linspace(poi_values_with_syst.min(), poi_values_with_syst.max(), 500)
spline_syst = make_interp_spline(poi_values_with_syst, nll_values_with_syst, k=3)
y_syst_smooth = spline_syst(x_syst_smooth)
plt.plot(x_syst_smooth, y_syst_smooth, color='black', lw=2, label='Expected (spline)')
plt.scatter(poi_values_with_syst, nll_values_with_syst, color='black', s=15, marker='x', label='With syst (raw points)')

# Add the theoretical prediction with uncertainty
plt.plot([theory_value, theory_value], [0, 2], color='red', linestyle='-', linewidth=3, label=r'MG5_aMC@NLO, NNLOPS')
plt.fill_betweenx(y=[0, 2], x1=theory_value - theory_uncertainty, x2=theory_value + theory_uncertainty,
                  color='none', edgecolor='red', facecolor='none', hatch='//')

# Grey lines at -2ΔlnL = 1 and 2
plt.axhline(1.0, color='grey', linestyle='--', linewidth=2)
plt.axhline(2.0, color='grey', linestyle='--', linewidth=2)

# Find the crossing points at -2 Δln(L) = 1 (using raw points)
crossing_indices = np.where(np.diff(np.sign(nll_values_with_syst - 1)))[0]
crossing_points = []
for idx in crossing_indices:
    x0, x1 = poi_values_with_syst[idx], poi_values_with_syst[idx + 1]
    y0, y1 = nll_values_with_syst[idx], nll_values_with_syst[idx + 1]
    crossing_points.append(x0 + (1 - y0) * (x1 - x0) / (y1 - y0))

for cp in crossing_points:
    plt.plot([cp, cp], [0, 1], color='grey', linestyle='--')

print("POI stat-only:", poi_values_statonly)
print("NLL stat-only:", nll_values_statonly)
print("POI syst:", poi_values_with_syst)
print("NLL syst:", nll_values_with_syst)

# Customize the plot
plt.xlabel(r'$\sigma_{\mathrm{fid}}$ (fb)')
plt.ylabel(r'$-2 \Delta \ln L$')
plt.xlim(47.5, 88.5)
plt.ylim(0, 2.25)
plt.legend(loc='upper left', fontsize=12)
hep.cms.label('Private work (Simulation/Data)', data=True, lumi=27.3, com=13.6, fontsize=16)
plt.tight_layout()

# Save the plot
plt.savefig("likelihood_scan_with_theory_prediction.pdf")
plt.savefig("likelihood_scan_with_theory_prediction.png")
