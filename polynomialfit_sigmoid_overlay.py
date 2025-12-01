import uproot
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import os
# make timing plots with mcp delta t from same mcp channel in this borad / group and compare 
# ---------------------------
# User settings
# ---------------------------
file_path = "/lustre/research/hep/yofeng/HG-DREAM/CERN/ROOT/run1409_250925135843_converted.root"
channel = "DRS_Board5_Group3_Channel1"
output_dir = "rising_edge_polynomialfit_t50"
os.makedirs(output_dir, exist_ok=True)
max_events_to_plot = 50

# ---------------------------
# Sigmoid model for rising edge
# ---------------------------
def sigmoid(t, A, t50, k):
    return A / (1 + np.exp(-(t - t50)/k))

# ---------------------------
# Load waveforms
# ---------------------------
file = uproot.open(file_path)
tree = file["EventTree"]
waveforms = tree[channel].array(library="np")

# ---------------------------
# Loop over events
# ---------------------------
t50_sigmoid_list = []
t50_poly_list = []

plotted = 0

for i, wf in enumerate(waveforms):
    if plotted >= max_events_to_plot:
        break

    # Baseline subtraction
    baseline_samples = 200
    baseline = np.median(wf[:baseline_samples])
    baseline_std = 1.4826 * np.median(np.abs(wf[:baseline_samples] - baseline))
    wf_bs = wf - baseline
    wf_bs_smooth = medfilt(wf_bs, kernel_size=5)

    # Threshold for peak detection
    threshold = 3 * baseline_std
    peaks = np.where(wf_bs_smooth > threshold)[0]
    if len(peaks) == 0:
        continue

    # Split into contiguous regions
    groups = np.split(peaks, np.where(np.diff(peaks) != 1)[0] + 1)
    if len(groups) != 1:
        continue

    rising_region = groups[0]
    peak_idx = np.argmax(wf_bs_smooth[rising_region]) + rising_region[0]
    peak_value = np.max(wf_bs_smooth[rising_region])

    # Fit region: use rising part until ~90% of peak
    above90 = np.where(wf_bs_smooth[rising_region] >= 0.9 * peak_value)[0]
    if len(above90) == 0:
        end_idx = peak_idx + 2
    else:
        end_idx = rising_region[0] + above90[0] + 2

    start_idx = max(rising_region[0] - 5, 0)

    t = np.arange(len(wf_bs_smooth))
    t_fit = t[start_idx:end_idx]
    y_fit = wf_bs_smooth[start_idx:end_idx]

    # ---------------------------
    # Sigmoid Fit
    # ---------------------------
    p0 = [peak_value, t_fit[len(t_fit)//2], 2]

    try:
        params, cov = curve_fit(sigmoid, t_fit, y_fit, p0=p0, maxfev=10000)
        A, t50_sig, k = params
        t50_sigmoid_list.append(t50_sig)
    except:
        continue

    # ---------------------------
    # Polynomial Fit (3rd order)
    # ---------------------------
    # Find 10% and 90% rising-edge points
    try:
        ten_rel = np.where(wf_bs_smooth[rising_region] >= 0.1 * peak_value)[0][0]
        ninety_rel = np.where(wf_bs_smooth[rising_region] >= 0.9 * peak_value)[0][0]
    except:
        t50_poly = np.nan
        t50_poly_list.append(t50_poly)
        print(f"Event {i+1:03d}: Sigmoid t50={t50_sig:.2f}, Poly t50=nan (no 10-90 window)")
        continue

    ten_idx = rising_region[0] + ten_rel
    ninety_idx = rising_region[0] + ninety_rel

    # Skip if invalid window
    if ten_idx >= ninety_idx:
        t50_poly = np.nan
        t50_poly_list.append(t50_poly)
        print(f"Event {i+1:03d}: Sigmoid t50={t50_sig:.2f}, Poly t50=nan (zero-length window)")
        continue

    # Extract data
    t_poly = t[ten_idx:ninety_idx]
    y_poly = wf_bs_smooth[ten_idx:ninety_idx]

    # Ensure arrays are non-empty
    if len(t_poly) < 4:
        t50_poly = np.nan
        t50_poly_list.append(t50_poly)
        print(f"Event {i+1:03d}: Sigmoid t50={t50_sig:.2f}, Poly t50=nan (too few points)")
        continue

    # Perform cubic fit
    coeffs = np.polyfit(t_poly, y_poly, 3)
    a, b, c, d = coeffs

    # Solve cubic for 50% amplitude
    half_val = 0.5 * peak_value
    roots = np.roots([a, b, c, d - half_val])
    real_roots = [r.real for r in roots if np.isreal(r) and ten_idx <= r.real <= ninety_idx]

    t50_poly = real_roots[0] if len(real_roots) > 0 else np.nan
    t50_poly_list.append(t50_poly)


    print(f"Event {i+1:03d}: Sigmoid t50={t50_sig:.2f}, Poly t50={t50_poly:.2f}")

    # ---------------------------
    # Plot waveform with both fits
    # ---------------------------
    plt.figure(figsize=(10,5))
    plt.plot(t, wf_bs_smooth, lw=1.2, label="Smoothed waveform")

    # Fit region
    plt.scatter(t_fit, y_fit, color='red', s=12, label="Sigmoid fit region")

    # Sigmoid curve
    plt.plot(t, sigmoid(t, *params), '--k', lw=2, label="Sigmoid fit")

    # Polynomial curve
    tt = np.linspace(ten_idx, ninety_idx, 300)
    plt.plot(tt, np.polyval(coeffs, tt), 'g--', lw=2, label="Cubic polynomial fit")

    # t50 markers
    plt.axvline(t50_sig, color='blue', linestyle='--', label=f"Sigmoid t50={t50_sig:.1f}")
    plt.axvline(t50_poly, color='green', linestyle=':', label=f"Poly t50={t50_poly:.1f}")

    plt.title(f"Event {i+1:03d} — {channel}")
    plt.xlabel("Sample")
    plt.ylabel("ADC (baseline-subtracted)")
    plt.grid(True, alpha=0.3)
    plt.xlim(400,700)
    plt.legend()

    plt.savefig(os.path.join(output_dir, f"event_{i+1:03d}_{channel}.pdf"), bbox_inches="tight")
    plt.close()

    plotted += 1

# ---------------------------
# Histograms
# ---------------------------
plt.figure(figsize=(8,5))
plt.hist(t50_sigmoid_list, bins=20, alpha=0.6, label="Sigmoid t50")
plt.hist(t50_poly_list, bins=20, alpha=0.6, label="Polynomial t50")
plt.xlabel("t50 (samples)")
plt.ylabel("Counts")
plt.title(f"t50 Comparison — {channel}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "t50_histograms_sigmoid_vs_poly.pdf"))
plt.close()

# Print stats
print("\n=== Summary ===")
print(f"Sigmoid: mean={np.mean(t50_sigmoid_list):.2f}, std={np.std(t50_sigmoid_list):.2f}")
print(f"Poly:    mean={np.mean(t50_poly_list):.2f}, std={np.std(t50_poly_list):.2f}")
