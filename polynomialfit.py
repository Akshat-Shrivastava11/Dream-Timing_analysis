import uproot
import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import os

# ---------------------------
# User settings
# ---------------------------
file_path = "/lustre/research/hep/yofeng/HG-DREAM/CERN/ROOT/run1409_250925135843_converted.root"
channel = "DRS_Board5_Group3_Channel1"
output_dir = "PolynomialFitResults"
os.makedirs(output_dir, exist_ok=True)
max_events_to_plot = 20

# ---------------------------
# Load waveforms
# ---------------------------
file = uproot.open(file_path)
tree = file["EventTree"]
waveforms = tree[channel].array(library="np")

# ---------------------------
# Loop over events
# ---------------------------
t50_poly_list = []
plotted = 0

for i, wf in enumerate(waveforms):
    if plotted >= max_events_to_plot:
        break

    # ---------------------------
    # Baseline subtraction
    # ---------------------------
    baseline_samples = 200
    baseline = np.median(wf[:baseline_samples])
    baseline_std = 1.4826 * np.median(np.abs(wf[:baseline_samples] - baseline))
    wf_bs = wf - baseline

    # Smoothing only used for finding the window (NOT for fitting)
    wf_bs_smooth = medfilt(wf_bs, kernel_size=5)

    # ---------------------------
    # Peak detection
    # ---------------------------
    threshold = 3 * baseline_std
    peaks = np.where(wf_bs_smooth > threshold)[0]
    if len(peaks) == 0:
        continue

    groups = np.split(peaks, np.where(np.diff(peaks) != 1)[0] + 1)
    if len(groups) != 1:
        continue  # Skip multiple clusters

    rising_region = groups[0]
    peak_value = np.max(wf_bs_smooth[rising_region])

    # ---------------------------
    # 10% – 90% region for polynomial fit
    # ---------------------------
    try:
        ten_rel = np.where(wf_bs_smooth[rising_region] >= 0.10 * peak_value)[0][0]
        ninety_rel = np.where(wf_bs_smooth[rising_region] >= np.float64(0.90 * peak_value))[0][0]
    except Exception:
        print(f"Event {i+1:03d}: Bad 10–90% region. Skipped.")
        t50_poly_list.append(np.nan)
        continue

    ten_idx = rising_region[0] + ten_rel
    ninety_idx = rising_region[0] + ninety_rel

    if ten_idx >= ninety_idx:
        print(f"Event {i+1:03d}: Invalid window (ten >= ninety). Skipped.")
        t50_poly_list.append(np.nan)
        continue

    # ---------------------------
    # Extract RAW waveform values for fitting (no smoothing!)
    # ---------------------------
    t = np.arange(len(wf_bs))
    t_poly = t[ten_idx:ninety_idx+1]
    y_poly = wf_bs[ten_idx:ninety_idx+1]

    if len(t_poly) < 6:
        print(f"Event {i+1:03d}: Too few points for poly fit.")
        t50_poly_list.append(np.nan)
        continue

    # ---------------------------
    # Cubic polynomial fit
    # ---------------------------
    coeffs = np.polyfit(t_poly, y_poly, 3)
    a, b, c, d = coeffs

    half_height = 0.5 * peak_value

    # Solve cubic(a t^3 + b t^2 + c t + d = half_height)
    roots = np.roots([a, b, c, d - half_height])

    # Keep real roots inside window
    real_roots = [r.real for r in roots if np.isreal(r) and ten_idx <= r.real <= ninety_idx]

    t50_poly = real_roots[0] if len(real_roots) > 0 else np.nan
    t50_poly_list.append(t50_poly)

    print(f"Event {i+1:03d}: Polynomial t50 = {t50_poly:.2f}")

    # ---------------------------
    # Plot
    # ---------------------------
    plt.figure(figsize=(10,5))
    plt.plot(t, wf_bs, lw=1, label="Raw baseline-subtracted")
    plt.plot(t, wf_bs_smooth, lw=1, alpha=0.5, label="Smoothed (only for window)")

    # Polynomial curve for visualization
    tt = np.linspace(ten_idx, ninety_idx, 300)
    plt.plot(tt, np.polyval(coeffs, tt), 'g--', lw=2, label="Cubic polynomial fit")

    # Vertical line at t50
    if not np.isnan(t50_poly):
        plt.axvline(t50_poly, color='green', linestyle='--', label=f"t50 = {t50_poly:.1f}")

    plt.title(f"Event {i+1:03d} — {channel}")
    plt.xlabel("Sample")
    plt.ylabel("ADC (baseline-subtracted)")
    plt.grid(True, alpha=0.3)
    plt.xlim(400, 700)
    plt.legend()

    plt.savefig(os.path.join(output_dir, f"event_{i+1:03d}_{channel}.pdf"),
                bbox_inches="tight")
    plt.close()

    plotted += 1

# ---------------------------
# Histogram
# ---------------------------
plt.figure(figsize=(8,5))
plt.hist(t50_poly_list, bins=20, color="skyblue", edgecolor="black")
plt.xlabel("Polynomial t50 (samples)")
plt.ylabel("Counts")
plt.title(f"Polynomial t50 distribution — {channel}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "t50_polynomial_histogram.pdf"))
plt.close()

print("\n=== Polynomial t50 Summary ===")
print(f"Mean t50 = {np.nanmean(t50_poly_list):.2f}")
print(f"Std  t50 = {np.nanstd(t50_poly_list):.2f}")
