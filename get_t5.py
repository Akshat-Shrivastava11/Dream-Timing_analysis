import uproot
import numpy as np
from scipy.optimize import curve_fit, root_scalar
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import os

# ---------------------------
# User settings
# ---------------------------
file_path = "/lustre/research/hep/yofeng/HG-DREAM/CERN/ROOT/run1409_250925135843_converted.root"
channel = "DRS_Board5_Group3_Channel1"
output_dir = "rising_edge_t50"
os.makedirs(output_dir, exist_ok=True)
max_events_to_plot = 10  # Only plot first 10 events

# ---------------------------
# Rising-edge model
# ---------------------------
def rising_edge(t, A, t0, tau):
    y = np.zeros_like(t)
    mask = t >= t0
    y[mask] = A * (1 - np.exp(-(t[mask] - t0)/tau))
    return y

# ---------------------------
# Load waveforms
# ---------------------------
file = uproot.open(file_path)
tree = file["EventTree"]
waveforms = tree[channel].array(library="np")

# ---------------------------
# Loop over events
# ---------------------------
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
        continue  # no peak above 3 sigma

    # Split into contiguous regions
    groups = np.split(peaks, np.where(np.diff(peaks) != 1)[0]+1)
    if len(groups) != 1:
        continue  # skip if multiple peaks

    rising_region = groups[0]
    peak_idx = np.argmax(wf_bs_smooth[rising_region]) + rising_region[0]

    # Fit region: start before threshold, end shortly after peak
    start_idx = max(rising_region[0] - 5, 0)
    end_idx = peak_idx + 2
    t = np.arange(len(wf_bs_smooth))
    t_fit = t[start_idx:end_idx]
    y_fit = wf_bs_smooth[start_idx:end_idx]

    # Fit with bounds
    p0 = [np.max(y_fit), start_idx, 3]
    bounds = ([0, start_idx, 0], [np.max(y_fit)*1.05, end_idx, np.inf])
    try:
        params, cov = curve_fit(rising_edge, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=10000)
        A, t0, tau = params
    except:
        continue

    # ---------------------------
    # Compute t50 numerically
    # ---------------------------
    def half_amplitude_func(t_val):
        return rising_edge(np.array([t_val]), A, t0, tau)[0] - 0.5*A

    try:
        sol = root_scalar(half_amplitude_func, bracket=[t0, t0 + 10*tau])
        t50 = sol.root if sol.converged else np.nan
    except:
        t50 = np.nan

    print(f"Event {i+1:03d}: A={A:.2f}, t0={t0:.2f}, tau={tau:.2f}, t50={t50:.2f}")

    # ---------------------------
    # Plot
    # ---------------------------
    plt.figure(figsize=(10,5))
    plt.plot(t, wf_bs, lw=1, alpha=0.5, label="Baseline-subtracted")
    plt.plot(t, wf_bs_smooth, lw=1.2, label="Smoothed")
    plt.scatter(t_fit, y_fit, color='red', s=15, label="Fit region")
    plt.plot(t, rising_edge(t, *params), '--k', lw=2, label="Rising-edge fit")
    plt.axvline(t50, color='blue', linestyle='-', label=f"t50 = {t50:.1f}")
    plt.title(f"Event {i+1:03d} — {channel}")
    plt.xlabel("Sample")
    plt.xlim(400,650)
    plt.ylabel("ADC (baseline-subtracted)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"event_{i+1:03d}_{channel}.pdf"), bbox_inches="tight")
    plt.close()

    plotted += 1
