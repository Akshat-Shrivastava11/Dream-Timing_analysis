import uproot
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import os

# ---------------------------
# User settings
# ---------------------------
file_path = "/lustre/research/hep/yofeng/HG-DREAM/CERN/ROOT/run1409_250925135843_converted.root"
channel = "DRS_Board5_Group3_Channel1"
output_dir = "rising_edge_sigmoid_t50"
os.makedirs(output_dir, exist_ok=True)
max_events_to_plot = 50

# ---------------------------
# Sigmoid model for rising edge
# ---------------------------
def sigmoid(t, A, t50, k):
    """Sigmoid pulse: A / (1 + exp(-(t - t50)/k))"""
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
t50_list = []
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
    groups = np.split(peaks, np.where(np.diff(peaks) != 1)[0]+1)
    if len(groups) != 1:
        continue

    rising_region = groups[0]
    peak_idx = np.argmax(wf_bs_smooth[rising_region]) + rising_region[0]

    # Fit region: stop shortly after waveform reaches ~90% of peak
    peak_value = np.max(wf_bs_smooth[rising_region])
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
    # Fit rising edge with sigmoid
    # ---------------------------
    p0 = [peak_value, t_fit[len(t_fit)//2], 2]  # initial guesses: A, t50, k
    try:
        params, cov = curve_fit(sigmoid, t_fit, y_fit, p0=p0, maxfev=10000)
        A, t50, k = params
        t50_list.append(t50)  # collect t50
    except:
        continue

    print(f"Event {i+1:03d}: A={A:.2f}, t50={t50:.2f}, k={k:.2f}")

    # ---------------------------
    # Plot
    # ---------------------------
    plt.figure(figsize=(10,5))
    plt.plot(t, wf_bs, lw=1, alpha=0.5, label="Baseline-subtracted")
    plt.plot(t, wf_bs_smooth, lw=1.2, label="Smoothed")
    plt.scatter(t_fit, y_fit, color='red', s=15, label="Fit region")
    plt.plot(t, sigmoid(t, *params), '--k', lw=2, label="Sigmoid fit")
    plt.axvline(t50, color='blue', linestyle='--', label=f"t50 = {t50:.1f}")
    plt.title(f"Event {i+1:03d} — {channel}")
    plt.xlabel("Sample")
    plt.ylabel("ADC (baseline-subtracted)")
    plt.legend()
    plt.xlim(400,700)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"event_{i+1:03d}_{channel}.pdf"), bbox_inches="tight")
    plt.close()

    plotted += 1
# ---------------------------
# Plot histogram of t50
# ---------------------------
plt.figure(figsize=(8,5))
plt.hist(t50_list, bins=20, color='skyblue', edgecolor='black')
plt.xlabel("t50 (samples)")
plt.ylabel("Counts")
plt.title(f"Histogram of t50 for first {max_events_to_plot} events — {channel}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "t50_histogram.pdf"))
# Print mean and std
t50_array = np.array(t50_list)
print(f"Mean t50 = {np.mean(t50_array):.2f}, Std = {np.std(t50_array):.2f}")
plt.axvline(np.mean(t50_array), color='red', linestyle='--', label='Mean t50')
plt.close()