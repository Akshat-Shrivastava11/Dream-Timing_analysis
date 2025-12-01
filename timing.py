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
output_dir = "rising_edge_single_peak"
os.makedirs(output_dir, exist_ok=True)

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
for i, wf in enumerate(waveforms):
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
    
    # Only proceed if there is a single peak
    if len(groups) != 1:
        continue

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

    # Timing info
    t50 = t0 + tau*np.log(2)
    t90 = t0 + tau*np.log(10)

    print(f"Event {i+1:03d}: A={A:.2f}, t0={t0:.2f}, tau={tau:.2f}, t50={t50:.2f}, t90={t90:.2f}")

    # Plot and save
    plt.figure(figsize=(10,5))
    plt.plot(t, wf_bs, lw=1, alpha=0.5, label="Baseline-subtracted")
    plt.plot(t, wf_bs_smooth, lw=1.2, label="Smoothed")
    plt.scatter(t_fit, y_fit, color='red', s=15, label="Fit region")
    plt.plot(t, rising_edge(t, *params), '--k', lw=2, label="Rising-edge fit")
    plt.title(f"Event {i+1:03d} — {channel}")
    plt.xlabel("Sample")
    plt.ylabel("ADC (baseline-subtracted)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"event_{i+1:03d}_{channel}.pdf"), bbox_inches="tight")
    plt.close()

'''
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import os

# ---------------------------
# User settings
# ---------------------------
file_path = "/lustre/research/hep/yofeng/HG-DREAM/CERN/ROOT/run1409_250925135843_converted.root"
channel = "DRS_Board5_Group3_Channel1"
event_idx = 98  # Event 099
output_dir = "rising_edge_plots"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Rising-edge fit model
# ---------------------------
def rising_edge(t, A, t0, tau):
    y = np.zeros_like(t)
    mask = t >= t0
    y[mask] = A * (1 - np.exp(-(t[mask] - t0)/tau))
    return y

# ---------------------------
# Load waveform
# ---------------------------
file = uproot.open(file_path)
tree = file["EventTree"]
wf = tree[channel].array(library="np")[event_idx]

# ---------------------------
# Baseline subtraction
# ---------------------------
baseline_samples = 200
baseline_window = wf[:baseline_samples]
baseline = np.median(baseline_window)
baseline_std = 1.4826 * np.median(np.abs(baseline_window - baseline))
wf_bs = wf - baseline
wf_bs_smooth = medfilt(wf_bs, kernel_size=5)

# ---------------------------
# Peak detection
# ---------------------------
threshold = 3 * baseline_std
peaks = np.where(wf_bs_smooth > threshold)[0]
if len(peaks) == 0:
    raise RuntimeError("No peaks above threshold found.")

groups = np.split(peaks, np.where(np.diff(peaks) != 1)[0]+1)
rising_region = groups[0]

# ---------------------------
# Fit region: start before threshold, end at peak
# ---------------------------
peak_idx = np.argmax(wf_bs_smooth[rising_region]) + rising_region[0]
start_idx = max(rising_region[0] - 5, 0)  # a few samples before threshold
end_idx = peak_idx + 2                     # stop shortly after peak

t = np.arange(len(wf_bs_smooth))
t_fit = t[start_idx:end_idx]
y_fit = wf_bs_smooth[start_idx:end_idx]

# ---------------------------
# Fit rising edge
# ---------------------------
p0 = [np.max(y_fit), start_idx, 10]
params, cov = curve_fit(rising_edge, t_fit, y_fit, p0=p0, maxfev=10000)
A, t0, tau = params

# ---------------------------
# Print diagnostics
# ---------------------------
print(f"Event 099 — Channel {channel}")
print(f"Baseline: {baseline:.2f}, Noise: {baseline_std:.2f}, Threshold: {threshold:.2f}")
print(f"Rising edge fit region: start={start_idx}, end={end_idx}, n_samples={end_idx-start_idx}")
print(f"Fit parameters: A={A:.2f}, t0={t0:.2f}, tau={tau:.2f}")

# ---------------------------
# Plot and save PDF
# ---------------------------
plt.figure(figsize=(12,6))
plt.plot(t, wf_bs, label="Baseline-subtracted", lw=1.2, alpha=0.6)
plt.plot(t, wf_bs_smooth, label="Smoothed waveform", lw=1.2)
plt.scatter(t_fit, y_fit, color='red', s=15, label="Fit region")
plt.plot(t, rising_edge(t, *params), '--k', lw=2,
         label=f"Rising-edge fit:\nA={A:.2f}, t0={t0:.2f}, tau={tau:.2f}")
plt.xlabel("Sample")
plt.ylabel("ADC (baseline-subtracted)")
plt.title(f"Rising-edge Fit — {channel} — Event 099")
plt.legend()
plt.grid(True, alpha=0.3)

output_file = os.path.join(output_dir, f"no_fall_off_rising_edge_event_099_{channel}.pdf")
plt.savefig(output_file, bbox_inches="tight")
plt.close()
print(f"Saved plot as: {output_file}")



import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# ---------------------------
# User settings
# ---------------------------
file_path = "/lustre/research/hep/yofeng/HG-DREAM/CERN/ROOT/run1409_250925135843_converted.root"
channel = "DRS_Board5_Group3_Channel1"
output_dir = "rising_edge_fits"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Rising-edge fit model
# ---------------------------
def rising_edge(t, A, t0, tau):
    y = np.zeros_like(t)
    mask = t >= t0
    y[mask] = A * (1 - np.exp(-(t[mask] - t0)/tau))
    return y

# ---------------------------
# Load tree and waveform array
# ---------------------------
file = uproot.open(file_path)
tree = file["EventTree"]
waveforms = tree[channel].array(library="np")
n_events = len(waveforms)

# ---------------------------
# Loop over events
# ---------------------------
for i, wf in enumerate(waveforms):
    # Baseline subtraction
    baseline = np.mean(wf[:200])
    baseline_std = np.std(wf[:200])
    wf_bs = wf - baseline

    # Find peaks above 3 sigma
    peak_threshold = baseline + 3*baseline_std
    peaks = np.where(wf_bs > 3*baseline_std)[0]

    # Check for single peak
    if len(peaks) == 0:
        continue
    # simple heuristic: one contiguous region above threshold
    groups = np.split(peaks, np.where(np.diff(peaks) != 1)[0]+1)
    if len(groups) != 1:
        continue  # skip if multiple peaks

    # Rising edge window: 5 samples before first crossing, 40 samples
    start_idx = max(groups[0][0] - 5, 0)
    end_idx = min(start_idx + 40, len(wf_bs))
    t = np.arange(len(wf_bs))
    t_fit = t[start_idx:end_idx]
    y_fit = wf_bs[start_idx:end_idx]

    # Fit rising edge
    p0 = [np.max(y_fit), start_idx, 10]  # initial guesses
    try:
        params, cov = curve_fit(rising_edge, t_fit, y_fit, p0=p0, maxfev=10000)
        A, t0, tau = params
    except RuntimeError:
        continue  # skip events where fit fails

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(t, wf_bs, label="Baseline-subtracted waveform", lw=1.2)
    plt.scatter(t_fit, y_fit, color='red', s=10, label="Fit region")
    plt.plot(t, rising_edge(t, *params), '--k', lw=2,
             label=f"Rising-edge fit:\nA={A:.1f}, t0={t0:.1f}, tau={tau:.1f}")
    plt.xlabel("Sample")
    plt.ylabel("ADC (baseline-subtracted)")
    plt.title(f"Rising-edge Fit — {channel} — Event {i+1:03d}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save PDF
    plt.savefig(f"{output_dir}/rising_edge_event_{i+1:03d}.pdf", bbox_inches="tight")
    plt.close()
'''