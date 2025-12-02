#!/usr/bin/env python3
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import os

# ============================
# Configuration
# ============================
FILE = "/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1468_250927145556_TimingDAQ.root"
OUTPUT_DIR = "./MCP_diff_200ev"
SAMPLE_TIME_NS = 0.2
MAX_EVENTS = 200

ALL_CHANNELS = [
    f"DRS_Board{b}_Group3_Channel{ch}" 
    for b in range(4) for ch in [6, 7]
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# T50 extraction (first peak, medium-tight)
# ============================
def t50_medium_firstpeak(wf):
    wf = np.asarray(wf, dtype=float)
    baseline_samples = 200
    baseline = np.median(wf[:baseline_samples])
    baseline_std = 1.4826 * np.median(np.abs(wf[:baseline_samples] - baseline))
    wf_bs = wf - baseline

    # Smooth and threshold
    wf_smooth = medfilt(wf_bs, kernel_size=5)
    threshold = 2.5 * baseline_std

    above = np.where(wf_smooth > threshold)[0]
    if len(above) == 0:
        return None, wf_bs, None

    groups = np.split(above, np.where(np.diff(above) > 3)[0] + 1)
    rising_region = groups[0]
    peak_value = np.max(wf_smooth[rising_region])

    ten_candidates = np.where(wf_smooth[rising_region] >= 0.10 * peak_value)[0]
    ninety_candidates = np.where(wf_smooth[rising_region] >= 0.90 * peak_value)[0]

    ten_idx = rising_region[ten_candidates[0]] if len(ten_candidates) else rising_region[0]
    ninety_idx = rising_region[ninety_candidates[0]] if len(ninety_candidates) else rising_region[-1]

    if ten_idx >= ninety_idx:
        ninety_idx = min(ten_idx + 1, len(wf_bs) - 1)

    t_poly = np.arange(ten_idx, ninety_idx + 1)
    y_poly = wf_bs[ten_idx:ninety_idx + 1]

    if len(t_poly) < 4:
        t50_idx = ten_idx + (ninety_idx - ten_idx) / 2
    else:
        coeffs = np.polyfit(t_poly, y_poly, 3)
        a, b, c, d = coeffs
        half_height = 0.5 * peak_value
        roots = np.roots([a, b, c, d - half_height])
        real_roots = [r.real for r in roots if np.isreal(r) and ten_idx <= r.real <= ninety_idx]
        t50_idx = real_roots[0] if real_roots else ten_idx + (ninety_idx - ten_idx) / 2

    return t50_idx * SAMPLE_TIME_NS, wf_bs, (ten_idx, ninety_idx, peak_value)

# ============================
# Main loop: first 200 events
# ============================
print("Opening ROOT file...")
with uproot.open(FILE) as f:
    tree = f["EventTree"]
    t50_map = {ch: [] for ch in ALL_CHANNELS}
    waveform_map = {ch: [] for ch in ALL_CHANNELS}  # store post-baseline waveforms
    fit_regions = {ch: [] for ch in ALL_CHANNELS}   # store t10-t90 + peak for plotting

    for iev, arrays in enumerate(tree.iterate(ALL_CHANNELS, step_size=1)):
        if iev >= MAX_EVENTS:
            break
        event = {ch: arrays[ch][0] for ch in ALL_CHANNELS}
        for ch in ALL_CHANNELS:
            t50, wf_bs, fit_info = t50_medium_firstpeak(event[ch])
            t50_map[ch].append(t50)
            waveform_map[ch].append(wf_bs)
            fit_regions[ch].append(fit_info)
    print(f"Processed {min(MAX_EVENTS, tree.num_entries)} events.")

# ============================
# Compute per-board differences (Ch6 - Ch7)
# ============================
diffs = []
for board in range(4):
    ch6 = f"DRS_Board{board}_Group3_Channel6"
    ch7 = f"DRS_Board{board}_Group3_Channel7"

    t6 = np.array(t50_map[ch6])
    t7 = np.array(t50_map[ch7])
    mask = ~np.isnan(t6) & ~np.isnan(t7)
    valid = (t6 - t7)[mask]
    diffs.extend(valid)

    print(f"Board {board}: Valid events = {np.sum(mask)}, Mean Δt50 = {np.mean(valid):.3f} ns")

# ============================
# Save histogram of Δt50
# ============================
plt.figure(figsize=(8,6))
plt.hist(diffs, bins=50, alpha=0.7)
plt.xlabel("Δt50 = t50(Ch6) - t50(Ch7) [ns]")
plt.ylabel("Events")
plt.title("Per-board t50 difference distribution (first 200 events)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Delta_t50_hist.png"))
plt.close()

# ============================
# Plot post-fit waveforms as subplots per board
# ============================
for board in range(4):
    ch6 = f"DRS_Board{board}_Group3_Channel6"
    ch7 = f"DRS_Board{board}_Group3_Channel7"

    fig, axes = plt.subplots(2, 1, figsize=(12,6), sharex=True)
    axes[0].set_title(f"Board {board} - Ch6 post-baseline waveforms with T50")
    axes[1].set_title(f"Board {board} - Ch7 post-baseline waveforms with T50")

    for iev in range(len(waveform_map[ch6])):
        wf6 = waveform_map[ch6][iev]
        wf7 = waveform_map[ch7][iev]
        fit6 = fit_regions[ch6][iev]
        fit7 = fit_regions[ch7][iev]
        x6 = np.arange(len(wf6)) * SAMPLE_TIME_NS
        x7 = np.arange(len(wf7)) * SAMPLE_TIME_NS

        axes[0].plot(x6, wf6, alpha=0.3, color='b')
        axes[1].plot(x7, wf7, alpha=0.3, color='r')

        # Plot t10-t90 region
        if fit6 is not None:
            ten, ninety, peak = fit6
            axes[0].axvspan(ten*SAMPLE_TIME_NS, ninety*SAMPLE_TIME_NS, color='g', alpha=0.2)
        if fit7 is not None:
            ten, ninety, peak = fit7
            axes[1].axvspan(ten*SAMPLE_TIME_NS, ninety*SAMPLE_TIME_NS, color='g', alpha=0.2)

    axes[1].set_xlabel("Time [ns]")
    axes[0].set_ylabel("Amplitude [a.u.]")
    axes[1].set_ylabel("Amplitude [a.u.]")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Board{board}_waveforms.png"))
    plt.close()

print(f"Plots and histogram saved in {OUTPUT_DIR}")
print("Done.")
