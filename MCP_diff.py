#!/usr/bin/env python3
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, find_peaks
from scipy.stats import norm
import os

# ============================
# Configuration
# ============================
FILE = "/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1468_250927145556_TimingDAQ.root"
OUTPUT_DIR = "./MCP/Delta_t50_Results"
SAMPLE_TIME_NS = 0.2
MAX_EVENTS = 200
THRESHOLD_LEVEL = 12
MIN_PEAK_WIDTH = 3  # minimum samples in a peak

os.makedirs(OUTPUT_DIR, exist_ok=True)
for board in range(4):
    os.makedirs(os.path.join(OUTPUT_DIR, f"Board{board}"), exist_ok=True)

# ============================
# T50 extraction using find_peaks
# ============================
def t50_firstpeak_threshold(wf, threshold_level=15):
    wf = np.asarray(wf, dtype=float)
    wf = -wf  # Invert waveform
    baseline_samples = 200
    baseline = np.median(wf[:baseline_samples])
    wf_bs = wf - baseline

    # Smooth waveform first
    wf_smooth = medfilt(wf_bs, kernel_size=5)

    # Find all points above threshold
    above = np.where(wf_smooth > threshold_level)[0]
    if len(above) == 0:
        return None, wf_bs, None  # No peaks above threshold

    # Split into continuous groups (candidate peaks)
    groups = np.split(above, np.where(np.diff(above) > 3)[0] + 1)

    # Pick first group that has a peak above threshold
    first_peak_group = None
    for g in groups:
        if np.max(wf_smooth[g]) >= threshold_level:
            first_peak_group = g
            break

    if first_peak_group is None:
        return None, wf_bs, None

    rising_region = first_peak_group
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
# Process per board
# ============================
with uproot.open(FILE) as f:
    tree = f["EventTree"]

    for board in range(4):
        print(f"\nProcessing Board {board} ...")
        channels = [f"DRS_Board{board}_Group3_Channel6",
                    f"DRS_Board{board}_Group3_Channel7"]

        t50_map = {ch: [] for ch in channels}
        waveform_map = {ch: [] for ch in channels}
        fit_regions = {ch: [] for ch in channels}

        # Loop over events
        for iev, arrays in enumerate(tree.iterate(channels, step_size=1)):
            if iev >= MAX_EVENTS:
                break
            event = {ch: arrays[ch][0] for ch in channels}
        
            for ch in channels:
                t50, wf_bs, fit_info = t50_firstpeak_threshold(event[ch])
                t50_map[ch].append(t50)
                waveform_map[ch].append(wf_bs)
                fit_regions[ch].append(fit_info)
                print(  f"Board {board} Event {iev} Channel {ch}: T50 = {t50} ns")

        # Compute Δt50
        t6 = np.array([v if v is not None else np.nan for v in t50_map[channels[0]]], dtype=float)
        t7 = np.array([v if v is not None else np.nan for v in t50_map[channels[1]]], dtype=float)
        mask = ~np.isnan(t6) & ~np.isnan(t7)
        delta_t = (t6 - t7)[mask]
        print(f"Board {board}: Found {len(delta_t)} valid Δt50 out of {MAX_EVENTS} events.")

        if len(delta_t) == 0:
            print("No valid Δt50 found for this board.")
            continue
        # Optional: save per-event waveform plots only if T50 was found
        # for iev in range(MAX_EVENTS):
        #     # Only plot if both channels have a valid T50
        #     if t50_map[channels[0]][iev] is None or t50_map[channels[1]][iev] is None:
        #         continue  # skip this event

        #     fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        #     for i, ch in enumerate(channels):
        #         wf = waveform_map[ch][iev]
        #         fit = fit_regions[ch][iev]
        #         x = np.arange(len(wf)) * SAMPLE_TIME_NS
        #         axes[i].plot(x, wf, color='b' if i == 0 else 'r', alpha=0.7)
        #         axes[i].set_title(f"{ch} Event {iev}")
        #         axes[i].set_xlabel("Time [ns]")
        #         axes[i].set_ylabel("Amplitude [a.u.]")

        #         if fit is not None:
        #             ten, ninety, peak = fit
        #             axes[i].axvspan(ten * SAMPLE_TIME_NS, ninety * SAMPLE_TIME_NS, color='g', alpha=0.3)
        #             t50_val = t50_map[ch][iev]
        #             axes[i].axvline(t50_val, color='k', linestyle='--', label='T50')

        #     plt.tight_layout()
        #     plt.savefig(os.path.join(OUTPUT_DIR, f"Board{board}", f"Event{iev}.png"))
        #     plt.close()

        # Gaussian fit
        mu, sigma = norm.fit(delta_t)

        # Plot histogram
        plt.figure(figsize=(8,6))
        n, bins, _ = plt.hist(delta_t, bins=50,histtype="step", alpha=0.7,  label="Δt50 data")
        x = np.linspace(min(delta_t), max(delta_t), 400)
        y = norm.pdf(x, mu, sigma)
        plt.plot(x, y, 'k--', linewidth=2, label=f"Mean={mu:.2f} ns, σ={sigma:.2f} ns, Width={np.ptp(delta_t):.2f} ns")
        plt.xlabel("Δt50 = t50(Ch6) - t50(Ch7) [ns]")
        plt.ylabel("Probability density")
        plt.title(f"Board {board} Δt50 histogram (first {MAX_EVENTS} events)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"Board{board}_Delta_t50_hist.png"))
        plt.close()

        # Optional: save per-event waveform plots
        


        print(f"Board {board}: Δt50 histogram saved, mean={mu:.3f} ns, std={sigma:.3f} ns, width={np.ptp(delta_t):.3f} ns")
