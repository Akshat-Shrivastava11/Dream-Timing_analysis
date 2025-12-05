#!/usr/bin/env python3
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.stats import norm
import os
from multiprocessing import Pool, cpu_count

# ============================
# Configuration
# ============================
FILE = "/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1468_250927145556_TimingDAQ.root"
OUTPUT_DIR = "./MCP/Delta_t50_Results2"
SAMPLE_TIME_NS = 0.2
MAX_EVENTS = 200000
THRESHOLD_LEVEL = 12

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# FAST T50 extraction
# =====================================================
def t50_firstpeak_threshold_fast(wf, threshold=15):
    wf = -wf.astype(np.float32)

    baseline = np.median(wf[:200])
    wf -= baseline

    wf_smooth = medfilt(wf, kernel_size=5)

    above = np.flatnonzero(wf_smooth > threshold)
    if above.size == 0:
        return np.nan

    diffs = np.diff(above)
    breaks = np.where(diffs > 3)[0] + 1
    groups = np.split(above, breaks)

    for g in groups:
        if wf_smooth[g].max() >= threshold:
            first = g
            break

    peak_val = wf_smooth[first].max()
    rel_vals = wf_smooth[first]

    # --- safe t10 ---
    idx10 = np.searchsorted(rel_vals, 0.1 * peak_val)
    idx10 = min(idx10, len(first) - 1)
    t10 = first[idx10]

    # --- safe t90 ---
    idx90 = np.searchsorted(rel_vals, 0.9 * peak_val)
    idx90 = min(idx90, len(first) - 1)
    t90 = first[idx90]

    if t90 <= t10:
        t90 = min(t10 + 1, len(wf) - 1)

    # Polynomial fit
    if t90 - t10 >= 4:
        x = np.arange(t10, t90 + 1)
        y = wf[t10:t90 + 1]
        a, b, c, d = np.polyfit(x, y, 3)
        roots = np.roots([a, b, c, d - 0.5 * peak_val])
        real = roots[np.isreal(roots)].real
        real = real[(real >= t10) & (real <= t90)]
        if real.size > 0:
            t50 = real[0]
        else:
            t50 = t10 + 0.5 * (t90 - t10)
    else:
        t50 = t10 + 0.5 * (t90 - t10)

    return t50 * SAMPLE_TIME_NS


# =====================================================
# Multiprocessing wrapper
# =====================================================
def _t50_worker(args):
    wf, thr = args
    return t50_firstpeak_threshold_fast(wf, threshold=thr)


# =====================================================
# Main processing per board
# =====================================================
with uproot.open(FILE) as f:
    tree = f["EventTree"]

    for board in range(4):
        print(f"\n==============================")
        print(f"Processing Board {board} with multiprocessing...")
        print(f"==============================")

        ch6 = f"DRS_Board{board}_Group3_Channel6"
        ch7 = f"DRS_Board{board}_Group3_Channel7"
        channels = [ch6, ch7]

        arrays = tree.arrays(channels, library="np", entry_stop=MAX_EVENTS)

        wf6_all = arrays[ch6]
        wf7_all = arrays[ch7]
        n_ev = wf6_all.shape[0]

        print(f"Loaded {n_ev} waveforms for board {board}")

        # Prepare multiprocessing inputs
        args6 = [(wf6_all[i], THRESHOLD_LEVEL) for i in range(n_ev)]
        args7 = [(wf7_all[i], THRESHOLD_LEVEL) for i in range(n_ev)]

        ncpu = cpu_count()
        print(f"Using {ncpu} CPU cores...")

        # Run multiprocessing
        with Pool(processes=ncpu) as pool:
            t6 = np.array(pool.map(_t50_worker, args6), dtype=np.float32)
            t7 = np.array(pool.map(_t50_worker, args7), dtype=np.float32)

        # Compute Δt50
        mask = ~np.isnan(t6) & ~np.isnan(t7)
        delta_t = t6[mask] - t7[mask]


        # ---------------------------------------------------------
        # Select 10 zero-peak events and 10 valid Δt50 events
        # ---------------------------------------------------------

        zero_mask = ((np.isnan(t6) | (t6 == 0)) | (np.isnan(t7) | (t7 == 0)))
        zero_indices = np.where(zero_mask)[0][:10]

        nonzero_indices = np.where(valid)[0][:10]

        # Prepare subdirectories
        zero_dir = os.path.join(OUTPUT_DIR, f"Board{board}", "zero_peaks")
        nonzero_dir = os.path.join(OUTPUT_DIR, f"Board{board}", "nonzero_peaks")
        os.makedirs(zero_dir, exist_ok=True)
        os.makedirs(nonzero_dir, exist_ok=True)

        def plot_waveform_pair(idx, outdir, label):
            """Plots both waveforms for an event, highlighting fit region if available."""
            wf6 = wf6_all[idx]
            wf7 = wf7_all[idx]

            # baseline subtract like inside T50 function
            wf6_bs = -(wf6 - np.median(wf6[:200]))
            wf7_bs = -(wf7 - np.median(wf7[:200]))

            t_axis = np.arange(len(wf6)) * SAMPLE_TIME_NS

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

            ax.plot(t_axis, wf6_bs, label="Ch6", alpha=0.8)
            ax.plot(t_axis, wf7_bs, label="Ch7", alpha=0.8)

            # Add T50 markers if nonzero
            if not (np.isnan(t6[idx]) or t6[idx] == 0):
                ax.axvline(t6[idx], color='b', linestyle='--', alpha=0.8, label="T50 Ch6")

            if not (np.isnan(t7[idx]) or t7[idx] == 0):
                ax.axvline(t7[idx], color='r', linestyle='--', alpha=0.8, label="T50 Ch7")

            ax.set_title(f"{label} — Event {idx}")
            ax.set_xlabel("Time [ns]")
            ax.set_ylabel("Amplitude [a.u.]")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()

            plt.savefig(os.path.join(outdir, f"event_{idx}.png"))
            plt.close()


        # ---------------------------------------------------------
        # Generate the 10 + 10 plots
        # ---------------------------------------------------------
        print(f"Board {board}: Plotting {len(zero_indices)} zero-peak events...")
        for idx in zero_indices:
            plot_waveform_pair(idx, zero_dir, "Zero peak (T50 = 0)")

        print(f"Board {board}: Plotting {len(nonzero_indices)} nonzero-peak events...")
        for idx in nonzero_indices:
            plot_waveform_pair(idx, nonzero_dir, "Good peak (Δt50 ≠ 0)")


        print(f"Board {board}: {mask.sum()} valid Δt50 events")

        if mask.sum() == 0:
            print("No valid events. Skipping.")
            continue

        # --------------------------------------------------------
        # Histogram & Gaussian Fit (only non-zero Δt50 values)
        # --------------------------------------------------------

        # extract only valid Δt values (non-zero peaks)
        delta_valid = delta_t[valid]

        # Fit a gaussian ONLY to good events
        mu, sigma = norm.fit(delta_valid)

        plt.figure(figsize=(8, 6))

        # Histogram normalized to probability density
        plt.hist(delta_valid, bins=2000, density=True,
                histtype="step", linewidth=1.4, alpha=0.8, label="Δt50 (valid)")

        # Generate Gaussian curve
        x = np.linspace(delta_valid.min(), delta_valid.max(), 400)
        plt.plot(x, norm.pdf(x, mu, sigma),
                "k--", linewidth=1.5,
                label=f"Gaussian Fit\nμ = {mu:.3f} ns\nσ = {sigma:.3f} ns")

        plt.xlabel("Δt50 = t50(Ch6) - t50(Ch7) [ns]")
        plt.ylabel("Probability Density")
        plt.title(f"Board {board} — Δt50 Distribution (Valid Only)\nN = {len(delta_valid)}")
        plt.legend()

        plt.xlim(-1, 1)
        plt.tight_layout()

        outfile = os.path.join(OUTPUT_DIR, f"Board{board}_Delta_t50_MP.png")
        plt.savefig(outfile)
        plt.close()

        print(f"Board {board}: DONE — saved Δt50 histogram.")
        print(f"Mean = {mu:.3f} ns   |   Sigma = {sigma:.3f} ns   |   N = {len(delta_valid)}")
