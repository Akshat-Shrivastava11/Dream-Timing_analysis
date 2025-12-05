#!/usr/bin/env python3
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.stats import norm
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ============================
# Configuration
# ============================
FILE = "/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1468_250927145556_TimingDAQ.root"
OUTPUT_DIR = "./MCP/Delta_t50_Results3"
SAMPLE_TIME_NS = 0.2
MAX_EVENTS = 2000000
THRESHOLD_LEVEL = 25

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

    idx10 = np.searchsorted(rel_vals, 0.1 * peak_val)
    idx10 = min(idx10, len(first) - 1)
    t10 = first[idx10]

    idx90 = np.searchsorted(rel_vals, 0.9 * peak_val)
    idx90 = min(idx90, len(first) - 1)
    t90 = first[idx90]

    if t90 <= t10:
        t90 = min(t10 + 1, len(wf) - 1)

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
# Main processing
# =====================================================
with uproot.open(FILE) as f:
    tree = f["EventTree"]
    board_fits = []
    for board in range(4):

        print(f"\n==============================")
        print(f"Processing Board {board} with multiprocessing...")
        print(f"==============================")

        ch6 = f"DRS_Board{board}_Group3_Channel6"
        ch7 = f"DRS_Board{board}_Group3_Channel7"

        arrays = tree.arrays([ch6, ch7], library="np", entry_stop=MAX_EVENTS)
        wf6_all = arrays[ch6]
        wf7_all = arrays[ch7]
        n_ev = wf6_all.shape[0]

        print(f"Loaded {n_ev} waveforms for board {board}")

        args6 = [(wf6_all[i], THRESHOLD_LEVEL) for i in range(n_ev)]
        args7 = [(wf7_all[i], THRESHOLD_LEVEL) for i in range(n_ev)]

        ncpu = cpu_count()
        print(f"Using {ncpu} CPU cores...")

        # Multiprocessing with progress bars
        with Pool(processes=ncpu) as pool:
            t6 = np.array(
                list(tqdm(pool.imap(_t50_worker, args6),
                          total=n_ev, desc=f"Board {board} Ch6")),
                dtype=np.float32
            )
            t7 = np.array(
                list(tqdm(pool.imap(_t50_worker, args7),
                          total=n_ev, desc=f"Board {board} Ch7")),
                dtype=np.float32
            )

        # Δt50
        valid = ~np.isnan(t6) & ~np.isnan(t7)
        delta_t = t6[valid] - t7[valid]
        print(f"Board {board}: {valid.sum()} valid Δt50 events")

        # ---------------------------------------------------------
        # Zero peak and good peak selection
        # ---------------------------------------------------------
        zero_mask = (np.isnan(t6) | (t6 == 0)) | (np.isnan(t7) | (t7 == 0))
        zero_indices = np.where(zero_mask)[0][:10]
        nonzero_indices = np.where(valid)[0][:10]

        zero_dir = os.path.join(OUTPUT_DIR, f"Board{board}", "zero_peaks")
        nonzero_dir = os.path.join(OUTPUT_DIR, f"Board{board}", "nonzero_peaks")
        os.makedirs(zero_dir, exist_ok=True)
        os.makedirs(nonzero_dir, exist_ok=True)

        # Waveform plotting with x-lim for good peaks
        def plot_waveform_pair(idx, outdir, label):
            wf6 = wf6_all[idx]
            wf7 = wf7_all[idx]

            wf6_bs = -(wf6 - np.median(wf6[:200]))
            wf7_bs = -(wf7 - np.median(wf7[:200]))

            t_axis = np.arange(len(wf6)) * SAMPLE_TIME_NS

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(t_axis, wf6_bs, label="Ch6", alpha=0.8)
            ax.plot(t_axis, wf7_bs, label="Ch7", alpha=0.8)

            if not np.isnan(t6[idx]) and t6[idx] != 0:
                ax.axvline(t6[idx], color='b', linestyle='--', alpha=0.8)
            if not np.isnan(t7[idx]) and t7[idx] != 0:
                ax.axvline(t7[idx], color='r', linestyle='--', alpha=0.8)

            ax.set_title(f"{label} — Event {idx}")
            ax.set_xlabel("Time [ns]")
            ax.set_ylabel("Amplitude [a.u.]")
            ax.grid(True)
            ax.legend()

            if "Good peak" in label:
                ax.set_xlim(100, 150)

            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"event_{idx}.png"))
            plt.close()

        # print(f"Board {board}: Plotting zero-peak examples...")
        # for idx in zero_indices:
        #     plot_waveform_pair(idx, zero_dir, "Zero peak (T50 = 0)")

        # print(f"Board {board}: Plotting nonzero-peak examples...")
        # for idx in nonzero_indices:
        #     plot_waveform_pair(idx, nonzero_dir, "Good peak (Δt50 ≠ 0)")

       


        # --------------------------------------------------------
        # Histogram + Robust Gaussian Fit
        # --------------------------------------------------------

        delta_valid = delta_t                  # non-zero peak Δt only
        fit_data = delta_valid[delta_valid > 0.25]

        if len(fit_data) == 0:
            print(f"Board {board}: No Δt > 0.25 — skipping fit.")
            continue

        # ======================================================
        # Step 1: Make histogram so we can find mode
        # ======================================================
        bin_width = 0.01
        bins = np.arange(np.min(delta_valid), np.max(delta_valid) + bin_width, bin_width)

        counts, edges = np.histogram(delta_valid, bins=bins)

        # Histogram mode (less sensitive to outliers)
        mode_idx = np.argmax(counts)
        mode = 0.5 * (edges[mode_idx] + edges[mode_idx+1])

        # ======================================================
        # Step 2: Restrict fit to a tight window around the mode
        # ======================================================
        fit_window = 0.20  # ns; adjust if needed

        fit_window_mask = (fit_data > (mode - fit_window)) & (fit_data < (mode + fit_window))
        fit_core = fit_data[fit_window_mask]

        if len(fit_core) < 20:
            print(f"Board {board}: Not enough events in core window — skipping.")
            continue

        # Gaussian fit on the cleaned core
        mu, sigma = norm.fit(fit_core)

        # ======================================================
        # Plot histogram (counts)
        # ======================================================
        plt.figure(figsize=(8, 6))
        counts_full, edges_full, _ = plt.hist(
            delta_valid,
            bins=bins,
            color='red',
            histtype="step",
            linewidth=1.4,
            density=False,
            label="Δt50 (non-zero peaks)"
        )

        # ======================================================
        # Step 3: Plot Gaussian scaled to histogram peak
        # ======================================================
        x = np.linspace(mode - fit_window, mode + fit_window, 600)
        hist_peak = counts_full[(edges_full[:-1] >= mode - fit_window) & 
                                (edges_full[:-1] <  mode + fit_window)].max()

        gauss = hist_peak * np.exp(-0.5 * ((x - mu) / sigma)**2)
        board_fits.append({"board": board,"x": x.copy(),"y": gauss.copy(),"mu": mu,"sigma": sigma})


        plt.plot(
            x, gauss,
            "k--", linewidth=2,
            label=f"Gaussian Fit\nμ={mu:.3f} ns, σ={sigma:.3f} ns, FWHM={2.355*sigma:.3f} ns"
        )

        plt.xlabel("Δt50 = t50(Ch6) - t50(Ch7) [ns]")
        plt.ylabel("Entries")
        plt.title(
            f"Board {board} — Δt50 Distribution \n"
            #f"N={len(delta_valid)}, FitN={len(fit_core)}"
        )
        plt.legend()
        plt.xlim(0.4,1.4)
        plt.tight_layout()

        outfile = os.path.join(OUTPUT_DIR, f"Board{board}_Delta_t50_MP.pdf")
        plt.savefig(outfile)
        plt.close()

        print(f"Board {board}: Robust Gaussian FIT: mu={mu:.3f}, sigma={sigma:.3f}, N_fit={len(fit_core)}")




    # ==========================================================
    # FINAL COMPARISON PLOT FOR ALL BOARDS
    # ==========================================================
    if len(board_fits) > 0:
        plt.figure(figsize=(10, 6))

        for bf in board_fits:
            plt.plot(bf["x"], bf["y"], linewidth=2, label=f"Board {bf['board']} (μ={bf['mu']:.3f}, σ={bf['sigma']:.3f})")

        plt.xlabel("Δt50 [ns]")
        plt.ylabel("Entries (scaled)")
        plt.title("Comparison of Gaussian timing fits across all boards")
        plt.legend()
        plt.xlim(0.25, 2)
        plt.tight_layout()

        out_all = os.path.join(OUTPUT_DIR, "AllBoards_GaussianFits.pdf")
        plt.savefig(out_all)
        plt.close()

        print(f"\nSaved all-board comparison: {out_all}")
    else:
        print("\nNo fits to plot for all boards.")


    print("\nAll boards processed.")        