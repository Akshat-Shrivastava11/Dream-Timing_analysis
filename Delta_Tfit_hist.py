#!/usr/bin/env python3
import uproot
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================
# Configuration
# ============================
FILE = "/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1468_250927145556_TimingDAQ.root"
OUTPUT_DIR = "./MCP/Hist_delta_T"
SAMPLE_TIME_ps = 0.2
THRESHOLD_LEVEL = 12

os.makedirs(OUTPUT_DIR, exist_ok=True)
for board in range(4):
    os.makedirs(os.path.join(OUTPUT_DIR, f"Board{board}"), exist_ok=True)

# ============================
# T50 extraction (NO smoothing)
# ============================
def t50_firstpeak_threshold(wf, threshold_level=15):
    wf = np.asarray(wf, dtype=float)

    # Invert waveform
    wf = -wf

    # Baseline subtract
    baseline = np.median(wf[:200])
    wf_bs = wf - baseline

    # Threshold crossing
    above = np.where(wf_bs > threshold_level)[0]
    if len(above) == 0:
        return None, wf_bs, None

    # Split into contiguous groups
    splits = np.where(np.diff(above) > 3)[0] + 1
    groups = np.split(above, splits)

    # Pick first real peak
    first_peak_group = None
    for g in groups:
        if np.max(wf_bs[g]) >= threshold_level:
            first_peak_group = g
            break
    if first_peak_group is None:
        return None, wf_bs, None

    rising = first_peak_group
    peak = np.max(wf_bs[rising])

    # 10% and 90%
    ten_idx = rising[np.where(wf_bs[rising] >= 0.10 * peak)[0][0]]
    ninety_idx = rising[np.where(wf_bs[rising] >= 0.90 * peak)[0][0]]

    if ten_idx >= ninety_idx:
        ninety_idx = min(ten_idx + 1, len(wf_bs)-1)

    # 50% crossing
    t50_level = 0.5 * peak
    region = np.arange(ten_idx, ninety_idx + 1)
    y = wf_bs[region]

    idx = np.argmin(np.abs(y - t50_level))
    t50_idx = region[idx]

    return t50_idx * SAMPLE_TIME_ps, wf_bs, (ten_idx, ninety_idx, peak)

# ============================
# Gaussian fit (NumPy-only)
# ============================
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def fit_gaussian_numpy(bin_centers, counts):
    A0 = np.max(counts)
    mu0 = np.sum(bin_centers * counts) / np.sum(counts)
    sigma0 = np.sqrt(np.sum(counts * (bin_centers - mu0)**2) / np.sum(counts))

    def residual(params):
        A, mu, sigma = params
        return np.sum((counts - gaussian(bin_centers, A, mu, sigma))**2)

    A, mu, sigma = A0, mu0, sigma0
    for _ in range(400):  # coordinate descent
        for i, p in enumerate([A, mu, sigma]):
            for d in [1e-3, -1e-3]:
                trial = [A, mu, sigma]
                trial[i] += d
                if residual(trial) < residual([A, mu, sigma]):
                    A, mu, sigma = trial
    return A, mu, abs(sigma)


# ============================
# PROCESS ALL EVENTS
# ============================
with uproot.open(FILE) as f:
    tree = f["EventTree"]
    combined_results = []
    for board in range(4):
        print(f"\nProcessing Board {board} ...")
        channels = [
            f"DRS_Board{board}_Group3_Channel6",
            f"DRS_Board{board}_Group3_Channel7"
        ]

        t50_map = {ch: [] for ch in channels}

        # iterate over ALL events
        for arrays in tree.iterate(channels, step_size=500):
            ch6_arr = arrays[channels[0]]
            ch7_arr = arrays[channels[1]]

            for i in range(len(ch6_arr)):
                wf6 = ch6_arr[i]
                wf7 = ch7_arr[i]

                t50_6, _, _ = t50_firstpeak_threshold(wf6)
                t50_7, _, _ = t50_firstpeak_threshold(wf7)

                t50_map[channels[0]].append(t50_6)
                t50_map[channels[1]].append(t50_7)

        # compute Δt50
        t6 = np.array([v if v is not None else np.nan for v in t50_map[channels[0]]])
        t7 = np.array([v if v is not None else np.nan for v in t50_map[channels[1]]])
        mask = ~np.isnan(t6) & ~np.isnan(t7)
        delta_t = (t6 - t7)[mask]

        print(f"Board {board}: valid Δt50 = {len(delta_t)} events")

        if len(delta_t) == 0:
            continue

        # Histogram + Gaussian fit
        fig, ax = plt.subplots(figsize=(8,6))
        counts, bips, _ = ax.hist(delta_t, bins=1000, histtype="step", label="Δt50",color='black')
        centers = 0.5*(bips[:-1] + bips[1:])
        
        A, mu, sigma = fit_gaussian_numpy(centers, counts)
        fwhm = 2.355 * sigma

        xfit = np.linspace(min(delta_t), max(delta_t), 400)
        yfit = gaussian(xfit, A, mu, sigma)
        ax.plot(xfit, yfit, 'k-', lw=2, color = 'red',
                label=(f"Gaussian Fit\n"
                       f"μ = {mu:.3f} ns\n"
                       f"σ = {sigma:.3f} ns\n"
                       f"FWHM = {fwhm:.3f} ns"))

        ax.set_xlabel("Δt50 [ps]")
        ax.set_ylabel("Counts")
        ax.set_title(f"Board {board} Δt50 Histogram")
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"Board{board}_Delta_t50_hist.png"))
        plt.close()

        print(f"Board {board}: fit → μ={mu:.3f} ps, σ={sigma:.3f} ps")
        combined_results.append({"board": board,
            "delta_t": delta_t,
            "A": A,
            "mu": mu,
            "sigma": sigma
        })
    # ============================
    # Combined Δt50 Plot (all boards)
    # ============================
    if len(combined_results) > 0:
        fig, ax = plt.subplots(figsize=(9,6))

        colors = ['C0', 'C1', 'C2', 'C3']

        for i, res in enumerate(combined_results):
            board = res["board"]
            dt    = res["delta_t"]
            A     = res["A"]
            mu    = res["mu"]
            sigma = res["sigma"]
            fwhm  = 2.355 * sigma
            
            # Histogram
            # counts, bins, _ = ax.hist(
            #      dt,
            #      bins=50,
            #      histtype="step",
            #      color=colors[i],
            #      linewidth=1.8,
            #      label=f"Board {board}: μ={mu:.3f}, σ={sigma:.3f}, FWHM={fwhm:.3f}"
            #  )

            # # Gaussian curve
            # centers = 0.5 * (bins[:-1] + bins[1:])
            xfit = np.linspace(min(dt), max(dt), 400)
            yfit = gaussian(xfit, A, mu, sigma)
            ax.plot(xfit, yfit, color=colors[i], linestyle='--',label=f"Board {board}: μ={mu:.3f}, σ={sigma:.3f}, FWHM={fwhm:.3f}")

        ax.set_xlabel("Δt50 [ns]")
        ax.set_ylabel("Counts")
        ax.set_title("Δt50 Comparison — All Boards")
        ax.grid(True)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "AllBoards_Delta_t50_Combined.png"))
        plt.close()

        print("\nSaved combined histogram: AllBoards_Delta_t50_Combined.png")

