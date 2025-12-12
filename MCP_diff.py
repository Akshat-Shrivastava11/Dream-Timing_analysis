#!/usr/bin/env python3
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# Configuration
# =====================================================
FILE = "/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1468_250927145556_TimingDAQ.root"
OUTPUT_DIR = "./MCP/Delta_t50_LinearFit"
SAMPLE_TIME_NS = 0.2     # 200 ps → 0.2 ns
MAX_EVENTS = 2000000

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# === LOGIC FROM DeltaT IMPLEMENTATION ===
# =====================================================

def find_baseline(event_info):
    """Baseline = mean of first 10% of samples."""
    try:
        baseline_region = np.asarray(event_info[: int(0.1 * len(event_info))])
        avg = np.mean(baseline_region)
        rms = np.std(baseline_region)
    except Exception:
        avg = 0
        rms = 99999
    return avg, rms


def fit_region(event_info, baseline):
    """
    Returns rising-edge region (90%→10%) of amplitude, same as original DeltaT.
    """
    try:
        event_info = np.asarray(event_info)
        if event_info.size == 0:
            return np.array([]), np.array([]), 0.0

        min_idx = np.argmin(event_info)
        min_val = event_info[min_idx]
        amplitude = baseline - min_val

        th_low = baseline - 0.1 * amplitude
        th_high = baseline - 0.9 * amplitude

        idxs = np.arange(len(event_info))
        mask = (
            (event_info >= th_high) &
            (event_info <= th_low) &
            (idxs < min_idx) &
            (np.abs(idxs - min_idx) < 10)
        )
        return event_info[mask], idxs[mask], amplitude

    except Exception:
        return np.array([]), np.array([]), 0.0


def fit_rising_edge(region, idxs):
    if len(region) < 2:
        return None, None
    return np.polyfit(idxs, region, 1)


def extract_true_time(slope, intercept, baseline, amplitude):
    """
    === T50 Extraction ===
    Solve: slope * t + intercept = baseline - 0.5 * amplitude
    """
    if slope is None:
        return np.nan

    target = baseline - 0.5 * amplitude
    try:
        t_samp = (target - intercept) / slope  # in samples
    except Exception:
        return np.nan

    return t_samp * SAMPLE_TIME_NS  # convert to ns


# =====================================================
# Multiprocessing worker (returns only T50 in ns)
# =====================================================

def t50_linearfit_worker(wf):
    wf = np.asarray(wf, dtype=np.float32)

    baseline, rms = find_baseline(wf)
    region, idxs, amplitude = fit_region(wf, baseline)

    if amplitude < 5 * rms:
        return np.nan  # below threshold

    slope, intercept = fit_rising_edge(region, idxs)
    t50 = extract_true_time(slope, intercept, baseline, amplitude)

    return t50


# =====================================================
# Detailed analysis for plotting/classification
# =====================================================

def analyze_waveform_for_plot(wf, t50):
    """
    Recompute all relevant quantities for plotting and classify status.
    status ∈ {'good', 'below_threshold', 'bad_fit'}
    """
    wf = np.asarray(wf, dtype=np.float32)
    baseline, rms = find_baseline(wf)
    region, idxs, amplitude = fit_region(wf, baseline)

    slope = intercept = None
    status = "good"

    if amplitude < 5 * rms:
        status = "below_threshold"
    else:
        if len(region) < 2:
            status = "bad_fit"
        else:
            slope, intercept = fit_rising_edge(region, idxs)
            if slope is None or np.isnan(t50):
                status = "bad_fit"

    return {
        "wf": wf,
        "baseline": baseline,
        "rms": rms,
        "amplitude": amplitude,
        "region": region,
        "idxs": idxs,
        "slope": slope,
        "intercept": intercept,
        "t50": t50,
        "status": status,
    }


# =====================================================
# Plotting: side-by-side Ch6 / Ch7
# =====================================================

def plot_event_pair(board, idx, info6, info7, outpath, main_title):
    """
    Side-by-side plot:
      Left  = Ch6
      Right = Ch7

    Draw:
      - baseline-subtracted waveform
      - 10% and 90% horizontal lines
      - rising-edge points
      - linear fit (if available)
      - vertical T50 line (if t50 is not NaN)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, info, ch_label in zip(axes, [info6, info7], ["Ch6", "Ch7"]):
        wf = info["wf"]
        baseline = info["baseline"]
        amplitude = info["amplitude"]
        region = info["region"]
        idxs = info["idxs"]
        slope = info["slope"]
        intercept = info["intercept"]
        t50 = info["t50"]
        status = info["status"]

        t_axis = np.arange(len(wf)) * SAMPLE_TIME_NS
        wf_bs = wf - baseline

        # Waveform
        ax.plot(t_axis, wf_bs, label="Waveform", alpha=0.7)

        # 10% & 90% horizontal lines
        if amplitude is not None and amplitude > 0 and not np.isnan(amplitude):
            y10 = -0.1 * amplitude   # baseline-subtracted
            y90 = -0.9 * amplitude
            ax.axhline(y10, color="gray", linestyle="--", linewidth=1,
                       label="10% level" if ch_label == "Ch6" else None)
            ax.axhline(y90, color="gray", linestyle="-.", linewidth=1,
                       label="90% level" if ch_label == "Ch6" else None)

        # Rising-edge region points
        if region is not None and len(region) > 0:
            ax.scatter(idxs * SAMPLE_TIME_NS, region - baseline,
                       s=12, color="orange",
                       label="Fit region" if ch_label == "Ch6" else None)

        # Linear fit
        if slope is not None and intercept is not None and len(idxs) > 0:
            xfit = idxs
            yfit = slope * xfit + intercept
            ax.plot(xfit * SAMPLE_TIME_NS, yfit - baseline,
                    "k--", linewidth=1.2,
                    label="Linear fit" if ch_label == "Ch6" else None)

        # Vertical T50 line (only if previously computed and not NaN)
        if t50 is not None and not np.isnan(t50):
            ax.axvline(t50, color="red", linestyle="--",
                       label="T50" if ch_label == "Ch6" else None)

        ax.set_title(f"{ch_label} ({status})")
        ax.set_xlabel("Time [ns]")
        ax.grid(True)

    axes[0].set_ylabel("Amplitude (baseline-subtracted)")

    # Build legend from left axis only
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=8, loc="best")

    fig.suptitle(main_title)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


# =====================================================
# Main processing per Board
# =====================================================

with uproot.open(FILE) as f:
    tree = f["EventTree"]
    board_fits = []

    for board in range(4):

        print("\n==============================")
        print(f"Processing Board {board} with LinearFit T50...")
        print("==============================")

        ch6 = f"DRS_Board{board}_Group3_Channel6"
        ch7 = f"DRS_Board{board}_Group3_Channel7"

        arrays = tree.arrays([ch6, ch7], library="np", entry_stop=MAX_EVENTS)
        wf6_all = arrays[ch6]
        wf7_all = arrays[ch7]
        n_ev = wf6_all.shape[0]

        print(f"Loaded {n_ev} waveforms")

        ncpu = cpu_count()
        print(f"Using {ncpu} CPU cores")

        # --- Parallel T50 calculation ---
        with Pool(processes=ncpu) as pool:
            t6 = np.array(
                list(tqdm(pool.imap(t50_linearfit_worker, wf6_all),
                          total=n_ev, desc=f"T50 Ch6 Board {board}")),
                dtype=np.float32
            )
            t7 = np.array(
                list(tqdm(pool.imap(t50_linearfit_worker, wf7_all),
                          total=n_ev, desc=f"T50 Ch7 Board {board}")),
                dtype=np.float32
            )

        # --- Detailed analysis for each waveform for plotting & classification ---
        info6_list = []
        info7_list = []

        for i in range(n_ev):
            info6_list.append(analyze_waveform_for_plot(wf6_all[i], t6[i]))
            info7_list.append(analyze_waveform_for_plot(wf7_all[i], t7[i]))

        # good event mask: both channels are "good"
        status_good_mask = np.array(
            [
                (info6_list[i]["status"] == "good") and (info7_list[i]["status"] == "good")
                for i in range(n_ev)
            ],
            dtype=bool,
        )

        usable_events = status_good_mask.sum()
        print(f"Board {board}: {usable_events}/{n_ev} usable events ({100.0*usable_events/n_ev:.2f}%)")

        # --- Directories for plots ---
        board_dir = os.path.join(OUTPUT_DIR, f"Board{board}")
        sample_good_dir = os.path.join(board_dir, "sample_good")
        failed_bt_dir = os.path.join(board_dir, "failed_waveforms", "below_threshold")
        failed_fit_dir = os.path.join(board_dir, "failed_waveforms", "bad_fit")

        os.makedirs(sample_good_dir, exist_ok=True)
        os.makedirs(failed_bt_dir, exist_ok=True)
        os.makedirs(failed_fit_dir, exist_ok=True)

        # =====================================================
        # Plot 10 GOOD events per board
        # =====================================================
        good_indices = np.where(status_good_mask)[0][:10]
        print(f"Board {board}: plotting {len(good_indices)} good sample events...")

        # for idx in good_indices:
        #     outpath = os.path.join(sample_good_dir, f"event_{idx}.png")
        #     plot_event_pair(
        #         board,
        #         idx,
        #         info6_list[idx],
        #         info7_list[idx],
        #         outpath,
        #         main_title=f"Board {board} — GOOD Event {idx}",
        #     )

        # =====================================================
        # Plot FAILED events (all) into separate folders
        # =====================================================
        print(f"Board {board}: plotting failed events...")

        # for idx in range(n_ev):
        #     if status_good_mask[idx]:
        #         continue

        #     s6 = info6_list[idx]["status"]
        #     s7 = info7_list[idx]["status"]

        #     if ("below_threshold" in (s6, s7)):
        #         fail_dir = failed_bt_dir
        #     else:
        #         fail_dir = failed_fit_dir

        #     outpath = os.path.join(fail_dir, f"event_{idx}.png")
        #     plot_event_pair(
        #         board,
        #         idx,
        #         info6_list[idx],
        #         info7_list[idx],
        #         outpath,
        #         main_title=f"Board {board} — FAILED Event {idx}",
        #     )

        # =====================================================
        # Δt50 histogram and Gaussian fit (using only 'good' events)
        # =====================================================
        delta_t = t6[status_good_mask] - t7[status_good_mask]

        delta_valid = delta_t[
            (~np.isnan(delta_t)) &
            (~np.isinf(delta_t)) &
            (np.abs(delta_t) < 5.0)
        ]

        if len(delta_valid) == 0:
            print(f"Board {board}: no valid Δt after cleaning, skipping.")
            continue

        fit_data = delta_valid[delta_valid > 0.05]
        if len(fit_data) < 50:
            print(f"Board {board}: not enough entries for fit, skipping.")
            continue

        low = np.percentile(delta_valid, 1)
        high = np.percentile(delta_valid, 99)
        bin_width = 0.01
        bins = np.arange(low, high + bin_width, bin_width)

        counts, edges = np.histogram(delta_valid, bins=bins)
        mode = edges[np.argmax(counts)]

        fit_window = 0.20
        core = fit_data[(fit_data > mode - fit_window) & (fit_data < mode + fit_window)]
        if len(core) < 20:
            print(f"Board {board}: not enough entries in core window, skipping fit.")
            continue

        mu, sigma = norm.fit(core)
        print(f"Board {board}: Fit μ = {mu:.3f} ns, σ = {sigma:.3f} ns")

        # Plot histogram + fit
        plt.figure(figsize=(8, 6))
        plt.hist(delta_valid, bins=bins, histtype="step", color="red", label="Δt50 (Linear Fit)")

        x = np.linspace(mode - fit_window, mode + fit_window, 500)
        gauss = counts.max() * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        plt.plot(x, gauss, "k--", label=f"Gaussian Fit\nμ={mu:.3f} ns, σ={sigma:.3f} ns , FWHM={2.355*sigma:.3f} ns")
        plt.xlim(0.5, 1.5)
        plt.xlabel("Δt50 [ns]")
        plt.ylabel("Entries")
        plt.title(f"Board {board} — Δt50 (Linear-Fit T50)")
        plt.legend()
        plt.tight_layout()

        outfile = os.path.join(OUTPUT_DIR, f"Board{board}_Delta_t50_LinearFit.pdf")
        plt.savefig(outfile)
        plt.close()

        # For multi-board comparison
        board_fits.append(
            {
                "board": board,
                "x": x.copy(),
                "y": gauss.copy(),
                "mu": mu,
                "sigma": sigma,
            }
        )

# =====================================================
# Final comparison across boards
# =====================================================

if len(board_fits) > 0:
    plt.figure(figsize=(10, 6))
    for bf in board_fits:
        plt.plot(bf["x"], bf["y"], linewidth=2,
                 label=f"Board {bf['board']} (μ={bf['mu']:.3f}, σ={bf['sigma']:.3f})")

    plt.xlabel("Δt50 [ns]")
    plt.ylabel("Entries (scaled)")
    plt.title("Comparison of Linear-Fit T50 Timing Across Boards")
    plt.legend()
    plt.tight_layout()

    out_all = os.path.join(OUTPUT_DIR, "AllBoards_T50_LinearFit_Comparison.pdf")
    plt.savefig(out_all)
    plt.close()

print("\nAll boards processed.")
