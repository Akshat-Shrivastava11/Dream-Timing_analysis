#!/usr/bin/env python3
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
from matplotlib.ticker import AutoMinorLocator
warnings.filterwarnings("ignore")

# =====================================================
# Configuration
# =====================================================

positron_files = {
    'run1527_250929001555.root': '100',
    'run1411_250925154340.root': '120',
    'run1422_250926102502.root': '110',
    'run1409_250925135843.root': '80',
    'run1526_250928235028.root': '60',
    'run1525_250928232144.root': '40',
    'run1416_250925230347.root': '30',
    'run1423_250926105310.root': '20',
    'run1424_250926124313.root': '10',
}

BASE_DIR = "/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT"
OUTPUT_DIR = "./MCP/Delta_t50_LinearFit_PositronCombined"
SAMPLE_TIME_NS = 0.2     # 200 ps → 0.2 ns
MAX_EVENTS = 20000000

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
# Plotting: side-by-side Ch6 / Ch7 (kept, still optional)
# =====================================================

def plot_event_pair(board, idx, info6, info7, outpath, main_title):
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
# Main processing across positron files
# =====================================================

# Collect Δt from ALL positron files, per board
board_delta_all = {board: [] for board in range(4)}

for fname, label in positron_files.items():
    filepath = os.path.join(BASE_DIR, fname)
    print("\n====================================================")
    print(f"Processing positron file: {filepath} (label {label})")
    print("====================================================")

    with uproot.open(filepath) as f:
        tree = f["EventTree"]

        for board in range(4):

            print("\n------------------------------")
            print(f"File {fname} — Board {board} with LinearFit T50...")
            print("------------------------------")

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
                              total=n_ev, desc=f"{fname} T50 Ch6 Board {board}")),
                    dtype=np.float32
                )
                t7 = np.array(
                    list(tqdm(pool.imap(t50_linearfit_worker, wf7_all),
                              total=n_ev, desc=f"{fname} T50 Ch7 Board {board}")),
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
            print(f"File {fname} — Board {board}: {usable_events}/{n_ev} usable events "
                  f"({100.0*usable_events/n_ev:.2f}%)")

            # --- Directories for plots (per file/board if you want to enable them) ---
            board_dir = os.path.join(OUTPUT_DIR, f"{os.path.splitext(fname)[0]}", f"Board{board}")
            sample_good_dir = os.path.join(board_dir, "sample_good")
            failed_bt_dir = os.path.join(board_dir, "failed_waveforms", "below_threshold")
            failed_fit_dir = os.path.join(board_dir, "failed_waveforms", "bad_fit")

            os.makedirs(sample_good_dir, exist_ok=True)
            os.makedirs(failed_bt_dir, exist_ok=True)
            os.makedirs(failed_fit_dir, exist_ok=True)

            # =====================================================
            # (Optional) Plot 10 GOOD events per board per file
            # =====================================================
            # good_indices = np.where(status_good_mask)[0][:10]
            # for idx in good_indices:
            #     outpath = os.path.join(sample_good_dir, f"event_{idx}.png")
            #     plot_event_pair(
            #         board,
            #         idx,
            #         info6_list[idx],
            #         info7_list[idx],
            #         outpath,
            #         main_title=f"{fname} — Board {board} — GOOD Event {idx}",
            #     )

            # =====================================================
            # (Optional) Plot FAILED events per file
            # =====================================================
            # for idx in range(n_ev):
            #     if status_good_mask[idx]:
            #         continue
            #
            #     s6 = info6_list[idx]["status"]
            #     s7 = info7_list[idx]["status"]
            #
            #     if ("below_threshold" in (s6, s7)):
            #         fail_dir = failed_bt_dir
            #     else:
            #         fail_dir = failed_fit_dir
            #
            #     outpath = os.path.join(fail_dir, f"event_{idx}.png")
            #     plot_event_pair(
            #         board,
            #         idx,
            #         info6_list[idx],
            #         info7_list[idx],
            #         outpath,
            #         main_title=f"{fname} — Board {board} — FAILED Event {idx}",
            #     )

            # =====================================================
            # Δt50 for this file & board (only 'good' events)
            # =====================================================
            delta_t = t6[status_good_mask] - t7[status_good_mask]

            delta_valid = delta_t[
                (~np.isnan(delta_t)) &
                (~np.isinf(delta_t)) &
                (np.abs(delta_t) < 5.0)
            ]

            print(f"File {fname} — Board {board}: {len(delta_valid)} valid Δt entries after cleaning")

            # Store for global combination per board
            if len(delta_valid) > 0:
                board_delta_all[board].append(delta_valid)

# =====================================================
# Combined Δt50 histograms across ALL positron files
# =====================================================

board_fits = []

for board in range(4):
    if len(board_delta_all[board]) == 0:
        print(f"\nBoard {board}: no valid Δt across positron files, skipping.")
        continue

    delta_valid_all = np.concatenate(board_delta_all[board])
    print(f"\nBoard {board}: total combined valid Δt entries = {len(delta_valid_all)}")

    if len(delta_valid_all) == 0:
        print(f"Board {board}: no valid Δt after combination, skipping.")
        continue

    # Same selection for fit as before
    fit_data = delta_valid_all[delta_valid_all > 0.05]
    if len(fit_data) < 50:
        print(f"Board {board}: not enough entries for fit, skipping.")
        continue

    low = np.percentile(delta_valid_all, 1)
    high = np.percentile(delta_valid_all, 99)
    bin_width = 0.01
    bins = np.arange(low, high + bin_width, bin_width)

    counts, edges = np.histogram(delta_valid_all, bins=bins)
    mode = edges[np.argmax(counts)]

    fit_window = 0.20
    core = fit_data[(fit_data > mode - fit_window) & (fit_data < mode + fit_window)]
    if len(core) < 20:
        print(f"Board {board}: not enough entries in core window, skipping fit.")
        continue

    mu, sigma = norm.fit(core)
    print(f"Board {board} (combined positron files): Fit μ = {mu:.3f} ns, σ = {sigma:.3f} ns")

    # Plot histogram + fit for combined data
    plt.figure(figsize=(8, 6))
    plt.hist(
        delta_valid_all,
        bins=bins,
        histtype="step",
        color="red",
        label="Δt50 (Linear Fit, all positron files)"
    )

    # Gaussian overlay (CORRECT normalization)
    bin_width = bins[1] - bins[0]
    N_fit = len(core)

    x = np.linspace(bins[0], bins[-1], 1000)
    gauss = N_fit * bin_width * norm.pdf(x, mu, sigma)

    plt.plot(
        x, gauss, "k--",
        label=(f"Gaussian fit (core)\n"
            f"μ={mu:.3f} ns, σ={sigma:.3f} ns, "
            f"FWHM={2.355*sigma:.3f} ns")
    )

    plt.xlim(0, 1.5)
    ax = plt.gca()
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=6)
    ax.tick_params(which="minor", length=3)
    plt.xlabel("Δt50 [ns]")
    plt.ylabel("Entries")
    plt.title(f"Board {board} — Δt50 (Linear-Fit T50, positrons)")
    plt.legend()
    plt.tight_layout()


    outfile = os.path.join(OUTPUT_DIR, f"Board{board}_Delta_t50_LinearFit_CombinedPositrons.pdf")
    plt.savefig(outfile)
    plt.close()

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
# Final comparison across boards (combined positron data)
# =====================================================

if len(board_fits) > 0:
    plt.figure(figsize=(10, 6))
    for bf in board_fits:
        plt.plot(bf["x"], bf["y"], linewidth=2,
                 label=f"Board {bf['board']} (μ={bf['mu']:.3f}, σ={bf['sigma']:.3f})")

    plt.xlabel("Δt50 [ns]")
    plt.ylabel("Entries (scaled)")
    plt.title("Comparison of Linear-Fit T50 Timing Across Boards\n(all positron files combined)")
    plt.legend()
    plt.tight_layout()
    plt.xlim(0,1.5) 
    
    ax = plt.gca()
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=6)
    ax.tick_params(which="minor", length=3)
    out_all = os.path.join(OUTPUT_DIR, "AllBoards_T50_LinearFit_CombinedPositrons_Comparison.pdf")
    plt.savefig(out_all)
    plt.close()

print("\nAll positron files processed and combined histograms created.")
