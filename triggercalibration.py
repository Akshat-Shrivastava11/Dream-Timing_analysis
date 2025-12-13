#!/usr/bin/env python3
import uproot
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# Configuration
# =====================================================
FILE = "/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1468_250927145556_TimingDAQ.root"
OUTPUT_DIR = "./MCP/T50_Channel8_Triggers"
SAMPLE_TIME_NS = 0.2
MAX_EVENTS = 2_000_000

# --- waveform QA options ---
PLOT_WAVEFORMS = True     # set False to disable
N_PLOT_EVENTS = 10        # number of random events per board/group
RNG_SEED = 12345

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# Trigger T50 extraction (edge-based, fixed window)
# =====================================================

def trigger_t50_linearfit(wf):
    wf = np.asarray(wf, dtype=np.float32)
    n = len(wf)
    if n < 10:
        return np.nan

    idxs = np.arange(n)

    # 1. Find falling edge (largest negative slope)
    d = np.diff(wf)
    edge_idx = np.argmin(d)

    if edge_idx < 3 or edge_idx > n - 4:
        return np.nan

    # 2. Local baseline (before edge)
    pre = wf[max(0, edge_idx - 20):edge_idx]
    if len(pre) < 5:
        return np.nan

    baseline = np.mean(pre)
    rms = np.std(pre)

    # 3. Local final level (after edge, before ringing)
    post = wf[edge_idx + 1:edge_idx + 20]
    if len(post) < 5:
        return np.nan

    final = np.mean(post)
    amp = baseline - final

    if amp < 5 * rms:
        return np.nan

    # 4. Fixed window around edge
    i0 = max(0, edge_idx - 2)
    i1 = min(n, edge_idx + 3)

    x = idxs[i0:i1]
    y = wf[i0:i1]

    if len(x) < 2:
        return np.nan

    # 5. Linear fit
    slope, intercept = np.polyfit(x, y, 1)

    # 6. Solve for T50
    target = baseline - 0.5 * amp
    t_samp = (target - intercept) / slope

    return t_samp * SAMPLE_TIME_NS


# =====================================================
# Optional waveform plotting (post-fit QA)
# =====================================================

def plot_trigger_waveforms(wf_all, board, group, outdir,
                           n_events=10, seed=0):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(wf_all), size=n_events, replace=False)

    fig, axes = plt.subplots(
        n_events, 1,
        figsize=(8, 1.8 * n_events),
        sharex=True
    )

    if n_events == 1:
        axes = [axes]

    for ax, idx in tqdm(
        zip(axes, indices),
        total=len(indices),
        desc=f"    QA plots Board {board} Group {group}",
        leave=False
    ):
        wf = np.asarray(wf_all[idx], dtype=np.float32)
        n = len(wf)
        t = np.arange(n) * SAMPLE_TIME_NS

        d = np.diff(wf)
        edge_idx = np.argmin(d)

        pre = wf[max(0, edge_idx - 20):edge_idx]
        post = wf[edge_idx + 1:edge_idx + 20]

        if len(pre) < 5 or len(post) < 5:
            continue

        baseline = np.mean(pre)
        final = np.mean(post)
        amp = baseline - final

        i0 = max(0, edge_idx - 2)
        i1 = min(n, edge_idx + 3)

        x = np.arange(n)[i0:i1]
        y = wf[i0:i1]

        if len(x) < 2:
            continue

        slope, intercept = np.polyfit(x, y, 1)
        target = baseline - 0.5 * amp
        t50 = (target - intercept) / slope * SAMPLE_TIME_NS

        ax.plot(t, wf, color="black", linewidth=1)
        ax.plot(
            x * SAMPLE_TIME_NS,
            slope * x + intercept,
            "r--", linewidth=1.2
        )
        ax.axvline(t50, color="blue", linestyle="--")

        ax.set_ylabel("ADC")
        ax.grid(True)

    axes[-1].set_xlabel("Time [ns]")

    fig.suptitle(
        f"Board {board} Group {group} Channel 8 — Trigger Waveforms",
        fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = os.path.join(
        outdir,
        f"Board{board}_Group{group}_Channel8_Waveforms.pdf"
    )
    fig.savefig(out)
    plt.close(fig)


# =====================================================
# Main
# =====================================================

with uproot.open(FILE) as f:
    tree = f["EventTree"]

    for board in range(4):
        print(f"\nProcessing Board {board}")
        plt.figure(figsize=(8, 6))

        for group in range(4):
            branch = f"DRS_Board{board}_Group{group}_Channel8"
            print(f"  Group {group} → {branch}")

            wf_all = tree.arrays(
                [branch],
                library="np",
                entry_stop=MAX_EVENTS
            )[branch]

            # Optional waveform QA
            if PLOT_WAVEFORMS:
                plot_trigger_waveforms(
                    wf_all,
                    board=board,
                    group=group,
                    outdir=OUTPUT_DIR,
                    n_events=N_PLOT_EVENTS,
                    seed=RNG_SEED + 10 * board + group
                )

            # T50 extraction with progress bar
            with Pool(cpu_count()) as pool:
                t50 = np.array(
                    list(
                        tqdm(
                            pool.imap(trigger_t50_linearfit, wf_all),
                            total=len(wf_all),
                            desc=f"    T50 Board {board} Group {group}"
                        )
                    ),
                    dtype=np.float32
                )

            # Clean absolute trigger time
            t50 = t50[
                (~np.isnan(t50)) &
                (~np.isinf(t50)) &
                (t50 > 0) &
                (t50 < 300)
            ]

            print(f"    {len(t50)} valid T50 entries")

            if len(t50) > 0:
                plt.hist(
                    t50,
                    bins=200,
                    histtype="step",
                    linewidth=1.5,
                    label=f"Group {group}"
                )

        plt.xlabel("Trigger T50 [ns]")
        plt.ylabel("Entries")
        plt.title(f"Board {board} — Channel 8 Trigger T50")
        plt.legend()
        plt.tight_layout()

        out = os.path.join(
            OUTPUT_DIR,
            f"Board{board}_Channel8_T50_Triggers.pdf"
        )
        plt.savefig(out)
        plt.close()

print("\nAll boards processed.")
