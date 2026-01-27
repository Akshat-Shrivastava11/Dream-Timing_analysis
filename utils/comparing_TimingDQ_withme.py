import os
import numpy as np
import uproot
import matplotlib.pyplot as plt

# -------------------------
# INPUTS
# -------------------------
chris_file_path = "/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1468_250927145556_TimingDAQ.root"
my_file_path    = "/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitskims/run1468_250927145556_converted_timingskim.root"

TREE = "EventTree"
OUTDIR = "Chris_t50/LP2_50_compare_side_by_side"
os.makedirs(OUTDIR, exist_ok=True)

ALL_CHANNELS = [
    "DRS_Board0_Group0_Channel0", "DRS_Board0_Group0_Channel1", "DRS_Board0_Group0_Channel2",
    "DRS_Board0_Group0_Channel3", "DRS_Board0_Group0_Channel4", "DRS_Board0_Group0_Channel5",
    "DRS_Board0_Group0_Channel6", "DRS_Board0_Group0_Channel7", "DRS_Board0_Group0_Channel8",
]

# -------------------------
# HELPERS
# -------------------------
def load_clean_vals(root_path: str, tree_name: str, branch: str):
    """Load branch as numpy, keep finite and >0 values."""
    tree = uproot.open(root_path)[tree_name]
    vals = tree[branch].array(library="np")
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    return vals

def summarize(x: np.ndarray) -> str:
    if x.size == 0:
        return "N=0"
    return (f"N={x.size}  mean={np.mean(x):.3f}  std={np.std(x):.3f}  "
            f"median={np.median(x):.3f}  p16={np.percentile(x,16):.3f}  p84={np.percentile(x,84):.3f}")

def plot_side_by_side(chris_vals, my_vals, branch, outdir, bins=80):
    """
    Two-panel plot:
      Left: Chris
      Right: Mine
    Uses a shared x-range and identical bins so shapes are directly comparable.
    """
    # Shared x-range (robust) from combined data
    allv = np.concatenate([chris_vals, my_vals]) if (chris_vals.size + my_vals.size) else np.array([])
    if allv.size == 0:
        print(f"[SKIP] {branch}: both empty after cleaning")
        return

    # robust x-lims to avoid long tails dominating
    lo = np.percentile(allv, 0.5)
    hi = np.percentile(allv, 99.5)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(allv)), float(np.max(allv))

    edges = np.linspace(lo, hi, bins + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)

    ax[0].hist(chris_vals, bins=edges, histtype="step", linewidth=1.8)
    ax[0].set_title("Chris (TimingDAQ)")
    ax[0].set_xlabel(branch)
    ax[0].set_ylabel("Entries")
    ax[0].grid(True, alpha=0.25)

    ax[1].hist(my_vals, bins=edges, histtype="step", linewidth=1.8)
    ax[1].set_title("Mine (Python)")
    ax[1].set_xlabel(branch)
    ax[1].grid(True, alpha=0.25)

    # Stats text
    ax[0].text(0.02, 0.98, summarize(chris_vals), transform=ax[0].transAxes,
               va="top", ha="left", fontsize=9)
    ax[1].text(0.02, 0.98, summarize(my_vals), transform=ax[1].transAxes,
               va="top", ha="left", fontsize=9)

    fig.suptitle(f"Side-by-side comparison: {branch}", y=1.05)
    fig.tight_layout()

    outpath = os.path.join(outdir, f"{branch}_side_by_side.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] {branch}")
    print("  Chris:", summarize(chris_vals))
    print("  Mine :", summarize(my_vals))
    print("  Saved:", outpath)

def plot_overlay(chris_vals, my_vals, branch, outdir, bins=80):
    """
    Optional: single-panel overlay (same bins). Nice for quick visual deltas.
    """
    allv = np.concatenate([chris_vals, my_vals]) if (chris_vals.size + my_vals.size) else np.array([])
    if allv.size == 0:
        return

    lo = np.percentile(allv, 0.5)
    hi = np.percentile(allv, 99.5)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(allv)), float(np.max(allv))

    edges = np.linspace(lo, hi, bins + 1)

    plt.figure(figsize=(8, 5))
    plt.hist(chris_vals, bins=edges, histtype="step", linewidth=1.8, label="Chris (TimingDAQ)")
    plt.hist(my_vals,    bins=edges, histtype="step", linewidth=1.8, label="Mine (Python)")
    plt.xlabel(branch)
    plt.ylabel("Entries")
    plt.title(f"Overlay comparison: {branch}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    outpath = os.path.join(outdir, f"{branch}_overlay.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print("  Overlay saved:", outpath)

# -------------------------
# RUN
# -------------------------
for channel in ALL_CHANNELS:
    branch = f"{channel}_LP2_50"
    print(f"\n=== {branch} ===")

    try:
        chris_vals = load_clean_vals(chris_file_path, TREE, branch)
    except Exception as e:
        print(f"[SKIP] Chris missing {branch} ({e})")
        continue

    try:
        my_vals = load_clean_vals(my_file_path, TREE, branch)
    except Exception as e:
        print(f"[SKIP] Mine missing {branch} ({e})")
        continue

    plot_side_by_side(chris_vals, my_vals, branch, OUTDIR, bins=80)
    # Uncomment if you also want overlays:
    # plot_overlay(chris_vals, my_vals, branch, OUTDIR, bins=80)
