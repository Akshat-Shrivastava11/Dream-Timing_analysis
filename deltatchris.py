#!/usr/bin/env python3
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import os
import warnings
from matplotlib.ticker import AutoMinorLocator

warnings.filterwarnings("ignore")

# =====================================================
# Configuration
# =====================================================

INPUT_FILE = "/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1468_250927145556_TimingDAQ.root"
TREE_NAME  = "EventTree"

OUTPUT_DIR = "./Chris_t50/deltat_mcp_lp2_50"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOARDS = [0, 1, 2, 3]

# =====================================================
# Helpers
# =====================================================

def br(b, ch):
    return f"DRS_Board{b}_Group3_Channel{ch}_LP2_50"


def style_axes(ax):
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=6)
    ax.tick_params(which="minor", length=3)


def gaussian_main_peak_fit(data, min_entries=50):
    """
    Main-peak Gaussian fit:
    - keep only 0 < Δt < 1.5 ns
    - find mode
    - fit ±0.20 ns around mode
    """
    data = data[np.isfinite(data)]
    data = data[(data > 0.0) & (data < 1.5)]

    if len(data) < min_entries:
        return None

    # Histogram for mode finding (same philosophy as reference)
    bin_width = 0.01
    bins = np.arange(-1.5, 1.5 + bin_width, bin_width)
    counts, edges = np.histogram(data, bins=bins)
    mode = edges[np.argmax(counts)]

    # Core window around peak
    fit_window = 0.20
    core = data[(data > mode - fit_window) & (data < mode + fit_window)]

    if len(core) < min_entries:
        return None

    mu, sigma = norm.fit(core)
    return mu, sigma, core, bins


# =====================================================
# Load ROOT file
# =====================================================

print("Opening ROOT file")
with uproot.open(INPUT_FILE) as f:
    arrays = f[TREE_NAME].arrays(library="np")

print("File loaded")

# =====================================================
# MCP Δt LP2_50 plots
# =====================================================

print("\n=== MCP Δt (LP2_50, -1.5–1.5 ns window) === but plotting only 0–1.5 ns")

for b in tqdm(BOARDS, desc="Boards"):
    k6 = br(b, 6)
    k7 = br(b, 7)

    if k6 not in arrays or k7 not in arrays:
        print(f"Board {b}: missing MCP branches, skipping")
        continue

    dt = arrays[k6] - arrays[k7]
    dt = dt[np.isfinite(dt)]

    if len(dt) < 100:
        continue

    fit = gaussian_main_peak_fit(dt)

    plt.figure(figsize=(8, 6))
    # Restrict to physical window for plotting
    dt_plot = dt[(dt > -1.5) & (dt < 1.5)]

    # Consistent binning (same as T50 reference)
    low = np.percentile(dt_plot, 1)
    high = np.percentile(dt_plot, 99)
    bin_width = 0.01
    bins = np.arange(low, high + bin_width, bin_width)
    print(f"amoutn of bins: {len(bins)}")
    plt.hist(
        dt_plot,
        bins=bins,
        histtype="step",
        color="black",
        label="Δt MCP (LP2_50)"
    )


    if fit is not None:
        mu, sigma, core, bins = fit
        bin_width = bins[1] - bins[0]

        x = np.linspace(bins[0], bins[-1], 1000)
        gauss = len(core) * bin_width * norm.pdf(x, mu, sigma)

        plt.plot(
            x, gauss, "r--",
            label=(
                f"Gaussian fit (main peak)\n"
                f"μ = {mu:.3f} ns\n"
                f"σ = {sigma:.3f} ns\n"
                f"FWHM = {2.355*sigma:.3f} ns"
            )
        )
    plt.xlim(0,1.5)
    plt.xlabel("Δt MCP [ns]")
    plt.ylabel("Entries")
    plt.title(f"Board {b} — MCP Δt (LP2_50)")
    style_axes(plt.gca())
    plt.legend()
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, f"Board{b}_MCP_DeltaT_LP2_50.pdf")
    plt.savefig(out)
    plt.close()

print("\nDone.")
print("Plots written to:", OUTPUT_DIR)
