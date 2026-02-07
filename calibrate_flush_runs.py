

#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ================= CONFIG =================
TREE_NAME = "EventTree"
NBINS = 200
XLIM = (7.0, 14.0)

MIN_RAW = 500
MIN_ENTRIES = 200

# ================= GRIDS =================
# ================= GRIDS =================
QUARTZ_GRID = [
    [None,"002",None,None],
    ["006","004","206","204"],
    ["016","014","216","214"],
    ["026","024","226","224"],
    [None,"030",None,None],
    [None,"034",None,None],
    ["106","104","306","304"],
    ["116","114","316","314"],
    ["126","124","326","324"],
    [None,"134",None,"334"],
]

PLASTIC_GRID = [
    [None,"000","202","200"],
    ["012","010","212","210"],
    ["022","020","222","220"],
    ["032",None,"232","230"],
    ["102","100","302","300"],
    ["112","110","312","310"],
    ["122","120","322","320"],
    ["132","130","332","330"],
]

SCI_GRID = [
    ["003","001","203","201"],
    ["007","005","207","205"],
    ["013","011","213","211"],
    ["017","015","217","215"],
    ["023","021","223","221"],
    ["027","025","227","225"],
    ["033","031","233","231"],
    [None,"035",None,"235"],
    ["103","101","303","301"],
    ["107","105","307","305"],
    ["113","111","313","311"],
    ["117","115","317","315"],
    ["123","121","323","321"],
    ["127","125","327","325"],
    ["133","131","333","331"],
    [None,"135",None,"335"],
]

FAMILIES = {
    "CER-Quartz": QUARTZ_GRID,
    "CER-Plastic": PLASTIC_GRID,
    "SCI": SCI_GRID,
}

# ================= HELPERS =================
def parse_code(code):
    return int(code[0]), int(code[1]), int(code[2])

def branch_name(b, g, c):
    return f"tfinal_Board{b}_Group{g}_Channel{c}"

def prep_array(arr):
    arr = np.abs(arr)
    arr = arr[np.isfinite(arr)]
    arr = arr[(arr >= XLIM[0]) & (arr <= XLIM[1])]
    if arr.size < MIN_ENTRIES:
        return None
    return arr

def hist_stats(arr, bins):
    h, edges = np.histogram(arr, bins=bins)
    if h.sum() == 0:
        return None
    mu = arr.mean()
    mode = 0.5 * (edges[np.argmax(h)] + edges[np.argmax(h)+1])
    return mu, mode, h

from scipy.optimize import curve_fit

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


def gaussian_peak_mean(arr, bins, window=0.4):
    """
    Fit a Gaussian to the peak of |tfinal| and return the fitted mean.
    Falls back to simple mean if fit fails.
    """
    h, edges = np.histogram(arr, bins=bins)
    centers = 0.5 * (edges[1:] + edges[:-1])

    imax = np.argmax(h)
    x_peak = centers[imax]

    mask = (centers > x_peak - window) & (centers < x_peak + window)
    x_fit = centers[mask]
    y_fit = h[mask]

    if len(x_fit) < 5:
        return arr.mean()  # fallback

    try:
        p0 = [y_fit.max(), x_peak, 0.15]
        popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0)
        mu_fit = popt[1]
        return float(mu_fit)
    except RuntimeError:
        return arr.mean()




# ================= CALIBRATION =================
def derive_family_calibration(root_file, grid):
    bins = np.linspace(*XLIM, NBINS+1)
    stats = {}
    arrays = {}   # <-- ADD THIS

    with uproot.open(root_file) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        for row in grid:
            for code in row:
                if code is None:
                    continue

                b, g, c = parse_code(code)
                k = branch_name(b, g, c)
                if k not in keys:
                    continue

                raw = tree[k].array(library="np")
                if raw.size < MIN_RAW:
                    continue

                arr = prep_array(raw)
                if arr is None:
                    continue

                # store the array so we can fit the anchor later
                arrays[(b, g, c)] = arr   # <-- ADD THIS

                # whatever stats you compute per channel
                mu = float(arr.mean())
                h, edges = np.histogram(arr, bins=bins)
                mode = float(0.5 * (edges[np.argmax(h)] + edges[np.argmax(h)+1])) if h.sum() else np.nan

                stats[(b, g, c)] = {"N": int(arr.size), "mu": mu, "mode": mode}

    if not stats:
        raise RuntimeError("No usable channels found for this family/grid.")

    # pick anchor by max N
    anchor_key, anchor_stats = max(stats.items(), key=lambda x: x[1]["N"])

    # define anchor_array properly (THIS FIXES YOUR ERROR)
    anchor_array = arrays[anchor_key]
    anchor_mu = gaussian_peak_mean(anchor_array, bins)

    shifts = {}
    for key, st in stats.items():
        shifts[key] = float(anchor_mu - st["mu"])

    return shifts, (anchor_key, {"mu": anchor_mu, "N": anchor_stats["N"]}), stats


# ================= PLOTTING =================
def mosaic_pre_post(root_file, grid, shifts, outname, title):
    bins = np.linspace(*XLIM, NBINS+1)
    centers = 0.5*(bins[1:] + bins[:-1])

    nrows = len(grid)
    ncols = max(len(r) for r in grid)

    with uproot.open(root_file) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.2*nrows), sharex=True, sharey=True)
        axes = np.atleast_2d(axes)

        for r,row in enumerate(grid):
            for c,code in enumerate(row):
                ax = axes[r,c]
                if code is None:
                    ax.axis("off")
                    continue

                b,g,ch = parse_code(code)
                k = branch_name(b,g,ch)
                if k not in keys:
                    ax.text(0.5,0.5,"missing",ha="center",va="center")
                    continue

                raw = prep_array(tree[k].array(library="np"))
                if raw is None:
                    ax.text(0.5,0.5,"nostats",ha="center",va="center")
                    continue

                mu_pre, _, h_pre = hist_stats(raw, bins)
                post = raw + shifts.get((b,g,ch), 0.0)
                mu_post, _, h_post = hist_stats(post, bins)

                ax.step(centers, h_pre, where="mid", label="PRE", lw=1)
                ax.step(centers, h_post, where="mid", label="POST", lw=1.5)
                ax.set_title(code, fontsize=9)
                ax.text(0.02,0.95,
                        f"μpre={mu_pre:.2f}\nμpost={mu_post:.2f}",
                        transform=ax.transAxes,
                        va="top",fontsize=8)

        fig.suptitle(title)
        fig.text(0.04,0.5,"Events",rotation=90,va="center")
        for ax in axes[-1]:
            ax.set_xlabel("|tfinal| [ns]")

        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.tight_layout(rect=[0.04,0.03,0.98,0.95])

        plt.savefig(outname)
        plt.close(fig)

def heatmap(root_file, grid, shifts, outname, quantity, apply_shift):
    nrows = len(grid)
    ncols = max(len(r) for r in grid)
    mat = np.full((nrows,ncols), np.nan)

    bins = np.linspace(*XLIM, NBINS+1)

    with uproot.open(root_file) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        for r,row in enumerate(grid):
            for c,code in enumerate(row):
                if code is None:
                    continue
                b,g,ch = parse_code(code)
                k = branch_name(b,g,ch)
                if k not in keys:
                    continue

                arr = prep_array(tree[k].array(library="np"))
                if arr is None:
                    continue

                if apply_shift:
                    arr = arr + shifts.get((b,g,ch),0.0)

                mu, mode, _ = hist_stats(arr, bins)
                mat[r,c] = mu if quantity=="mean" else mode

    fig,ax = plt.subplots(figsize=(10,0.7*nrows+1))
    im = ax.imshow(mat,aspect="auto")
    for r in range(nrows):
        for c in range(ncols):
            if np.isfinite(mat[r,c]):
                ax.text(c,r,f"{mat[r,c]:.2f}",ha="center",va="center",fontsize=8)
    fig.colorbar(im,label=quantity)
    ax.set_title(outname)
    plt.savefig(outname)
    plt.close(fig)

# ================= MAIN =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference",default= '/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1501_250928105227_converted_timingskim.root' )
    ap.add_argument("--test", default = '/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1511_250928180741_converted_timingskim.root')
    ap.add_argument("--outdir", default="/lustre/research/hep/akshriva/Dream-Timing/TRUE-HGtiming/calibration_studiesZ/MODE_CALIB_OUTPUT")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    calibrations = {}

    for fam,grid in FAMILIES.items():
        shifts, anchor, stats = derive_family_calibration(args.reference, grid)
        calibrations[fam] = shifts
        print(f"[{fam}] anchor={anchor[0]}  μ={anchor[1]['mu']:.4f}")

        # Reference run
        mosaic_pre_post(
            args.reference, grid, shifts,
            f"{args.outdir}/MOSAIC_{fam}_REF_PRE_POST.pdf",
            f"{fam} — Reference"
        )

        # Test run
        mosaic_pre_post(
            args.test, grid, shifts,
            f"{args.outdir}/MOSAIC_{fam}_TEST_PRE_POST.pdf",
            f"{fam} — Test"
        )

        for qty in ["mean","mode"]:
            heatmap(args.reference, grid, shifts,
                    f"{args.outdir}/HEATMAP_{fam}_REF_{qty}_PRE.pdf",
                    qty, apply_shift=False)
            heatmap(args.reference, grid, shifts,
                    f"{args.outdir}/HEATMAP_{fam}_REF_{qty}_POST.pdf",
                    qty, apply_shift=True)

            heatmap(args.test, grid, shifts,
                    f"{args.outdir}/HEATMAP_{fam}_TEST_{qty}_PRE.pdf",
                    qty, apply_shift=False)
            heatmap(args.test, grid, shifts,
                    f"{args.outdir}/HEATMAP_{fam}_TEST_{qty}_POST.pdf",
                    qty, apply_shift=True)

if __name__ == "__main__":
    main()
