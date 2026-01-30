#!/usr/bin/env python3
import os
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ================= USER SETTINGS =================
ANA_FILE  = "TRUE-HGtiming/skimmed_files/run1513_250928194230_TimingDAQ_postaskim_allchannels_newmethod.root"
TREE_NAME = "EventTree"

OUTDIR = "./TRUE-HGtiming/3mmplots_histonly/90deg_calibration"
os.makedirs(OUTDIR, exist_ok=True)

BOARDS = [0, 1, 2, 3]
NG = 4
NC = 9

NBINS = 200
CUT_MIN = 1.0
MIN_ENTRIES = 200
MIN_RAW = 500


# ---------------- CHANNEL MASKS ----------------
def _base_ok(g, c):
    if c == 8:
        return False
    if g == 3 and c in (6, 7):
        return False
    return True

def _ok(g, c, parity):
    if not _base_ok(g, c):
        return False
    return (c % 2 == 1) if parity == "odd" else (c % 2 == 0)

def _xlabel():
    return r"$|t_{\mathrm{final}}|$ [ns]"

def _prep(arr, xlim):
    if arr.size < MIN_RAW:
        return None
    arr = np.abs(arr)
    arr = arr[arr >= CUT_MIN]
    if arr.size < MIN_ENTRIES:
        return None
    arr = arr[(arr >= xlim[0]) & (arr <= xlim[1])]
    if arr.size < 50:
        return None
    return arr

def _code(b, g, c):
    # "bgc" with each as a single digit -> e.g. 0,0,0 => "000"
    return f"{b}{g}{c}"

def _branch(b, g, c):
    return f"tfinal_Board{b}_Group{g}_Channel{c}"


# ---------------- 2×2 BOARDS (overlay channels per board) ----------------
def make_boards(parity, label, xlim):
    out = f"{OUTDIR}/HISTONLY_{label}_Boards_vertical.pdf"
    bins = np.linspace(xlim[0], xlim[1], NBINS + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    with uproot.open(ANA_FILE) as f, PdfPages(out) as pdf:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        fig, axes = plt.subplots(2, 2, figsize=(8, 14), sharex=True)
        axes = axes.flatten()

        for ax, b in zip(axes, BOARDS):
            for g in range(NG):
                for c in range(NC):
                    if not _ok(g, c, parity):
                        continue
                    k = _branch(b, g, c)
                    if k not in keys:
                        continue

                    arr = _prep(tree[k].array(library="np"), xlim)
                    if arr is None:
                        continue

                    h, _ = np.histogram(arr, bins=bins)
                    ax.step(centers, h, where="mid", label=_code(b, g, c))

            ax.set_xlim(*xlim)
            ax.set_ylabel("Events")
            ax.legend(fontsize=6, ncol=4, frameon=False, title=None)

        for ax in axes:
            ax.set_xlabel(_xlabel())

        # pack tighter
        fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.06, hspace=0.15, wspace=0.12)
        pdf.savefig(fig)
        plt.close(fig)

    print("Saved:", out)


# ---------------- 16 SUBPLOTS (overlay channels per board-group) ----------------
def make_16(parity, label, xlim):
    out = f"{OUTDIR}/HISTONLY_{label}_16Subplots_vertical.pdf"
    bins = np.linspace(xlim[0], xlim[1], NBINS + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    layout = []
    for g in range(NG):
        layout.append((0, g, 2, g))
    for g in range(NG):
        layout.append((1, g, 3, g))

    with uproot.open(ANA_FILE) as f, PdfPages(out) as pdf:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        fig, axes = plt.subplots(8, 2, figsize=(9, 28), sharex=True)

        for r, (bL, gL, bR, gR) in enumerate(layout):
            for cidx, (b, g) in enumerate([(bL, gL), (bR, gR)]):
                ax = axes[r, cidx]

                for ch in range(NC):
                    if not _ok(g, ch, parity):
                        continue
                    k = _branch(b, g, ch)
                    if k not in keys:
                        continue

                    arr = _prep(tree[k].array(library="np"), xlim)
                    if arr is None:
                        continue

                    h, _ = np.histogram(arr, bins=bins)
                    ax.step(centers, h, where="mid", label=_code(b, g, ch))

                ax.set_xlim(*xlim)
                ax.set_ylabel("Events")
                ax.legend(fontsize=6, ncol=4, frameon=False, title=None)

        for ax in axes[-1]:
            ax.set_xlabel(_xlabel())

        # remove titles entirely (you asked)
        # pack tighter
        fig.subplots_adjust(left=0.06, right=0.99, top=0.995, bottom=0.03, hspace=0.22, wspace=0.10)
        pdf.savefig(fig)
        plt.close(fig)

    print("Saved:", out)


# ---------------- QUARTZ / PLASTIC MOSAICS ----------------
# Define the exact "pattern" you gave as a rectangular grid of codes (or None for blanks).
# Codes are "bgc" (b=board, g=group, c=channel).
QUARTZ_GRID = [
    [None,  "002", None,  None],
    ["006", "004", "206", "204"],
    ["016", "014", "216", "214"],
    ["026", "024", "226", "224"],
    [None,  "030", None,  None],
    [None,  "034", None,  None],
    ["106", "104", "306", "304"],
    ["116", "114", "316", "314"],
    ["126", "124", "326", "324"],
    [None,  "134", None,  "334"],
]

PLASTIC_GRID = [
    [None,  "000", "202", "200"],
    ["012", "010", "212", "210"],
    ["022", "020", "222", "220"],
    ["032", None,  "232", "230"],
    ["102", "100", "302", "300"],
    ["112", "110", "312", "310"],
    ["122", "120", "322", "320"],
    ["132", "130", "332", "330"],
]

def _parse_code(code_str):
    # code_str like "214" -> b=2,g=1,c=4
    b = int(code_str[0])
    g = int(code_str[1])
    c = int(code_str[2])
    return b, g, c

def make_mosaic_hist(grid, label, xlim):
    out = f"{OUTDIR}/HISTONLY_{label}_mosaic.pdf"
    bins = np.linspace(xlim[0], xlim[1], NBINS + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    nrows = len(grid)
    ncols = max(len(r) for r in grid)

    with uproot.open(ANA_FILE) as f, PdfPages(out) as pdf:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.3 * nrows), sharex=True)
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        elif ncols == 1:
            axes = np.array([[ax] for ax in axes])

        for r in range(nrows):
            row = grid[r]
            for c in range(ncols):
                ax = axes[r, c]
                ax.set_axis_on()
                ax.set_xlim(*xlim)
                ax.set_ylabel("")

                if c >= len(row) or row[c] is None:
                    ax.axis("off")
                    continue

                code = row[c]
                b, g, ch = _parse_code(code)

                # enforce your existing base vetoes too
                if not _base_ok(g, ch):
                    ax.axis("off")
                    continue

                k = _branch(b, g, ch)
                if k not in keys:
                    ax.text(0.5, 0.5, f"{code}\n(missing)", ha="center", va="center", transform=ax.transAxes, fontsize=9)
                    continue

                arr = _prep(tree[k].array(library="np"), xlim)
                if arr is None:
                    ax.text(0.5, 0.5, f"{code}\n(no stats)", ha="center", va="center", transform=ax.transAxes, fontsize=9)
                    continue

                h, _ = np.histogram(arr, bins=bins)
                ax.step(centers, h, where="mid", label=code)

                # No titles; use legend as the "label"
                ax.legend(fontsize=8, frameon=False, loc="upper right", handlelength=1.0, borderaxespad=0.2)

                # cleaner ticks
                ax.tick_params(labelsize=8)

        # xlabels only on bottom row, for visible axes
        for ax in axes[-1, :]:
            if ax.has_data():
                ax.set_xlabel(_xlabel())

        # tight packing
        fig.subplots_adjust(left=0.05, right=0.995, top=0.995, bottom=0.04, hspace=0.18, wspace=0.08)
        pdf.savefig(fig)
        plt.close(fig)

    print("Saved:", out)

def make_mosaic_heatmap(grid, label, xlim):
    out = f"{OUTDIR}/HEATMAP_{label}_mean.pdf"

    nrows = len(grid)
    ncols = max(len(r) for r in grid)

    means = np.full((nrows, ncols), np.nan, dtype=float)

    with uproot.open(ANA_FILE) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        for r in range(nrows):
            row = grid[r]
            for c in range(ncols):
                if c >= len(row) or row[c] is None:
                    continue
                code = row[c]
                b, g, ch = _parse_code(code)

                if not _base_ok(g, ch):
                    continue

                k = _branch(b, g, ch)
                if k not in keys:
                    continue

                arr = _prep(tree[k].array(library="np"), xlim)
                if arr is None:
                    continue

                means[r, c] = float(arr.mean())

    with PdfPages(out) as pdf:
        fig, ax = plt.subplots(1, 1, figsize=(8, 1.0 + 0.6 * nrows))
        im = ax.imshow(means, origin="upper", aspect="equal")

        # annotate codes (and means if present)
        for r in range(nrows):
            row = grid[r]
            for c in range(ncols):
                if c >= len(row) or row[c] is None:
                    continue
                code = row[c]
                if np.isfinite(means[r, c]):
                    txt = f"{code}\n{means[r,c]:.2f}"
                else:
                    txt = f"{code}\n—"
                ax.text(c, r, txt, ha="center", va="center", fontsize=8)

        ax.set_xticks(range(ncols))
        ax.set_yticks(range(nrows))
        ax.set_xticklabels([""] * ncols)
        ax.set_yticklabels([""] * nrows)
        ax.tick_params(length=0)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Mean(|tfinal|) [ns]")

        ax.set_title(f"{label} mean(|tfinal|) map")
        fig.subplots_adjust(left=0.04, right=0.92, top=0.94, bottom=0.04)
        pdf.savefig(fig)
        plt.close(fig)

    print("Saved:", out)


# ---------------- MAIN ----------------
def main():
    # Your original SCI/CER sets
    make_boards("odd",  "SCI", (7.0, 14.0))
    make_16(   "odd",  "SCI", (7.0, 14.0))

    make_boards("even", "CER", (7.0, 14.0))
    make_16(   "even", "CER", (7.0, 14.0))

    # New: CER-Quartz
    make_mosaic_hist(QUARTZ_GRID, "CER-Quartz", (7.0, 14.0))
    make_mosaic_heatmap(QUARTZ_GRID, "CER-Quartz", (7.0, 14.0))

    # New: CER-Plastic
    make_mosaic_hist(PLASTIC_GRID, "CER-Plastic", (7.0, 14.0))
    make_mosaic_heatmap(PLASTIC_GRID, "CER-Plastic", (7.0, 14.0))

    print("All done.")

if __name__ == "__main__":
    main()
