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
    return f"{b}{g}{c}"

def _branch(b, g, c):
    return f"tfinal_Board{b}_Group{g}_Channel{c}"

def _mode_from_hist(arr, bins):
    # mode = center of the max bin
    h, edges = np.histogram(arr, bins=bins)
    if h.sum() == 0:
        return np.nan, h
    imax = int(np.argmax(h))
    mode = 0.5 * (edges[imax] + edges[imax + 1])
    return float(mode), h


# ---------------- Styling helpers ----------------
def _global_ylabel(fig, text="Events"):
    fig.text(0.012, 0.5, text, va="center", rotation=90)

def _tight_layout(fig, left=0.06, right=0.995, top=0.995, bottom=0.045, hspace=0.18, wspace=0.08):
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace)


# ---------------- 2×2 BOARDS (overlay channels per board) ----------------
def make_boards(parity, label, xlim):
    out = f"{OUTDIR}/HISTONLY_{label}_Boards_vertical.pdf"
    bins = np.linspace(xlim[0], xlim[1], NBINS + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    with uproot.open(ANA_FILE) as f, PdfPages(out) as pdf:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        fig, axes = plt.subplots(2, 2, figsize=(8, 14), sharex=True, sharey=True)
        axes = axes.flatten()

        # First pass: compute max bin height across the whole figure
        global_ymax = 1
        cached = {b: [] for b in BOARDS}  # list of tuples (b,g,c, h)
        for b in BOARDS:
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
                    mode, h = _mode_from_hist(arr, bins=bins)
                    cached[b].append((g, c, arr, mode, h))
                    if h.max() > global_ymax:
                        global_ymax = int(h.max())

        # Second pass: draw
        for ax, b in zip(axes, BOARDS):
            for (g, c, arr, mode, h) in cached[b]:
                mu = float(arr.mean())
                sig = float(arr.std())
                # filled, red histogram (use precomputed h for consistent y max)
                ax.fill_between(
                    centers, h, step="mid", alpha=0.30
                )
                ax.step(
                    centers, h, where="mid", linewidth=1.0,
                    label=f"{_code(b,g,c)}  μ={mu:.2f}  m={mode:.2f}  σ={sig:.2f}"
                )

            ax.set_xlim(*xlim)
            ax.set_ylim(0, global_ymax * 1.05)

            # no per-axes ylabel; legend carries identification
            ax.legend(fontsize=6, ncol=1, frameon=False, title=None, loc="upper right")
            ax.tick_params(labelsize=9)

        for ax in axes:
            ax.set_xlabel(_xlabel())

        _global_ylabel(fig, "Events")
        _tight_layout(fig, left=0.07, right=0.995, top=0.995, bottom=0.05, hspace=0.12, wspace=0.08)

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

        fig, axes = plt.subplots(8, 2, figsize=(9, 28), sharex=True, sharey=True)

        # precompute histograms for all panels → global y max
        global_ymax = 1
        cache = {}  # (b,g) -> list of (ch, arr, mode, h)
        for r, (bL, gL, bR, gR) in enumerate(layout):
            for (b, g) in [(bL, gL), (bR, gR)]:
                if (b, g) in cache:
                    continue
                cache[(b, g)] = []
                for ch in range(NC):
                    if not _ok(g, ch, parity):
                        continue
                    k = _branch(b, g, ch)
                    if k not in keys:
                        continue
                    arr = _prep(tree[k].array(library="np"), xlim)
                    if arr is None:
                        continue
                    mode, h = _mode_from_hist(arr, bins=bins)
                    cache[(b, g)].append((ch, arr, mode, h))
                    if h.max() > global_ymax:
                        global_ymax = int(h.max())

        # draw
        for r, (bL, gL, bR, gR) in enumerate(layout):
            for cidx, (b, g) in enumerate([(bL, gL), (bR, gR)]):
                ax = axes[r, cidx]

                for (ch, arr, mode, h) in cache.get((b, g), []):
                    mu = float(arr.mean())
                    sig = float(arr.std())

                    ax.fill_between(centers, h, step="mid", alpha=0.30)
                    ax.step(
                        centers, h, where="mid", linewidth=1.0,
                        label=f"{_code(b,g,ch)}  μ={mu:.2f}  m={mode:.2f}  σ={sig:.2f}"
                    )

                ax.set_xlim(*xlim)
                ax.set_ylim(0, global_ymax * 1.05)
                ax.legend(fontsize=6, ncol=1, frameon=False, loc="upper right")
                ax.tick_params(labelsize=8)

        for ax in axes[-1]:
            ax.set_xlabel(_xlabel())

        _global_ylabel(fig, "Events")
        _tight_layout(fig, left=0.06, right=0.995, top=0.997, bottom=0.03, hspace=0.16, wspace=0.06)

        pdf.savefig(fig)
        plt.close(fig)

    print("Saved:", out)


# ---------------- QUARTZ / PLASTIC MOSAICS ----------------
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

# Combined CER grid you provided (normalized + obvious typo fixes)
CER_ALL_GRID = [
    ["002", "000", "202", "200"],
    ["006", "004", "206", "204"],
    ["012", "010", "212", "210"],
    ["016", "014", "216", "214"],
    ["022", "020", "222", "220"],
    ["026", "024", "226", "224"],
    ["032", "030", "232", "230"],
    [None,  "034", None,  "234"],
    ["102", "100", "302", "300"],
    ["106", "104", "306", "304"],  # fixed "1061004"
    ["112", "110", "312", "310"],
    ["116", "114", "316", "314"],
    ["122", "120", "322", "320"],
    ["126", "124", "326", "324"],
    ["132", "130", "332", "330"],
    [None,  "134", None,  "334"],
]

def _parse_code(code_str):
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

        # precompute all hists + stats to set global y max
        global_ymax = 1
        cell = {}  # (r,c) -> dict or None
        for r in range(nrows):
            row = grid[r]
            for c in range(ncols):
                if c >= len(row) or row[c] is None:
                    cell[(r, c)] = None
                    continue
                code = row[c]
                b, g, ch = _parse_code(code)

                if not _base_ok(g, ch):
                    cell[(r, c)] = {"code": code, "status": "veto"}
                    continue

                k = _branch(b, g, ch)
                if k not in keys:
                    cell[(r, c)] = {"code": code, "status": "missing"}
                    continue

                arr = _prep(tree[k].array(library="np"), xlim)
                if arr is None:
                    cell[(r, c)] = {"code": code, "status": "nostats"}
                    continue

                mode, h = _mode_from_hist(arr, bins=bins)
                mu = float(arr.mean())
                sig = float(arr.std())
                cell[(r, c)] = {"code": code, "status": "ok", "h": h, "mu": mu, "mode": mode, "sig": sig}

                if h.max() > global_ymax:
                    global_ymax = int(h.max())

        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.05 * nrows), sharex=True, sharey=True)
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        elif ncols == 1:
            axes = np.array([[ax] for ax in axes])

        for r in range(nrows):
            for c in range(ncols):
                ax = axes[r, c]
                ax.set_xlim(*xlim)
                ax.set_ylim(0, global_ymax * 1.05)
                ax.tick_params(labelsize=8)

                entry = cell.get((r, c))
                if entry is None:
                    ax.axis("off")
                    continue

                code = entry["code"]
                status = entry["status"]
                if status != "ok":
                    ax.text(0.5, 0.5, f"{code}\n({status})", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9)
                    continue

                h = entry["h"]
                ax.fill_between(centers, h, step="mid", alpha=0.30)
                ax.step(
                    centers, h, where="mid", linewidth=1.0,
                    label=f"{code}  μ={entry['mu']:.2f}  m={entry['mode']:.2f}  σ={entry['sig']:.2f}"
                )
                ax.legend(fontsize=7, frameon=False, loc="upper right", handlelength=1.0, borderaxespad=0.2)

        # xlabels only on bottom row
        for ax in axes[-1, :]:
            if ax.get_visible() and ax.axison:
                ax.set_xlabel(_xlabel())

        _global_ylabel(fig, "Events")
        _tight_layout(fig, left=0.05, right=0.995, top=0.995, bottom=0.035, hspace=0.14, wspace=0.05)

        pdf.savefig(fig)
        plt.close(fig)

    print("Saved:", out)

def make_mosaic_heatmaps(grid, label, xlim):
    """
    Produces a 2-page PDF:
      page 1: mean(|tfinal|)
      page 2: mode(|tfinal|) from histogram max bin
    Uses inferno colormap.
    """
    out = f"{OUTDIR}/HEATMAP_{label}_mean_mode.pdf"

    nrows = len(grid)
    ncols = max(len(r) for r in grid)

    means = np.full((nrows, ncols), np.nan, dtype=float)
    modes = np.full((nrows, ncols), np.nan, dtype=float)

    bins = np.linspace(xlim[0], xlim[1], NBINS + 1)

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
                mode, _ = _mode_from_hist(arr, bins=bins)
                modes[r, c] = float(mode)

    with PdfPages(out) as pdf:
        for mat, title, cbar_label in [
            (means, f"{label} mean(|tfinal|)", "Mean(|tfinal|) [ns]"),
            (modes, f"{label} mode(|tfinal|)", "Mode(|tfinal|) [ns]"),
        ]:
            fig, ax = plt.subplots(1, 1, figsize=(8.4, 1.0 + 0.55 * nrows))
            im = ax.imshow(mat, origin="upper", aspect="equal", cmap="viridis")

            # annotate codes + value
            for rr in range(nrows):
                row = grid[rr]
                for cc in range(ncols):
                    if cc >= len(row) or row[cc] is None:
                        continue
                    code = row[cc]
                    val = mat[rr, cc]
                    txt = f"{code}\n{val:.2f}" if np.isfinite(val) else f"{code}\n—"
                    ax.text(cc, rr, txt, ha="center", va="center", fontsize=8)

            ax.set_xticks(range(ncols))
            ax.set_yticks(range(nrows))
            ax.set_xticklabels([""] * ncols)
            ax.set_yticklabels([""] * nrows)
            ax.tick_params(length=0)

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(cbar_label)

            ax.set_title(title)
            fig.subplots_adjust(left=0.04, right=0.92, top=0.93, bottom=0.04)
            pdf.savefig(fig)
            plt.close(fig)

    print("Saved:", out)


# ---------------- MAIN ----------------
def main():
    # SCI/CER standard
    make_boards("odd",  "SCI", (7.0, 14.0))
    make_16(   "odd",  "SCI", (7.0, 14.0))

    make_boards("even", "CER", (7.0, 14.0))
    make_16(   "even", "CER", (7.0, 14.0))

    # Quartz + Plastic mosaics + heatmaps (mean+mode)
    make_mosaic_hist(QUARTZ_GRID, "CER-Quartz", (7.0, 14.0))
    make_mosaic_heatmaps(QUARTZ_GRID, "CER-Quartz", (8.0, 14.0))

    make_mosaic_hist(PLASTIC_GRID, "CER-Plastic", (7.0, 14.0))
    make_mosaic_heatmaps(PLASTIC_GRID, "CER-Plastic", (7.0, 14.0))

    # Combined CER heatmaps (mean+mode) using your mapping
    make_mosaic_heatmaps(CER_ALL_GRID, "CER-AllChannels", (7.0, 14.0))

    print("All done.")

if __name__ == "__main__":
    main()
