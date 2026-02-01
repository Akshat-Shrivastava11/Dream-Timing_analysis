#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ================= DEFAULTS (overridden by argparse) =================
ANA_FILE  = "TRUE-HGtiming/skimmed_files/run1513_250928194230_TimingDAQ_postaskim_allchannels_newmethod.root"
TREE_NAME = "EventTree"

NBINS = 200
CUT_MIN = 1.0
MIN_ENTRIES = 200
MIN_RAW = 500

BOARDS = [0, 1, 2, 3]
NG = 4
NC = 9

# Heatmap colormap: flipped so yellow=low, blue/purple=high
HEATMAP_CMAP = "viridis_r"

# Tight packing defaults
HSPACE = 0.10
WSPACE = 0.05

# ================= MOSAIC GRIDS =================
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

# Combined CER grid you provided
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
    ["106", "104", "306", "304"],
    ["112", "110", "312", "310"],
    ["116", "114", "316", "314"],
    ["122", "120", "322", "320"],
    ["126", "124", "326", "324"],
    ["132", "130", "332", "330"],
    [None,  "134", None,  "334"],
]

# Combined SCI grid you provided (normalized + obvious typo fixes)
SCI_ALL_GRID = [
    ["003", "001", "203", "201"],
    ["007", "005", "207", "205"],
    ["013", "011", "213", "211"],
    ["017", "015", "217", "215"],
    ["023", "021", "223", "221"],
    ["027", "025", "227", "225"],
    ["033", "031", "233", "231"],
    [None,  "035", None,  "235"],

    ["103", "101", "303", "301"],
    ["107", "105", "307", "305"],
    ["113", "111", "313", "311"],
    ["117", "115", "317", "315"],
    ["123", "121", "323", "321"],
    ["127", "125", "327", "325"],
    ["133", "131", "333", "331"],
    [None,  "135", None,  "335"],
]

# ================= HELPERS =================
# ================= HELPERS =================
def _infer_run_label(path: str) -> str:
    base = os.path.basename(path)

    # run#### part
    m_run = re.search(r"run(\d+)", base)
    run_part = f"run{m_run.group(1)}" if m_run else "runUnknown"

    # timestamp-ish part (11–12 digits after an underscore)
    # examples:
    #   run1513_250928194230_converted_timingskim.root
    #   run1513_250928194230_TimingDAQ_postaskim.root
    m_ts = re.search(r"_(\d{11,12})(?:_|\.|$)", base)
    ts_part = m_ts.group(1) if m_ts else None

    if ts_part:
        return f"{run_part}_{ts_part}_heatmap_and_hists"
    return f"{run_part}_heatmap_and_hists_"


def _default_outdir_for(ana_file: str) -> str:
    return os.path.join("./TRUE-HGtiming/2Dplots_for_timing", _infer_run_label(ana_file))

def _global_ylabel(fig, text="Events"):
    fig.text(0.010, 0.5, text, va="center", rotation=90)

def _xlabel():
    return r"$|t_{\mathrm{final}}|$ [ns]"

def _branch(b, g, c):
    return f"tfinal_Board{b}_Group{g}_Channel{c}"

def _code(b, g, c):
    return f"{b}{g}{c}"

def _parse_code(code_str):
    b = int(code_str[0])
    g = int(code_str[1])
    c = int(code_str[2])
    return b, g, c

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

def _mode_and_hist(arr, bins):
    h, edges = np.histogram(arr, bins=bins)
    if h.sum() == 0:
        return np.nan, h
    imax = int(np.argmax(h))
    mode = 0.5 * (edges[imax] + edges[imax + 1])
    return float(mode), h

def _legend_label(code, mu, mode, sig):
    return f"{code}  μ={mu:.2f}  m={mode:.2f}  σ={sig:.2f}"

def _tighten(fig, left=0.055, right=0.995, top=0.995, bottom=0.04, hspace=HSPACE, wspace=WSPACE):
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace)

# ================= PLOTS =================
def make_boards(parity, label, xlim, outdir):
    out = f"{outdir}/HISTONLY_{label}_Boards_vertical.pdf"
    bins = np.linspace(xlim[0], xlim[1], NBINS + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    with uproot.open(ANA_FILE) as f, PdfPages(out) as pdf:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        fig, axes = plt.subplots(2, 2, figsize=(8, 14), sharex=True, sharey=True)
        axes = axes.flatten()

        # precompute all hists to set a global y-max
        global_ymax = 1
        cache = {b: [] for b in BOARDS}  # list of (g,c, mu, mode, sig, h)
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
                    mu = float(arr.mean())
                    sig = float(arr.std())
                    mode, h = _mode_and_hist(arr, bins=bins)
                    cache[b].append((g, c, mu, mode, sig, h))
                    global_ymax = max(global_ymax, int(h.max()))

        # draw
        for ax, b in zip(axes, BOARDS):
            for (g, c, mu, mode, sig, h) in cache[b]:
                code = _code(b, g, c)
                ax.fill_between(centers, h, step="mid", color="red", alpha=0.25)
                ax.step(centers, h, where="mid", color="red", linewidth=1.0,
                        label=_legend_label(code, mu, mode, sig))

            ax.set_xlim(*xlim)
            ax.set_ylim(0, global_ymax * 1.05)
            ax.legend(fontsize=6, ncol=1, frameon=False, loc="upper right")

        for ax in axes:
            ax.set_xlabel(_xlabel())

        _global_ylabel(fig, "Events")
        _tighten(fig, left=0.07, bottom=0.05, hspace=0.10, wspace=0.05)
        pdf.savefig(fig)
        plt.close(fig)

    print("Saved:", out)

def make_16(parity, label, xlim, outdir):
    out = f"{outdir}/HISTONLY_{label}_16Subplots_vertical.pdf"
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

        # cache + global ymax
        global_ymax = 1
        cache = {}  # (b,g) -> list of (ch, mu, mode, sig, h)
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
                    mu = float(arr.mean())
                    sig = float(arr.std())
                    mode, h = _mode_and_hist(arr, bins=bins)
                    cache[(b, g)].append((ch, mu, mode, sig, h))
                    global_ymax = max(global_ymax, int(h.max()))

        # draw
        for r, (bL, gL, bR, gR) in enumerate(layout):
            for cidx, (b, g) in enumerate([(bL, gL), (bR, gR)]):
                ax = axes[r, cidx]
                for (ch, mu, mode, sig, h) in cache.get((b, g), []):
                    code = _code(b, g, ch)
                    ax.fill_between(centers, h, step="mid", color="red", alpha=0.25)
                    ax.step(centers, h, where="mid", color="red", linewidth=1.0,
                            label=_legend_label(code, mu, mode, sig))
                ax.set_xlim(*xlim)
                ax.set_ylim(0, global_ymax * 1.05)
                ax.legend(fontsize=6, ncol=1, frameon=False, loc="upper right")

        for ax in axes[-1]:
            ax.set_xlabel(_xlabel())

        _global_ylabel(fig, "Events")
        _tighten(fig, left=0.06, right=0.995, top=0.997, bottom=0.03, hspace=0.12, wspace=0.04)
        pdf.savefig(fig)
        plt.close(fig)

    print("Saved:", out)

def make_mosaic_hist(grid, label, xlim, outdir):
    out = f"{outdir}/HISTONLY_{label}_mosaic.pdf"
    bins = np.linspace(xlim[0], xlim[1], NBINS + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    nrows = len(grid)
    ncols = max(len(r) for r in grid)

    with uproot.open(ANA_FILE) as f, PdfPages(out) as pdf:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        # precompute + global ymax
        global_ymax = 1
        cell = {}
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

                mu = float(arr.mean())
                sig = float(arr.std())
                mode, h = _mode_and_hist(arr, bins=bins)
                cell[(r, c)] = {"code": code, "status": "ok", "mu": mu, "sig": sig, "mode": mode, "h": h}
                global_ymax = max(global_ymax, int(h.max()))

        fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 2.0 * nrows), sharex=True, sharey=True)
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
                    ax.text(0.5, 0.5, f"{code}\n({status})",
                            ha="center", va="center", transform=ax.transAxes, fontsize=9)
                    continue

                h = entry["h"]
                ax.fill_between(centers, h, step="mid", color="red", alpha=0.25)
                ax.step(centers, h, where="mid", color="red", linewidth=1.0,
                        label=_legend_label(code, entry["mu"], entry["mode"], entry["sig"]))
                ax.legend(fontsize=7, frameon=False, loc="upper right",
                          handlelength=1.0, borderaxespad=0.2)

        for ax in axes[-1, :]:
            if ax.axison and ax.get_visible():
                ax.set_xlabel(_xlabel())

        _global_ylabel(fig, "Events")
        _tighten(fig, left=0.05, right=0.995, top=0.995, bottom=0.035, hspace=0.10, wspace=0.04)
        pdf.savefig(fig)
        plt.close(fig)

    print("Saved:", out)

def make_mosaic_heatmaps_mean_mode(grid, label, xlim, outdir):
    """
    2-page PDF:
      page 1: mean(|tfinal|)
      page 2: mode(|tfinal|) from histogram max bin
    """
    out = f"{outdir}/HEATMAP_{label}_mean_mode.pdf"

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
                mode, _ = _mode_and_hist(arr, bins=bins)
                modes[r, c] = float(mode)

    with PdfPages(out) as pdf:
        for mat, title, cbar_label in [
            (means, f"{label} mean(|tfinal|)", "Mean(|tfinal|) [ns]"),
            (modes, f"{label} mode(|tfinal|)", "Mode(|tfinal|) [ns]"),
        ]:
            fig, ax = plt.subplots(1, 1, figsize=(12.5, 0.65 * nrows + 1.2))
            im = ax.imshow(mat, origin="upper", aspect="auto", cmap=HEATMAP_CMAP)

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
            fig.subplots_adjust(left=0.03, right=0.90, top=0.90, bottom=0.05)
            pdf.savefig(fig)
            plt.close(fig)

    print("Saved:", out)


# ================= MAIN =================
def main():
    global ANA_FILE, TREE_NAME, NBINS, CUT_MIN, MIN_ENTRIES, MIN_RAW, HSPACE, WSPACE

    ap = argparse.ArgumentParser()
    ap.add_argument("--ana-file", default=ANA_FILE, help="Input ROOT file")
    ap.add_argument("--tree", default=TREE_NAME, help="Tree name")
    ap.add_argument("--outdir", default=None,
                    help="Output directory. Default: ./TRUE-HGtiming/3mmplots_histonly/<runXXXX>")
    ap.add_argument("--xmin", type=float, default=4.0, help="Min |tfinal| for plots/heatmaps")
    ap.add_argument("--xmax", type=float, default=20.0, help="Max |tfinal| for plots/heatmaps")
    ap.add_argument("--nbins", type=int, default=NBINS, help="Histogram bins")
    ap.add_argument("--cut-min", type=float, default=CUT_MIN, help="Ignore |tfinal| < cut-min")
    ap.add_argument("--min-entries", type=int, default=MIN_ENTRIES, help="Min entries after cuts")
    ap.add_argument("--min-raw", type=int, default=MIN_RAW, help="Min raw entries before cuts")
    ap.add_argument("--hspace", type=float, default=HSPACE, help="subplot hspace (smaller=tighter)")
    ap.add_argument("--wspace", type=float, default=WSPACE, help="subplot wspace (smaller=tighter)")
    args = ap.parse_args()

    ANA_FILE = args.ana_file
    TREE_NAME = args.tree
    NBINS = args.nbins
    CUT_MIN = args.cut_min
    MIN_ENTRIES = args.min_entries
    MIN_RAW = args.min_raw
    HSPACE = args.hspace
    WSPACE = args.wspace

    outdir = args.outdir or _default_outdir_for(ANA_FILE)
    os.makedirs(outdir, exist_ok=True)

    print("ANA_FILE :", ANA_FILE)
    print("TREE_NAME:", TREE_NAME)
    print("OUTDIR   :", outdir)

    xlim = (args.xmin, args.xmax)

    # KEEP ALL ORIGINAL OUTPUTS (but upgraded styling/packing)
    make_boards("odd",  "SCI", xlim, outdir)
    make_16(   "odd",  "SCI", xlim, outdir)

    make_boards("even", "CER", xlim, outdir)
    make_16(   "even", "CER", xlim, outdir)

    make_mosaic_hist(QUARTZ_GRID, "CER-Quartz", xlim, outdir)
    make_mosaic_heatmaps_mean_mode(QUARTZ_GRID, "CER-Quartz", xlim, outdir)

    make_mosaic_hist(PLASTIC_GRID, "CER-Plastic", xlim, outdir)
    make_mosaic_heatmaps_mean_mode(PLASTIC_GRID, "CER-Plastic", xlim, outdir)

    # ADD: combined CER mosaic hist + mean/mode heatmaps
    make_mosaic_hist(CER_ALL_GRID, "CER-AllChannels", xlim, outdir)
    make_mosaic_heatmaps_mean_mode(CER_ALL_GRID, "CER-AllChannels", xlim, outdir)

    # ADD: combined SCI mosaic hist + mean/mode heatmaps
    make_mosaic_hist(SCI_ALL_GRID, "SCI-AllChannels", xlim, outdir)
    make_mosaic_heatmaps_mean_mode(SCI_ALL_GRID, "SCI-AllChannels", xlim, outdir)


    print("All done.")


if __name__ == "__main__":
    main()

