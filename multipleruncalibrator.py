#!/usr/bin/env python3
import os
import re
import glob
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator

# ================= DEFAULTS =================
TREE_NAME = "EventTree"

NBINS = 200
CUT_MIN = 1.0
MIN_ENTRIES = 200
MIN_RAW = 500

NG = 4
NC = 9

HSPACE = 0.10
WSPACE = 0.05

# how many runs to print (mu,sigma) inside each cell
CELL_STATS_MAXLINES = 3

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
def _xlabel():
    return r"$|t_{\mathrm{final}}|$ [ns]"

def _global_ylabel(fig, text="Events"):
    fig.text(0.010, 0.5, text, va="center", rotation=90)

def _tighten(fig, left=0.05, right=0.98, top=0.985, bottom=0.035, hspace=HSPACE, wspace=WSPACE):
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace)

def _parse_code(code_str):
    b = int(code_str[0])
    g = int(code_str[1])
    c = int(code_str[2])
    return b, g, c

def _branch(b, g, c):
    return f"tfinal_Board{b}_Group{g}_Channel{c}"

def _base_ok(g, c):
    if c == 8:
        return False
    if g == 3 and c in (6, 7):
        return False
    return True

def _prep(arr, xlim, cut_min, min_entries, min_raw):
    if arr.size < min_raw:
        return None
    arr = np.abs(arr)
    arr = arr[arr >= cut_min]
    if arr.size < min_entries:
        return None
    arr = arr[(arr >= xlim[0]) & (arr <= xlim[1])]
    if arr.size < 50:
        return None
    return arr

def _run_label(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"(run\d+_\d{11,12})", base)
    if m:
        return m.group(1)
    return os.path.splitext(base)[0]

def _extract_runs(files):
    runs = []
    for f in files:
        m = re.search(r"run(\d+)", os.path.basename(f))
        if m:
            runs.append(int(m.group(1)))
    return runs

def _fileset_tag(files):
    runs = _extract_runs(files)
    if not runs:
        return f"files_n{len(files)}"
    return f"runs{min(runs)}-{max(runs)}_n{len(files)}"

def _resolve_files(args):
    if args.ana_files:
        files = list(args.ana_files)
    else:
        files = sorted(glob.glob(args.ana_glob))

    if args.run_min is not None and args.run_max is not None:
        keep = []
        for f in files:
            m = re.search(r"run(\d+)", os.path.basename(f))
            if not m:
                continue
            r = int(m.group(1))
            if args.run_min <= r <= args.run_max:
                keep.append(f)
        files = keep

    def _sort_key(p):
        b = os.path.basename(p)
        mrun = re.search(r"run(\d+)", b)
        r = int(mrun.group(1)) if mrun else 10**9
        mts = re.search(r"_(\d{11,12})(?:_|\.|$)", b)
        ts = int(mts.group(1)) if mts else 10**18
        return (r, ts, b)

    return sorted(files, key=_sort_key)

def _extract_int(s: str, pattern: str) -> int:
    m = re.search(pattern, s)
    if not m:
        return 10**18
    try:
        return int(m.group(1))
    except Exception:
        return 10**18

def _build_color_map(runlabels):
    """
    Guarantee distinct colors for each runlabel by sampling a colormap at N points.
    For N ~ 10-30, tab20 is good; fall back to turbo/hsv for larger N.
    """
    n = len(runlabels)
    if n <= 20:
        cmap = plt.get_cmap("tab20")
    elif n <= 256:
        cmap = plt.get_cmap("turbo")
    else:
        cmap = plt.get_cmap("hsv")

    xs = np.linspace(0.0, 1.0, n, endpoint=False)
    colors = [cmap(x) for x in xs]
    return {rl: colors[i] for i, rl in enumerate(runlabels)}

def _mode_from_hist(arr, bins):
    """
    Histogram mode = center of the bin with maximum counts.
    Returns (mode_value, counts_max, hist_counts).
    """
    h, _ = np.histogram(arr, bins=bins)
    if h.sum() == 0:
        return (np.nan, 0, h)
    idx = int(np.argmax(h))
    centers = 0.5 * (bins[1:] + bins[:-1])
    return (float(centers[idx]), int(h[idx]), h)

# ================= CORE: overlay mosaic =================
def make_mosaic_hist_overlay(files, grid, label, xlim, outdir,
                            tree_name, nbins, cut_min, min_entries, min_raw):
    os.makedirs(outdir, exist_ok=True)
    tag = _fileset_tag(files)
    out = os.path.join(outdir, f"HISTONLY_OVERLAY_{label}_{tag}.pdf")

    bins = np.linspace(xlim[0], xlim[1], nbins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    nrows = len(grid)
    ncols = max(len(r) for r in grid)

    # open
    opened = []
    labels_in_order = []
    for fpath in files:
        try:
            uf = uproot.open(fpath)
            tree = uf[tree_name]
            keys = set(tree.keys())
            rl = _run_label(fpath)
            opened.append((fpath, uf, tree, keys, rl))
            labels_in_order.append(rl)
        except Exception as e:
            print(f"[WARN] failed to open {fpath}: {e}")

    if len(opened) == 0:
        raise RuntimeError("No readable input files.")

    # unique colors per file label
    color_map = _build_color_map(labels_in_order)

    # cell[(r,c)] = None OR {"code":..., "status":..., "items":[(rlabel,h,mu,sig,n), ...]}
    cell = {}
    global_ymax = 1

    for r in range(nrows):
        row = grid[r]
        for c in range(ncols):
            if c >= len(row) or row[c] is None:
                cell[(r, c)] = None
                continue

            code = row[c]
            b, g, ch = _parse_code(code)

            if not _base_ok(g, ch):
                cell[(r, c)] = {"code": code, "status": "veto", "items": []}
                continue

            k = _branch(b, g, ch)
            items = []

            for (_, _, tree, keys, rl) in opened:
                if k not in keys:
                    continue
                try:
                    arr = tree[k].array(library="np")
                except Exception:
                    continue
                arr = _prep(arr, xlim, cut_min, min_entries, min_raw)
                if arr is None:
                    continue

                mu = float(arr.mean())
                sig = float(arr.std())
                n = int(arr.size)

                h, _ = np.histogram(arr, bins=bins)
                if h.sum() == 0:
                    continue

                items.append((rl, h, mu, sig, n))
                global_ymax = max(global_ymax, int(h.max()))

            if len(items) == 0:
                cell[(r, c)] = {"code": code, "status": "nostats", "items": []}
            else:
                items = sorted(items, key=lambda x: (_extract_int(x[0], r"run(\d+)"),
                                                    _extract_int(x[0], r"_(\d{11,12})")))
                cell[(r, c)] = {"code": code, "status": "ok", "items": items}

    # close files
    for (_, uf, _, _, _) in opened:
        try:
            uf.close()
        except Exception:
            pass

    # ---------- PDF ----------
    with PdfPages(out) as pdf:
        # PAGE 1: mosaic
        fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 2.0 * nrows), sharex=True, sharey=True)
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        elif ncols == 1:
            axes = np.array([[ax] for ax in axes])

        legend_handles = {}

        for rr in range(nrows):
            for cc in range(ncols):
                ax = axes[rr, cc]
                ax.set_xlim(*xlim)
                ax.set_ylim(0, global_ymax * 1.05)
                ax.tick_params(labelsize=8)

                entry = cell.get((rr, cc))
                if entry is None:
                    ax.axis("off")
                    continue

                code = entry["code"]
                status = entry["status"]

                if status != "ok":
                    ax.text(0.5, 0.5, f"{code}\n({status})",
                            ha="center", va="center", transform=ax.transAxes, fontsize=9)
                    continue

                for (rl, h, mu, sig, n) in entry["items"]:
                    color = color_map[rl]
                    ln, = ax.step(centers, h, where="mid", linewidth=1.0, alpha=0.95, color=color)
                    ax.fill_between(centers, h, step="mid", alpha=0.16, color=color)
                    if rl not in legend_handles:
                        legend_handles[rl] = ln

                ax.set_title(code, fontsize=9, pad=2)

                top = sorted(entry["items"], key=lambda x: x[4], reverse=True)[:CELL_STATS_MAXLINES]
                if len(top) > 0:
                    lines = [f"{rl}: μ={mu:.2f}, σ={sig:.2f}" for (rl, _, mu, sig, _) in top]
                    if len(entry["items"]) > CELL_STATS_MAXLINES:
                        lines.append(f"+{len(entry['items']) - CELL_STATS_MAXLINES} more")
                    ax.text(0.02, 0.98, "\n".join(lines),
                            transform=ax.transAxes, ha="left", va="top",
                            fontsize=7,
                            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"))

        for ax in axes[-1, :]:
            if ax.axison and ax.get_visible():
                ax.set_xlabel(_xlabel())

        _global_ylabel(fig, "Events")
        _tighten(fig, left=0.05, right=0.98, top=0.985, bottom=0.035, hspace=0.10, wspace=0.04)
        pdf.savefig(fig)
        plt.close(fig)

        # PAGE 2: legend-only
        legend_labels = sorted(list(legend_handles.keys()),
                               key=lambda s: (_extract_int(s, r"run(\d+)"), _extract_int(s, r"_(\d{11,12})")))
        handles = [legend_handles[k] for k in legend_labels]

        nitems = len(legend_labels)
        max_per_col = 28
        ncol = max(1, int(np.ceil(nitems / max_per_col)))
        fontsize = 11 if ncol == 1 else 10 if ncol == 2 else 9

        fig2 = plt.figure(figsize=(8.5, 11))
        ax2 = fig2.add_subplot(111)
        ax2.axis("off")
        ax2.set_title(f"{label} overlays legend ({tag})\n(Colors are unique per file; μ,σ are per-channel in the mosaic cells)",
                      fontsize=13, pad=12)

        fig2.legend(handles, legend_labels,
                    loc="center",
                    fontsize=fontsize,
                    frameon=False,
                    ncol=ncol,
                    columnspacing=1.2,
                    handlelength=2.2,
                    handletextpad=0.8)

        pdf.savefig(fig2)
        plt.close(fig2)

    print("Saved:", out)

# ================= NEW: single-channel overlay for 104 =================
def make_channel_overlay_with_modes(files, code_str, label, xlim, outdir,
                                    tree_name, nbins, cut_min, min_entries, min_raw):
    """
    Make a standalone overlay plot for one channel code (e.g. "104").
      - Overlaid histograms for each run
      - Vertical dashed line per run at MODE
      - Top axis with minor ticks at each mode
      - Legend: includes mode and sigma per run
    """
    os.makedirs(outdir, exist_ok=True)
    tag = _fileset_tag(files)
    out = os.path.join(outdir, f"CHANNEL_{code_str}_OVERLAY_{label}_{tag}.pdf")

    b, g, ch = _parse_code(code_str)
    k = _branch(b, g, ch)

    bins = np.linspace(xlim[0], xlim[1], nbins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    # open and collect
    opened = []
    labels_in_order = []
    for fpath in files:
        try:
            uf = uproot.open(fpath)
            tree = uf[tree_name]
            keys = set(tree.keys())
            rl = _run_label(fpath)
            opened.append((uf, tree, keys, rl))
            labels_in_order.append(rl)
        except Exception as e:
            print(f"[WARN] failed to open {fpath}: {e}")

    if len(opened) == 0:
        raise RuntimeError("No readable input files.")

    color_map = _build_color_map(labels_in_order)

    items = []  # (rl, h, mode, sigma, n)
    global_ymax = 1

    for (uf, tree, keys, rl) in opened:
        if k not in keys:
            continue
        try:
            arr = tree[k].array(library="np")
        except Exception:
            continue

        arr = _prep(arr, xlim, cut_min, min_entries, min_raw)
        if arr is None:
            continue

        mode, _, h = _mode_from_hist(arr, bins)
        sig = float(arr.std())
        n = int(arr.size)

        if h.sum() == 0 or not np.isfinite(mode):
            continue

        items.append((rl, h, mode, sig, n))
        global_ymax = max(global_ymax, int(h.max()))

    for (uf, _, _, _) in opened:
        try:
            uf.close()
        except Exception:
            pass

    if len(items) == 0:
        print(f"[WARN] no valid data found for channel {code_str} (branch {k}); skipping {out}")
        return

    # stable sort
    items = sorted(items, key=lambda x: (_extract_int(x[0], r"run(\d+)"),
                                        _extract_int(x[0], r"_(\d{11,12})")))

    modes = [it[2] for it in items]

    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(11.0, 6.5))
        ax = fig.add_subplot(111)

        ax.set_xlim(*xlim)
        ax.set_ylim(0, global_ymax * 1.10)
        from matplotlib.ticker import AutoMinorLocator

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis="x", which="minor", length=4)
        ax.tick_params(axis="x", which="major", length=7)

        ax.set_xlabel(_xlabel())
        ax.set_ylabel("Events")
        ax.set_title(f"Channel {code_str} overlay ({label})  —  branch: {k}\n{tag}", fontsize=13)

        handles = []
        labels = []

        # draw hist + mode lines
        # draw hist + mode lines
        for (rl, h, mode, sig, n) in items:
            color = color_map[rl]

            
            stair = ax.step(centers, h, where="mid", linewidth=1.2, alpha=0.95, color=color)[0]
            ax.fill_between(centers, h, step="mid", alpha=0.18, color=color)

            
            ax.axvline(mode, linestyle="--", linewidth=2.0, alpha=0.95, color=color)

            # legend handle: use the histogram line (not the vline)
            handles.append(stair)
            labels.append(f"{rl}  (mode={mode:.2f}, σ={sig:.2f}, N={n})")



        # --- top axis with MINOR ticks at each mode ---
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xlabel("Mode markers (minor ticks)", labelpad=8)

        # no major ticks/labels
        ax_top.set_xticks([])
        ax_top.set_xticklabels([])

        # minor ticks at each mode
        ax_top.xaxis.set_minor_locator(FixedLocator(modes))
        ax_top.tick_params(axis="x", which="minor", length=6)
        ax_top.tick_params(axis="x", which="major", length=0)

        # legend
        nitems = len(labels)
        ncol = 1 if nitems <= 8 else 2 if nitems <= 18 else 3
        # ax.legend(handles, labels, fontsize=9, frameon=False, ncol=ncol,
        #           loc="upper right", handlelength=2.4, columnspacing=1.2, handletextpad=0.6)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # -------- PAGE 2: legend-only --------
        figL = plt.figure(figsize=(8.5, 11))
        axL = figL.add_subplot(111)
        axL.axis("off")

        axL.set_title(
            f"Channel {code_str} legend\n"
            f"Dashed lines = histogram mode, σ = RMS\n{tag}",
            fontsize=14,
            pad=14
        )

        nitems = len(labels)
        max_per_col = 28
        ncol = max(1, int(np.ceil(nitems / max_per_col)))
        fontsize = 11 if ncol == 1 else 10 if ncol == 2 else 9

        figL.legend(
            handles,
            labels,
            loc="center",
            frameon=False,
            fontsize=fontsize,
            ncol=ncol,
            handlelength=2.6,
            columnspacing=1.4,
            handletextpad=0.8,
        )

        pdf.savefig(figL)
        plt.close(figL)


    print("Saved:", out)

# ================= MAIN =================
def main():
    global NBINS, CUT_MIN, MIN_ENTRIES, MIN_RAW, HSPACE, WSPACE, CELL_STATS_MAXLINES

    ap = argparse.ArgumentParser()
    ap.add_argument("--ana-files", nargs="+", default=None,
                    help="Explicit list of input ROOT files (full paths). Keeps duplicates.")
    ap.add_argument("--ana-glob", default=None,
                    help="Glob for input ROOT files (e.g. '/path/run*_converted_timingskim.root')")

    ap.add_argument("--run-min", type=int, default=None, help="Keep only runs >= run-min")
    ap.add_argument("--run-max", type=int, default=None, help="Keep only runs <= run-max")

    ap.add_argument("--tree", default=TREE_NAME, help="Tree name")
    ap.add_argument("--outdir", default="./TRUE-HGtiming/calibration_studiesZ/overlay_runs_1499_1501",
                    help="Output directory")

    ap.add_argument("--xmin", type=float, default=4.0, help="Min |tfinal|")
    ap.add_argument("--xmax", type=float, default=25.0, help="Max |tfinal|")
    ap.add_argument("--nbins", type=int, default=NBINS, help="Histogram bins")
    ap.add_argument("--cut-min", type=float, default=CUT_MIN, help="Ignore |tfinal| < cut-min")
    ap.add_argument("--min-entries", type=int, default=MIN_ENTRIES, help="Min entries after cuts")
    ap.add_argument("--min-raw", type=int, default=MIN_RAW, help="Min raw entries before cuts")

    ap.add_argument("--cell-stats-lines", type=int, default=CELL_STATS_MAXLINES,
                    help="How many run μ,σ lines to print inside each channel cell (top by entries).")

    # new: choose which channel code to plot as the standalone overlay (default 104)
    ap.add_argument("--single-channel", default="104",
                    help="3-digit code bgc to make a standalone overlay plot for (default: 104). "
                         "Example: 104 => Board1 Group0 Chan4")

    args = ap.parse_args()

    if args.ana_files is None and args.ana_glob is None:
        raise SystemExit("ERROR: provide either --ana-files or --ana-glob")

    NBINS = args.nbins
    CUT_MIN = args.cut_min
    MIN_ENTRIES = args.min_entries
    MIN_RAW = args.min_raw
    CELL_STATS_MAXLINES = args.cell_stats_lines

    files = _resolve_files(args)
    if len(files) == 0:
        raise SystemExit("ERROR: no files matched your selection")

    print(f"Found {len(files)} files.")
    for f in files:
        print("  ", os.path.basename(f))

    xlim = (args.xmin, args.xmax)
    os.makedirs(args.outdir, exist_ok=True)

    # make_mosaic_hist_overlay(files, QUARTZ_GRID,  "CER-Quartz",      xlim, args.outdir,
    #                         args.tree, NBINS, CUT_MIN, MIN_ENTRIES, MIN_RAW)
    # make_mosaic_hist_overlay(files, PLASTIC_GRID, "CER-Plastic",     xlim, args.outdir,
    #                         args.tree, NBINS, CUT_MIN, MIN_ENTRIES, MIN_RAW)
    # make_mosaic_hist_overlay(files, SCI_ALL_GRID, "SCI-AllChannels", xlim, args.outdir,
    #                         args.tree, NBINS, CUT_MIN, MIN_ENTRIES, MIN_RAW)
    # make_mosaic_hist_overlay(files, CER_ALL_GRID, "CER-AllChannels", xlim, args.outdir,
    #                         args.tree, NBINS, CUT_MIN, MIN_ENTRIES, MIN_RAW)

    # ---- NEW: standalone channel overlay with modes ----
    make_channel_overlay_with_modes(files, args.single_channel, "ALL-RUNS", xlim, args.outdir,
                                    args.tree, NBINS, CUT_MIN, MIN_ENTRIES, MIN_RAW)

    print("All done.")

if __name__ == "__main__":
    main()
