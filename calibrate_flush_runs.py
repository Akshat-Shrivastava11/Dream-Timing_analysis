#!/usr/bin/env python3
import os
import re
import json
import csv
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ================= DEFAULTS =================
TREE_NAME = "EventTree"

NBINS = 200
CUT_MIN = 1.0
MIN_ENTRIES = 200
MIN_RAW = 500

NG = 4
NC = 9

HEATMAP_CMAP = "viridis_r"
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

# ================= QC / ALIGNMENT =================
DO_ALIGN = True

QC_MIN_GOOD_N = 10
QC_MIN_PEAK = 8
QC_MIN_PROM = 8.0
QC_MAX_SIGMA = 1.82  # None disables

# Computed from reference run
ABS_SHIFTS = {}     # (b,g,c) -> delta added to |tfinal|
TARGETS = {}        # group -> target mean
ANCHORS = {}        # group -> anchor channel info
QC_STATS = {}       # group -> {(b,g,c): stats}

# ================= HELPERS =================
def _infer_run_label(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"(run\d+_\d{11,12})", base)
    return m.group(1) if m else os.path.splitext(base)[0]

def _mkdir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _xlabel():
    return r"$|t_{\mathrm{final}}|$ [ns]"

def _global_ylabel(fig, text="Events"):
    fig.text(0.010, 0.5, text, va="center", rotation=90)

def _tighten(fig, left=0.05, right=0.995, top=0.995, bottom=0.035, hspace=None, wspace=None):
    if hspace is None: hspace = HSPACE
    if wspace is None: wspace = WSPACE
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace)

def _parse_code(code_str):
    b = int(code_str[0]); g = int(code_str[1]); c = int(code_str[2])
    return b, g, c

def _branch(b, g, c):
    return f"tfinal_Board{b}_Group{g}_Channel{c}"

def _base_ok(g, c):
    if c == 8:
        return False
    if g == 3 and c in (6, 7):
        return False
    return True

def _prep_abs(arr_abs, xlim):
    if arr_abs.size < MIN_RAW:
        return None
    arr_abs = arr_abs[np.isfinite(arr_abs)]
    arr_abs = arr_abs[arr_abs >= CUT_MIN]
    if arr_abs.size < MIN_ENTRIES:
        return None
    arr_abs = arr_abs[(arr_abs >= xlim[0]) & (arr_abs <= xlim[1])]
    if arr_abs.size < 50:
        return None
    return arr_abs

def _mode_and_hist(arr_abs, bins):
    h, edges = np.histogram(arr_abs, bins=bins)
    if h.sum() == 0:
        return np.nan, h
    imax = int(np.argmax(h))
    mode = 0.5 * (edges[imax] + edges[imax + 1])
    return float(mode), h

def _hist_quality(arr_abs, bins):
    h, _ = np.histogram(arr_abs, bins=bins)
    peak = float(h.max()) if h.size else 0.0
    baseline = float(np.median(h)) if h.size else 0.0
    prom = peak / max(1.0, baseline)
    mu = float(np.mean(arr_abs)) if arr_abs.size else np.nan
    sig = float(np.std(arr_abs)) if arr_abs.size else np.nan
    N = int(arr_abs.size)

    ok = True
    if N < QC_MIN_GOOD_N: ok = False
    if peak < QC_MIN_PEAK: ok = False
    if prom < QC_MIN_PROM: ok = False
    if QC_MAX_SIGMA is not None and np.isfinite(sig) and sig > QC_MAX_SIGMA: ok = False

    return ok, {"N": N, "peak": peak, "baseline": baseline, "prom": prom, "mu": mu, "sig": sig, "ok": ok}

def _grid_codes(grid):
    for row in grid:
        for code in row:
            if code is None:
                continue
            yield code

def _apply_shift(arr_abs, b, g, c, do_shift):
    if not do_shift:
        return arr_abs
    return arr_abs + float(ABS_SHIFTS.get((b, g, c), 0.0))

# ================= CALIBRATION TABLE OUTPUT =================
def _calib_entry(code, b, g, ch, branch, group_name, st, target, anchor_code):
    mu = float(st.get("mu", np.nan))
    sig = float(st.get("sig", np.nan))
    ok = bool(st.get("ok", False))
    shift = (float(target) - mu) if (ok and np.isfinite(mu) and np.isfinite(target)) else 0.0

    return {
        "code": code,
        "board": int(b),
        "group": int(g),
        "channel": int(ch),
        "branch": branch,
        "material_group": group_name,
        "mean_abs": mu,
        "sigma_abs": sig,
        "N": int(st.get("N", 0)),
        "peak": float(st.get("peak", 0.0)),
        "prom": float(st.get("prom", 0.0)),
        "ok": ok,
        "anchor_code": anchor_code,
        "target_mean_abs": float(target) if np.isfinite(target) else np.nan,
        "shift_abs": float(shift),
    }

def _write_calibration_json(entries, outpath, meta=None):
    payload = {"meta": meta or {}, "entries": entries}
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print("Saved calibration JSON:", outpath)

def _write_calibration_csv(entries, outpath):
    if not entries:
        return
    fieldnames = list(entries[0].keys())
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for e in entries:
            w.writerow(e)
    print("Saved calibration CSV:", outpath)

# ================= BUILD SHIFTS FROM REFERENCE (ANCHOR=MAX EVENTS) =================
def build_reference_shifts(reference_file, xlim):
    """
    For each group (Quartz, Plastic, SCI):
      - compute stats for each channel
      - choose anchor = channel with max N among GOOD channels
      - target = mean_abs(anchor)
      - shift = target - mean_abs(channel)
    """
    global ABS_SHIFTS, TARGETS, ANCHORS, QC_STATS

    ABS_SHIFTS = {}
    TARGETS = {}
    ANCHORS = {}
    QC_STATS = {}

    bins = np.linspace(xlim[0], xlim[1], NBINS + 1)

    def _group(grid, group_name):
        with uproot.open(reference_file) as f:
            tree = f[TREE_NAME]
            keys = set(tree.keys())

            stats = {}
            good = []

            for code in _grid_codes(grid):
                b, g, ch = _parse_code(code)
                if not _base_ok(g, ch):
                    continue
                k = _branch(b, g, ch)
                if k not in keys:
                    continue
                raw = tree[k].array(library="np")
                arr_abs = _prep_abs(np.abs(raw), xlim)
                if arr_abs is None:
                    continue
                ok, st = _hist_quality(arr_abs, bins=bins)
                stats[(b, g, ch)] = st
                if ok and np.isfinite(st["mu"]):
                    good.append((code, b, g, ch, st["N"], st["mu"]))

        if len(good) == 0:
            return {}, np.nan, stats, None

        # anchor = max N
        good_sorted = sorted(good, key=lambda t: t[4], reverse=True)
        anchor_code, ab, ag, ach, aN, amu = good_sorted[0]
        target = float(amu)

        shifts = {}
        for (code, b, g, ch, N, mu) in good_sorted:
            shifts[(b, g, ch)] = float(target - float(mu))

        anchor_info = {
            "anchor_code": anchor_code,
            "anchor_board": int(ab),
            "anchor_group": int(ag),
            "anchor_channel": int(ach),
            "anchor_N": int(aN),
            "anchor_mean_abs": float(amu),
        }
        return shifts, target, stats, anchor_info

    q_shifts, q_target, q_stats, q_anchor = _group(QUARTZ_GRID, "CER-Quartz")
    p_shifts, p_target, p_stats, p_anchor = _group(PLASTIC_GRID, "CER-Plastic")
    s_shifts, s_target, s_stats, s_anchor = _group(SCI_ALL_GRID, "SCI")

    ABS_SHIFTS.update(q_shifts)
    ABS_SHIFTS.update(p_shifts)
    ABS_SHIFTS.update(s_shifts)

    TARGETS = {"CER-Quartz": q_target, "CER-Plastic": p_target, "SCI": s_target}
    QC_STATS = {"CER-Quartz": q_stats, "CER-Plastic": p_stats, "SCI": s_stats}
    ANCHORS = {"CER-Quartz": q_anchor, "CER-Plastic": p_anchor, "SCI": s_anchor}

# ================= PLOTTING =================
def mosaic_overlay_pre_post(run_file, grid, label, xlim, outdir, apply_post_shift):
    """
    One PDF: each channel cell shows PRE and POST overlayed.
    - PRE: no shift
    - POST: apply ABS_SHIFTS if apply_post_shift True
    """
    out = os.path.join(outdir, f"OVERLAY_PREPOST_{label}_mosaic.pdf")
    bins = np.linspace(xlim[0], xlim[1], NBINS + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    nrows = len(grid)
    ncols = max(len(r) for r in grid)

    with uproot.open(run_file) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

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
                    cell[(r, c)] = {"code": code, "status": "veto"}
                    continue

                k = _branch(b, g, ch)
                if k not in keys:
                    cell[(r, c)] = {"code": code, "status": "missing"}
                    continue

                raw = tree[k].array(library="np")
                pre = _prep_abs(np.abs(raw), xlim)
                if pre is None:
                    cell[(r, c)] = {"code": code, "status": "nostats"}
                    continue

                post = _apply_shift(pre, b, g, ch, apply_post_shift)

                pre_mode, pre_h = _mode_and_hist(pre, bins=bins)
                post_mode, post_h = _mode_and_hist(post, bins=bins)

                global_ymax = max(global_ymax, int(pre_h.max()), int(post_h.max()))

                cell[(r, c)] = {
                    "code": code, "status": "ok",
                    "pre": {"mu": float(pre.mean()), "sig": float(pre.std()), "mode": pre_mode, "h": pre_h},
                    "post": {"mu": float(post.mean()), "sig": float(post.std()), "mode": post_mode, "h": post_h},
                }

    with PdfPages(out) as pdf:
        fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 2.0 * nrows), sharex=True, sharey=True)
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        elif ncols == 1:
            axes = np.array([[ax] for ax in axes])

        # One global legend: PRE vs POST
        pre_handle = None
        post_handle = None

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

                pre = entry["pre"]
                post = entry["post"]

                # PRE (filled)
                ln1, = ax.step(centers, pre["h"], where="mid", linewidth=1.0, alpha=0.95)
                ax.fill_between(centers, pre["h"], step="mid", alpha=0.18)

                # POST (line only)
                ln2, = ax.step(centers, post["h"], where="mid", linewidth=1.4, alpha=0.95)

                if pre_handle is None: pre_handle = ln1
                if post_handle is None: post_handle = ln2

                ax.set_title(code, fontsize=9, pad=2)

                # concise stats box (so you can see "flushed")
                txt = (f"PRE  μ={pre['mu']:.2f} σ={pre['sig']:.2f}\n"
                       f"POST μ={post['mu']:.2f} σ={post['sig']:.2f}")
                ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                        ha="left", va="top", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"))

        for ax in axes[-1, :]:
            if ax.axison and ax.get_visible():
                ax.set_xlabel(_xlabel())

        _global_ylabel(fig, "Events")
        _tighten(fig, left=0.05, right=0.86, top=0.995, bottom=0.035, hspace=0.08, wspace=0.04)

        # global legend placed to the right
        if pre_handle is not None and post_handle is not None:
            fig.legend([pre_handle, post_handle], ["PRE (raw)", "POST (shifted)"],
                       loc="center left", bbox_to_anchor=(0.87, 0.5),
                       fontsize=10, frameon=False)

        pdf.savefig(fig)
        plt.close(fig)

    print("Saved:", out)

def heatmaps_mean_mode(run_file, grid, label, xlim, outdir, apply_shift_flag):
    """
    Separate heatmap PDF (2 pages: mean + mode). No overlay here.
    """
    out = os.path.join(outdir, f"HEATMAP_{label}_mean_mode.pdf")

    nrows = len(grid)
    ncols = max(len(r) for r in grid)

    means = np.full((nrows, ncols), np.nan, dtype=float)
    modes = np.full((nrows, ncols), np.nan, dtype=float)

    bins = np.linspace(xlim[0], xlim[1], NBINS + 1)

    with uproot.open(run_file) as f:
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
                raw = tree[k].array(library="np")
                arr_abs = _prep_abs(np.abs(raw), xlim)
                if arr_abs is None:
                    continue
                arr_abs = _apply_shift(arr_abs, b, g, ch, apply_shift_flag)
                means[r, c] = float(arr_abs.mean())
                mode, _ = _mode_and_hist(arr_abs, bins=bins)
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

# ================= MAIN PIPELINE =================
def save_calibration_tables(reference_file, xlim, outdir):
    """
    Save per-channel means used for calibration (and shifts) from reference file.
    """
    runlab = _infer_run_label(reference_file)

    entries = []
    for group_name, grid in [("CER-Quartz", QUARTZ_GRID), ("CER-Plastic", PLASTIC_GRID), ("SCI", SCI_ALL_GRID)]:
        stats_map = QC_STATS.get(group_name, {})
        target = TARGETS.get(group_name, np.nan)
        anchor_code = (ANCHORS.get(group_name) or {}).get("anchor_code", "")
        for code in _grid_codes(grid):
            b, g, ch = _parse_code(code)
            if not _base_ok(g, ch):
                continue
            st = stats_map.get((b, g, ch), {"ok": False})
            entries.append(_calib_entry(code, b, g, ch, _branch(b, g, ch), group_name, st, target, anchor_code))

    meta = {
        "reference_file": reference_file,
        "reference_run_label": runlab,
        "xlim": {"xmin": xlim[0], "xmax": xlim[1]},
        "cuts": {"cut_min": CUT_MIN, "min_entries": MIN_ENTRIES, "min_raw": MIN_RAW},
        "qc": {"min_good_n": QC_MIN_GOOD_N, "min_peak": QC_MIN_PEAK, "min_prom": QC_MIN_PROM, "max_sigma": QC_MAX_SIGMA},
        "targets": {k: (float(v) if np.isfinite(v) else None) for k, v in TARGETS.items()},
        "anchors": ANCHORS,
        "notes": "target_mean_abs is mean(|tfinal|) of the GOOD channel with the most events (anchor). shift_abs = target - mean_abs(channel).",
    }

    _write_calibration_json(entries, os.path.join(outdir, f"calibration_{runlab}.json"), meta=meta)
    _write_calibration_csv(entries, os.path.join(outdir, f"calibration_{runlab}.csv"))

def run_all_for_file(run_file, xlim, outdir, tag, apply_post_shift, do_heatmaps):
    runlab = _infer_run_label(run_file)
    base = _mkdir(os.path.join(outdir, f"{tag}_{runlab}"))

    # overlays (pre vs post) mosaics
    mosaic_overlay_pre_post(run_file, QUARTZ_GRID,  f"{tag}_CER-Quartz",      xlim, base, apply_post_shift)
    mosaic_overlay_pre_post(run_file, PLASTIC_GRID, f"{tag}_CER-Plastic",     xlim, base, apply_post_shift)
    mosaic_overlay_pre_post(run_file, CER_ALL_GRID, f"{tag}_CER-AllChannels", xlim, base, apply_post_shift)
    mosaic_overlay_pre_post(run_file, SCI_ALL_GRID, f"{tag}_SCI-AllChannels", xlim, base, apply_post_shift)

    # heatmaps separate
    if do_heatmaps:
        heatmaps_mean_mode(run_file, QUARTZ_GRID,  f"{tag}_CER-Quartz",      xlim, base, apply_post_shift)
        heatmaps_mean_mode(run_file, PLASTIC_GRID, f"{tag}_CER-Plastic",     xlim, base, apply_post_shift)
        heatmaps_mean_mode(run_file, CER_ALL_GRID, f"{tag}_CER-AllChannels", xlim, base, apply_post_shift)
        heatmaps_mean_mode(run_file, SCI_ALL_GRID, f"{tag}_SCI-AllChannels", xlim, base, apply_post_shift)

def main():
    global TREE_NAME, NBINS, CUT_MIN, MIN_ENTRIES, MIN_RAW
    global HSPACE, WSPACE, DO_ALIGN
    global QC_MIN_GOOD_N, QC_MIN_PEAK, QC_MIN_PROM, QC_MAX_SIGMA

    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-file", required=True,
                    help="Reference run ROOT file (derive shifts), e.g. run1501_250928105227_*.root")
    ap.add_argument("--test-files", nargs="+", required=True,
                    help="Test run ROOT files, e.g. run1511_..., run1507_...")

    ap.add_argument("--tree", default=TREE_NAME)
    ap.add_argument("--xmin", type=float, default=7.0)
    ap.add_argument("--xmax", type=float, default=14.0)
    ap.add_argument("--nbins", type=int, default=NBINS)
    ap.add_argument("--cut-min", type=float, default=CUT_MIN)
    ap.add_argument("--min-entries", type=int, default=MIN_ENTRIES)
    ap.add_argument("--min-raw", type=int, default=MIN_RAW)
    ap.add_argument("--hspace", type=float, default=HSPACE)
    ap.add_argument("--wspace", type=float, default=WSPACE)

    ap.add_argument("--qc-min-good-n", type=int, default=QC_MIN_GOOD_N)
    ap.add_argument("--qc-min-peak", type=float, default=QC_MIN_PEAK)
    ap.add_argument("--qc-min-prom", type=float, default=QC_MIN_PROM)
    ap.add_argument("--qc-max-sigma", type=float, default=QC_MAX_SIGMA)

    ap.add_argument("--no-align", action="store_true", help="Do not apply shifts in POST overlays (still computes shifts).")
    ap.add_argument("--no-heatmaps", action="store_true", help="Skip heatmap PDFs (mosaic overlays still produced).")

    ap.add_argument("--outdir", default="./TRUE-HGtiming/calibration_anchorMaxN",
                    help="Base output directory")

    args = ap.parse_args()

    TREE_NAME = args.tree
    NBINS = args.nbins
    CUT_MIN = args.cut_min
    MIN_ENTRIES = args.min_entries
    MIN_RAW = args.min_raw
    HSPACE = args.hspace
    WSPACE = args.wspace

    QC_MIN_GOOD_N = args.qc_min_good_n
    QC_MIN_PEAK = args.qc_min_peak
    QC_MIN_PROM = args.qc_min_prom
    QC_MAX_SIGMA = args.qc_max_sigma

    DO_ALIGN = (not args.no_align)
    do_heatmaps = (not args.no_heatmaps)

    xlim = (args.xmin, args.xmax)
    _mkdir(args.outdir)

    # 1) build reference shifts using anchor=channel with most events
    build_reference_shifts(args.reference_file, xlim)

    print("\n[Anchors]")
    for g in ["CER-Quartz", "CER-Plastic", "SCI"]:
        a = ANCHORS.get(g)
        if a is None:
            print(f"  {g}: NONE (no good channels)")
        else:
            print(f"  {g}: anchor={a['anchor_code']}  N={a['anchor_N']}  mean={a['anchor_mean_abs']:.4f} ns")

    # 2) save per-channel means/shifts used for calibration
    ref_label = _infer_run_label(args.reference_file)
    ref_dir = _mkdir(os.path.join(args.outdir, f"REFERENCE_{ref_label}"))
    save_calibration_tables(args.reference_file, xlim, ref_dir)

    # 3) overlays for reference run (pre vs post using its own shifts)
    run_all_for_file(args.reference_file, xlim, args.outdir,
                     tag="REFERENCE_PRE_vs_POST", apply_post_shift=DO_ALIGN, do_heatmaps=do_heatmaps)

    # 4) overlays for test runs (pre vs post using reference shifts)
    for tf in args.test_files:
        run_all_for_file(tf, xlim, args.outdir,
                         tag=f"TEST_PRE_vs_POST_using_{ref_label}", apply_post_shift=DO_ALIGN, do_heatmaps=do_heatmaps)

    print("\nAll done.")

if __name__ == "__main__":
    main()
