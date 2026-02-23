#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
import json
import multiprocessing
import time

# Attempt to import PDF merging library
try:
    from pypdf import PdfWriter, PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfWriter, PdfReader
    except ImportError:
        print("CRITICAL ERROR: 'pypdf' or 'PyPDF2' is required for multiprocessing.")
        print("Please run: pip install pypdf")
        exit(1)

# ================= CONFIG =================
TREE_NAME = "EventTree"
NBINS = 200
XLIM = (8.0, 15.0)
MIN_RAW = 500
MIN_ENTRIES = 200

# ================= GRIDS =================
QUARTZ_GRID = [
    [None,  "002", None,  None],
    ["006", "004", "206", "204"],
    ["016", "014", "216", "214"],
    ["026", "024", "226", "224"],
    [None,  "030", None,  None],
    [None,  "034", None,  None],
    ["106", "104", None, "304"],
    ["116", "114", None, "314"],
    # ["126", "124", "326", "324"],
    # [None,  "134", None,  "334"],
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
    ["102", "100", None, "300"],
    ["106", "104", None, "304"],
    ["112", "110", None, "310"],
    ["116", "114", None, "314"],
    ["122", "120", "322", "320"],
    ["126", "124", "326", "324"],
    ["132", "130", "332", "330"],
    [None,  "134", None,  "334"],
]

SCI_GRID = [
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

FAMILIES = {
    "CER-Quartz": QUARTZ_GRID,
    "CER-Plastic": PLASTIC_GRID,
    "SCI": SCI_GRID,
    "CER-All": CER_ALL_GRID,
}

ANCHORS = {
    "SCI": (0, 3, 3),
    "CER-Quartz": (0, 3, 0),
    "CER-Plastic": (0, 1, 0),
}

# ================= HELPERS =================
def _infer_run_label(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"(run\d+_\d{11,12})", base)
    return m.group(1) if m else os.path.splitext(base)[0]

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
    mu = float(arr.mean())
    mode = float(0.5 * (edges[np.argmax(h)] + edges[np.argmax(h) + 1]))
    return mu, mode, h

# ================= GAUSS FIT =================
def _gauss(x, A, mu, sig):
    return A * np.exp(-0.5 * ((x - mu) / sig) ** 2)

def fit_gaussian_to_peak(arr_abs, bins, window=0.5):
    h, edges = np.histogram(arr_abs, bins=bins)
    centers = 0.5 * (edges[1:] + edges[:-1])
    if h.sum() == 0:
        return False, np.nan, np.nan, np.nan

    imax = int(np.argmax(h))
    x0 = float(centers[imax])

    m = (centers >= x0 - window) & (centers <= x0 + window)
    x = centers[m]
    y = h[m]

    if x.size < 6 or y.max() < 5:
        return False, np.nan, np.nan, np.nan

    p0 = [float(y.max()), x0, 0.15]
    bounds = ([0.0, x0 - window, 0.02], [np.inf, x0 + window, 2.0])

    try:
        popt, _ = curve_fit(_gauss, x, y, p0=p0, bounds=bounds, maxfev=10000)
        A, mu, sig = map(float, popt)
        return True, mu, sig, A
    except Exception:
        return False, np.nan, np.nan, np.nan

# ================= CALIBRATION LOGIC =================
def derive_family_calibration_fixed_anchor(root_file, grid, anchor_key, calib_stat="mode"):
    if calib_stat not in ("mean", "mode"):
        raise ValueError(f"--calib-stat must be 'mean' or 'mode' (got {calib_stat})")

    bins = np.linspace(*XLIM, NBINS + 1)
    stats = {}
    arrays = {}

    with uproot.open(root_file) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        for row in grid:
            for code in row:
                if code is None: continue
                b, g, c = parse_code(code)
                k = branch_name(b, g, c)
                if k not in keys: continue

                raw = tree[k].array(library="np")
                if raw.size < MIN_RAW: continue
                arr = prep_array(raw)
                if arr is None: continue

                arrays[(b, g, c)] = arr
                hstats = hist_stats(arr, bins)
                if hstats is None: continue
                mu, mode, _ = hstats
                stats[(b, g, c)] = {"N": int(arr.size), "mu": float(mu), "mode": float(mode)}

    if anchor_key not in arrays or anchor_key not in stats:
        print(f"WARNING: Anchor {anchor_key} not found in {root_file}. Skipping family calibration.")
        dummy_info = {"mu": 0, "N":0, "fit_ok":False, "sig_fit":0, "calib_stat":calib_stat}
        return {}, (anchor_key, dummy_info), {}

    anchor_arr = arrays[anchor_key]
    fit_ok, mu_fit, sig_fit, _A = fit_gaussian_to_peak(anchor_arr, bins, window=0.5)
    anchor_mu = float(mu_fit) if (fit_ok and np.isfinite(mu_fit)) else float(stats[anchor_key]["mu"])

    shifts = {}
    for key, st in stats.items():
        loc = st["mu"] if calib_stat == "mean" else st["mode"]
        shifts[key] = float(anchor_mu - float(loc))

    anchor_info = {
        "mu": anchor_mu,
        "N": int(stats[anchor_key]["N"]),
        "fit_ok": bool(fit_ok),
        "sig_fit": float(sig_fit) if np.isfinite(sig_fit) else np.nan,
        "calib_stat": calib_stat,
    }
    return shifts, (anchor_key, anchor_info), stats

# ================= DATA PROCESSING =================
def compute_test_mode_table(root_file, grid, shifts):
    bins = np.linspace(*XLIM, NBINS + 1)
    rows = []

    with uproot.open(root_file) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        for row in grid:
            for code in row:
                if code is None: continue
                b, g, ch = parse_code(code)
                k = branch_name(b, g, ch)
                if k not in keys: continue

                raw0 = tree[k].array(library="np")
                if raw0.size < MIN_RAW: continue
                arr = prep_array(raw0)
                if arr is None: continue

                st_pre = hist_stats(arr, bins)
                if st_pre is None: continue
                _, mode_pre, _ = st_pre

                shift_used = float(shifts.get((b, g, ch), 0.0))
                arr_post = arr + shift_used

                st_post = hist_stats(arr_post, bins)
                if st_post is None: continue
                _, mode_post, _ = st_post

                # Fit Gaussian to extract sigma, then calculate Standard Error
                fit_ok, mu_fit, sig_fit, _ = fit_gaussian_to_peak(arr_post, bins, window=0.5)
                
                N = arr.size
                
                # Calculate sigma / sqrt(N)
                if fit_ok and np.isfinite(sig_fit) and N > 0:
                    mode_err = float(sig_fit / np.sqrt(N))
                else:
                    mode_err = 0.0

                rows.append({
                    "code": code,
                    "N": int(N),
                    "mode_pre": float(mode_pre),
                    "mode_post": float(mode_post),
                    "mode_err": float(mode_err),  # Now stores standard error
                    "dmode_test": float(mode_post - mode_pre),
                    "shift_used": float(shift_used),
                })
    return rows

def save_shifts_json(outpath, meta, shifts_by_family):
    payload = {"meta": meta, "shifts_by_family": {}}
    for fam, shifts in shifts_by_family.items():
        d = {}
        for (b, g, ch), s in shifts.items():
            d[f"{b}_{g}_{ch}"] = float(s)
        payload["shifts_by_family"][fam] = d
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print("Saved JSON:", outpath)

def write_mode_shift_txt(outpath, tag, per_family_rows, calib_stat, ref_label, test_label):
    def _format_float(x, w=9, p=3):
        if x is None or not np.isfinite(x): return " " * (w - 1) + "—"
        return f"{x:{w}.{p}f}"

    with open(outpath, "w") as f:
        f.write(f"{tag}\n")
        f.write(f"reference_label={ref_label}\n")
        f.write(f"test_label={test_label}\n")
        f.write(f"XLIM={XLIM}, NBINS={NBINS}, MIN_RAW={MIN_RAW}, MIN_ENTRIES={MIN_ENTRIES}\n")
        f.write(f"calib_stat={calib_stat}\n\n")

        header = f"{'code':>5}  {'N':>8}  {'mode_pre':>9}  {'mode_post':>9}  {'dmode_test':>10}  {'shift_used':>10}\n"
        sep = "-" * (len(header) - 1) + "\n"

        for fam, rows in per_family_rows.items():
            f.write(f"\n=== {fam} ===\n")
            f.write(header)
            f.write(sep)
            
            # --- SORTING LOGIC FOR TEXT FILE (BY GRID ORDER) ---
            grid = FAMILIES.get(fam)
            if grid:
                flat_order = []
                for row in grid:
                    for code in row:
                        if code is not None: flat_order.append(code)
                order_map = {c: i for i, c in enumerate(flat_order)}
                # Sort by index in grid; Put unknowns at end
                rows = sorted(rows, key=lambda x: order_map.get(x['code'], 9999))
            else:
                rows = sorted(rows, key=lambda x: x['code'])
            # ---------------------------------------------------

            for r in rows:
                f.write(
                    f"{r['code']:>5}  {r['N']:8d}  "
                    f"{_format_float(r['mode_pre'])}  "
                    f"{_format_float(r['mode_post'])}  "
                    f"{_format_float(r['dmode_test'], w=10)}  "
                    f"{_format_float(r['shift_used'], w=10)}\n"
                )
    print("Saved TXT:", outpath)

# ================= PLOTTING FUNCTIONS =================
def plot_family_summary_bars_figure(fam_name, multi_run_data):
    """
    Plots points with error bars for multiple test runs on the same figure,
    AND generates individual plots for each run.
    
    Returns:
        List[plt.Figure]: A list containing [Overlay_Figure, Run1_Figure, Run2_Figure, ...]
    """
    if not multi_run_data:
        return []

    generated_figures = []

    # --- 1. DETERMINE SORT ORDER (Based on Grid) ---
    grid = FAMILIES.get(fam_name)
    if grid:
        flat_order = []
        for row in grid:
            for code in row:
                if code is not None: flat_order.append(code)
        order_map = {code: i for i, code in enumerate(flat_order)}
    else:
        # Fallback: get all codes from data
        all_codes = set()
        for item in multi_run_data:
            for r in item['rows']:
                all_codes.add(r['code'])
        flat_order = sorted(list(all_codes))
        order_map = {c: i for i, c in enumerate(flat_order)}

    # Collect all codes present in the data
    present_codes = set()
    for item in multi_run_data:
        for r in item['rows']:
            present_codes.add(r['code'])
    
    # Sort them according to the grid order
    sorted_codes = sorted(list(present_codes), key=lambda x: order_map.get(x, 9999))
    if not sorted_codes: return []

    indices = np.arange(len(sorted_codes))
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    # ==========================================
    # A. GENERATE OVERLAY PLOT (All Runs)
    # ==========================================
    fig_overlay, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                                           gridspec_kw={'height_ratios': [3, 1]})
    
    n_runs = len(multi_run_data)
    total_width = 0.6 # Adjust spread of points slightly for overlay
    point_offset = total_width / max(1, n_runs)
    
    min_y, max_y = 999, -999

    for i, run_item in enumerate(multi_run_data):
        label = run_item['label']
        rows = run_item['rows']
        anchor_val = run_item['anchor']
        color = run_item.get('color', default_colors[i % len(default_colors)])
        run_item['color'] = color 

        # Map modes and errors
        row_map = {r['code']: r['mode_post'] for r in rows}
        err_map = {r['code']: r.get('mode_err', 0.0) for r in rows} # fallback to 0.0 if old data
        
        modes = [row_map.get(c, np.nan) for c in sorted_codes]
        errors = [err_map.get(c, 0.0) for c in sorted_codes]
        residuals = [m - anchor_val if np.isfinite(m) else np.nan for m in modes]

        offset = (i - (n_runs - 1) / 2) * point_offset
        
        # Points with Error Bars
        ax1.errorbar(indices + offset, modes, yerr=errors, fmt='o', label=label, 
                     color=color, alpha=0.85, capsize=3, elinewidth=1.5, markersize=6)
        
        # Anchor
        ax1.axhline(anchor_val, color=color, linestyle='--', linewidth=1.5, alpha=0.8,
                    label=f"{label} Anchor")

        # Track limits using modes AND errors
        for m, e in zip(modes, errors):
            if np.isfinite(m):
                min_y = min(min_y, m - e)
                max_y = max(max_y, m + e)

        # Residuals
        ax2.errorbar(indices + offset, residuals, yerr=errors, fmt='o', color=color, 
                     alpha=0.8, capsize=2, elinewidth=1.0, markersize=5)
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    # Format Overlay
    margin = 0.8
    if min_y != 999: ax1.set_ylim(min_y - margin, max_y + margin)
    ax1.set_ylabel("Mode TOA [ns]", fontsize=12)
    ax1.set_title(f"Family Timing & Residuals: {fam_name} (OVERLAY)", fontsize=16, pad=20)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.grid(True, axis='both', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.9)
    ax2.set_ylabel(r"$\Delta$ (Mode - Anchor)", fontsize=10)
    ax2.set_xlabel("Channel Code (Ordered by Grid)", fontsize=12)
    ax2.grid(True, axis='both', linestyle='--', alpha=0.5)
    ax2.set_xticks(indices)
    ax2.set_xticklabels(sorted_codes, rotation=45, ha='right', fontsize=9)
    ax2.set_xlim(-0.6, len(sorted_codes) - 0.4)
    y_limits = ax2.get_ylim()
    max_res = max(abs(y_limits[0]), abs(y_limits[1]), 0.1)
    ax2.set_ylim(-max_res*1.15, max_res*1.15)
    fig_overlay.tight_layout()
    fig_overlay.subplots_adjust(hspace=0.05)
    
    generated_figures.append(fig_overlay)

    # ==========================================
    # B. GENERATE INDIVIDUAL PLOTS (One per Run)
    # ==========================================
    for run_item in multi_run_data:
        label = run_item['label']
        rows = run_item['rows']
        anchor_val = run_item['anchor']
        color = run_item.get('color', 'blue') 
        
        fig_single, (sax1, sax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                                                gridspec_kw={'height_ratios': [3, 1]})
        
        row_map = {r['code']: r['mode_post'] for r in rows}
        err_map = {r['code']: r.get('mode_err', 0.0) for r in rows}
        
        modes = [row_map.get(c, np.nan) for c in sorted_codes]
        errors = [err_map.get(c, 0.0) for c in sorted_codes]
        residuals = [m - anchor_val if np.isfinite(m) else np.nan for m in modes]

        # Single Points with Error Bars
        sax1.errorbar(indices, modes, yerr=errors, fmt='o', label=label, 
                      color=color, alpha=0.9, capsize=4, elinewidth=1.5, markersize=8)
        
        sax1.axhline(anchor_val, color='red', linestyle='--', linewidth=2.0, alpha=0.9,
                     label=f"Anchor ({anchor_val:.2f} ns)")

        # Single Residuals with Errors
        sax2.errorbar(indices, residuals, yerr=errors, fmt='o', color=color, 
                      alpha=0.8, capsize=3, elinewidth=1.2, markersize=6)
        sax2.axhline(0, color='red', linestyle='-', linewidth=1, alpha=0.5)

        # Format Single
        s_min_y, s_max_y = 999, -999
        for m, e in zip(modes, errors):
            if np.isfinite(m):
                s_min_y = min(s_min_y, m - e)
                s_max_y = max(s_max_y, m + e)
        
        if s_min_y != 999:
            sax1.set_ylim(s_min_y - margin, s_max_y + margin)

        sax1.set_ylabel("Mode TOA [ns]", fontsize=12)
        sax1.set_title(f"Family Timing: {fam_name} - {label}", fontsize=16, pad=20)
        sax1.grid(axis='y', linestyle=':', alpha=0.6)
        sax1.legend(loc='upper left', fontsize=10)
        
        sax2.set_ylabel(r"$\Delta$ (Mode - Anchor)", fontsize=10)
        sax2.set_xlabel("Channel Code", fontsize=12)
        sax2.grid(axis='y', linestyle='--', alpha=0.4)
        sax2.set_xticks(indices)
        sax2.set_xticklabels(sorted_codes, rotation=45, ha='right', fontsize=9)
        sax2.set_xlim(-0.6, len(sorted_codes) - 0.4)
        
        # Auto Scale Residuals Y
        s_max_res = max(max([abs(r) + e for r, e in zip(residuals, errors) if np.isfinite(r)] + [0.1]), 0.1)
        sax2.set_ylim(-s_max_res*1.2, s_max_res*1.2)

        fig_single.tight_layout()
        fig_single.subplots_adjust(hspace=0.05)
        
        generated_figures.append(fig_single)

    return generated_figures
# ================= LEGACY PLOTS (Defined but calls commented out) =================
def mosaic_pre_post_to_pdf_pages(pdf, root_files, grid, shifts, title_prefix):
    """
    Generates a Mosaic Plot (Pre vs Post shift) for EACH file in root_files.
    """
    bins = np.linspace(*XLIM, NBINS + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    nrows = len(grid)
    ncols = max(len(r) for r in grid)

    # Loop through each file to create a separate page/mosaic for it
    for fpath in root_files:
        run_label = _infer_run_label(fpath)
        full_title = f"{title_prefix} - {run_label}"

        try:
            with uproot.open(fpath) as f:
                if TREE_NAME not in f: continue
                tree = f[TREE_NAME]
                keys = set(tree.keys())

                fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.2 * nrows), sharex=True, sharey=True)
                axes = np.atleast_2d(axes)

                global_ymax = 1
                cache = {}
                
                # 1. Collect Data
                for r, row in enumerate(grid):
                    for c, code in enumerate(row):
                        if code is None:
                            cache[(r, c)] = None
                            continue
                        b, g, ch = parse_code(code)
                        k = branch_name(b, g, ch)
                        
                        if k not in keys:
                            cache[(r, c)] = ("missing", code)
                            continue

                        raw = prep_array(tree[k].array(library="np"))
                        if raw is None:
                            cache[(r, c)] = ("nostats", code)
                            continue

                        pre_stats = hist_stats(raw, bins)
                        if pre_stats is None:
                            cache[(r, c)] = ("nostats", code)
                            continue
                        mu_pre, mode_pre, h_pre = pre_stats

                        # Apply Shift
                        shift_val = shifts.get((b, g, ch), 0.0)
                        post = raw + shift_val
                        post_stats = hist_stats(post, bins)
                        
                        if post_stats is None:
                            cache[(r, c)] = ("nostats", code)
                            continue
                        mu_post, mode_post, h_post = post_stats
                        
                        global_ymax = max(global_ymax, int(h_pre.max()), int(h_post.max()))
                        cache[(r, c)] = ("ok", code, mode_pre, mode_post, h_pre, h_post, shift_val)

                # 2. Plot Data
                ln1 = ln2 = None
                for r, row in enumerate(grid):
                    for c, code in enumerate(row):
                        if c >= len(axes[r]): continue 
                        ax = axes[r, c]
                        
                        entry = cache.get((r, c))
                        if entry is None:
                            ax.axis("off")
                            continue
                        
                        status = entry[0]
                        if status != "ok":
                            ax.text(0.5, 0.5, f"{entry[1]}\n({status})", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                            continue
                        
                        _, code, mode_pre, mode_post, h_pre, h_post, shift = entry
                        
                        # Plot Pre (Blue) and Post (Orange)
                        ln1, = ax.step(centers, h_pre, where="mid", lw=1.0, alpha=0.6, color='tab:blue')
                        ln2, = ax.step(centers, h_post, where="mid", lw=1.2, alpha=0.9, color='tab:orange')
                        
                        ax.set_ylim(0, global_ymax * 1.15)
                        ax.set_title(f"{code} (s={shift:.2f})", fontsize=8, pad=1)
                        
                        # Vertical lines for Mode
                        ax.axvline(mode_pre, color='blue', ls=':', lw=0.8, alpha=0.5)
                        ax.axvline(mode_post, color='orange', ls='--', lw=1.0, alpha=0.8)

                fig.suptitle(full_title, fontsize=14)
                if ln1 and ln2:
                    fig.legend([ln1, ln2], ["Raw", "Shifted"], loc="upper right", fontsize=10)
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)

        except Exception as e:
            print(f"Failed Mosaic for {run_label}: {e}")
            plt.close()

def heatmap_to_pdf_pages(pdf, root_files, grid, shifts, quantity, apply_shift, title_prefix):
    """
    Generates a Heatmap for EACH file in root_files.
    """
    nrows = len(grid)
    ncols = max(len(r) for r in grid)
    bins = np.linspace(*XLIM, NBINS + 1)

    for fpath in root_files:
        run_label = _infer_run_label(fpath)
        full_title = f"{title_prefix} - {run_label} ({'Shifted' if apply_shift else 'Raw'} {quantity})"
        
        try:
            mat = np.full((nrows, ncols), np.nan, dtype=float)
            
            with uproot.open(fpath) as f:
                if TREE_NAME not in f: continue
                tree = f[TREE_NAME]
                keys = set(tree.keys())

                for r, row in enumerate(grid):
                    for c, code in enumerate(row):
                        if code is None: continue
                        b, g, ch = parse_code(code)
                        k = branch_name(b, g, ch)
                        
                        if k not in keys: continue
                        
                        raw = tree[k].array(library="np")
                        arr = prep_array(raw)
                        if arr is None: continue
                        
                        if apply_shift:
                            arr = arr + shifts.get((b, g, ch), 0.0)
                        
                        st = hist_stats(arr, bins)
                        if st is None: continue
                        mu, mode, _ = st
                        mat[r, c] = mu if quantity == "mean" else mode

            # Plot Heatmap
            fig, ax = plt.subplots(figsize=(10, 0.8 * nrows + 1.5))
            
            # Mask NaNs for better visualization
            masked_mat = np.ma.masked_invalid(mat)
            
            # Determine range based on data or fixed limits
            vmin = np.nanmin(mat) if np.nanmin(mat) > XLIM[0] else XLIM[0]
            vmax = np.nanmax(mat) if np.nanmax(mat) < XLIM[1] else XLIM[1]
            
            im = ax.imshow(masked_mat, origin="upper", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            
            # Text annotations
            for rr in range(nrows):
                for cc in range(ncols):
                    if cc >= len(grid[rr]) or grid[rr][cc] is None: continue
                    val = mat[rr, cc]
                    code = grid[rr][cc]
                    txt_val = f"{val:.2f}" if np.isfinite(val) else "—"
                    
                    # Contrast text color based on value
                    text_color = "white" if np.isfinite(val) and val < (vmin + (vmax-vmin)*0.5) else "black"
                    
                    ax.text(cc, rr, f"{code}\n{txt_val}", ha="center", va="center", fontsize=8, color=text_color)
                    
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(f"{quantity} [ns]")
            ax.set_title(full_title)
            ax.set_xticks([])
            ax.set_yticks([])
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        except Exception as e:
            print(f"Failed Heatmap for {run_label}: {e}")
            plt.close()

def column_overlay_to_pdf_pages(pdf, root_file, grid, shifts, title, apply_shift):
    bins = np.linspace(*XLIM, NBINS + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    nrows = len(grid)
    ncols = max(len(r) for r in grid)

    with uproot.open(root_file) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        for cc in range(ncols):
            codes = []
            for rr in range(nrows):
                if cc >= len(grid[rr]): continue
                code = grid[rr][cc]
                if code is not None: codes.append(code)

            fig, ax = plt.subplots(figsize=(12, 9))
            ax.set_title(f"{title}\nColumn {cc}", fontsize=14)
            ax.set_xlim(*XLIM)
            
            any_plotted = False
            ymax = 0
            for code in codes:
                b, g, ch = parse_code(code)
                k = branch_name(b, g, ch)
                if k not in keys: continue
                raw0 = tree[k].array(library="np")
                if raw0.size < MIN_RAW: continue
                arr = prep_array(raw0)
                if arr is None: continue
                if apply_shift:
                    arr = arr + shifts.get((b, g, ch), 0.0)
                
                st = hist_stats(arr, bins)
                if st is None: continue
                _, mode, h = st
                ymax = max(ymax, int(h.max()))
                ln, = ax.step(centers, h, where="mid", lw=1.6, label=f"{code} (mode={mode:.3f})")
                ax.axvline(mode, color=ln.get_color(), linestyle="--", linewidth=1.8, alpha=0.95)
                any_plotted = True
            
            if not any_plotted:
                ax.text(0.5, 0.5, "No usable channels", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.set_ylim(0, max(1, int(1.15 * ymax)))
                ax.legend(loc="upper right")
            pdf.savefig(fig)
            plt.close(fig)

def heatmap_to_pdf_pages(pdf, root_file, grid, shifts, quantity, apply_shift, title):
    nrows = len(grid)
    ncols = max(len(r) for r in grid)
    mat = np.full((nrows, ncols), np.nan, dtype=float)
    bins = np.linspace(*XLIM, NBINS + 1)

    with uproot.open(root_file) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())
        for r, row in enumerate(grid):
            for c, code in enumerate(row):
                if code is None: continue
                b, g, ch = parse_code(code)
                k = branch_name(b, g, ch)
                if k not in keys: continue
                arr = prep_array(tree[k].array(library="np"))
                if arr is None: continue
                if apply_shift:
                    arr = arr + shifts.get((b, g, ch), 0.0)
                st = hist_stats(arr, bins)
                if st is None: continue
                mu, mode, _ = st
                mat[r, c] = mu if quantity == "mean" else mode

    fig, ax = plt.subplots(figsize=(12, 0.65 * nrows + 1.2))
    im = ax.imshow(mat, origin="upper", aspect="auto", vmin=XLIM[0], vmax=XLIM[1])
    
    for rr in range(nrows):
        for cc in range(ncols):
            if cc >= len(grid[rr]) or grid[rr][cc] is None: continue
            val = mat[rr, cc]
            txt = f"{grid[rr][cc]}\n{val:.2f}" if np.isfinite(val) else f"{grid[rr][cc]}\n—"
            ax.text(cc, rr, txt, ha="center", va="center", fontsize=8)
            
    fig.colorbar(im, ax=ax).set_label(f"{quantity} [ns]")
    ax.set_title(title)
    pdf.savefig(fig)
    plt.close(fig)

# ================= MULTIPROCESSING WORKER =================
def process_family_pdf(args_tuple):
    """
    Worker function executed by multiprocessing pool.
    """
    # Unpack 8 items
    (fam_name, file_list, grid, shifts, tag, out_dir, is_test, multi_run_data) = args_tuple
    
    pid = os.getpid()
    safe_fam = fam_name.replace(" ", "_").replace("-", "_")
    temp_pdf_name = os.path.join(out_dir, f"temp_{pid}_{safe_fam}_{'TEST' if is_test else 'REF'}.pdf")
    
    print(f"  [Worker {pid}] Processing {fam_name} ({'TEST' if is_test else 'REF'})...")

    with PdfPages(temp_pdf_name) as pdf: 
        mosaic_pre_post_to_pdf_pages(pdf, file_list, grid, shifts, f"{tag} Mosaic")
        # 1. Error Bar Plots (Overlay + Individual Runs)
        if is_test and multi_run_data:
            figs_list = plot_family_summary_bars_figure(fam_name, multi_run_data)
            if figs_list:
                for fig in figs_list:
                    pdf.savefig(fig)
                    plt.close(fig)
        
        # 2. Overlays and Heatmaps
        if file_list and len(file_list) > 0:
            # FIX: Loop through the tuple and pass files individually
            for file_path in file_list:
                # Extract run label for a cleaner title if dealing with multiple test files
                run_label = _infer_run_label(file_path)
                current_tag = f"{tag} - {run_label}" if is_test else f"{tag}"
                
                heatmap_to_pdf_pages(pdf, file_path, grid, shifts, "mode", True, f"{current_tag} Heatmap")

    return temp_pdf_name


def row_overlay_to_pdf_pages(pdf, root_files, grid, shifts, title, apply_shift):
    """
    Overlays histograms from MULTIPLE root files on the same plot.
    root_files is a LIST of paths.
    """
    bins = np.linspace(*XLIM, NBINS + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    
    # Pre-assign colors for files
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    with uproot.open(root_files[0]) as f_dummy: # Just to get tree check? No, open in loop.
        pass

    for rr, row in enumerate(grid):
        codes = [code for code in row if code is not None]
        fig, ax = plt.subplots(figsize=(12, 9))
        codes_str = ", ".join(codes) if codes else "—"
        ax.set_title(f"{title}\nRow {rr}: [{codes_str}]", fontsize=13)
        ax.set_xlim(*XLIM)
        ax.set_xlabel(r"$|t_{\mathrm{final}}|$ [ns]")
        ax.set_ylabel("Events")
        ax.tick_params(direction="in", top=True, right=True)
        
        any_plotted = False
        ymax = 0
        
        # Loop over Grid Codes in this Row
        for code in codes:
            b, g, ch = parse_code(code)
            k = branch_name(b, g, ch)
            
            # Loop over input files to overlay them for this code
            for idx, fpath in enumerate(root_files):
                label = _infer_run_label(fpath)
                color = default_colors[idx % len(default_colors)]
                
                try:
                    with uproot.open(fpath) as f:
                        if TREE_NAME not in f: continue
                        tree = f[TREE_NAME]
                        if k not in tree: continue
                        
                        raw0 = tree[k].array(library="np")
                        if raw0.size < MIN_RAW: continue
                        arr = prep_array(raw0)
                        if arr is None: continue
                        
                        if apply_shift:
                            arr = arr + shifts.get((b, g, ch), 0.0)
                        
                        st = hist_stats(arr, bins)
                        if st is None: continue
                        _, mode, h = st
                        ymax = max(ymax, int(h.max()))
                        
                        # Plot
                        # Use Alpha to see overlaps
                        ln, = ax.step(centers, h, where="mid", lw=1.5, color=color, alpha=0.8,
                                      label=f"{code} {label} (m={mode:.2f})")
                        # Add thin vertical line for mode
                        ax.axvline(mode, color=color, linestyle="--", linewidth=1.0, alpha=0.6)
                        any_plotted = True
                except:
                    continue

        if not any_plotted:
            plt.close(fig)
            continue
        else:
            ax.set_ylim(0, max(1, int(1.15 * ymax)))
            ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
        
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def merge_pdfs(output_path, input_paths):
    merger = PdfWriter()
    valid_files = 0
    for path in input_paths:
        # 1. Check if file exists and is not empty
        if not os.path.exists(path) or os.path.getsize(path) == 0:
             print(f"WARNING: Skipping empty or missing file: {path}")
             continue
        
        # 2. Try to read the PDF
        try:
            with open(path, "rb") as f:
                reader = PdfReader(f)
                # 3. Check if it actually has pages
                if len(reader.pages) > 0:
                     for page in reader.pages:
                        merger.add_page(page)
                     valid_files += 1
                else:
                     print(f"WARNING: File has no pages: {path}")
        except Exception as e:
            print(f"WARNING: Failed to read {path}. Error: {e}")
            continue

    if valid_files > 0:
        with open(output_path, "wb") as f_out:
            merger.write(f_out)
        print(f"Merged PDF saved to: {output_path}")
    else:
        print(f"WARNING: No valid pages found. PDF not saved: {output_path}")
# ================= MAIN =================
# ================= MAIN =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1501_250928105227_converted_timingskim.root")
    ap.add_argument("--test", nargs="+", default=["/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1511_250928180741_converted_timingskim.root"])
    ap.add_argument("--outdir", default="/lustre/research/hep/akshriva/Dream-Timing/TRUE-HGtiming/calibration_studiesZ/MODE_CALIB_OUTPUT_y_1000")
    ap.add_argument("--calib-stat", choices=["mean", "mode"], default="mode")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count()), help="Number of parallel processes")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ref_label = _infer_run_label(args.reference)
    suffix = f"calib_{args.calib_stat}"

    # --- 1. Derive shifts (Reference) ---
    print(f"Deriving Calibration Shifts from Reference: {ref_label}...")
    shifts_by_family = {}
    anchor_targets = {}
    
    for fam in ["CER-Quartz", "CER-Plastic", "SCI"]:
        s, a_info, _ = derive_family_calibration_fixed_anchor(
            args.reference, FAMILIES[fam], ANCHORS[fam], args.calib_stat
        )
        shifts_by_family[fam] = s
        anchor_targets[fam] = a_info[1]['mu'] 
        
    shifts_by_family["CER-All"] = {**shifts_by_family["CER-Quartz"], **shifts_by_family["CER-Plastic"]}
    anchor_targets["CER-All"] = anchor_targets["CER-Quartz"]

    # Save JSON
    shifts_json_path = os.path.join(args.outdir, f"shifts_{ref_label}_{suffix}.json")
    meta = {"reference": args.reference, "calib_stat": args.calib_stat, "families": list(shifts_by_family.keys())}
    save_shifts_json(shifts_json_path, meta, shifts_by_family)

    # --- 2. Process TEST Files (Loop) ---
    print(f"Processing {len(args.test)} Test Files...")
    
    multi_run_by_fam = {fam: [] for fam in FAMILIES.keys()}
    all_test_labels = []

    for test_file in args.test:
        label = _infer_run_label(test_file)
        all_test_labels.append(label)
        print(f" -> Analyzing {label}...")
        
        per_family_test_rows = {}
        
        for fam in ["CER-Quartz", "CER-Plastic", "SCI"]:
            _, anchor_info, _ = derive_family_calibration_fixed_anchor(
                test_file, FAMILIES[fam], ANCHORS[fam], args.calib_stat
            )
            this_anchor = anchor_info[1]['mu']
            
            rows = compute_test_mode_table(test_file, FAMILIES[fam], shifts_by_family[fam])
            per_family_test_rows[fam] = rows
            
            multi_run_by_fam[fam].append({
                'label': label,
                'rows': rows,
                'anchor': this_anchor
            })

        q_anchor = next((item['anchor'] for item in multi_run_by_fam["CER-Quartz"] if item['label'] == label), 0.0)
        all_rows = per_family_test_rows["CER-Quartz"] + per_family_test_rows["CER-Plastic"]
        per_family_test_rows["CER-All"] = all_rows
        
        multi_run_by_fam["CER-All"].append({
            'label': label,
            'rows': all_rows,
            'anchor': q_anchor
        })

        txt_path = os.path.join(args.outdir, f"TEST_table_{label}_{suffix}.txt")
        write_mode_shift_txt(txt_path, f"TEST {label}", per_family_test_rows, args.calib_stat, ref_label, label)

    # --- 3. Multiprocessing Plot Generation ---
    fam_list = ["CER-Quartz", "CER-Plastic", "SCI", "CER-All"]
    
    tasks_ref = []
    for fam in fam_list:
        # FIX: Use tuple for file list: (args.reference,)
        tasks_ref.append((
            fam, (args.reference,), FAMILIES[fam], shifts_by_family[fam],
            f"REF {ref_label}", args.outdir, False, None
        ))

    tasks_test = []
    combined_test_tag = f"TEST ({len(args.test)} Runs)"
    for fam in fam_list:
        # FIX: Use tuple for file list: tuple(args.test)
        tasks_test.append((
            fam, tuple(args.test), FAMILIES[fam], shifts_by_family[fam],
            combined_test_tag, args.outdir, True, multi_run_by_fam[fam]
        ))

    print(f"Starting PDF generation with {args.workers} workers...")
    
    with multiprocessing.Pool(args.workers) as pool:
        print(" -> Generating Reference pages...")
        ref_temp_files = pool.map(process_family_pdf, tasks_ref)
        
        print(" -> Generating Test pages (Multi-Run)...")
        test_temp_files = pool.map(process_family_pdf, tasks_test)

    # --- 4. Merge ---
    final_ref_pdf = os.path.join(args.outdir, f"REF_file_{ref_label}_{suffix}.pdf")
    if len(all_test_labels) > 1:
        test_label_short = f"MultiRun_{len(all_test_labels)}_files"
    else:
        test_label_short = all_test_labels[0]
        
    final_test_pdf = os.path.join(args.outdir, f"TEST_file_{test_label_short}_{suffix}.pdf")

    print("Merging PDFs...")
    merge_pdfs(final_ref_pdf, ref_temp_files)
    merge_pdfs(final_test_pdf, test_temp_files)

    for f in ref_temp_files + test_temp_files:
        if os.path.exists(f): os.remove(f)

    print("Done!")
    print(f"Ref PDF: {final_ref_pdf}")
    print(f"Test PDF: {final_test_pdf}")

if __name__ == "__main__":
    main()