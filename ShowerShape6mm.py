#!/usr/bin/env python3
import os
import re
import glob
import argparse
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

# ================= DEFAULTS =================
TREE_NAME = "EventTree"

NBINS = 100
CUT_MIN = 1.0
MIN_ENTRIES = 100
MIN_RAW = 500

HSPACE = 0.10
WSPACE = 0.05
# ================= ADC CUT CONFIG =================
AMP_THRESHOLD = 100.0  
MIN_ADC_CUT = -100.0

CELL_STATS_MAXLINES = 3

# ================= PID CONFIGURATION =================
PID_BRANCH_MAP = {
    "PSD": "DRS_Board7_Group1_Channel1",
    "HoleVeto": "DRS_Board7_Group1_Channel6",
    "NC": "DRS_Board7_Group1_Channel7",
    "T3": "DRS_Board7_Group2_Channel0",
    "T4": "DRS_Board7_Group2_Channel1",
    "KT1": "DRS_Board7_Group2_Channel2",
    "KT2": "DRS_Board7_Group2_Channel3",
    "TTUMuonVeto": "DRS_Board7_Group2_Channel4",
    "Cer474": "DRS_Board7_Group2_Channel5",
    "Cer519": "DRS_Board7_Group2_Channel6",
    "Cer537": "DRS_Board7_Group2_Channel7",
}

def get_service_drs_cut(service_drs: str) -> tuple:
    cut_default = (0, 1000, -5e4, "Sum")
    cuts = {
        "HoleVeto": (100, 350, -2e3, "Sum"),
        "PSD": (100, 400, -3500.0, "Sum"),
        "TTUMuonVeto": (200, 400, -2e3, "Sum"),
        "Cer474": (800, 900, -2000.0, "Sum"),
        "Cer519": (450, 550, -1000.0, "Sum"),
        "Cer537": (400, 500, -500.0, "Sum"),
    }
    return cuts.get(service_drs, cut_default)

def get_particle_selection(particle_type: str) -> dict:
    selections = {
        "muon": {
            "TTUMuonVeto": True,
            "PSD": False,
        },
        "pion": {
            "TTUMuonVeto": False,
            "PSD": False,
            "Cer474": True, "Cer519": True, "Cer537": True
        },
        "electron": {
            "TTUMuonVeto": False,
            "PSD": True,
            "Cer474": True, "Cer519": True, "Cer537": True
        },
        "proton": {
            "TTUMuonVeto": False,
            "PSD": False,
            "Cer474": False, "Cer519": False, "Cer537": False
        },
    }
    return selections.get(particle_type.lower(), {})

def compute_pid_mask(tree, particle_type):
    requirements = get_particle_selection(particle_type)
    if not requirements:
        print(f"[PID] Warning: No requirements defined for '{particle_type}'. Using all events.")
        return None

    n_entries = tree.num_entries
    final_mask = np.ones(n_entries, dtype=bool)
    available_keys = set(tree.keys())

    print(f"  [PID] Applying selection for {particle_type} (with baseline subtraction)...")

    for det, must_fire in requirements.items():
        branch_name = PID_BRANCH_MAP.get(det)
        if not branch_name or branch_name not in available_keys:
            continue

        ts_min, ts_max, val_cut, method = get_service_drs_cut(det)

        try:
            waveforms = tree[branch_name].array(library="ak")
            if method == "Sum":
                baseline = ak.mean(waveforms[:, :30], axis=1)
                waveforms_blsub = waveforms - baseline
                window = waveforms_blsub[:, int(ts_min):int(ts_max)]
                window_sum = ak.sum(window, axis=1)
                window_sum_np = ak.to_numpy(window_sum)
                is_fired = window_sum_np < val_cut
            else:
                continue

            initial_count = np.sum(final_mask)
            if must_fire:
                final_mask = final_mask & is_fired
            else:
                final_mask = final_mask & (~is_fired)
            
            removed = initial_count - np.sum(final_mask)
            status = "FIRED" if must_fire else "VETOED"
            print(f"    [PID] {det:<12} ({status}): -> Removed {removed} events")

        except Exception as e:
            print(f"    [PID] CRITICAL ERROR processing {det}: {e}")
            continue

    return final_mask


# ================= 6mm MOSAIC GRIDS =================
QUARTZ_GRID = [
    # --- Missing Top Section ---
    
    # --- Upper Body ---
    [None,  None,  "617", "616", "615", "614", None,  None],
    [None,  None,  "625", "624", "623", "622", None,  None],
    [None,  "637", "631", "630", "627", "626", "636", None],
    
    # --- Main Body with 500-Series Side Wings ---
    ["515", "514", "635", "634", "633", "632", "501", "500"],
    [None,  None,  "002", None,  None,  None,  None,  None],
    ["517", "516", "006", "004", "206", "204", "503", "502"],
    [None,  None,  "016", "014", "216", "214", None,  None],
    ["521", "520", "026", "024", "226", "224", "505", "504"],
    [None,  None,  "030", None,  None,  None,  None,  None],
    [None,  None,  "530", "034", "534", "234", None,  None],
    ["523", "522", "106", "104", "306", "304", "507", "506"],
    [None,  None,  "116", "114", "316", "314", None,  None],
    ["525", "524", "126", "124", "326", "324", "511", "510"],
    [None,  None,  "532", "134", "536", "334", None,  None],
    ["527", "526", "403", "402", "401", "400", "513", "512"],
    
    # --- Lower Body ---
    [None,  "437", "407", "406", "405", "404", "436", None],
    [None,  None,  "413", "412", "411", "410", None,  None],
    [None,  None,  "417", "416", "415", "414", None,  None],
]

PLASTIC_GRID = [
    # --- Top Section (Olive) ---
    [None,  None,  "603", "602", "601", "600", None,  None],
    [None,  None,  None,  "607", "606", None,  None,  None],
    [None,  None,  "613", "612", "611", "610", None,  None],
    
    # --- Physical Gap ---
    [None,  None,  None,  None,  None,  None,  None,  None],
    [None,  None,  None,  None,  None,  None,  None,  None],
    [None,  None,  None,  None,  None,  None,  None,  None],

    # --- Bottom Section (Teal) ---
    [None,  None,  "425", "424", "423", "422", None,  None],
    [None,  None,  None,  "427", "426", None,  None,  None],
    [None,  None,  "433", "432", "431", "430", None,  None],
]

SCI_GRID = [
    [None,None,"605","604",None,None],
    [None,None,None,None,None,None],
    [None,None,"621","620",None,None],
    [None,None,None,None,None,None],

    [None,"533","135","537","335",None],
    [None,None,None,None,None,None],
    [None,None,None,None,None,None],
    [None,None,"421","420",None,None],
    [None,None,None,None,None,None],
    [None,None,"425","434",None,None],
]


# ================= HELPERS =================
def _xlabel(): return r"$|t_{\mathrm{final}}|$ [ns]"

def _tighten(fig, left=0.05, right=0.98, top=0.985, bottom=0.035, hspace=HSPACE, wspace=WSPACE):
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace)

def _parse_code(code_str): return int(code_str[0]), int(code_str[1]), int(code_str[2])

def _base_ok(g, c): return False if c == 8 else True

def _run_label(path: str) -> str:
    m = re.search(r"(run\d+_\d{11,12})", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]

def _extract_runs(files):
    runs = []
    for f in files:
        m = re.search(r"run(\d+)", os.path.basename(f))
        if m: runs.append(int(m.group(1)))
    return runs

def _fileset_tag(files, pid_tag=""):
    runs = _extract_runs(files)
    base = f"files_n{len(files)}"
    if runs: base = f"runs{min(runs)}-{max(runs)}_n{len(files)}"
    if pid_tag: base += f"_{pid_tag}"
    return base

def _resolve_files(args):
    files = list(args.ana_files) if args.ana_files else sorted(glob.glob(args.ana_glob))
    if args.run_min is not None and args.run_max is not None:
        files = [f for f in files if args.run_min <= int(re.search(r"run(\d+)", os.path.basename(f)).group(1)) <= args.run_max]

    def _sort_key(p):
        b = os.path.basename(p)
        mrun = re.search(r"run(\d+)", b)
        mts = re.search(r"_(\d{11,12})(?:_|\.|$)", b)
        return (int(mrun.group(1)) if mrun else 10**9, int(mts.group(1)) if mts else 10**18, b)
    return sorted(files, key=_sort_key)

def _extract_int(s: str, pattern: str) -> int:
    m = re.search(pattern, s)
    return int(m.group(1)) if m else 10**18

def _build_color_map(runlabels):
    n = len(runlabels)
    cmap = plt.get_cmap("tab20") if n <= 20 else plt.get_cmap("turbo") if n <= 256 else plt.get_cmap("hsv")
    return {rl: cmap(x) for x, rl in zip(np.linspace(0.0, 1.0, n, endpoint=False), runlabels)}


# ================= 6MM TIMING CALCULATION =================
def get_tfinal_6mm(tree, b, g, c, suffix):
    """
    Computes the ABSOLUTE VALUE of t_final for the 6mm setup on the fly:
    |t_final(b,g,c)| = |( t(b,g,c) - t(b,g,8) ) - ( t(0,3,7) - t(0,3,8) )|
    """
    br_sig     = f"DRS_Board{b}_Group{g}_Channel{c}{suffix}"
    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8{suffix}"
    br_trg     = f"DRS_Board0_Group3_Channel7{suffix}"
    br_trg_ref = f"DRS_Board0_Group3_Channel8{suffix}"
    
    keys = tree.keys()
    for br in [br_sig, br_sig_ref, br_trg, br_trg_ref]:
        if br not in keys:
            return None
            
    arr_sig     = tree[br_sig].array(library="np")
    arr_sig_ref = tree[br_sig_ref].array(library="np")
    arr_trg     = tree[br_trg].array(library="np")
    arr_trg_ref = tree[br_trg_ref].array(library="np")
    
    if not (arr_sig.shape == arr_sig_ref.shape == arr_trg.shape == arr_trg_ref.shape):
        return None
        
    t_final = (arr_sig - arr_sig_ref) - (arr_trg - arr_trg_ref)
    
    return np.abs(t_final)


# ================= SHOWER SHAPE WAVEFORM MOSAIC =================
def make_waveform_pdf(files, code_str, label, xlim, outdir, tree_name, particle_type=None, n_events=200):
    """
    Finds valid events for the given channel strictly inside the xlim window.
    Selects the first N valid events.
    Plots one positive waveform per page. Optimized for speed using ax.clear().
    Ignores t_peak completely.
    """
    wf_dir = os.path.join(outdir, "WaveformPlots")
    os.makedirs(wf_dir, exist_ok=True)
    
    clean_code = re.sub(r'[^0-9]', '', code_str)[:3]
    b, g, c = int(clean_code[0]), int(clean_code[1]), int(clean_code[2])
    
    pid_tag = particle_type if particle_type else "NoPID"
    tag = _fileset_tag(files, pid_tag)
    
    safe_label = label.replace(":", "").replace(" ", "_")
    out_name = os.path.join(wf_dir, f"Waveforms_CH{clean_code}_{safe_label}_{tag}_L50.pdf")
    
    main_br   = f"DRS_Board{b}_Group{g}_Channel{c}"
    timing_br = f"{main_br}_LP2_50"
    suffix    = "_LP2_50"
    
    print(f"--- Processing Individual Waveforms for {label} (Ch {clean_code}) in Window: {xlim} ---")
    
    with PdfPages(out_name) as pdf:
        # Create a single figure/axis outside the loop to keep plotting fast
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for fpath in files:
            rl = _run_label(fpath)
            try:
                with uproot.open(fpath) as f:
                    tree = f[tree_name]
                    pid_mask = compute_pid_mask(tree, particle_type) if particle_type else np.ones(tree.num_entries, dtype=bool)
                    
                    # 6MM TIMING LOGIC
                    tf_array = get_tfinal_6mm(tree, b, g, c, suffix)
                    if tf_array is None: continue
                    
                    # Apply constraints
                    valid_mask = pid_mask & ~np.isnan(tf_array) & (tf_array > 0)
                    valid_mask = valid_mask & (tf_array >= xlim[0]) & (tf_array <= xlim[1])
                    
                    valid_indices = np.where(valid_mask)[0]
                    if len(valid_indices) == 0: continue
                    
                    # Grab the first n_events (no mode calculations needed)
                    selected_evs = valid_indices[:n_events]

                    for ev in selected_evs:
                        ax.clear() # Clear the axis instead of making a new figure (Super Fast)
                        
                        tf_val = tf_array[ev]
                        
                        w = tree[main_br].array(entry_start=ev, entry_stop=ev+1, library="np")[0]
                        timing_raw = tree[timing_br].array(entry_start=ev, entry_stop=ev+1, library="np")[0]
                        
                        baseline = np.mean(w[:30])
                        
                        # Positive Waveform
                        w_sub = -(w - baseline)
                        
                        time_axis = np.arange(len(w)) * 0.2
                        
                        ax.plot(time_axis, w_sub, color="tab:blue", lw=2.0)
                        
                        if not np.isnan(timing_raw):
                            ax.axvline(x=timing_raw, color='black', linestyle='--', alpha=0.8, lw=2.0, label=f"L50: {timing_raw:.2f} ns")
                            ax.set_xlim(timing_raw - 15, timing_raw + 25)
                        else:
                            ax.set_xlim(0, 200) 
                            ax.text(0.5, 0.5, "NaN Timing", transform=ax.transAxes, ha='center', color='red', fontsize=16)

                        # Clean Formatting
                        ax.minorticks_on()
                        ax.tick_params(axis='both', which='major', labelsize=12, direction='in', top=True, right=True)
                        ax.set_xlabel("Time [ns]", fontsize=14, loc='right')
                        ax.set_ylabel("Amplitude [ADC]", fontsize=14, loc='top')
                        
                        ax.set_title(f"$\mathbf{{CaloX}}$ {label} (Ch {clean_code}) | Event: {ev} | $|t_{{final}}|$: {tf_val:.2f} ns", fontsize=14, pad=12)
                        ax.legend(loc='lower right', frameon=False, fontsize=12)
                        
                        fig.tight_layout()
                        pdf.savefig(fig)

            except Exception as e:
                print(f"Failed extracting waveforms from {fpath}: {e}")
                
        plt.close(fig)

    print(f"Saved Waveforms to: {out_name}")


# ================= HISTOGRAM MOSAIC OVERLAY =================
def make_mosaic_hist_overlay(files, grid, label, xlim, outdir, tree_name, nbins, cut_min, min_entries, min_raw, suffix, particle_type=None):
    os.makedirs(outdir, exist_ok=True)
    
    pid_tag = particle_type if particle_type else "NoPID"
    tag = _fileset_tag(files, pid_tag)
    safe_suffix = suffix.strip("_")
    out = os.path.join(outdir, f"HISTONLY_OVERLAY_{label}_{safe_suffix}_{tag}.pdf")

    bins = np.linspace(xlim[0], xlim[1], nbins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    nrows = len(grid)
    ncols = max(len(r) for r in grid)

    opened = []
    labels_in_order = []
    
    print(f"--- Processing t_final Hist Mosaic for {label} with suffix {suffix} ---")

    for fpath in files:
        try:
            uf = uproot.open(fpath)
            tree = uf[tree_name]
            rl = _run_label(fpath)
            pid_mask = compute_pid_mask(tree, particle_type) if particle_type else None
            opened.append((fpath, uf, tree, set(tree.keys()), rl, pid_mask))
            labels_in_order.append(rl)
        except Exception as e:
            print(f"[WARN] failed to open {fpath}: {e}")

    if len(opened) == 0:
        raise RuntimeError("No readable input files.")

    color_map = _build_color_map(labels_in_order)
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

            items = []
            for (_, _, tree, keys, rl, pid_mask) in opened:
                try:
                    # 6MM TIMING LOGIC
                    arr = get_tfinal_6mm(tree, b, g, ch, suffix)
                    if arr is None: continue
                    
                    combined_mask = pid_mask if pid_mask is not None else np.ones(tree.num_entries, dtype=bool)

                    if arr.shape[0] == combined_mask.shape[0]:
                        arr = arr[combined_mask]
                    else:
                        continue
                        
                    arr = arr[~np.isnan(arr)]
                    arr = arr[(arr >= xlim[0]) & (arr <= xlim[1])]
                    
                    if len(arr) < 25:
                        continue

                    mu = float(arr.mean())
                    sig = float(arr.std())
                    n = int(arr.size)

                    h, _ = np.histogram(arr, bins=bins)
                    if h.sum() == 0: continue

                    items.append((rl, h, mu, sig, n))
                    global_ymax = max(global_ymax, int(h.max()))

                except Exception as e:
                    continue

            if len(items) == 0:
                cell[(r, c)] = {"code": code, "status": "nostats", "items": []}
            else:
                items = sorted(items, key=lambda x: (_extract_int(x[0], r"run(\d+)"), _extract_int(x[0], r"_(\d{11,12})")))
                cell[(r, c)] = {"code": code, "status": "ok", "items": items}

    for (_, uf, _, _, _, _) in opened:
        try: uf.close()
        except: pass

    with PdfPages(out) as pdf:
        fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 2.0 * nrows), sharex=True, sharey=True)
        if nrows == 1 and ncols == 1: axes = np.array([[axes]])
        elif nrows == 1: axes = np.array([axes])
        elif ncols == 1: axes = np.array([[ax] for ax in axes])

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
                    ax.text(0.5, 0.5, f"{code}\n({status})", ha="center", va="center", transform=ax.transAxes, fontsize=9)
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
                    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=7,
                            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"))

        for ax in axes[-1, :]:
            if ax.axison and ax.get_visible():
                ax.set_xlabel(_xlabel())
        
        _tighten(fig, left=0.05, right=0.98, top=0.985, bottom=0.035, hspace=0.10, wspace=0.04)
        pdf.savefig(fig)
        plt.close(fig)

        # Legend Page
        legend_labels = sorted(list(legend_handles.keys()), key=lambda s: (_extract_int(s, r"run(\d+)"), _extract_int(s, r"_(\d{11,12})")))
        handles = [legend_handles[k] for k in legend_labels]

        nitems = len(legend_labels)
        ncol = max(1, int(np.ceil(nitems / 28)))
        fontsize = 11 if ncol == 1 else 10 if ncol == 2 else 9

        fig2 = plt.figure(figsize=(8.5, 11))
        ax2 = fig2.add_subplot(111)
        ax2.axis("off")
        ax2.set_title(f"{label} overlays legend ({tag})\nPID: {pid_tag} | Suffix: {suffix}", fontsize=13, pad=12)

        fig2.legend(handles, legend_labels, loc="center", fontsize=fontsize, frameon=False, ncol=ncol)
        pdf.savefig(fig2)
        plt.close(fig2)

    print("Saved Hist Mosaic:", out)


# ================= MAIN =================
def main():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--ana-files", nargs="+", 
                    default=["/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1474_250927193729_converted_timingskim.root" 
                        "/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1475_250927200644_converted_timingskim.root"],
                    help="Explicit list of input ROOT files.")
    ap.add_argument("--ana-glob", default=None, help="Glob for input ROOT files.")
    ap.add_argument("--run-min", type=int, default=None, help="Keep only runs >= run-min")
    ap.add_argument("--run-max", type=int, default=None, help="Keep only runs <= run-max")

    ap.add_argument("--tree", default=TREE_NAME, help="Tree name")
    
    # Locked to Showershape directory
    ap.add_argument("--outdir", default="./Showershape/6mm/", help="Output directory")

    ap.add_argument("--xmin", type=float, default=9.0, help="Min |tfinal|")
    ap.add_argument("--xmax", type=float, default=13.0, help="Max |tfinal|")
    ap.add_argument("--nbins", type=int, default=NBINS, help="Histogram bins")
    ap.add_argument("--cut-min", type=float, default=CUT_MIN, help="Ignore |tfinal| < cut-min")
    ap.add_argument("--min-entries", type=int, default=MIN_ENTRIES, help="Min entries after cuts")
    ap.add_argument("--min-raw", type=int, default=MIN_RAW, help="Min raw entries before cuts")

    # Locked PID default to pion
    ap.add_argument("--pid", default='pion', choices=["muon", "pion", "electron", "proton"],
                    help="Apply PID selection. Default: pion.")

    args = ap.parse_args()

    files = _resolve_files(args)
    if len(files) == 0:
        raise SystemExit("ERROR: no files matched your selection")

    print(f"Found {len(files)} files.")
    xlim = (args.xmin, args.xmax)
    os.makedirs(args.outdir, exist_ok=True)
    
    # Strictly using L50, absolutely no t_peak
    suffixes = ["_LP2_50"]
    
    # Bundle the 6MM Grids
    grids = {
        "6MM-Quartz": QUARTZ_GRID,
        "6MM-Plastic": PLASTIC_GRID,
        "6MM-Sci": SCI_GRID
    }
    
    # 1. Plot the Individual Waveform grids ONLY for the targeted channels
    target_channels = ["617", "616", "615", "614",
                       "603", "602", "601", "600",
                       "605","604",
                       ]
    print(f"\n--- Extracting Waveforms for Target Channels: {target_channels} ---")
    
    for code in target_channels:
        try:
            b, g, c = _parse_code(code)
            if _base_ok(g, c):
                make_waveform_pdf(
                    files, code, f"6MM-Target_{code}", xlim, args.outdir, 
                    args.tree, args.pid, n_events=200
                )
        except Exception as e:
            print(f"[WARN] Skipping code {code}: {e}")
            
    # 2. Make the t_final histogram mosaics for the full grids
    for name, grid in grids.items():
        for suffix in suffixes:
            make_mosaic_hist_overlay(
                files, grid, name, xlim, args.outdir,
                args.tree, args.nbins, args.cut_min, args.min_entries, args.min_raw, suffix, args.pid
            )

    print("All done.")

if __name__ == "__main__":
    main()