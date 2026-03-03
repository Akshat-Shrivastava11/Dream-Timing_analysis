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

NBINS = 200
CUT_MIN = 1.0
MIN_ENTRIES = 200
MIN_RAW = 500

HSPACE = 0.10
WSPACE = 0.05
# ================= ADC CUT CONFIG =================
AMP_THRESHOLD = 100.0  # Waveform must peak above this (baseline subtracted)
MIN_ADC_CUT = -100.0

# how many runs to print (mu,sigma) inside each cell
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
    # returns (ts_min, ts_max, val_cut, method)
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
    """
    Returns a dictionary of detector requirements.
    True: Must Fire (Signal < Cut)
    False: Must Veto (Signal >= Cut, i.e. did NOT fire)
    """
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
    """
    Computes PID mask with BASELINE SUBTRACTION.
    """
    requirements = get_particle_selection(particle_type)
    if not requirements:
        print(f"[PID] Warning: No requirements defined for '{particle_type}'. Using all events.")
        return None

    # Initialize mask as all True
    n_entries = tree.num_entries
    final_mask = np.ones(n_entries, dtype=bool)
    
    available_keys = set(tree.keys())

    print(f"  [PID] Applying selection for {particle_type} (with baseline subtraction)...")

    for det, must_fire in requirements.items():
        branch_name = PID_BRANCH_MAP.get(det)
        if not branch_name or branch_name not in available_keys:
            print(f"    [PID] SKIP: Branch '{branch_name}' ({det}) not found.")
            continue

        # Get cut parameters
        ts_min, ts_max, val_cut, method = get_service_drs_cut(det)

        try:
            # 1. Load Raw Waveform using Awkward Array
            waveforms = tree[branch_name].array(library="ak")
            
            if method == "Sum":
                # 2. CALCULATE BASELINE (First 30 bins)
                baseline = ak.mean(waveforms[:, :30], axis=1)
                
                # 3. SUBTRACT BASELINE
                waveforms_blsub = waveforms - baseline
                
                # 4. INTEGRATE WINDOW
                window = waveforms_blsub[:, int(ts_min):int(ts_max)]
                
                # Sum along axis 1 (the time samples)
                window_sum = ak.sum(window, axis=1)
                
                # Convert back to numpy for boolean logic
                window_sum_np = ak.to_numpy(window_sum)
                
                # 5. CHECK CUT
                is_fired = window_sum_np < val_cut
            else:
                print(f"    [PID] Method {method} not implemented. Skipping {det}.")
                continue

            # Update Mask Logic
            initial_count = np.sum(final_mask)
            if must_fire:
                final_mask = final_mask & is_fired
            else:
                final_mask = final_mask & (~is_fired)
            
            removed = initial_count - np.sum(final_mask)
            status = "FIRED" if must_fire else "VETOED"
            print(f"    [PID] {det:<12} ({status}): cut {val_cut:.1f} in [{ts_min}:{ts_max}] -> Removed {removed} events")

        except Exception as e:
            print(f"    [PID] CRITICAL ERROR processing {det}: {e}")
            continue

    return final_mask

def compute_adc_mask(tree, code_str):
    """
    Computes ADC quality cuts for the GRID channel.
    Uses drs_channel (raw waveforms) to cut on Amp and Min.
    """
    b, g, c = int(code_str[0]), int(code_str[1]), int(code_str[2])
    drs_br = f"DRS_Board{b}_Group{g}_Channel{c}"
    if drs_br not in tree:
        return np.ones(tree.num_entries, dtype=bool)
    
    waves = tree[drs_br].array(library="ak")
    baseline = ak.mean(waves[:, :30], axis=1)
    waves_blsub = waves - baseline
    
    peak = ak.max(waves_blsub, axis=1)
    min_adc = ak.min(waves_blsub, axis=1)
    
    mask = (peak >= AMP_THRESHOLD) & (min_adc >= MIN_ADC_CUT)
    return ak.to_numpy(mask)

# ================= 6mm MOSAIC GRIDS =================
# ================= 6mm MOSAIC GRIDS =================
# A visual "hole" of empty rows is inserted between the top section 
# (0xx/2xx) and bottom section (1xx/3xx) to represent the missing 3mm modules.

SCI_GRID = [
    [None,None,"605","604",None,None],
    [None,None,None,None,None,None],
    [None,None,"621","620",None,None],
    [None,None,None,None,None,None],
    [None,None,None,None,None,None],
    [None,"003","001","203","201",None],
    [None,"007","005","207","205",None],
    [None,"013","011","213","211",None],
    [None,"017","015","217","215",None],
    [None,"023","021","223","221",None],
    [None,"027","025","227","225",None],
    [None,"033","031","233","231",None],
    [None,None,"035",None,"235",None],
    # --- 3MM HOLE START ---
    [None,None,None,None,None,None],
    [None,None,None,None,None,None],
    [None,None,None,None,None,None],
    [None,None,None,None,None,None],
    # --- 3MM HOLE END ---
    [None,"103","101","303","301",None],
    [None,"107","105","307","305",None],
    [None,"113","111","313","311",None],
    [None,"117","115","317","315",None],
    [None,"123","121","323","321",None],
    [None,"127","125","327","325",None],
    [None,"133","131","333","331",None],
    [None,"533","135","537","335",None],
    [None,None,None,None,None,None],
    [None,None,None,None,None,None],
    [None,None,"421","420",None,None],
    [None,None,None,None,None,None],
    [None,None,"425","434",None,None],
]

QUARTZ_GRID = [
    [None,"603","602","601","600",None],
    [None,"697","606",None,None,None],
    [None,"613","612","611","610",None],
    [None,"617","616","615","614",None],
    [None,"625","624","623","622",None],
    ["637","631","630","627","626","636"],
    [None,"635","634","633","632",None],
    [None,None,None,None,None,None], 
    [None,None,None,None,None,None], 
    [None,"016","014","216","214",None],
    [None,"026","024","226","224",None],
    [None,None,"030",None,None,None],
    [None,None,"034",None,None,None],
    # --- 3MM HOLE START ---
    [None,None,None,None,None,None],
    [None,None,None,None,None,None],
    [None,None,None,None,None,None],
    [None,None,None,None,None,None],
    # --- 3MM HOLE END ---
    [None,"106","104","306","304",None],
    [None,"116","114","316","314",None],
    [None,"126","124","326","324",None],
    [None,"532","134","536","334",None],
    [None,"403","402","401","400",None],
    ["437","407","406","405","404","436"],
    [None,"413","412","411","410",None],
    [None,"417","416","415","414",None],
    [None,"425","424","423","422",None],
    [None,None,"427","426",None,None],
    [None,"433","432","431","430",None],
]

PLASTIC_GRID = [
    [None,"603","602","601","600",None],
    [None,"697","606",None,None,None],
    [None,"613","612","611","610",None],
    [None,None,None,None,None,None],
    [None,None,None,None,None,None],
    [None,"000","202","200",None,None],
    [None,"012","010","212","210",None],
    [None,"022","020","222","220",None],
    [None,"032",None,"232","230",None],
    # --- 3MM HOLE START ---
    [None,None,None,None,None,None],
    [None,None,None,None,None,None],
    [None,None,None,None,None,None],
    [None,None,None,None,None,None],
    # --- 3MM HOLE END ---
    [None,"102","100","302","300",None],
    [None,"112","110","312","310",None],
    [None,"122","120","322","320",None],
    [None,"132","130","332","330",None],
    [None,None,None,None,None,None],
    [None,None,None,None,None,None],
    [None,"425","424","423","422",None],
    [None,None,"427","426",None,None],
    [None,"433","432","431","430",None],
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

def _base_ok(g, c):
    # Channel 8 is reserved for reference
    if c == 8:
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

def _fileset_tag(files, pid_tag=""):
    runs = _extract_runs(files)
    base = f"files_n{len(files)}"
    if runs:
        base = f"runs{min(runs)}-{max(runs)}_n{len(files)}"
    if pid_tag:
        base += f"_{pid_tag}"
    return base

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
    h, _ = np.histogram(arr, bins=bins)
    if h.sum() == 0:
        return (np.nan, 0, h)
    idx = int(np.argmax(h))
    centers = 0.5 * (bins[1:] + bins[:-1])
    return (float(centers[idx]), int(h[idx]), h)

# ================= 6MM DYNAMIC t_final CALCULATION =================
def get_tfinal_6mm(tree, b, g, c):
    """
    Computes t_final for the 6mm setup on the fly from the LP2_50 branches:
    t_final(b,g,c) = ( t(b,g,c) - t(b,g,8) ) - ( t(0,3,7) - t(0,3,8) )
    """
    br_sig     = f"DRS_Board{b}_Group{g}_Channel{c}_t_peak"
    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8_t_peak"
    br_trg     = f"DRS_Board0_Group3_Channel7_t_peak"
    br_trg_ref = f"DRS_Board0_Group3_Channel8_t_peak"
    
    keys = tree.keys()
    for br in [br_sig, br_sig_ref, br_trg, br_trg_ref]:
        if br not in keys:
            return None
            
    # Load raw timing arrays
    arr_sig     = tree[br_sig].array(library="np")
    arr_sig_ref = tree[br_sig_ref].array(library="np")
    arr_trg     = tree[br_trg].array(library="np")
    arr_trg_ref = tree[br_trg_ref].array(library="np")
    
    # Ensure they're the same shape before doing math
    if not (arr_sig.shape == arr_sig_ref.shape == arr_trg.shape == arr_trg_ref.shape):
        return None
        
    return (arr_sig - arr_sig_ref) - (arr_trg - arr_trg_ref)


# ================= CORE: overlay mosaic =================
def make_mosaic_hist_overlay(files, grid, label, xlim, outdir,
                             tree_name, nbins, cut_min, min_entries, min_raw,
                             particle_type=None):
    os.makedirs(outdir, exist_ok=True)
    
    pid_tag = particle_type if particle_type else "NoPID"
    tag = _fileset_tag(files, pid_tag)
    out = os.path.join(outdir, f"HISTONLY_OVERLAY_{label}_{tag}.pdf")

    bins = np.linspace(xlim[0], xlim[1], nbins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    nrows = len(grid)
    ncols = max(len(r) for r in grid)

    # open
    opened = []
    labels_in_order = []
    
    print(f"--- Processing Mosaic for {pid_tag} ---")

    for fpath in files:
        try:
            uf = uproot.open(fpath)
            tree = uf[tree_name]
            keys = set(tree.keys())
            rl = _run_label(fpath)
            
            # --- COMPUTE PID MASK ONCE PER FILE ---
            pid_mask = None
            if particle_type:
                pid_mask = compute_pid_mask(tree, particle_type)

            opened.append((fpath, uf, tree, keys, rl, pid_mask))
            labels_in_order.append(rl)
        except Exception as e:
            print(f"[WARN] failed to open {fpath}: {e}")

    if len(opened) == 0:
        raise RuntimeError("No readable input files.")

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

            items = []

            for (_, _, tree, keys, rl, pid_mask) in opened:
                try:
                    # Load tfinal dynamically using 6mm formula
                    arr = get_tfinal_6mm(tree, b, g, ch)
                    if arr is None: 
                        continue
                    
                    # --- COMPUTE COMBINED MASK (PID + ADC CUTS) ---
                    # Start with PID mask if it exists, else all True
                    combined_mask = pid_mask if pid_mask is not None else np.ones(tree.num_entries, dtype=bool)
                    
                    # Compute ADC Cut mask for this specific grid channel
                    adc_mask = compute_adc_mask(tree, code)
                    combined_mask = combined_mask & adc_mask

                    # Apply the combined mask
                    if arr.shape[0] == combined_mask.shape[0]:
                        arr = arr[combined_mask]
                    else:
                        print(f"[WARN] Shape mismatch in {rl}: len={arr.shape[0]}, mask len={combined_mask.shape[0]}")
                        continue

                except Exception as e:
                    print(f"[ERROR] processing channel {code} in {rl}: {e}")
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
    for (_, uf, _, _, _, _) in opened:
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
        
        # Include ADC Cut info in the title/label for clarity
        adc_info = f" | ADC: Amp>{AMP_THRESHOLD}, Min>{MIN_ADC_CUT}"
        _global_ylabel(fig, f"Events ({pid_tag}){adc_info}")
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
        ax2.set_title(f"{label} overlays legend ({tag})\nPID: {pid_tag}{adc_info}",
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

def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-(x - mean)**2 / (2 * sigma**2))

# ================= NEW: single-channel overlay for 104 =================
def make_channel_overlay_with_modes(files, code_str, label, xlim, outdir,
                                    tree_name, nbins, cut_min, min_entries, min_raw,
                                    particle_type=None):
    os.makedirs(outdir, exist_ok=True)
    
    pid_tag = particle_type if particle_type else "NoPID"
    tag = _fileset_tag(files, pid_tag)
    out = os.path.join(outdir, f"CHANNEL_{code_str}_FIT_OVERLAY_{label}_{tag}_w_adccuts.pdf")

    # Parse which hardware channel corresponds to this code
    b, g, ch = _parse_code(code_str)

    bins = np.linspace(xlim[0], xlim[1], nbins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    opened = []
    labels_in_order = []
    
    print(f"--- Processing Single Channel {code_str} for {pid_tag} ---")
    
    for fpath in files:
        try:
            uf = uproot.open(fpath)
            tree = uf[tree_name]
            keys = set(tree.keys())
            rl = _run_label(fpath)
            
            # --- COMPUTE PID MASK ---
            pid_mask = None
            if particle_type:
                pid_mask = compute_pid_mask(tree, particle_type)

            opened.append((uf, tree, keys, rl, pid_mask))
            labels_in_order.append(rl)
        except Exception as e:
            print(f"[WARN] failed to open {fpath}: {e}")

    if not opened: return

    color_map = _build_color_map(labels_in_order)
    items = []  

    for (uf, tree, keys, rl, pid_mask) in opened:
        try:
            # 1. LOAD TFINAL DATA dynamically using 6mm formula
            arr = get_tfinal_6mm(tree, b, g, ch)
            if arr is None: 
                continue
            
            # 2. COMPUTE COMBINED MASK (PID + ADC CUTS)
            combined_mask = pid_mask if pid_mask is not None else np.ones(tree.num_entries, dtype=bool)
            
            # Use compute_adc_mask which analyzes raw waveforms
            adc_mask = compute_adc_mask(tree, code_str)
            combined_mask = combined_mask & adc_mask
            
            # 3. APPLY MASK
            if arr.shape[0] == combined_mask.shape[0]:
                arr = arr[combined_mask]
            else:
                print(f" [WARN] Shape mismatch in {rl}: data={arr.shape[0]}, mask={combined_mask.shape[0]}")
                continue
        except Exception as e:
            print(f" [ERROR] Failed masking in {rl}: {e}")
            continue

        arr = _prep(arr, xlim, cut_min, min_entries, min_raw)
        if arr is None: continue

        mode, max_counts, h = _mode_from_hist(arr, bins)
        if h.sum() == 0: continue

        # --- NORMALIZATION ---
        if h.max() > 0:
            h = h / h.max()

        # Focus tightly on the main peak
        fit_window = 1.5 
        mask = (centers >= mode - fit_window) & (centers <= mode + fit_window)
        x_fit = centers[mask]
        y_fit = h[mask]

        fit_mu, fit_sig, fwhm = np.nan, np.nan, np.nan
        y_gauss = None

        if len(x_fit) > 4:
            try:
                # p0: [Amplitude, Mean, Sigma]
                p0 = [1.0, mode, 0.3]
                popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0)
                fit_amp = popt[0]

                fit_mu = popt[1]
                fit_sig = abs(popt[2])
                # --- 2. THE PEAK=1.0 FIX ---
                # Re-normalize data so the peak of the smooth curve is exactly 1.0
                if fit_amp > 0:
                    h = h / fit_amp
                    y_gauss = gaussian(x_fit, 1.0, fit_mu, fit_sig)
                else:
                    y_gauss = gaussian(x_fit, *popt)

                fwhm = 2.355 * fit_sig
            except:
                fit_mu, fit_sig = mode, float(arr.std())
                fwhm = 2.355 * fit_sig

        items.append((rl, h, mode, fit_mu, fit_sig, fwhm, int(arr.size), x_fit, y_gauss))

    # Close handles
    for (uf, _, _, _, _) in opened:
        try: uf.close()
        except: pass

    items = sorted(items, key=lambda x: (_extract_int(x[0], r"run(\d+)"), _extract_int(x[0], r"_(\d{11,12})")))

    # --- Plotting ---
    with PdfPages(out) as pdf:
        fig, ax = plt.subplots(figsize=(12, 8)) 
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1.3)
        ax.set_xlabel(_xlabel())
        ax.set_ylabel(f"Normalized Events (Peak=1.0)")
        
        # Updated title to include ADC cut confirmation
        adc_title = f"Amp>{AMP_THRESHOLD}, Min>{MIN_ADC_CUT}"
        ax.set_title(f"Channel {code_str}: Peak Fitting (PID: {pid_tag} | {adc_title})", fontsize=14)

        handles, labels = [], []

        for (rl, h, mode, f_mu, f_sig, fwhm, n, x_f, y_f) in items:
            color = color_map[rl]
            ax.step(centers, h, where="mid", lw=1.0, alpha=0.3, color=color)
            
            if y_f is not None:
                line, = ax.plot(x_f, y_f, color=color, lw=2.5)
                handles.append(line)
            else:
                line = ax.axvline(mode, color=color, ls='--')
                handles.append(line)
            
            legend_str = (f"{rl}: Mean={f_mu:.3f}, $\sigma$={f_sig:.3f}, "
                          f"FWHM={fwhm:.3f} (N={n})")
            labels.append(legend_str)

        ax.legend(handles, labels, fontsize=8, ncol=2, frameon=True, 
                  loc="upper right", bbox_to_anchor=(1.0, 1.0))
        
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print("Saved:", out)


# ================= NEW: INVESTIGATE RAW TIMINGS =================
# ================= NEW: INVESTIGATE RAW TIMINGS =================
# ================= NEW: INVESTIGATE RAW TIMINGS =================
def investigate_raw_timings(files, code_str, outdir, tree_name, particle_type=None):
    """
    Plots the raw timing variables on Page 1, calculates t_final using exactly 
    MCP trigger board references (0, 1, 2, 3) on Page 2, and plots the absolute 
    value of t_final for all MCPs (zoomed 0-20ns) on Page 3.
    """
    os.makedirs(outdir, exist_ok=True)
    b, g, c = _parse_code(code_str)

    pid_tag = particle_type if particle_type else "NoPID"
    tag = _fileset_tag(files, pid_tag)
    out = os.path.join(outdir, f"RAW_TIMING_INVESTIGATION_CH{code_str}_{tag}.pdf")

    suffixes = ["_t_peak", "_LP2_50"]
    trg_boards_to_test = [0, 1, 2, 3]

    print(f"\n========================================================")
    print(f"--- Investigating Raw Timings & Multi-MCP t_final for CH {code_str} ({pid_tag}) ---")
    print(f"========================================================\n")

    with PdfPages(out) as pdf:
        pages_saved = 0
        
        for fpath in files:
            rl = _run_label(fpath)
            print(f">>> Opening file: {os.path.basename(fpath)} (Run: {rl})")
            
            try:
                uf = uproot.open(fpath)
                tree = uf[tree_name]
                keys = set(tree.keys())
                
                print(f"    Tree '{tree_name}' loaded with {tree.num_entries} total events.")
                
                pid_mask = None
                if particle_type:
                    pid_mask = compute_pid_mask(tree, particle_type)
                    if pid_mask is not None:
                        print(f"    [PID RESULT] {np.sum(pid_mask)} events passed.")

                for suffix in suffixes:
                    print(f"\n  -> Testing suffix: '{suffix}'")
                    
                    # -------------------------------------------------------------
                    # PAGE 1: RAW DISTRIBUTIONS (Original 4 Plots)
                    # -------------------------------------------------------------
                    fig_raw, axes_raw = plt.subplots(2, 2, figsize=(14, 10))
                    fig_raw.suptitle(f"Run: {rl} | Suffix: {suffix} | PID: {pid_tag}\nRaw Distributions (Channel {code_str})", fontsize=16)
                    axes_raw = axes_raw.flatten()

                    branches_raw = {
                        f"Signal (Board {b})": f"DRS_Board{b}_Group{g}_Channel{c}{suffix}",
                        f"Sig_Ref (Board {b})": f"DRS_Board{b}_Group{g}_Channel8{suffix}",
                        f"Trigger (Board 0)": f"DRS_Board0_Group3_Channel7{suffix}",
                        f"Trg_Ref (Board 0)": f"DRS_Board0_Group3_Channel8{suffix}"
                    }
                    
                    found_raw = False
                    for i, (label, br) in enumerate(branches_raw.items()):
                        ax = axes_raw[i]
                        if br in keys:
                            found_raw = True
                            arr = tree[br].array(library="np")
                            
                            if pid_mask is not None and len(arr) == len(pid_mask):
                                arr = arr[pid_mask]
                                
                            clean_arr = arr[~np.isnan(arr)]
                            if len(clean_arr) > 0:
                                ax.hist(clean_arr, bins=100, color='steelblue', alpha=0.8)
                                ax.set_title(f"{label}: {br}\nMean: {clean_arr.mean():.1f}, Std: {clean_arr.std():.1f}")
                                ax.set_xlabel("Raw Value")
                                ax.set_ylabel("Events")
                            else:
                                ax.set_title(f"{label}: {br}\n(No valid data)")
                        else:
                            ax.set_title(f"{label}: {br}\n(BRANCH NOT FOUND)")
                            ax.axis('off')

                    plt.tight_layout()
                    if found_raw:
                        pdf.savefig(fig_raw)
                        pages_saved += 1
                    plt.close(fig_raw)

                    # -------------------------------------------------------------
                    # PAGE 2: MULTI-MCP t_final COMPARISONS
                    # -------------------------------------------------------------
                    fig_tf, axes_tf = plt.subplots(2, 3, figsize=(18, 10))
                    fig_tf.suptitle(f"Run: {rl} | Suffix: {suffix} | PID: {pid_tag}\nt_final Comparisons (Channel {code_str})", fontsize=16)
                    axes_tf = axes_tf.flatten()

                    br_sig = f"DRS_Board{b}_Group{g}_Channel{c}{suffix}"
                    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8{suffix}"
                    
                    found_base = False
                    t_finals_all_boards = {} # Dictionary to store calculations for Page 3
                    
                    if br_sig in keys and br_sig_ref in keys:
                        found_base = True
                        arr_sig = tree[br_sig].array(library="np")
                        arr_sig_ref = tree[br_sig_ref].array(library="np")
                        
                        if pid_mask is not None:
                            arr_sig = arr_sig[pid_mask]
                            arr_sig_ref = arr_sig_ref[pid_mask]

                        # Plot (Sig - Sig_Ref)
                        delta_sig = arr_sig - arr_sig_ref
                        clean_delta = delta_sig[~np.isnan(delta_sig)]
                        if len(clean_delta) > 0:
                            axes_tf[0].hist(clean_delta, bins=100, color='purple', alpha=0.8)
                            axes_tf[0].set_title(f"(Signal - Sig_Ref) Only\nMean: {clean_delta.mean():.1f}, Std: {clean_delta.std():.1f}")
                        else:
                            axes_tf[0].set_title(f"(Signal - Sig_Ref)\n(All NaN)")

                        # Calculate t_final using boards 0, 1, 2, 3
                        plot_idx = 1
                        
                        for trg_b in trg_boards_to_test:
                            br_trg = f"DRS_Board{trg_b}_Group3_Channel7{suffix}"
                            br_trg_ref = f"DRS_Board{trg_b}_Group3_Channel8{suffix}"
                            
                            ax = axes_tf[plot_idx]
                            if br_trg in keys and br_trg_ref in keys:
                                arr_trg = tree[br_trg].array(library="np")
                                arr_trg_ref = tree[br_trg_ref].array(library="np")
                                
                                if pid_mask is not None:
                                    arr_trg = arr_trg[pid_mask]
                                    arr_trg_ref = arr_trg_ref[pid_mask]
                                
                                t_final_arr = delta_sig - (arr_trg - arr_trg_ref)
                                t_finals_all_boards[trg_b] = t_final_arr # Store for Page 3
                                
                                clean_t_final = t_final_arr[~np.isnan(t_final_arr)]
                                
                                if len(clean_t_final) > 0:
                                    ax.hist(clean_t_final, bins=100, color='darkorange', alpha=0.9)
                                    ax.set_title(f"t_final (Using Board {trg_b} MCP)\nMean: {clean_t_final.mean():.1f}, Std: {clean_t_final.std():.1f}")
                                    print(f"     [OK] t_final with Board {trg_b} MCP -> Mean: {clean_t_final.mean():.1f}")
                                else:
                                    ax.set_title(f"t_final (Board {trg_b} MCP)\n(All NaN)")
                            else:
                                ax.set_title(f"t_final (Board {trg_b} MCP)\n(Trigger Branches Missing)")
                                ax.axis('off')
                                print(f"     [MISSING] Trigger branches for Board {trg_b}")
                            
                            plot_idx += 1
                            
                        # --- 6th Plot on Page 2: Absolute value of t_final (Board 0) focused on 0-20 ---
                        ax_abs = axes_tf[5]
                        if 0 in t_finals_all_boards:
                            abs_t_final_0 = np.abs(t_finals_all_boards[0])
                            clean_abs_0 = abs_t_final_0[~np.isnan(abs_t_final_0)]
                            
                            if len(clean_abs_0) > 0:
                                ax_abs.hist(clean_abs_0, bins=100, range=(0, 20), color='forestgreen', alpha=0.9)
                                ax_abs.set_xlim(0, 20)
                                ax_abs.set_title(f"|t_final| (Board 0 MCP)\nxlim = (0, 20)\nMean: {clean_abs_0.mean():.1f}, Std: {clean_abs_0.std():.1f}")
                            else:
                                ax_abs.set_title(f"|t_final| (Board 0 MCP)\n(All NaN)")
                        else:
                            ax_abs.set_title(f"|t_final| (Board 0 MCP)\n(Missing Base Branches)")
                            ax_abs.axis('off')
                            
                    else:
                        axes_tf[0].set_title(f"Cannot compute t_final: Missing local Signal or Ref")
                        print(f"     [MISSING] Local Signal/Ref branches for t_final calculation.")
                        for j in range(1, 6):
                            axes_tf[j].axis('off')

                    plt.tight_layout()
                    if found_base:
                        pdf.savefig(fig_tf)
                        pages_saved += 1
                    plt.close(fig_tf)

                    # -------------------------------------------------------------
                    # PAGE 3: ABSOLUTE VALUE t_final (0-20ns) FOR ALL MCPs
                    # -------------------------------------------------------------
                    fig_abs, axes_abs = plt.subplots(2, 2, figsize=(14, 10))
                    fig_abs.suptitle(f"Run: {rl} | Suffix: {suffix} | PID: {pid_tag}\n|t_final| for All MCPs (0-20 Range)", fontsize=16)
                    axes_abs = axes_abs.flatten()

                    for i, trg_b in enumerate(trg_boards_to_test):
                        ax = axes_abs[i]
                        if found_base and trg_b in t_finals_all_boards:
                            abs_t_final = np.abs(t_finals_all_boards[trg_b])
                            clean_abs = abs_t_final[~np.isnan(abs_t_final)]
                            
                            if len(clean_abs) > 0:
                                ax.hist(clean_abs, bins=100, range=(0, 20), color='forestgreen', alpha=0.9)
                                ax.set_xlim(0, 20)
                                ax.set_title(f"|t_final| (Using Board {trg_b} MCP)\nMean: {clean_abs.mean():.1f}, Std: {clean_abs.std():.1f}")
                                ax.set_xlabel("|t_final| [ns]")
                                ax.set_ylabel("Events")
                            else:
                                ax.set_title(f"|t_final| (Board {trg_b} MCP)\n(All NaN)")
                        else:
                            ax.set_title(f"|t_final| (Board {trg_b} MCP)\n(Missing Branches)")
                            ax.axis('off')

                    plt.tight_layout()
                    if found_base and len(t_finals_all_boards) > 0:
                        pdf.savefig(fig_abs)
                        pages_saved += 1
                    plt.close(fig_abs)


            except Exception as e:
                print(f"[ERROR] Failed to process {fpath}: {e}")

        if pages_saved == 0:
            print(f"\n[CRITICAL] No pages were saved! Writing a dummy page to prevent PDF corruption.")
            fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "ERROR: NO VALID SIGNAL BRANCHES FOUND", ha='center', va='center', fontsize=16, color='red')
            plt.axis('off')
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\n========================================================")
    print(f"Saved Raw Investigation & Multi-MCP t_final to: {out}")
    print(f"========================================================\n")
# ================= MAIN =================
def main():
    global NBINS, CUT_MIN, MIN_ENTRIES, MIN_RAW, HSPACE, WSPACE, CELL_STATS_MAXLINES

    ap = argparse.ArgumentParser()
    
    # Updated default to the specific 6mm run mentioned
    ap.add_argument("--ana-files", nargs="+", 
                    default=["/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1504_250928133854_converted_timingskim.root"],
                    help="Explicit list of input ROOT files.")
    ap.add_argument("--ana-glob", default=None,
                    help="Glob for input ROOT files.")

    ap.add_argument("--run-min", type=int, default=None, help="Keep only runs >= run-min")
    ap.add_argument("--run-max", type=int, default=None, help="Keep only runs <= run-max")

    ap.add_argument("--tree", default=TREE_NAME, help="Tree name")
    ap.add_argument("--outdir", default="./TRUE-HGtiming/calibration_studiesZ/6mmtrial",
                    help="Output directory")

    # Updated default xlims for 6mm
    ap.add_argument("--xmin", type=float, default=0.0, help="Min |tfinal|")
    ap.add_argument("--xmax", type=float, default=500.0, help="Max |tfinal|")
    ap.add_argument("--nbins", type=int, default=NBINS, help="Histogram bins")
    ap.add_argument("--cut-min", type=float, default=CUT_MIN, help="Ignore |tfinal| < cut-min")
    ap.add_argument("--min-entries", type=int, default=MIN_ENTRIES, help="Min entries after cuts")
    ap.add_argument("--min-raw", type=int, default=MIN_RAW, help="Min raw entries before cuts")

    ap.add_argument("--cell-stats-lines", type=int, default=CELL_STATS_MAXLINES,
                    help="How many run μ,σ lines to print inside each channel cell.")

    ap.add_argument("--single-channel", default="605",
                    help="3-digit code bgc to make a standalone overlay plot for (default: 104).")

    # PID ARGUMENT
    ap.add_argument("--pid", default='electron', choices=["muon", "pion", "electron", "proton"],
                    help="Apply PID selection (muon, pion, electron, proton). Default: None.")

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
    
    pid_label = f"PID_{args.pid}" if args.pid else "AllParticles"

    # Only running the 6mm grids now
    # make_mosaic_hist_overlay(files, QUARTZ_GRID,  f"6MM-Quartz_{pid_label}",       xlim, args.outdir,
    #                          args.tree, NBINS, CUT_MIN, MIN_ENTRIES, MIN_RAW, args.pid)
    # make_mosaic_hist_overlay(files, PLASTIC_GRID, f"6MM-Plastic_{pid_label}",      xlim, args.outdir,
    #                          args.tree, NBINS, CUT_MIN, MIN_ENTRIES, MIN_RAW, args.pid)
    # make_mosaic_hist_overlay(files, SCI_GRID,     f"6MM-Sci_{pid_label}",          xlim, args.outdir,
    #                          args.tree, NBINS, CUT_MIN, MIN_ENTRIES, MIN_RAW, args.pid)

    # make_channel_overlay_with_modes(files, args.single_channel, f"6MM-ALL-RUNS_{pid_label}", xlim, args.outdir,
    #                                 args.tree, NBINS, CUT_MIN, MIN_ENTRIES, MIN_RAW, args.pid)

    investigate_raw_timings(files, args.single_channel, args.outdir, args.tree, args.pid)

    print("All done.")
    print("All done.")

if __name__ == "__main__":
    main()