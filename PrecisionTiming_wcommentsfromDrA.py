#!/usr/bin/env python3
import os
import re
import glob
import argparse
import numpy as np
import uproot
import awkward as ak
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator
import mplhep as hep

# Apply the CMS style globally
plt.style.use(hep.style.CMS)
# ================= GRIDS & CHANNELS =================
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

def extract_channels(grid):
    return [ch for row in grid for ch in row if ch is not None]

# ================= CONFIGURATION =================
TREE_NAME = "EventTree"
AMP_THRESHOLD = 100.0  
MIN_ADC_CUT = -100.0


# Define your wirechamber branch here
WC_CHANNELS = {
    "L1": "DRS_Board7_Group0_Channel0",
    "R1": "DRS_Board7_Group0_Channel1",
}
WC_X_CUT = 100.0
# Using your requested negative limits
# FAMILIES = {
#     "Plastic": {"channels": ["100","102","112", "110"], "tmin": -14.5, "tmax": -11.5, "legend": "Cherenkov-Plastic", "color": "red"},
#     "Quartz":  {"channels": ["104","106", "304","114"], "tmin": -15.0, "tmax": -11.5, "legend": "Cherenkov-Quartz",  "color": "blue"},
#     "SCI":     {"channels": ["105", "107","111","117"], "tmin": -13.5, "tmax":  -9.5, "legend": "Scintillating",     "color": "green"}
# }

#all channels
FAMILIES = {
    "Plastic": {
        "channels": extract_channels(PLASTIC_GRID), 
        "tmin": -14.5, "tmax": -11.5, 
        "legend": "Cherenkov-Plastic", "color": "red"
    },
    "Quartz":  {
        "channels": extract_channels(QUARTZ_GRID),  
        "tmin": -15.0, "tmax": -11.5, 
        "legend": "Cherenkov-Quartz",  "color": "blue"
    },
    # "SCI":     {
    #     "channels": extract_channels(SCI_ALL_GRID), 
    #     "tmin": -13.5, "tmax":  -9.5, 
    #     "legend": "Scintillating",     "color": "green"
    # }

    "SCI":     {
        "channels": [
            "103", "101", "303", "301",
            "107", "105", "307", "305",
            "113", "111", "313", "311"
        ], 
        "tmin": -13.5, "tmax":  -9.5, 
        "legend": "Scintillating", "color": "green"
    }
}


FAMILY_DISPLAY_NAMES = {
    "Plastic": "Toray PJR-FB750 (Plastic)",
    "SCI": "SCSF-81J (Scintillator)",
    "Quartz": "FSHA (Fused-silica)",
}

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

# ================= PID & ADC MASKS =================
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

# ================= Z POSITION MAPPING ================
def get_z_position(run_label):
    if "run1513" in run_label:
        # if "192918" in run_label: return -54.5
        # if "194230" in run_label: return -400.3
        if "192918" in run_label: return 163.5   # Was -54.5  (+ 218.0)
        if "194230" in run_label: return -182.3  # Was -400.3 (+ 218.0)
    match = re.search(r"run(\d+)", run_label)
    run_num = int(match.group(1)) if match else None
    #z_map = {1501: -168.0, 1507: -218.0, 1511: -268.0}
    z_map = {1501: 50.0, 1507: 0.0, 1511: -50.0}
    return z_map.get(run_num, -999.0)

def get_particle_selection(particle_type: str) -> dict:
    selections = {
        "muon": {"TTUMuonVeto": True, "PSD": False},
        "pion": {"TTUMuonVeto": False, "PSD": False, "Cer474": True, "Cer519": True, "Cer537": True},
        "electron": {"TTUMuonVeto": False, "PSD": True, "Cer474": True, "Cer519": True, "Cer537": True},
        "proton": {"TTUMuonVeto": False, "PSD": False, "Cer474": False, "Cer519": False, "Cer537": False},
    }
    return selections.get(particle_type.lower(), {})

def compute_pid_mask(tree, particle_type):
    requirements = get_particle_selection(particle_type)
    if not requirements: return None

    n_entries = tree.num_entries
    final_mask = np.ones(n_entries, dtype=bool)
    available_keys = set(tree.keys())

    for det, must_fire in requirements.items():
        branch_name = PID_BRANCH_MAP.get(det)
        if not branch_name or branch_name not in available_keys: continue
        ts_min, ts_max, val_cut, method = get_service_drs_cut(det)

        try:
            waveforms = tree[branch_name].array(library="ak")
            if method == "Sum":
                baseline = ak.mean(waveforms[:, :30], axis=1)
                waveforms_blsub = waveforms - baseline
                window_sum = ak.sum(waveforms_blsub[:, int(ts_min):int(ts_max)], axis=1)
                is_fired = ak.to_numpy(window_sum) < val_cut
            else:
                continue

            final_mask = final_mask & is_fired if must_fire else final_mask & (~is_fired)
        except Exception:
            continue
    return final_mask

def compute_adc_mask(tree, code_str):
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


def get_hit_times_vectorized(events):
    """Finds the bin index of the waveform minimum."""
    if events.ndim != 2:
        return np.zeros(len(events))
    baselines = np.mean(events[:, :20], axis=1, keepdims=True)
    corrected = events - baselines
    return np.argmin(corrected, axis=1)

def compute_wc_mask(tree, limit=WC_X_CUT):
    """Generates a boolean mask based on the raw waveform minimum bin difference."""
    br_l1 = WC_CHANNELS["L1"]
    br_r1 = WC_CHANNELS["R1"]
    
    if br_l1 not in tree.keys() or br_r1 not in tree.keys():
        print("    [WARN] Wirechamber waveform branches missing. Skipping WC cut.")
        return np.ones(tree.num_entries, dtype=bool)
    
    # Pull the raw waveforms (Notice: no suffix here!)
    L1 = ak.to_numpy(tree[br_l1].array(library="ak"))
    R1 = ak.to_numpy(tree[br_r1].array(library="ak"))
    
    # Get the bin index of the peak
    L1_t = get_hit_times_vectorized(L1)
    R1_t = get_hit_times_vectorized(R1)
    
    # Calculate X value (difference in indices)
    x_positions = L1_t - R1_t
    
    # Apply the cut |X| < 80
    mask = np.abs(x_positions) < limit
    
    return mask


# ================= CORE TIMING =================
def get_tfinal_3mm(tree, b, g, c, suffix):
    """
    Computes the raw t_final for the 3mm/6mm setup on the fly:
    t_final(b,g,c) = ( t(b,g,c) - t(b,g,8) ) - ( t(b,3,7) - t(b,3,8) )
    """
    br_sig     = f"DRS_Board{b}_Group{g}_Channel{c}{suffix}"
    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8{suffix}"
    br_trg     = f"DRS_Board0_Group3_Channel7{suffix}"
    br_trg_ref = f"DRS_Board0_Group3_Channel8{suffix}"
    
    keys = tree.keys()
    if any(br not in keys for br in [br_sig, br_sig_ref, br_trg, br_trg_ref]):
        return None
            
    arr_sig     = tree[br_sig].array(library="np")
    arr_sig_ref = tree[br_sig_ref].array(library="np")
    arr_trg     = tree[br_trg].array(library="np")
    arr_trg_ref = tree[br_trg_ref].array(library="np")
        
    return (arr_sig - arr_sig_ref) - (arr_trg - arr_trg_ref)


    

def gaussian_peak_1(x, mean, sigma):
    # Gaussian normalized to peak at 1.0
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

# ================= HELPERS =================
def _parse_code(code_str):
    return int(code_str[0]), int(code_str[1]), int(code_str[2])

def _run_label(path: str) -> str:
    m = re.search(r"(run\d+_\d{11,12})", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]

def _fileset_tag(files, pid_tag):
    return f"{pid_tag}_{len(files)}files"

def _build_color_map(labels):
    cmap = plt.get_cmap("tab10")
    return {lbl: cmap(i % 10) for i, lbl in enumerate(labels)}

def _extract_int(s, pattern):
    m = re.search(pattern, s)
    return int(m.group(1)) if m else 0

def _mode_from_hist(arr, bins):
    h, _ = np.histogram(arr, bins=bins)
    if h.sum() == 0: return (np.nan, 0, h)
    idx = int(np.argmax(h))
    centers = 0.5 * (bins[1:] + bins[:-1])
    return (float(centers[idx]), int(h[idx]), h)


# ================= INDIVIDUAL CHANNEL PLOTTING =================
def make_channel_overlay_with_modes(files, code_str, label, xlim, outdir,
                                    tree_name, nbins, suffixes, particle_type=None): 
    os.makedirs(outdir, exist_ok=True)
    
    pid_tag = particle_type if particle_type else "NoPID"
    tag = _fileset_tag(files, pid_tag)
    out = os.path.join(outdir, f"FIT_CH{code_str}_{label}_CombinedTimings_{tag}.pdf")

    b, g, ch = _parse_code(code_str)
    
    bins = np.linspace(xlim[0], xlim[1], nbins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    opened = []
    labels_in_order = []
    
    print(f"\n  [Individual] ----------------------------------------------------")
    print(f"  [Individual] Processing Channel {code_str} | Hard Cut Window: [{xlim[0]:.2f}, {xlim[1]:.2f}]")
    
    for fpath in files:
        print(f"    [FILE] Opening {os.path.basename(fpath)}...")
        try:
            uf = uproot.open(fpath)
            tree = uf[tree_name]
            keys = set(tree.keys())
            rl = _run_label(fpath)
            pid_mask = compute_pid_mask(tree, particle_type) if particle_type else None
            
            if pid_mask is not None:
                print(f"      [INFO] PID Mask calculated: {np.sum(pid_mask)} events passed out of {tree.num_entries}")
            else:
                print(f"      [INFO] No PID mask applied. Total events: {tree.num_entries}")
                
            opened.append((uf, tree, keys, rl, pid_mask))
            labels_in_order.append(rl)
        except Exception as e:
            print(f"      [ERROR] Failed to open {fpath}: {e}")
            continue

    if not opened: 
        print("    [WARN] No files successfully opened. Skipping.")
        return
        
    color_map = _build_color_map(labels_in_order)

    with PdfPages(out) as pdf:
        for suffix in suffixes:
            print(f"    [SUFFIX] Processing timing suffix: {suffix}")
            safe_suffix = suffix.strip("_")
            items = []  

            for (uf, tree, keys, rl, pid_mask) in opened:
                print(f"      -> Extracting Run {rl} data...")
                try:
                    arr = get_tfinal_3mm(tree, b, g, ch, suffix)
                    if arr is None: 
                        print(f"        [SKIP] Branches missing for {rl}")
                        continue
                    
                    print(f"        [CUTS] Raw array size: {len(arr)}")
                    
                    # --- ADDED ADC CUT HERE ---
                    adc_mask = compute_adc_mask(tree, code_str)
                    combined_mask = pid_mask & adc_mask if pid_mask is not None else adc_mask
                    
                    if arr.shape[0] == combined_mask.shape[0]:
                        arr = arr[combined_mask]
                        print(f"        [CUTS] Size after PID & ADC mask: {len(arr)}")
                    else: 
                        print(f"        [ERROR] Shape mismatch between data ({arr.shape[0]}) and mask ({combined_mask.shape[0]})")
                        continue
                        
                    arr = arr[~np.isnan(arr)]
                    print(f"        [CUTS] Size after removing NaNs: {len(arr)}")
                    
                    arr = arr[(arr >= xlim[0]) & (arr <= xlim[1])]
                    print(f"        [CUTS] Size after Hard Window cut {xlim}: {len(arr)}")
                    
                    if len(arr) < 25: 
                        print(f"        [SKIP] Too few events left in window ({len(arr)} < 25). Cannot fit.")
                        continue

                except Exception as e: 
                    print(f"        [ERROR] Processing {rl}: {e}")
                    continue

                mode, max_counts, h = _mode_from_hist(arr, bins)
                if h.sum() == 0 or h.max() == 0: 
                    print(f"        [SKIP] Histogram is empty for {rl}")
                    continue
                    
                print(f"        [INFO] Mode found at {mode:.3f} with {max_counts} counts. Attempting fit...")

                h_plot = h / h.max()
                x_fit = centers
                y_fit = h_plot

                fit_mu, fit_sig, fwhm = np.nan, np.nan, np.nan
                x_smooth, y_gauss = None, None

                if len(x_fit) >= 4:
                    try:
                        p0 = [mode, arr.std()]
                        bounds = ([xlim[0] - 2.0, 0.001], [xlim[1] + 2.0, 10.0])
                        popt, _ = curve_fit(gaussian_peak_1, x_fit, y_fit, p0=p0, bounds=bounds)
                        
                        fit_mu = popt[0]
                        fit_sig = abs(popt[1])
                        fwhm = 2.355 * fit_sig
                        print(f"        [FIT OK] Mu: {fit_mu:.3f}, Sigma: {fit_sig:.3f}, FWHM: {fwhm:.3f}")
                            
                    except Exception as e:
                        fit_mu = float(arr.mean())
                        fit_sig = float(arr.std())
                        fwhm = 2.355 * fit_sig
                        print(f"        [FIT FAIL] Exception: {e}")
                        print(f"        [FIT FAIL] Falling back to raw stats -> Mu: {fit_mu:.3f}, Sigma: {fit_sig:.3f}")
                    
                    x_smooth = np.linspace(xlim[0], xlim[1], 500)
                    y_gauss = gaussian_peak_1(x_smooth, fit_mu, fit_sig)

                items.append((rl, h_plot, mode, fit_mu, fit_sig, fwhm, int(arr.size), x_smooth, y_gauss))

            items = sorted(items, key=lambda x: (_extract_int(x[0], r"run(\d+)"), _extract_int(x[0], r"_(\d{11,12})")))

            print(f"    [PLOT] Generating PDF page for {code_str} ({len(items)} successful fits)...")
            fig, ax = plt.subplots(figsize=(12, 8)) 
            ax.set_xlim(*xlim)
            ax.set_ylim(0, 1.4) 

            ax.set_xlabel("Time of Arrival [ns]", fontsize=14, loc='right')
            ax.set_ylabel("Normalized Events", fontsize=14, loc='top')

            display_name = "Positron" if particle_type and particle_type.lower() == "electron" else (particle_type.capitalize() if particle_type else "All")
            timing_label = "LP2_{50}" if "LP2" in safe_suffix else "t_{peak}"
            fam_name = label.split('_')[0].replace("6MM-", "").replace("3MM-", "")
            
            header_text = r"$\LARGE $\mathbf{CaloX}$ $\mathit{Data}$" + f"  40 GeV{display_name} | ${timing_label}$ | {fam_name} | {code_str}"
            ax.text(0.0, 1.02, header_text, transform=ax.transAxes, fontsize=12, va='bottom', ha='left')

            ax.minorticks_on()
            ax.tick_params(axis='both', which='major', labelsize=12, length=6, direction='in', top=True, right=True)
            ax.tick_params(axis='both', which='minor', length=3, direction='in', top=True, right=True)

            handles, labels_list = [], []
            for (rl, h_plot, mode, f_mu, f_sig, fwhm, n, x_smooth, y_gauss) in items:
                color = color_map[rl]
                ax.step(centers, h_plot, where="mid", lw=1.2, alpha=0.4, color=color)
                
                if y_gauss is not None:
                    line, = ax.plot(x_smooth, y_gauss, color=color, lw=2.5)
                    handles.append(line)
                else:
                    line = ax.axvline(mode, color=color, ls='--')
                    handles.append(line)
                
                legend_str = (f"{rl}: Mean={f_mu:.2f}, $\sigma$={f_sig:.2f}, FWHM={fwhm:.2f} (N={n})")
                labels_list.append(legend_str)

            if handles:
                ax.legend(handles, labels_list, fontsize=9, ncol=1, frameon=False, loc="upper right")
                
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    for (uf, _, _, _, _) in opened:
        try: uf.close()
        except: pass
    print(f"  [Individual] Saved: {out}")

# ================= FAMILY OVERLAY PLOTTING =================
def make_family_overlay(files, fam_name, channels, xlim, outdir, tree_name, suffix, particle_type=None):
    os.makedirs(outdir, exist_ok=True)
    pid_tag = particle_type if particle_type else "NoPID"
    tag = _fileset_tag(files, pid_tag)
    safe_suffix = suffix.strip("_")
    out = os.path.join(outdir, f"FIT_FAMILY_OVERLAY_{fam_name}_{tag}.pdf")

    nbins = 50
    cmap = plt.get_cmap("tab10")
    color_map = {ch: cmap(i % 10) for i, ch in enumerate(channels)}

    print(f"\n  [Family] --------------------------------------------------------")
    print(f"  [Family] Generating Family-Level Overlay for: {fam_name} | Window: {xlim}")

    with PdfPages(out) as pdf:
        for fpath in files:
            rl = _run_label(fpath)
            print(f"    [RUN] Processing Run {rl}...")
            
            try:
                uf = uproot.open(fpath)
                tree = uf[tree_name]
                pid_mask = compute_pid_mask(tree, particle_type) if particle_type else None
            except Exception as e:
                print(f"      [ERROR] Could not open {rl}: {e}")
                continue
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_xlim(*xlim)
            ax.set_ylim(0, 1.4)
            
            ax.set_xlabel("Time of Arrival [ns]")
            ax.set_ylabel("Normalized Events")
            
            display_name = "Positron" if particle_type and particle_type.lower() == "electron" else (particle_type.capitalize() if particle_type else "All")
            timing_label = "LP2_{50}" if "LP2" in safe_suffix else "t_{peak}"
            
            # --- USE MPLHEP FOR THE HEADER ---
            right_label = f"40 GeV {display_name} | ${timing_label}$ | Family: {fam_name} | {rl}"
            hep.cms.label(ax=ax, exp="CaloX", data=True, rlabel=right_label)

            handles, labels_list = [], []

            for ch_str in channels:
                b, g, ch = _parse_code(ch_str)
                print(f"      -> Extracting Ch {ch_str}...")
                
                try:
                    arr = get_tfinal_3mm(tree, b, g, ch, suffix)
                    if arr is None: 
                        print(f"        [SKIP] Ch {ch_str}: Branches missing")
                        continue
                    
                    # --- ADDED ADC CUT HERE ---
                    adc_mask = compute_adc_mask(tree, ch_str)
                    combined_mask = pid_mask & adc_mask if pid_mask is not None else adc_mask
                    
                    if arr.shape[0] == combined_mask.shape[0]:
                        arr = arr[combined_mask]
                    else: 
                        print(f"        [ERROR] Ch {ch_str}: Shape mismatch")
                        continue
                    
                    arr = arr[~np.isnan(arr)]
                    arr_cut = arr[(arr >= xlim[0]) & (arr <= xlim[1])]
                    
                    if len(arr_cut) < 25: 
                        print(f"        [SKIP] Ch {ch_str}: Too few events in window ({len(arr_cut)})")
                        continue
                        
                    print(f"        [INFO] Ch {ch_str}: {len(arr_cut)} events ready for fitting.")
                    
                    bins_ch = np.linspace(xlim[0], xlim[1], nbins + 1)
                    mode, max_counts, h = _mode_from_hist(arr_cut, bins_ch)
                    if h.sum() == 0 or h.max() == 0: 
                        print(f"        [SKIP] Ch {ch_str}: Empty histogram")
                        continue
                    
                    h_plot = h / h.max()
                    centers_ch = 0.5 * (bins_ch[1:] + bins_ch[:-1])
                    
                    p0 = [mode, arr_cut.std()]
                    bounds = ([xlim[0] - 2.0, 0.001], [xlim[1] + 2.0, 10.0])
                    popt, _ = curve_fit(gaussian_peak_1, centers_ch, h_plot, p0=p0, bounds=bounds)
                    
                    fit_mu, fit_sig = popt[0], abs(popt[1])
                    fwhm = 2.355 * fit_sig
                    
                    x_smooth = np.linspace(xlim[0], xlim[1], 500)
                    y_gauss = gaussian_peak_1(x_smooth, fit_mu, fit_sig)
                    
                    color = color_map[ch_str]
                    ax.step(centers_ch, h_plot, where="mid", lw=1.2, alpha=0.3, color=color)
                    line, = ax.plot(x_smooth, y_gauss, color=color, lw=2.5)
                    handles.append(line)
                    
                    legend_str = f"Ch {ch_str}: $\mu$={fit_mu:.2f}, $\sigma$={fit_sig:.2f} (N={len(arr_cut)})"
                    labels_list.append(legend_str)
                    print(f"        [FIT OK] Ch {ch_str}: Mu={fit_mu:.2f}, Sig={fit_sig:.2f}")
                    
                except Exception as e:
                    fit_mu = float(arr_cut.mean())
                    fit_sig = float(arr_cut.std())
                    x_smooth = np.linspace(xlim[0], xlim[1], 500)
                    y_gauss = gaussian_peak_1(x_smooth, fit_mu, fit_sig)
                    
                    color = color_map[ch_str]
                    line, = ax.plot(x_smooth, y_gauss, color=color, lw=2.5, linestyle="--")
                    handles.append(line)
                    labels_list.append(f"Ch {ch_str} (Fit Failed): $\mu$={fit_mu:.2f}, $\sigma$={fit_sig:.2f}")
                    print(f"        [FIT FAIL] Ch {ch_str}: Exception -> {e}")
                    print(f"        [FIT FAIL] Ch {ch_str}: Using raw stats. Mu={fit_mu:.2f}, Sig={fit_sig:.2f}")

            if handles:
                ax.legend(handles, labels_list, fontsize=12, ncol=1, frameon=False, loc="upper right")
                
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            uf.close()
            print(f"    [PLOT] Run {rl} page appended to PDF.")
    print(f"  [Family] Saved: {out}")

# ================= EXECUTION =================
def process_all_channels(files, outdir, tree_name, particle_type):
    print(f"\n[MAIN] Starting processing loop for {len(FAMILIES)} families.")
    for fam_name, config in FAMILIES.items():
        channels = config["channels"]
        xlim = [config["tmin"], config["tmax"]]
        print(f"\n================ Processing Family: {fam_name} ({len(channels)} Channels) ================")
        print(f"Family Hardcoded Bounds: {xlim}")
        
        suffix_to_test = "_LP2_50" 
        
        # 1. Individual channel mode overlays
        for ch_str in channels:
            make_channel_overlay_with_modes(
                files=files, 
                code_str=ch_str, 
                label=fam_name, 
                xlim=xlim, 
                outdir=outdir,
                tree_name=tree_name, 
                nbins=100, 
                suffixes=[suffix_to_test],
                particle_type=particle_type
            )
            
        # 2. Family-Level Overlay
        make_family_overlay(
            files=files, 
            fam_name=fam_name, 
            channels=channels, 
            xlim=xlim,
            outdir=outdir, 
            tree_name=tree_name, 
            suffix=suffix_to_test, 
            particle_type=particle_type
        )

def _resolve_files(args):
    if args.ana_files: files = list(args.ana_files)
    else: files = sorted(glob.glob(args.ana_glob))

    def _sort_key(p):
        b = os.path.basename(p)
        mrun = re.search(r"run(\d+)", b)
        r = int(mrun.group(1)) if mrun else 10**9
        mts = re.search(r"_(\d{11,12})(?:_|\.|$)", b)
        ts = int(mts.group(1)) if mts else 10**18
        return (r, ts, b)

    return sorted(files, key=_sort_key)


def style_paper_axes(ax, xlabel, ylabel, particle_type):
    ax.set_xlabel(xlabel)
    ax.set_xlim(-19, 19)
    ax.set_ylabel(ylabel)
    
    display_name = "Positron" if particle_type.lower() == "electron" else particle_type.capitalize()
    right_label = f"40 GeV {display_name}"
    
    # Set data=False to prevent mplhep from overriding your custom text
    hep.cms.label(ax=ax, exp="CaloX", data=False, llabel="Z-scan", rlabel=right_label)
# ================= UPDATED VELOCITY PLOT WITH FIT ERR# ================= UPDATED VELOCITY PLOT WITH FIT ERRORS & CM/NS =================
# ================= UPDATED VELOCITY PLOT WITH FIT ERRORS & CM/NS =================
def create_z_toa_plot(plot_data, txt_path, pid_label, particle_type):
    outdir = os.path.dirname(txt_path)
    pdf_path = os.path.join(outdir, f"Z_vs_TOA_Fits_{pid_label}.pdf")
    
    # Mapping dictionary for the new names
    NAME_MAP = {
        "Plastic": "Toray PJR-FB750 (Plastic)",
        "Quartz":  "FSHA (Fused-silica)",
        "SCI":     "SCSF-81J (Scintillator)"
    }
    
    print("\n[VELOCITY] --------------------------------------------------------")
    print(f"[VELOCITY] Calculating Independent Velocity Fits (PID: {pid_label})")
    
    with open(txt_path, "a") as f_out:
        f_out.write("\n" + "=" * 115 + "\n")
        f_out.write(f"{'FAMILY':<10} | {'VELOCITY [cm/ns]':<18} | {'V_ERROR [cm/ns]':<18} | {'FIT EQUATION'}\n")
        f_out.write("=" * 115 + "\n")
        
        with PdfPages(pdf_path) as pdf:
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Pass the new y-axis label
            style_paper_axes(ax, "Z Position [cm]", "Mean Time of Arrival [ns]", particle_type)
            
            # Force the new y-axis and x-axis limits to prevent squishing
            ax.set_ylim(-15.0, -9.5)
            ax.set_xlim(-20, 20) 

            # Starting Y position for the custom text (in axes coordinates 0-1)
            text_y_pos = 0.95 

            for fam, channels_dict in plot_data.items():
                combined_z = []
                combined_mu = []
                combined_sig = []
                
                for ch, data in channels_dict.items():
                    combined_z.extend(data["z"])
                    combined_mu.extend(data["mu"])
                    combined_sig.extend(data["sig"])
                    
                if not combined_z: 
                    continue
                    
                z_arr = np.array(combined_z) / 10.0 # Convert mm to cm for fitting
                mu_arr = np.array(combined_mu)
                sig_mean_arr = np.array(combined_sig)

                color = FAMILIES[fam]["color"]
                
                weights = 1.0 / sig_mean_arr
                params, cov = np.polyfit(z_arr, mu_arr, 1, w=weights, cov=True)
                slope, intercept = params[0], params[1]
                slope_err = np.sqrt(cov[0,0])
                intercept_err = np.sqrt(cov[1,1])
                
                # CORRECTED MATH: slope is already in ns/cm. Velocity v = 1/slope (cm/ns).
                v_cm_ns = (1.0 / abs(slope)) if slope != 0 else 0
                
                # CORRECTED ERROR: dv = (1/slope^2) * dslope
                v_err_cm_ns = (1.0 / (slope**2)) * slope_err if slope != 0 else 0
                
                eq_str = f"t = ({slope:.4f} +/- {slope_err:.4f})z {'+' if intercept >= 0 else '-'} ({abs(intercept):.2f} +/- {intercept_err:.2f})"
                
                f_out.write(f"{fam:<10} | {v_cm_ns:<18.3f} | {v_err_cm_ns:<18.3f} | {eq_str}\n")
                
                # Scale the fit line boundaries for cm
                z_fit = np.linspace(min(z_arr) - 2.0, max(z_arr) + 2.0, 200)
                
                ax.errorbar(z_arr, mu_arr, yerr=sig_mean_arr, fmt='o', color=color, capsize=3, markersize=6, elinewidth=2)
                ax.plot(z_fit, slope * z_fit + intercept, '-', color=color, linewidth=2)

                # Format the display string: e.g FSHA (Fused-silica) 20.587+/-0.443 cm/ns
                display_name = NAME_MAP.get(fam, fam)
                text_str = f"{display_name} {v_cm_ns:.3f} $\pm$ {v_err_cm_ns:.3f} cm/ns"
                
                ax.text(0.95, text_y_pos, text_str, 
                        transform=ax.transAxes, 
                        color=color, 
                        fontsize=20,          
                        fontweight='bold', 
                        verticalalignment='top',
                        horizontalalignment='right')
                
                # Shift down the Y position for the next family's text
                text_y_pos -= 0.06 

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"[VELOCITY] Saved Independent Z vs TOA plot to: {pdf_path}")

# ================= UPDATED SHARED INTERCEPT PLOT WITH RICH PRINTS =================
# ================= UPDATED SHARED INTERCEPT PLOT WITH FIT ERRORS & 10^8 =================
def create_shared_intercept_plot(plot_data, txt_path, pid_label, particle_type):
    outdir = os.path.dirname(txt_path)
    pdf_path = os.path.join(outdir, f"Z_vs_TOA_SharedInterceptFits_{pid_label}.pdf")
    
    print("\n[SHARED FIT] ------------------------------------------------------")
    print(f"[SHARED FIT] Calculating Global Shared Intercept Fit (PID: {pid_label})")
    
    active_families = [fam for fam, data in plot_data.items() if len(data["z"]) > 0]
    if len(active_families) < 2: 
        print("  [WARN] Not enough families with data to perform a shared intercept fit.")
        return

    all_fam_idx, all_z, all_mu, all_sig = [], [], [], []
    indiv_slopes, indiv_intercepts = [], []

    for i, fam in enumerate(active_families):
        z_arr, mu_arr, sig_arr = np.array(plot_data[fam]["z"]), np.array(plot_data[fam]["mu"]), np.array(plot_data[fam]["sig"])
        all_fam_idx.extend([i] * len(z_arr))
        all_z.extend(z_arr)
        all_mu.extend(mu_arr)
        all_sig.extend(sig_arr)
        
        m, b = np.polyfit(z_arr, mu_arr, 1, w=1.0/sig_arr)
        indiv_slopes.append(m); indiv_intercepts.append(b)

    X_data, Y_data, sig_data = np.vstack((all_fam_idx, all_z)), np.array(all_mu), np.array(all_sig)
    has_sci = "SCI" in active_families

    def global_fit(X, *params):
        idx, z = X[0].astype(int), X[1]
        b_shared, b_sci = (params[0], params[1]) if has_sci else (params[0], 0.0)
        m_arr = np.array(params[2:]) if has_sci else np.array(params[1:])
        
        y_calc = np.zeros_like(z)
        for j in range(len(z)):
            fam_name = active_families[idx[j]]
            y_calc[j] = m_arr[idx[j]] * z[j] + (b_sci if fam_name == "SCI" else b_shared)
        return y_calc

    cher_b_guess = np.mean([indiv_intercepts[i] for i, f in enumerate(active_families) if f != "SCI"])
    p0 = [cher_b_guess, indiv_intercepts[active_families.index("SCI")]] + indiv_slopes if has_sci else [cher_b_guess] + indiv_slopes
    
    print("  -> Running simultaneous global curve fit across all families...")
    try:
        popt, pcov = curve_fit(global_fit, X_data, Y_data, p0=p0, sigma=sig_data, absolute_sigma=True)
        shared_b = popt[0]
        shared_b_err = np.sqrt(pcov[0,0])
        
        if has_sci:
            sci_b = popt[1]
            sci_b_err = np.sqrt(pcov[1,1])
            fit_slopes = popt[2:]
            fit_slope_errs = np.sqrt(np.diag(pcov))[2:]
        else:
            sci_b, sci_b_err = None, None
            fit_slopes = popt[1:]
            fit_slope_errs = np.sqrt(np.diag(pcov))[1:]
            
        print(f"  [RESULT] Shared Cherenkov Intercept: {shared_b:.3f} +/- {shared_b_err:.3f} ns")
        if has_sci: print(f"  [RESULT] Independent SCI Intercept:  {sci_b:.3f} +/- {sci_b_err:.3f} ns")
        
    except Exception as e: 
        print(f"  [ERROR] Shared fit failed: {e}")
        return

    with open(txt_path, "a") as f_out:
        f_out.write("\n" + "=" * 115 + "\n")
        f_out.write(f"{'GLOBAL SHARED INTERCEPT FIT (Cherenkovs Combined)':^115}\n")
        f_out.write("=" * 115 + "\n")
        
        with PdfPages(pdf_path) as pdf:
            # INCREASED SIZE HERE
            fig, ax = plt.subplots(figsize=(14, 10))
            style_paper_axes(ax, "Z Position [mm]", "Mean Time of Arrival [ns]", particle_type)

            for i, fam in enumerate(active_families):
                z_arr, mu_arr, sig_arr = np.array(plot_data[fam]["z"]), np.array(plot_data[fam]["mu"]), np.array(plot_data[fam]["sig"])
                m, m_err = fit_slopes[i], fit_slope_errs[i]
                b = sci_b if fam == "SCI" else shared_b
                b_err = sci_b_err if fam == "SCI" else shared_b_err
                
                v = abs(1.0 / m) * 1e6
                v_err = (v**2) * (m_err * 1e-6)
                
                v8 = v / 1e8
                v8_err = v_err / 1e8
                v_str = f"{v8:.4f}e8"
                v_err_str = f"{v8_err:.4f}e8"
                
                eq_str = f"t = ({m:.4f} +/- {m_err:.4f})z {'+' if b >= 0 else '-'} ({abs(b):.2f} +/- {b_err:.2f})"
                f_out.write(f"{fam:<10} | {v_str:<15} | {v_err_str:<15} | {eq_str}\n")

                legend_label = f"{fam:<10} | {v_str:<15} | {v_err_str:<15} | {eq_str}"

                z_fit = np.linspace(min(all_z) - 20, max(all_z) + 20, 200)
                #ax.errorbar(z_arr, mu_arr, yerr=sig_arr, fmt='o', color=FAMILIES[fam]["color"],  capsize=2, markersize=2, elinewidth=1)
                ax.errorbar(z_arr, mu_arr, yerr=sig_arr, fmt='o', color=FAMILIES[fam]["color"], capsize=3, markersize=6, elinewidth=2)
                ax.plot(z_fit, m * z_fit + b, '-', color=FAMILIES[fam]["color"], lw=2, label=legend_label)

            legend_title = (
                f"SHARED INTERCEPTS: Cherenkov=({shared_b:.2f} +/- {shared_b_err:.2f}) ns" + 
                (f", SCI=({sci_b:.2f} +/- {sci_b_err:.2f}) ns\n" if has_sci else "\n") +
                f"{'FAMILY':<10} | {'VELOCITY [m/s]':<15} | {'V_ERROR [m/s]':<15} | {'FIT EQUATION'}"
            )
            
            # MOVED LEGEND INSIDE TO THE BOTTOM LEFT
            leg = ax.legend(
                loc="upper right", 
                frameon=True, 
                prop={'family': 'monospace', 'size': 9}, 
                title=legend_title
            )
            plt.setp(leg.get_title(), family='monospace', fontsize=9, weight='bold')
            
            fig.tight_layout() # Use tight_layout instead of manual subplots_adjust
            pdf.savefig(fig)
            plt.close(fig)
            print(f"[SHARED FIT] Saved Shared Intercept plot to: {pdf_path}")


def create_channel_velocity_plots(plot_data, txt_path, pid_label, particle_type):
    outdir = os.path.dirname(txt_path)
    pdf_path = os.path.join(outdir, f"Z_vs_TOA_PerChannelFits_{pid_label}.pdf")
    
    print("\n[VELOCITY] --------------------------------------------------------")
    print(f"[VELOCITY] Calculating Per-Channel Velocity Fits (PID: {pid_label})")
    
    # Store all individual velocities per family for the histogramming step
    fam_v_lists = {fam: [] for fam in FAMILIES.keys()}
    
    with open(txt_path, "a") as f_out:
        f_out.write("\n" + "=" * 120 + "\n")
        f_out.write(f"{'FAMILY':<10} | {'CHANNEL':<7} | {'VELOCITY [m/s]':<15} | {'V_ERROR [m/s]':<15} | {'FIT EQUATION'}\n")
        f_out.write("=" * 120 + "\n")
        
        with PdfPages(pdf_path) as pdf:
            # --- PAGE 1: Scatter plot of Z vs TOA (Linear Fits) ---
            fig1, ax1 = plt.subplots(figsize=(14, 10))
            style_paper_axes(ax1, "Z Position [mm]", "Time of Arrival Mean [ns]", particle_type)
            
            for fam, channels_dict in plot_data.items():
                color = FAMILIES[fam]["color"]
                fam_slopes, fam_slope_errs, fam_intercepts, fam_intercept_weights = [], [], [], []
                
                for ch, data in channels_dict.items():
                    if len(data["z"]) < 2: continue
                    z_arr, mu_arr, sig_arr = np.array(data["z"]), np.array(data["mu"]), np.array(data["sig"])
                    weights = 1.0 / sig_arr
                    
                    try:
                        params, cov = np.polyfit(z_arr, mu_arr, 1, w=weights, cov='unscaled')
                    except: continue
                        
                    slope, intercept = params[0], params[1]
                    slope_err = np.sqrt(cov[0,0])
                    
                    # Store variables for overall weighted avg
                    fam_slopes.append(slope)
                    fam_slope_errs.append(slope_err)
                    fam_intercepts.append(intercept)
                    fam_intercept_weights.append(1.0 / np.sqrt(cov[1,1]))
                    
                    # Calculate velocity and store for histogram
                    v = abs(1.0 / slope) * 1e6 if slope != 0 else 0
                    fam_v_lists[fam].append(v / 1e8) # Storing in units of 10^8 m/s
                    
                    v_err = (v**2) * (slope_err * 1e-6) if slope != 0 else 0
                    f_out.write(f"{fam:<10} | {ch:<7} | {v/1e8:.4f}e8      | {v_err/1e8:.4f}e8      | t = ({slope:.4f})z {'+' if intercept >= 0 else '-'} {abs(intercept):.2f}\n")
                    
                    # Plot thin lines
                    ax1.errorbar(z_arr, mu_arr, yerr=sig_arr, fmt='none', ecolor=color, alpha=0.3, capsize=3)
                    ax1.plot(z_arr, mu_arr, marker='o', color=color, alpha=0.3, markersize=6, linestyle='none')
                
                if len(fam_slopes) > 0:
                    w_slope = 1.0 / (np.array(fam_slope_errs)**2)
                    f_slope = np.sum(np.array(fam_slopes) * w_slope) / np.sum(w_slope)
                    f_slope_err = np.sqrt(1.0 / np.sum(w_slope))
                    f_intercept = np.average(fam_intercepts, weights=fam_intercept_weights)
                    
                    final_v = abs(1.0 / f_slope) * 1e6
                    legend_label = f"OVERALL {fam:<7} | {final_v/1e8:.3f}e8 m/s"
                    z_fit_global = np.linspace(-200, 200, 200)
                    ax1.plot(z_fit_global, f_slope * z_fit_global + f_intercept, '-', color=color, linewidth=4, label=legend_label)

            ax1.legend(loc="upper right", frameon=True, prop={'family': 'monospace', 'size': 9})
            fig1.tight_layout()
            pdf.savefig(fig1)
            plt.close(fig1)

            # --- PAGE 2: Velocity Histograms and Gaussian Fits ---
            fig2, ax2 = plt.subplots(figsize=(14, 10))
            ax2.set_xlabel("Velocity [$10^8$ m/s]", fontsize=14)
            ax2.set_ylabel("Counts (Normalized)", fontsize=14)
            hep.cms.label(ax=ax2, exp="CaloX", data=False, llabel="Velocity Distribution", rlabel=f"{particle_type.capitalize()}")

            for fam, v_data in fam_v_lists.items():
                if len(v_data) < 3: continue # Need enough points for a histogram/fit
                
                v_data = np.array(v_data)
                color = FAMILIES[fam]["color"]
                
                # Create histogram
                counts, bins = np.histogram(v_data, bins=15)
                bin_centers = 0.5 * (bins[1:] + bins[:-1])
                
                # Normalize counts for plotting against Gaussian
                norm_counts = counts / np.max(counts) if np.max(counts) > 0 else counts
                
                # Gaussian Fit
                try:
                    p0 = [np.mean(v_data), np.std(v_data)]
                    popt, _ = curve_fit(gaussian_peak_1, bin_centers, norm_counts, p0=p0)
                    v_mu, v_sig = popt[0], abs(popt[1])
                    
                    x_plot = np.linspace(np.min(v_data)*0.8, np.max(v_data)*1.2, 200)
                    y_plot = gaussian_peak_1(x_plot, v_mu, v_sig)
                    
                    ax2.step(bin_centers, norm_counts, where='mid', color=color, alpha=0.3)
                    ax2.plot(x_plot, y_plot, color=color, lw=3, label=f"{fam}: $\mu$={v_mu:.3f}, $\sigma$={v_sig:.3f} [$10^8$ m/s]")
                except:
                    ax2.hist(v_data, bins=15, histtype='step', color=color, label=f"{fam} (Fit Failed)")

            ax2.legend(loc="upper left", frameon=True)
            fig2.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)

    print(f"[VELOCITY] Saved Per-Channel Z vs TOA and Gaussian fits to: {pdf_path}")

def create_channel_velocity_plots(plot_data, txt_path, pid_label, particle_type):
    outdir = os.path.dirname(txt_path)
    pdf_path = os.path.join(outdir, f"Z_vs_TOA_PerChannelFits_{pid_label}.pdf")
    
    print("\n[VELOCITY] --------------------------------------------------------")
    print(f"[VELOCITY] Calculating Per-Channel Velocity Fits (PID: {pid_label})")
    
    # Store all individual velocities per family for histogramming
    fam_v_lists = {fam: [] for fam in FAMILIES.keys()}
    all_velocities = [] # For the stacked plot
    
    with open(txt_path, "a") as f_out:
        f_out.write("\n" + "=" * 120 + "\n")
        f_out.write(f"{'FAMILY':<10} | {'CHANNEL':<7} | {'VELOCITY [m/s]':<15} | {'V_ERROR [m/s]':<15} | {'FIT EQUATION'}\n")
        f_out.write("=" * 120 + "\n")
        
        with PdfPages(pdf_path) as pdf:
            # --- PAGE 1: Summary Scatter Plot ---
            fig1, ax1 = plt.subplots(figsize=(14, 10))
            style_paper_axes(ax1, "Z Position [cm]", "Time of Arrival Mean [ns]", particle_type)
            
            for fam, channels_dict in plot_data.items():
                color = FAMILIES[fam]["color"]
                fam_slopes, fam_slope_errs, fam_intercepts, fam_intercept_weights = [], [], [], []
                
                for ch, data in channels_dict.items():
                    if len(data["z"]) < 2: continue
                    # CONVERSION: mm -> cm
                    z_arr = np.array(data["z"]) / 10.0
                    mu_arr, sig_arr = np.array(data["mu"]), np.array(data["sig"])
                    weights = 1.0 / sig_arr
                    
                    try:
                        params, cov = np.polyfit(z_arr, mu_arr, 1, w=weights, cov='unscaled')
                    except: continue
                        
                    slope, intercept = params[0], params[1]
                    slope_err = np.sqrt(cov[0,0])
                    
                    fam_slopes.append(slope)
                    fam_slope_errs.append(slope_err)
                    fam_intercepts.append(intercept)
                    fam_intercept_weights.append(1.0 / np.sqrt(cov[1,1]))
                    
                    # Velocity Calculation: (1 / slope [ns/cm]) * (1e7 [cm/m * s/ns])
                    v = abs(1.0 / slope) * 1e7 if slope != 0 else 0
                    v_scaled = v / 1e8 # Units of 10^8 m/s
                    fam_v_lists[fam].append(v_scaled)
                    all_velocities.append(v_scaled)
                    
                    v_err = (v**2) * (slope_err * 1e-7) if slope != 0 else 0
                    f_out.write(f"{fam:<10} | {ch:<7} | {v/1e8:.4f}e8      | {v_err/1e8:.4f}e8      | t = ({slope:.4f})z {'+' if intercept >= 0 else '-'} {abs(intercept):.2f}\n")
                    
                    ax1.errorbar(z_arr, mu_arr, yerr=sig_arr, fmt='none', ecolor=color, alpha=0.2, capsize=2)
                    ax1.plot(z_arr, mu_arr, marker='o', color=color, alpha=0.2, markersize=4, linestyle='none')
                
                if len(fam_slopes) > 0:
                    w_slope = 1.0 / (np.array(fam_slope_errs)**2)
                    f_slope = np.sum(np.array(fam_slopes) * w_slope) / np.sum(w_slope)
                    f_intercept = np.average(fam_intercepts, weights=fam_intercept_weights)
                    final_v = abs(1.0 / f_slope) * 1e7
                    
                    z_fit_global = np.linspace(-20, 20, 100) # Range in cm
                    ax1.plot(z_fit_global, f_slope * z_fit_global + f_intercept, '-', color=color, linewidth=3, 
                            label=f"AVG {fam}: {final_v/1e8:.3f}e8 m/s")

            ax1.legend(loc="upper right", frameon=True, prop={'family': 'monospace', 'size': 9})
            pdf.savefig(fig1)
            plt.close(fig1)

            # --- PAGE 2: Stacked Velocity Plot (All Channels) ---
            fig_stack, ax_stack = plt.subplots(figsize=(12, 8))
            # Filter out extreme outliers for better plotting
            clean_v_data = [v for v in all_velocities if 0 < v < 5] 
            
            # Prepare data for stacked histogram
            hist_data = [fam_v_lists[f] for f in FAMILIES.keys() if len(fam_v_lists[f]) > 0]
            hist_labels = [f for f in FAMILIES.keys() if len(fam_v_lists[f]) > 0]
            hist_colors = [FAMILIES[f]["color"] for f in FAMILIES.keys() if len(fam_v_lists[f]) > 0]

            ax_stack.hist(hist_data, bins=25, stacked=True, color=hist_colors, label=hist_labels, alpha=0.8, edgecolor='black')
            ax_stack.set_xlabel("Velocity [$10^8$ m/s]", fontsize=14)
            ax_stack.set_ylabel("Number of Channels", fontsize=14)
            ax_stack.set_title(f"Stacked Velocity Distribution - {particle_type}", fontsize=16)
            ax_stack.legend()
            pdf.savefig(fig_stack)
            plt.close(fig_stack)

            # --- PAGE 3: Individual Channel Fits (Z in cm) ---
            for fam, channels_dict in plot_data.items():
                color = FAMILIES[fam]["color"]
                for ch, data in channels_dict.items():
                    if len(data["z"]) < 2: continue
                    z_cm = np.array(data["z"]) / 10.0
                    mu, sig = np.array(data["mu"]), np.array(data["sig"])
                    
                    try:
                        p, c = np.polyfit(z_cm, mu, 1, w=1.0/sig, cov='unscaled')
                        v_ch = (abs(1.0/p[0]) * 1e7) / 1e8
                        
                        fig_ch, ax_ch = plt.subplots(figsize=(10, 7))
                        style_paper_axes(ax_ch, "Z Position [cm]", "TOA Mean [ns]", f"{fam} {ch}")
                        ax_ch.errorbar(z_cm, mu, yerr=sig, fmt='o', color=color, capsize=3)
                        
                        z_range = np.linspace(min(z_cm)-2, max(z_cm)+2, 50)
                        ax_ch.plot(z_range, p[0]*z_range + p[1], 'k--', alpha=0.6, label=f"v = {v_ch:.4f} $10^8$ m/s")
                        
                        ax_ch.legend()
                        pdf.savefig(fig_ch)
                        plt.close(fig_ch)
                    except: continue

            # --- PAGE 4: Normalized Histograms with Gaussian Fits ---
            fig2, ax2 = plt.subplots(figsize=(14, 10))
            hep.cms.label(ax=ax2, exp="CaloX", llabel="Velocity Fits", rlabel=particle_type)
            
            for fam, v_data in fam_v_lists.items():
                if len(v_data) < 3: continue
                v_data = np.array(v_data)
                counts, bins = np.histogram(v_data, bins=15)
                bin_centers = 0.5 * (bins[1:] + bins[:-1])
                norm_counts = counts / np.max(counts) if np.max(counts) > 0 else counts
                
                try:
                    popt, _ = curve_fit(gaussian_peak_1, bin_centers, norm_counts, p0=[np.mean(v_data), np.std(v_data)])
                    x_p = np.linspace(min(v_data), max(v_data), 100)
                    ax2.plot(x_p, gaussian_peak_1(x_p, *popt), color=FAMILIES[fam]["color"], lw=2, label=f"{fam} $\mu$={popt[0]:.3f}")
                    ax2.step(bin_centers, norm_counts, where='mid', color=FAMILIES[fam]["color"], alpha=0.3)
                except:
                    ax2.hist(v_data, bins=15, histtype='step', color=FAMILIES[fam]["color"], label=f"{fam} (Fit Fail)")

            ax2.set_xlabel("Velocity [$10^8$ m/s]")
            ax2.legend()
            pdf.savefig(fig2)
            plt.close(fig2)

    print(f"[VELOCITY] Success. PDF saved with Z in [cm] and stacked histograms.")

def create_channel_velocity_plots(plot_data, txt_path, pid_label, particle_type):
    from matplotlib.lines import Line2D
    outdir = os.path.dirname(txt_path)
    pdf_path = os.path.join(outdir, f"Z_vs_TOA_PerChannelFits_{pid_label}.pdf")
    
    print("\n[VELOCITY] --------------------------------------------------------")
    print(f"[VELOCITY] Calculating Per-Channel Velocity Fits (PID: {pid_label})")
    
    fam_v_lists = {fam: [] for fam in FAMILIES.keys()}
    all_channel_fits = [] 
    
    with open(txt_path, "a") as f_out:
        f_out.write("\n" + "=" * 120 + "\n")
        f_out.write(f"{'FAMILY':<10} | {'CHANNEL':<7} | {'VELOCITY [m/s]':<15} | {'V_ERROR [m/s]':<15} | {'FIT EQUATION'}\n")
        f_out.write("=" * 120 + "\n")
        
        with PdfPages(pdf_path) as pdf:
            # --- PAGE 1: Summary Trend (Averages) ---
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            style_paper_axes(ax1, "Z Position [cm]", "Time of Arrival Mean [ns]", particle_type)
            
            for fam, channels_dict in plot_data.items():
                color = FAMILIES[fam]["color"]
                fam_slopes, fam_slope_errs, fam_intercepts, fam_intercept_weights = [], [], [], []
                
                for ch, data in channels_dict.items():
                    # REQUIREMENT: >= 4 points for the fit
                    if len(data["z"]) < 4: continue 
                    
                    z_cm = np.array(data["z"]) / 10.0
                    mu, sig = np.array(data["mu"]), np.array(data["sig"])
                    
                    try:
                        params, cov = np.polyfit(z_cm, mu, 1, w=1.0/sig, cov='unscaled')
                        slope, intercept = params[0], params[1]
                        all_channel_fits.append((slope, intercept, color))
                        
                        v = abs(1.0 / slope) * 1e7
                        v_err = (v**2) * (np.sqrt(cov[0,0]) * 1e-7) if slope != 0 else 0
                        fam_v_lists[fam].append(v / 1e8)
                        
                        fam_slopes.append(slope)
                        fam_slope_errs.append(np.sqrt(cov[0,0]))
                        fam_intercepts.append(intercept)
                        fam_intercept_weights.append(1.0 / np.sqrt(cov[1,1]))

                        f_out.write(f"{fam:<10} | {ch:<7} | {v/1e8:.4f}e8      | {v_err/1e8:.4f}e8      | t = {slope:.4f}z + {intercept:.2f}\n")
                    except: continue

                if fam_slopes:
                    w = 1.0 / (np.array(fam_slope_errs)**2)
                    avg_slope = np.sum(np.array(fam_slopes) * w) / np.sum(w)
                    avg_int = np.average(fam_intercepts, weights=fam_intercept_weights)
                    
                    # Calculate Stats for Legend: Mean and SEM (sigma/sqrt(N))
                    v_arr = np.array(fam_v_lists[fam])
                    v_mean = np.mean(v_arr)
                    v_sem = np.std(v_arr) / np.sqrt(len(v_arr)) if len(v_arr) > 1 else 0
                    
                    z_eval = np.linspace(-20, 20, 100)
                    ax1.plot(z_eval, avg_slope * z_eval + avg_int, color=color, lw=4, 
                             label=f"{fam}: {v_mean:.3f} ± {v_sem:.3f} [$10^8$ m/s]")

            ax1.legend(loc='upper right', fontsize=10)
            pdf.savefig(fig1); plt.close(fig1)

            # --- PAGE 2: Zoomed overlays, one page per family, color-coded by channel ---
            # --- PAGE 2+: Zoomed overlays per family (LINES ONLY) + y-intercept plot ---
            from matplotlib.lines import Line2D

            for fam, channels_dict in plot_data.items():
                valid_channels = []
                for ch, data in channels_dict.items():
                    if len(data["z"]) >= 4:
                        valid_channels.append(ch)

                if not valid_channels:
                    continue

                cmap = plt.get_cmap("turbo", len(valid_channels))
                ch_color_map = {ch: cmap(i) for i, ch in enumerate(sorted(valid_channels))}

                fam_fit_lines = []
                all_y_vals = []
                all_z_vals = []

                for ch in sorted(valid_channels):
                    data = channels_dict[ch]

                    try:
                        z_cm = np.array(data["z"]) / 10.0
                        mu = np.array(data["mu"])
                        sig = np.array(data["sig"])

                        p, cov = np.polyfit(z_cm, mu, 1, w=1.0 / sig, cov=True)
                        slope, intercept = p[0], p[1]
                        slope_err = np.sqrt(cov[0, 0]) if cov is not None else np.nan
                        intercept_err = np.sqrt(cov[1, 1]) if cov is not None else np.nan

                        v_ch = abs(1.0 / slope) * 1e7 / 1e8 if slope != 0 else np.nan
                        v_err = ((abs(1.0 / slope) * 1e7) ** 2) * (slope_err * 1e-7) / 1e8 if slope != 0 else np.nan

                        fam_fit_lines.append({
                            "ch": ch,
                            "z_cm": z_cm,
                            "mu": mu,
                            "sig": sig,
                            "slope": slope,
                            "intercept": intercept,
                            "intercept_err": intercept_err,
                            "v": v_ch,
                            "v_err": v_err,
                            "color": ch_color_map[ch],
                        })

                        all_y_vals.extend(mu.tolist())
                        all_z_vals.extend(z_cm.tolist())

                    except Exception:
                        continue

                if not fam_fit_lines:
                    continue

                # --------------------------------------------------
                # PAGE A: family overlay, lines only
                # --------------------------------------------------
                z_min = min(all_z_vals)
                z_max = max(all_z_vals)
                y_min = min(all_y_vals)
                y_max = max(all_y_vals)

                z_pad = max(1.5, 0.15 * (z_max - z_min) if z_max > z_min else 1.5)
                y_pad = max(0.15, 0.20 * (y_max - y_min) if y_max > y_min else 0.15)

                fig_fam, ax_fam = plt.subplots(figsize=(12, 8))
                style_paper_axes(
                    ax_fam,
                    "Z Position [cm]",
                    "Time of Arrival Mean [ns]",
                    particle_type
                )

                ax_fam.set_xlim(z_min - z_pad, z_max + z_pad)
                ax_fam.set_ylim(y_min - y_pad, y_max + y_pad)

                z_eval = np.linspace(z_min - z_pad, z_max + z_pad, 300)

                handles = []
                labels = []

                for entry in fam_fit_lines:
                    ch = entry["ch"]
                    slope = entry["slope"]
                    intercept = entry["intercept"]
                    color = entry["color"]
                    v_ch = entry["v"]
                    v_err = entry["v_err"]

                    ax_fam.plot(
                        z_eval,
                        slope * z_eval + intercept,
                        color=color,
                        lw=2.4,
                        alpha=0.95
                    )

                    handles.append(Line2D([0], [0], color=color, lw=2.5))
                    labels.append(f"Ch {ch}: v = {v_ch:.3f} ± {v_err:.3f} [$10^8$ m/s]")

                ax_fam.set_title(
                    f"{fam} Family: Overlayed Per-Channel Velocity Fits (Lines Only, N ≥ 4)",
                    fontsize=16
                )
                ax_fam.grid(True, linestyle=':', alpha=0.35)

                ax_fam.legend(
                    handles,
                    labels,
                    loc='best',
                    fontsize=9,
                    frameon=True,
                    ncol=1
                )

                pdf.savefig(fig_fam)
                plt.close(fig_fam)

                # --------------------------------------------------
                # PAGE B: y-intercept per channel
                # --------------------------------------------------
                fig_int, ax_int = plt.subplots(figsize=(12, 8))

                ch_sorted = sorted(
                    fam_fit_lines,
                    key=lambda x: int(x["ch"])
                )

                xvals = np.arange(len(ch_sorted))
                yvals = np.array([x["intercept"] for x in ch_sorted])
                yerrs = np.array([x["intercept_err"] for x in ch_sorted])
                colors = [x["color"] for x in ch_sorted]
                ch_labels = [x["ch"] for x in ch_sorted]

                for i, (x, y, ye, c) in enumerate(zip(xvals, yvals, yerrs, colors)):
                    ax_int.errorbar(
                        x, y,
                        yerr=ye if np.isfinite(ye) else None,
                        fmt='o',
                        color=c,
                        capsize=4,
                        markersize=8,
                        elinewidth=1.8
                    )

                mean_intercept = np.mean(yvals)
                sem_intercept = np.std(yvals, ddof=1) / np.sqrt(len(yvals)) if len(yvals) > 1 else 0.0

                ax_int.axhline(
                    mean_intercept,
                    color='black',
                    linestyle='--',
                    linewidth=2.0,
                    label=f"Mean = {mean_intercept:.3f} ± {sem_intercept:.3f} ns"
                )

                ax_int.set_xticks(xvals)
                ax_int.set_xticklabels(ch_labels, rotation=45, ha='right')
                ax_int.set_xlabel(f"{fam} Channel", fontsize=14)
                ax_int.set_ylabel("Fit y-intercept [ns]", fontsize=14)
                ax_int.set_title(f"{fam} Family: y-intercept by Channel (N ≥ 4)", fontsize=16)
                ax_int.grid(True, linestyle=':', alpha=0.35)
                ax_int.legend(loc='best', frameon=True)

                hep.cms.label(
                    ax=ax_int,
                    exp="CaloX",
                    data=False,
                    llabel="Per-channel intercepts",
                    rlabel=f"{particle_type.capitalize() if isinstance(particle_type, str) else particle_type}"
                )

                fig_int.tight_layout()
                pdf.savefig(fig_int)
                plt.close(fig_int)

            # --- PAGE 3: Stacked Velocity Histogram ---
            fig_st, ax_st = plt.subplots(figsize=(12, 8))
            v_lists = [fam_v_lists[f] for f in FAMILIES.keys() if fam_v_lists[f]]
            v_labs = [f for f in FAMILIES.keys() if fam_v_lists[f]]
            v_cols = [FAMILIES[f]["color"] for f in FAMILIES.keys() if fam_v_lists[f]]
            
            ax_st.hist(v_lists, bins=25, stacked=True, color=v_cols, label=v_labs, edgecolor='black', alpha=0.8)
            ax_st.set_xlabel("Velocity [$10^8$ m/s]", fontsize=14)
            ax_st.set_ylabel("Number of Channels", fontsize=14)
            ax_st.legend()
            pdf.savefig(fig_st); plt.close(fig_st)

            # --- PAGE 4+: Individual Channel Plots ---
            for fam, channels_dict in plot_data.items():
                for ch, data in channels_dict.items():
                    if len(data["z"]) < 4: continue # Same cut applied here
                    try:
                        z_cm = np.array(data["z"]) / 10.0
                        mu, sig = np.array(data["mu"]), np.array(data["sig"])
                        p, _ = np.polyfit(z_cm, mu, 1, w=1.0/sig, cov=False)
                        
                        fig_ch, ax_ch = plt.subplots(figsize=(10, 6))
                        style_paper_axes(ax_ch, "Z Position [cm]", "TOA [ns]", f"{fam} {ch}")
                        ax_ch.errorbar(z_cm, mu, yerr=sig, fmt='o', color=FAMILIES[fam]["color"], markersize=8)
                        
                        z_fit = np.linspace(min(z_cm)-2, max(z_cm)+2, 50)
                        v_val = (abs(1/p[0])*1e7)/1e8
                        ax_ch.plot(z_fit, p[0]*z_fit + p[1], 'k--', lw=2, label=f"v={v_val:.4f} $10^8$ m/s")
                        ax_ch.legend(); pdf.savefig(fig_ch); plt.close(fig_ch)
                    except: continue

    print(f"[VELOCITY] Success. Applied N>=4 cut and updated legend stats.")



def make_metric_mosaic(ax, grid, entries_by_channel, metric_key, title, cbar_label,
                       particle_type=None, family_name=None, cmap_name="viridis",
                       value_fmt=".4f"):
    """
    Draw a channel mosaic where each cell is colored by entries_by_channel[ch][metric_key].
    """
    import matplotlib as mpl

    nrows = len(grid)
    ncols = max(len(r) for r in grid)

    mat = np.full((nrows, ncols), np.nan, dtype=float)
    ch_labels = np.empty((nrows, ncols), dtype=object)
    ch_labels[:] = ""

    for i, row in enumerate(grid):
        for j, ch in enumerate(row):
            if ch is None:
                continue
            ch_labels[i, j] = ch
            if ch in entries_by_channel and metric_key in entries_by_channel[ch]:
                val = entries_by_channel[ch][metric_key]
                if np.isfinite(val):
                    mat[i, j] = val

    masked = np.ma.masked_invalid(mat)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="white")

    im = ax.imshow(masked, cmap=cmap, aspect="auto", origin="upper")

    # annotate each filled channel
    for i in range(nrows):
        for j in range(ncols):
            ch = ch_labels[i, j]
            if not ch:
                continue

            if np.isfinite(mat[i, j]):
                txt = f"{ch}\n{format(mat[i, j], value_fmt)}"
            else:
                txt = f"{ch}\nNA"

            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=8, color="black"
            )

    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_xticklabels([str(i) for i in range(ncols)], fontsize=10)
    ax.set_yticklabels([str(i) for i in range(nrows)], fontsize=10)
    ax.set_xlabel("Column", fontsize=12)
    ax.set_ylabel("Row", fontsize=12)
    ax.set_title(title, fontsize=15)

    # draw cell borders
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=12)

    hep.cms.label(
        ax=ax,
        exp="CaloX",
        data=False,
        llabel=f"{family_name} channel mosaic" if family_name else "Channel mosaic",
        rlabel=f"{particle_type.capitalize() if isinstance(particle_type, str) else particle_type}"
    )

def create_channel_velocity_plots(plot_data, txt_path, pid_label, particle_type):
    from matplotlib.lines import Line2D

    outdir = os.path.dirname(txt_path)
    pdf_path = os.path.join(outdir, f"Z_vs_TOA_PerChannelFits_{pid_label}_wyintercept.pdf")

    print("\n[VELOCITY] --------------------------------------------------------")
    print(f"[VELOCITY] Calculating Per-Channel Velocity Fits (PID: {pid_label})")

    fam_v_lists = {fam: [] for fam in FAMILIES.keys()}

    with open(txt_path, "a") as f_out:
        f_out.write("\n" + "=" * 150 + "\n")
        f_out.write(
            f"{'FAMILY':<10} | {'CHANNEL':<7} | {'SLOPE [ns/cm]':<15} | "
            f"{'SLOPE_ERR':<12} | {'INTERCEPT [ns]':<15} | {'INT_ERR':<12} | "
            f"{'VELOCITY [1e8 m/s]':<18} | {'V_ERR [1e8 m/s]':<18}\n"
        )
        f_out.write("=" * 150 + "\n")

        with PdfPages(pdf_path) as pdf:
            # ==========================================================
            # PAGE 1: Summary family-average lines
            # ==========================================================
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            style_paper_axes(ax1, "Z Position [cm]", "Time of Arrival Mean [ns]", particle_type)

            for fam, channels_dict in plot_data.items():
                color = FAMILIES[fam]["color"]
                fam_slopes = []
                fam_slope_errs = []
                fam_intercepts = []
                fam_intercept_weights = []

                for ch, data in channels_dict.items():
                    if len(data["z"]) < 4:
                        continue

                    z_cm = np.array(data["z"]) / 10.0
                    mu = np.array(data["mu"])
                    sig = np.array(data["sig"])

                    try:
                        params, cov = np.polyfit(z_cm, mu, 1, w=1.0 / sig, cov=True)
                    except Exception:
                        continue

                    slope = params[0]
                    intercept = params[1]
                    slope_err = np.sqrt(cov[0, 0]) if cov is not None else np.nan
                    intercept_err = np.sqrt(cov[1, 1]) if cov is not None else np.nan

                    if slope == 0:
                        continue

                    v = abs(1.0 / slope) * 1e7 / 1e8
                    v_err = ((abs(1.0 / slope) * 1e7) ** 2) * (slope_err * 1e-7) / 1e8

                    fam_v_lists[fam].append(v)
                    fam_slopes.append(slope)
                    fam_slope_errs.append(slope_err)
                    fam_intercepts.append(intercept)
                    if np.isfinite(intercept_err) and intercept_err > 0:
                        fam_intercept_weights.append(1.0 / intercept_err**2)
                    else:
                        fam_intercept_weights.append(1.0)

                    f_out.write(
                        f"{fam:<10} | {ch:<7} | {slope:<15.5f} | {slope_err:<12.5f} | "
                        f"{intercept:<15.5f} | {intercept_err:<12.5f} | "
                        f"{v:<18.5f} | {v_err:<18.5f}\n"
                    )

                if fam_slopes:
                    w = 1.0 / np.square(np.array(fam_slope_errs))
                    avg_slope = np.sum(np.array(fam_slopes) * w) / np.sum(w)
                    avg_intercept = np.average(fam_intercepts, weights=fam_intercept_weights)

                    v_arr = np.array(fam_v_lists[fam])
                    v_mean = np.mean(v_arr)
                    v_sem = np.std(v_arr, ddof=1) / np.sqrt(len(v_arr)) if len(v_arr) > 1 else 0.0

                    z_eval = np.linspace(-20, 20, 200)
                    ax1.plot(
                        z_eval,
                        avg_slope * z_eval + avg_intercept,
                        color=color,
                        lw=3,
                        label=f"{fam}: v = {v_mean:.3f} ± {v_sem:.3f} [$10^8$ m/s]"
                    )

            ax1.legend(loc="upper right", fontsize=10, frameon=True)
            ax1.grid(True, linestyle=":", alpha=0.35)
            fig1.tight_layout()
            pdf.savefig(fig1)
            plt.close(fig1)

            # ==========================================================
            # PAGE 2+: Per-family overlays + slope/intercept diagnostics
            # ==========================================================
            for fam, channels_dict in plot_data.items():
                valid_channels = [ch for ch, data in channels_dict.items() if len(data["z"]) >= 4]
                if not valid_channels:
                    continue

                cmap = plt.get_cmap("turbo", len(valid_channels))
                ch_color_map = {ch: cmap(i) for i, ch in enumerate(sorted(valid_channels))}

                fam_fit_lines = []
                all_y_vals = []
                all_z_vals = []

                for ch in sorted(valid_channels):
                    data = channels_dict[ch]

                    try:
                        z_cm = np.array(data["z"]) / 10.0
                        mu = np.array(data["mu"])
                        sig = np.array(data["sig"])

                        params, cov = np.polyfit(z_cm, mu, 1, w=1.0 / sig, cov=True)
                        slope = params[0]
                        intercept = params[1]
                        slope_err = np.sqrt(cov[0, 0]) if cov is not None else np.nan
                        intercept_err = np.sqrt(cov[1, 1]) if cov is not None else np.nan

                        if slope == 0:
                            continue

                        v_ch = abs(1.0 / slope) * 1e7 / 1e8
                        v_err = ((abs(1.0 / slope) * 1e7) ** 2) * (slope_err * 1e-7) / 1e8

                        fam_fit_lines.append({
                            "ch": ch,
                            "z_cm": z_cm,
                            "mu": mu,
                            "sig": sig,
                            "slope": slope,
                            "slope_err": slope_err,
                            "intercept": intercept,
                            "intercept_err": intercept_err,
                            "v": v_ch,
                            "v_err": v_err,
                            "color": ch_color_map[ch],
                        })

                        all_y_vals.extend(mu.tolist())
                        all_z_vals.extend(z_cm.tolist())

                    except Exception:
                        continue

                if not fam_fit_lines:
                    continue

                # --------------------------------------------------
                # PAGE A: overlayed fit lines, legend now includes slope + intercept
                # --------------------------------------------------
                z_min = min(all_z_vals)
                z_max = max(all_z_vals)
                y_min = min(all_y_vals)
                y_max = max(all_y_vals)

                z_pad = max(1.5, 0.15 * (z_max - z_min) if z_max > z_min else 1.5)
                y_pad = max(0.15, 0.20 * (y_max - y_min) if y_max > y_min else 0.15)

                fig_fam, ax_fam = plt.subplots(figsize=(13, 8))
                style_paper_axes(ax_fam, "Z Position [cm]", "Time of Arrival Mean [ns]", particle_type)
                ax_fam.set_xlim(z_min - z_pad, z_max + z_pad)
                ax_fam.set_ylim(y_min - y_pad, y_max + y_pad)

                z_eval = np.linspace(z_min - z_pad, z_max + z_pad, 300)

                handles = []
                labels = []

                for entry in fam_fit_lines:
                    ch = entry["ch"]
                    slope = entry["slope"]
                    intercept = entry["intercept"]
                    color = entry["color"]
                    v_ch = entry["v"]
                    v_err = entry["v_err"]

                    ax_fam.plot(z_eval, slope * z_eval + intercept, color=color, lw=2.4, alpha=0.95)

                    handles.append(Line2D([0], [0], color=color, lw=2.5))
                    labels.append(
                        f"Ch {ch}: m={slope:.4f} ns/cm, b={intercept:.3f} ns, "
                        f"v={v_ch:.3f}±{v_err:.3f} [$10^8$ m/s]"
                    )

                ax_fam.set_title(
                    f"{fam} Family: Per-Channel Linear Fits (compare slope vs intercept)",
                    fontsize=16
                )
                ax_fam.grid(True, linestyle=':', alpha=0.35)
                ax_fam.legend(handles, labels, loc='best', fontsize=8.5, frameon=True, ncol=1)

                fig_fam.tight_layout()
                pdf.savefig(fig_fam)
                plt.close(fig_fam)

                # --------------------------------------------------
                # PAGE B: y-intercept by channel
                # --------------------------------------------------
                fig_int, ax_int = plt.subplots(figsize=(12, 8))

                ch_sorted = sorted(fam_fit_lines, key=lambda x: int(x["ch"]))
                xvals = np.arange(len(ch_sorted))
                yvals = np.array([x["intercept"] for x in ch_sorted])
                yerrs = np.array([x["intercept_err"] for x in ch_sorted])
                colors = [x["color"] for x in ch_sorted]
                ch_labels = [x["ch"] for x in ch_sorted]

                for x, y, ye, c in zip(xvals, yvals, yerrs, colors):
                    ax_int.errorbar(
                        x, y,
                        yerr=ye if np.isfinite(ye) else None,
                        fmt='o',
                        color=c,
                        capsize=4,
                        markersize=8,
                        elinewidth=1.8
                    )

                mean_intercept = np.mean(yvals)
                sem_intercept = np.std(yvals, ddof=1) / np.sqrt(len(yvals)) if len(yvals) > 1 else 0.0

                ax_int.axhline(
                    mean_intercept,
                    color='black',
                    linestyle='--',
                    linewidth=2.0,
                    label=f"Mean = {mean_intercept:.3f} ± {sem_intercept:.3f} ns"
                )

                ax_int.set_xticks(xvals)
                ax_int.set_xticklabels(ch_labels, rotation=45, ha='right')
                ax_int.set_xlabel(f"{fam} Channel", fontsize=14)
                ax_int.set_ylabel("Fit y-intercept [ns]", fontsize=14)
                ax_int.set_title(f"{fam} Family: y-intercept by Channel", fontsize=16)
                ax_int.grid(True, linestyle=':', alpha=0.35)
                ax_int.legend(loc='best', frameon=True)

                hep.cms.label(
                    ax=ax_int,
                    exp="CaloX",
                    data=False,
                    llabel="Per-channel intercepts",
                    rlabel=f"{particle_type.capitalize() if isinstance(particle_type, str) else particle_type}"
                )

                fig_int.tight_layout()
                pdf.savefig(fig_int)
                plt.close(fig_int)

                # --------------------------------------------------
                # PAGE C: slope by channel
                # --------------------------------------------------
                fig_slope, ax_slope = plt.subplots(figsize=(12, 8))

                slope_vals = np.array([x["slope"] for x in ch_sorted])
                slope_errs = np.array([x["slope_err"] for x in ch_sorted])

                for x, y, ye, c in zip(xvals, slope_vals, slope_errs, colors):
                    ax_slope.errorbar(
                        x, y,
                        yerr=ye if np.isfinite(ye) else None,
                        fmt='o',
                        color=c,
                        capsize=4,
                        markersize=8,
                        elinewidth=1.8
                    )

                mean_slope = np.mean(slope_vals)
                sem_slope = np.std(slope_vals, ddof=1) / np.sqrt(len(slope_vals)) if len(slope_vals) > 1 else 0.0

                ax_slope.axhline(
                    mean_slope,
                    color='black',
                    linestyle='--',
                    linewidth=2.0,
                    label=f"Mean = {mean_slope:.4f} ± {sem_slope:.4f} ns/cm"
                )

                ax_slope.set_xticks(xvals)
                ax_slope.set_xticklabels(ch_labels, rotation=45, ha='right')
                ax_slope.set_xlabel(f"{fam} Channel", fontsize=14)
                ax_slope.set_ylabel("Fit slope [ns/cm]", fontsize=14)
                ax_slope.set_title(f"{fam} Family: slope by Channel", fontsize=16)
                ax_slope.grid(True, linestyle=':', alpha=0.35)
                ax_slope.legend(loc='best', frameon=True)

                hep.cms.label(
                    ax=ax_slope,
                    exp="CaloX",
                    data=False,
                    llabel="Per-channel slopes",
                    rlabel=f"{particle_type.capitalize() if isinstance(particle_type, str) else particle_type}"
                )

                fig_slope.tight_layout()
                pdf.savefig(fig_slope)
                plt.close(fig_slope)

                # --------------------------------------------------
                # PAGE D: intercept vs slope scatter
                # This is the cleanest plot for your stated goal
                # --------------------------------------------------
                fig_corr, ax_corr = plt.subplots(figsize=(10, 8))

                for entry in ch_sorted:
                    ax_corr.errorbar(
                        entry["slope"],
                        entry["intercept"],
                        xerr=entry["slope_err"] if np.isfinite(entry["slope_err"]) else None,
                        yerr=entry["intercept_err"] if np.isfinite(entry["intercept_err"]) else None,
                        fmt='o',
                        color=entry["color"],
                        capsize=3,
                        markersize=8,
                        elinewidth=1.4
                    )
                    ax_corr.text(
                        entry["slope"],
                        entry["intercept"],
                        f" {entry['ch']}",
                        fontsize=9,
                        va='bottom',
                        ha='left'
                    )

                ax_corr.set_xlabel("Slope [ns/cm]", fontsize=14)
                ax_corr.set_ylabel("y-intercept [ns]", fontsize=14)
                ax_corr.set_title(f"{fam} Family: slope vs y-intercept by Channel", fontsize=16)
                ax_corr.grid(True, linestyle=':', alpha=0.35)

                hep.cms.label(
                    ax=ax_corr,
                    exp="CaloX",
                    data=False,
                    llabel="Slope/intercept comparison",
                    rlabel=f"{particle_type.capitalize() if isinstance(particle_type, str) else particle_type}"
                )

                fig_corr.tight_layout()
                pdf.savefig(fig_corr)
                plt.close(fig_corr)

                            # --------------------------------------------------
                # PAGE E: slope mosaic + intercept mosaic
                # --------------------------------------------------
                grid_map = {
                    "Quartz": QUARTZ_GRID,
                    "Plastic": PLASTIC_GRID,
                    "SCI": SCI_ALL_GRID,
                }

                if fam in grid_map:
                    entries_by_channel = {
                        entry["ch"]: {
                            "slope": entry["slope"],
                            "intercept": entry["intercept"],
                            "slope_err": entry["slope_err"],
                            "intercept_err": entry["intercept_err"],
                        }
                        for entry in ch_sorted
                    }

                    fig_mos, (ax_m1, ax_m2) = plt.subplots(1, 2, figsize=(18, 8))

                    make_metric_mosaic(
                        ax=ax_m1,
                        grid=grid_map[fam],
                        entries_by_channel=entries_by_channel,
                        metric_key="slope",
                        title=f"{fam} Family: Slope Mosaic",
                        cbar_label="Slope [ns/cm]",
                        particle_type=particle_type,
                        family_name=fam,
                        cmap_name="coolwarm",
                        value_fmt=".4f",
                    )

                    make_metric_mosaic(
                        ax=ax_m2,
                        grid=grid_map[fam],
                        entries_by_channel=entries_by_channel,
                        metric_key="intercept",
                        title=f"{fam} Family: y-intercept Mosaic",
                        cbar_label="y-intercept [ns]",
                        particle_type=particle_type,
                        family_name=fam,
                        cmap_name="viridis",
                        value_fmt=".3f",
                    )

                    fig_mos.tight_layout()
                    pdf.savefig(fig_mos)
                    plt.close(fig_mos)
                    
            # ==========================================================
            # Final page: stacked velocity histogram
            # ==========================================================
            fig_st, ax_st = plt.subplots(figsize=(12, 8))
            v_lists = [fam_v_lists[f] for f in FAMILIES.keys() if fam_v_lists[f]]
            v_labs = [f for f in FAMILIES.keys() if fam_v_lists[f]]
            v_cols = [FAMILIES[f]["color"] for f in FAMILIES.keys() if fam_v_lists[f]]

            ax_st.hist(
                v_lists,
                bins=25,
                stacked=True,
                color=v_cols,
                label=v_labs,
                edgecolor='black',
                alpha=0.8
            )
            ax_st.set_xlabel("Velocity [$10^8$ m/s]", fontsize=14)
            ax_st.set_ylabel("Number of Channels", fontsize=14)
            ax_st.set_title("Stacked Per-Channel Velocity Distribution", fontsize=16)
            ax_st.legend()
            ax_st.grid(True, linestyle=':', alpha=0.35)

            fig_st.tight_layout()
            pdf.savefig(fig_st)
            plt.close(fig_st)

    print(f"[VELOCITY] Success. Added slope/intercept diagnostics and family intercept plots.")
# ================= UPDATED STATS TABLE WITH TIME_ERROR =================
def generate_stats_table(files, outpath, tree_name, particle_type=None):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    # Added Time_Err column to the format
    header_fmt = "{:<10} | {:<10} | {:<10} | {:<8} | {:<12} | {:<12} | {:<12} | {:<12} | {:<10}"
    row_fmt    = "{:<10} | {:<10.1f} | {:<10} | {:<8} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<10}"
    
    #plot_data = {fam: {"z": [], "mu": [], "sig": []} for fam in FAMILIES.keys()}
    # Now nested: plot_data[family][channel] = {"z": [], "mu": [], "sig": []}
    plot_data = {fam: {} for fam in FAMILIES.keys()}
    
    with open(outpath, "w") as f_out:
        f_out.write("=" * 120 + "\n")
        f_out.write(header_fmt.format("Run", "Position_Z", "Family", "Channel", "Time_Mean", "Time_Sigma", "Time_Err", "FWHM", "N_Events") + "\n")
        f_out.write("=" * 120 + "\n")
        
        for fpath in files:
            rl = _run_label(fpath)
            try:
                uf = uproot.open(fpath)
                tree = uf[tree_name]
                z_pos = get_z_position(rl)
                pid_mask = compute_pid_mask(tree, particle_type) if particle_type else None
            except Exception: continue

            for family_name, fam_cfg in FAMILIES.items():
                xlim = [fam_cfg["tmin"], fam_cfg["tmax"]]
                for code_str in fam_cfg["channels"]:
                    b, g, ch = _parse_code(code_str)
                    
                    try:
                        arr_raw = get_tfinal_3mm(tree, b, g, ch, "_LP2_50")
                        if arr_raw is None: continue
                        
                        # adc_mask = compute_adc_mask(tree, code_str)
                        # combined_mask = pid_mask & adc_mask if pid_mask is not None else adc_mask
                        adc_mask = compute_adc_mask(tree, code_str)
                        wc_mask = compute_wc_mask(tree) # Will default to the 80.0 limit

                        base_mask = pid_mask & adc_mask if pid_mask is not None else adc_mask
                        combined_mask = base_mask & wc_mask


                        arr_adc = arr_raw[combined_mask]
                        arr_time = arr_adc[~np.isnan(arr_adc)]
                        arr_time = arr_time[(arr_time >= xlim[0]) & (arr_time <= xlim[1])]
                        n_final = len(arr_time)
                    except Exception: continue

                    if n_final < 25: continue
                    
                    # Fitting logic...
                    bins = np.linspace(xlim[0], xlim[1], 200 + 1)
                    centers = 0.5 * (bins[1:] + bins[:-1])
                    mode, _, h = _mode_from_hist(arr_time, bins)
                    h = h / h.max() if h.max() > 0 else h
                    
                    try:
                        p0 = [mode, arr_time.std()]
                        popt, _ = curve_fit(gaussian_peak_1, centers, h, p0=p0)
                        fit_mu, fit_sig = popt[0], abs(popt[1])
                    except: 
                        fit_mu, fit_sig = mode, float(arr_time.std())

                    # ==========================================================
                    # NEW CUT: Ignore SCI outlier channels with unphysical low sigma
                    # ==========================================================
                    if family_name == "SCI" and fit_sig < 0.050:
                        # Optional: print statement so you know which ones got dropped
                        print(f"        [SKIP] Ch {code_str} (SCI): Sigma too low ({fit_sig:.4f} < 0.050 ns)")
                        continue
                    # ==========================================================

                    # Statistical precision calculation
                    time_err = fit_sig / np.sqrt(n_final)

                    run_display = re.search(r"run(\d+)", rl).group(1) if re.search(r"run(\d+)", rl) else rl
                    f_out.write(row_fmt.format(
                        run_display, z_pos, family_name, code_str, 
                        fit_mu, fit_sig, time_err, 2.355 * fit_sig, n_final
                    ) + "\n")
                    
                    # NEW DATA APPENDING LOGIC: Save by specific channel
                    if z_pos != -999.0:
                        if code_str not in plot_data[family_name]:
                            plot_data[family_name][code_str] = {"z": [], "mu": [], "sig": []}
                        
                        plot_data[family_name][code_str]["z"].append(z_pos)
                        plot_data[family_name][code_str]["mu"].append(fit_mu)
                        plot_data[family_name][code_str]["sig"].append(time_err)
            uf.close()
    
    pid_label = f"PID_{particle_type}" if particle_type else "AllParticles"
    #create_channel_velocity_plots(plot_data, outpath, pid_label, particle_type)
    #create_z_toa_plot(plot_data, outpath, pid_label, particle_type)
    create_channel_velocity_plots(plot_data, outpath, pid_label, particle_type)
    # 2. Shared Intercept plot (Combined Cherenkov fit)
    #create_shared_intercept_plot(plot_data, outpath, pid_label, particle_type)
    
    print("All done.")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ana-files", nargs="+", default=None, help="Explicit list of input ROOT files.")
    ap.add_argument("--ana-glob", default=None, help="Glob for input ROOT files.")
    ap.add_argument("--tree", default=TREE_NAME, help="Tree name")
    ap.add_argument("--outdir", default="./PresiseTiming/wWirechambermask_updatedwcomments", help="Output directory")
    ap.add_argument("--pid", default='electron', choices=["muon", "pion", "electron", "proton"], help="Apply PID selection")

    args = ap.parse_args()

    print("\n[INIT] Initializing precision timing script...")
    if args.ana_files is None and args.ana_glob is None:
        raise SystemExit("[FATAL ERROR] Provide either --ana-files or --ana-glob")

    files = _resolve_files(args)
    if len(files) == 0:
        raise SystemExit("[FATAL ERROR] No files matched your selection")

    print(f"[INIT] Successfully resolved {len(files)} files.")
    print(f"[INIT] Output directory set to: {args.outdir}")
    print(f"[INIT] PID selection active: {args.pid}")
    
    # Run the Overlay Plots
    #process_all_channels(files, args.outdir, args.tree, args.pid)
    
    # Run the Stats Table and Velocity Plots
    pid_label = f"PID_{args.pid}" if args.pid else "AllParticles"
    output_txt_path = os.path.join(args.outdir, f"Timing_Statistics_{pid_label}.txt")
    generate_stats_table(files, output_txt_path, args.tree, particle_type=args.pid)
    
    print("\n[DONE] All families and velocity plots processed. Data exported successfully.")
if __name__ == "__main__":
    main()