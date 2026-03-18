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

# Using your requested negative limits
FAMILIES = {
    "Plastic": {"channels": ["100","102","112", "110"], "tmin": -14.5, "tmax": -11.5, "legend": "Cherenkov-Plastic", "color": "red"},
    "Quartz":  {"channels": ["104","106", "304","114"], "tmin": -15.0, "tmax": -11.5, "legend": "Cherenkov-Quartz",  "color": "blue"},
    "SCI":     {"channels": ["105", "107","111","117"], "tmin": -13.5, "tmax":  -9.5, "legend": "Scintillating",     "color": "green"}
}

#all channels
# FAMILIES = {
#     "Plastic": {
#         "channels": extract_channels(PLASTIC_GRID), 
#         "tmin": -14.5, "tmax": -11.5, 
#         "legend": "Cherenkov-Plastic", "color": "red"
#     },
#     "Quartz":  {
#         "channels": extract_channels(QUARTZ_GRID),  
#         "tmin": -15.0, "tmax": -11.5, 
#         "legend": "Cherenkov-Quartz",  "color": "blue"
#     },
#     "SCI":     {
#         "channels": extract_channels(SCI_ALL_GRID), 
#         "tmin": -13.5, "tmax":  -9.5, 
#         "legend": "Scintillating",     "color": "green"
#     }
# }
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
        if "192918" in run_label: return -54.5
        if "194230" in run_label: return -400.3
    match = re.search(r"run(\d+)", run_label)
    run_num = int(match.group(1)) if match else None
    z_map = {1501: -168.0, 1507: -218.0, 1511: -268.0}
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

# ================= VELOCITY & STATS PLOTTING =================
# def style_paper_axes(ax, xlabel, ylabel, particle_type):
#     ax.set_xlabel(xlabel, loc='right', fontsize=14, fontweight='bold')
#     ax.set_ylabel(ylabel, loc='top', fontsize=14, fontweight='bold')
    
#     ax.xaxis.set_minor_locator(AutoMinorLocator())
#     ax.yaxis.set_minor_locator(AutoMinorLocator())
#     ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=12)
#     ax.tick_params(which='major', length=8)
#     ax.tick_params(which='minor', length=4)
    
#     display_name = "Positron" if particle_type.lower() == "electron" else particle_type.capitalize()
    
#     # Wrapped the trailing text in \\mathbf{{ }} and used \\ to preserve spaces
#     header_text = r"$\mathbf{CaloX}$ $\mathit{Data}$" + f"  $\\mathbf{{40\ GeV\ {display_name}}}$"
    
#     ax.text(0.0, 1.02, header_text, transform=ax.transAxes, fontsize=16, va='bottom', ha='left', color='black')


def style_paper_axes(ax, xlabel, ylabel, particle_type):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    display_name = "Positron" if particle_type.lower() == "electron" else particle_type.capitalize()
    
    # --- USE MPLHEP FOR THE HEADER ---
    right_label = f"40 GeV {display_name}"
    
    # Adding llabel="Data" ensures it prints "CaloX Data"
    hep.cms.label(ax=ax, exp="CaloX", llabel="Data", data=True, rlabel=right_label)

# ================= UPDATED VELOCITY PLOT WITH FIT ERRORS & 10^8 =================
def create_z_toa_plot(plot_data, txt_path, pid_label, particle_type):
    outdir = os.path.dirname(txt_path)
    pdf_path = os.path.join(outdir, f"Z_vs_TOA_Fits_{pid_label}.pdf")
    
    print("\n[VELOCITY] --------------------------------------------------------")
    print(f"[VELOCITY] Calculating Independent Velocity Fits (PID: {pid_label})")
    
    with open(txt_path, "a") as f_out:
        f_out.write("\n" + "=" * 115 + "\n")
        f_out.write(f"{'FAMILY':<10} | {'VELOCITY [m/s]':<15} | {'V_ERROR [m/s]':<15} | {'FIT EQUATION'}\n")
        f_out.write("=" * 115 + "\n")
        
        with PdfPages(pdf_path) as pdf:
            # INCREASED SIZE HERE
            fig, ax = plt.subplots(figsize=(14, 10))
            style_paper_axes(ax, "Z Position [mm]", "Time of Arrival Mean [ns]", particle_type)
            
            for fam, data in plot_data.items():
                if not data["z"]: continue
                z_arr, mu_arr, sig_mean_arr = np.array(data["z"]), np.array(data["mu"]), np.array(data["sig"])
                color = FAMILIES[fam]["color"]
                
                weights = 1.0 / sig_mean_arr
                params, cov = np.polyfit(z_arr, mu_arr, 1, w=weights, cov=True)
                slope, intercept = params[0], params[1]
                slope_err = np.sqrt(cov[0,0])
                intercept_err = np.sqrt(cov[1,1])
                
                speed_m_s = abs(1.0 / slope) * 1e6 if slope != 0 else 0
                speed_err = (speed_m_s**2) * (slope_err * 1e-6) if slope != 0 else 0
                
                v8 = speed_m_s / 1e8
                v8_err = speed_err / 1e8
                v_str = f"{v8:.4f}e8"
                v_err_str = f"{v8_err:.4f}e8"
                
                eq_str = f"t = ({slope:.4f} +/- {slope_err:.4f})z {'+' if intercept >= 0 else '-'} ({abs(intercept):.2f} +/- {intercept_err:.2f})"
                
                f_out.write(f"{fam:<10} | {v_str:<15} | {v_err_str:<15} | {eq_str}\n")
                legend_label = f"{fam:<10} | {v_str:<15} | {v_err_str:<15} | {eq_str}"
                
                z_fit = np.linspace(min(z_arr) - 20, 400, 200)
                ax.errorbar(z_arr, mu_arr, yerr=sig_mean_arr, fmt='o', color=color, capsize=3, markersize=4)
                ax.plot(z_fit, slope * z_fit + intercept, '-', color=color, linewidth=2, label=legend_label)

            legend_title = f"{'FAMILY':<10} | {'VELOCITY [m/s]':<15} | {'V_ERROR [m/s]':<15} | {'FIT EQUATION'}"
            
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
            style_paper_axes(ax, "Z Position [mm]", "Time of Arrival Mean [ns]", particle_type)

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
                ax.errorbar(z_arr, mu_arr, yerr=sig_arr, fmt='o', color=FAMILIES[fam]["color"], capsize=3, markersize=4)
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

# ================= UPDATED STATS TABLE WITH TIME_ERROR =================
def generate_stats_table(files, outpath, tree_name, particle_type=None):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    # Added Time_Err column to the format
    header_fmt = "{:<10} | {:<10} | {:<10} | {:<8} | {:<12} | {:<12} | {:<12} | {:<12} | {:<10}"
    row_fmt    = "{:<10} | {:<10.1f} | {:<10} | {:<8} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<10}"
    
    plot_data = {fam: {"z": [], "mu": [], "sig": []} for fam in FAMILIES.keys()}
    
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
                        
                        adc_mask = compute_adc_mask(tree, code_str)
                        combined_mask = pid_mask & adc_mask if pid_mask is not None else adc_mask
                        
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

                    # Statistical precision calculation
                    time_err = fit_sig / np.sqrt(n_final)

                    run_display = re.search(r"run(\d+)", rl).group(1) if re.search(r"run(\d+)", rl) else rl
                    f_out.write(row_fmt.format(
                        run_display, z_pos, family_name, code_str, 
                        fit_mu, fit_sig, time_err, 2.355 * fit_sig, n_final
                    ) + "\n")
                    
                    if z_pos != -999.0:
                        plot_data[family_name]["z"].append(z_pos)
                        plot_data[family_name]["mu"].append(fit_mu)
                        plot_data[family_name]["sig"].append(time_err) # Use SE for the plot weights
            uf.close()
    
    pid_label = f"PID_{particle_type}" if particle_type else "AllParticles"
    create_z_toa_plot(plot_data, outpath, pid_label, particle_type)
    # 2. Shared Intercept plot (Combined Cherenkov fit)
    create_shared_intercept_plot(plot_data, outpath, pid_label, particle_type)
    
    print("All done.")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ana-files", nargs="+", default=None, help="Explicit list of input ROOT files.")
    ap.add_argument("--ana-glob", default=None, help="Glob for input ROOT files.")
    ap.add_argument("--tree", default=TREE_NAME, help="Tree name")
    ap.add_argument("--outdir", default="./PresiseTiming", help="Output directory")
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