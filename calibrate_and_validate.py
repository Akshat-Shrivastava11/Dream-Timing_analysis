#!/usr/bin/env python3
import os
import re
import json
import argparse
import numpy as np
import uproot
import awkward as ak
from scipy.optimize import curve_fit
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mplhep as hep
import matplotlib.patheffects as pe

# Apply CMS style for publication-quality plots
plt.style.use(hep.style.CMS)

# =========================================================
# CONFIGURATION & CONSTANTS
# =========================================================
TREE_NAME = "EventTree"
AMP_THRESHOLD = 20.0  
MIN_ADC_CUT = -100.0
NBINS = 150
WC_X_CUT = 75.0
TARGETS = {
    "y1000": {
        "CER-Quartz": ["617", "616", "615", "614", "625", "624", "623", "622", "637", "631", "630", "627", "626", "636", "515", "514", "635", "634", "633", "632", "501", "500", "002", "517", "516", "006", "004", "206", "204", "503", "502", "016", "014", "216", "214", "521", "520", "026", "024", "226", "224", "505", "504"],
        "SCI": ["621", "620", "003", "001", "203", "201", "007", "005", "207", "205", "013", "011", "213", "211", "017", "015", "217", "215"]
    },
    "y936": {
        "SCI": ["605", "604"]
    },
    "y1028": {
        "CER-Quartz": ["437", "407", "406", "405", "404", "436", "413", "412", "411", "410", "417", "416", "415", "414"],
        "SCI": ["421", "420", "425", "434"]
    }
}
# Manual Overrides (Plastic Removed as it's frozen)
MANUAL_OVERRIDES = {
    "CER-Quartz": {
         "002": +0.00, 
         "006": +0.00, "004": +0.00, 
         "016": 0.00, "014": +0.01,
         "216": -0.00, "214": -0.01,
         "502": -1.00, "503": -2.12, 
         "504": -2.25, "505": -2.25, 
         "624":-1.41, "632":-1.70,
        "514":-1.41, "511":-1.37,


        # #"306": -2.62,
        # #"316": +0.78, 
        # "532": -1.61, "536": -1.80, "334": -0.98, 
        # "402": -1.00,
        # "522": +0.10, "523": +0.10, "506": +0.10, "507": +1.10, "524": +1.10,
        # "525": +0.10, "510": +1.10, "511": +0.10, "520": -1.40, "521": -1.40,
        # "016": -0.45, "206": -0.40, "526": -1.30, "416": -1.44
    },
    "SCI": {
        "605": -1.09, "604": -1.07, "621": -1.09, "620": -1.07, "003": -0.01,
        # "001": -0.01, "203": -0.05, "201": -0.05, "007": -0.01, "005": -0.01,
        # "207": -0.05, "205": -0.05, "013": -0.01, "011": -0.01, "213": -0.05,
        # "211": -0.05, "017": -0.01, "015": -0.01, "217": -0.05, "215": -0.05,
        # "023": -0.01, "021": -0.11, "223": -0.05, "221": -0.05, "027": -0.01,
        # "025": -0.01, "227": -0.05, "225": -0.05, "033": -0.01, "031": -0.01,
        # "233": -0.05, "231": -0.05, "531": -1.73, "035": -0.01, "535": -1.73,
        # "235": -0.05, "123": -0.18, "121": -0.07, "323": -0.05, "321": -0.05,
        # "127": -0.18, "125": -0.07, "327": -0.05, "325": -0.10, "133": -0.18,
        # "131": -0.07, "333": -0.06, "331": -0.10, "533": -2.90, "135": -0.07,
        # "537": -1.78, "335": -0.10, "425": -1.34, "434": -1.34,
    }
}

FIXED_PLASTIC = {
      "000": -1.75,
      "010": -0.1,
      "012": -0.1,
      "020": -0.05050532294829502,
      "022": 0.04118641160619951,
      "032": 0.028620678044344317,
      "100": -0.0785018554575867,
      "102": -0.0006136427951037859,
      "110": -0.1381511727166007,
      "112": -0.0707079617909514,
      "120": -0.006470491594718908,
      "122": 0.02548640516969769,
      "130": -0.08302194580099886,
      "132": -1.41,
      "200": -1.7395766614740076,
      "202": -2.44,
      "210": -0.1,
      "212": -0.1,
      "220": -0.19164926632867818,
      "222": -0.03334092601700078,
      "230": -0.16644938722802927,
      "232": -0.0098899326705979,
      "300": -0.1889256910574577,
      "302": -0.15,
      "310": -0.16956436833990907,
      "312": -0.25,
      "320": -0.10861013583613044,
      "322": 0.0002125455828618783,
      "330": -0.1133639666521784,
      "332": 0.0006590009603293367,
      "422": -2.16451181346358,
      "423": -2.0865525872770103,
      "424": -1.85,
      "425": -2.0055986693471723,
      "426": -2.071595646680482,
      "427": -2.034387642494142,
      "430": -2.2388266199446694,
      "431": -2.2264471424681638,
      "432": -1.9064865115507956,
      "433": -1.9080147695871368,
      "600": -1.7805099933239212,
      "601": -1.6849012166275354,
      "602": -1.6,
      "603": -1.55,
      "606": -1.5550457089197423,
      "607": -1.51482719179325,
      "610": -1.7143018120282818,
      "611": -1.6364207160084199,
      "612": -1.5628684475579604,
      "613": -1.562912329710704
    }

# =========================================================
# PID & WC BRANCH MAPPING
# =========================================================
WC_CHANNELS = {"L1": "DRS_Board7_Group0_Channel0", "R1": "DRS_Board7_Group0_Channel1"}
PID_BRANCH_MAP = {
    "PSD": "DRS_Board7_Group1_Channel1", "HoleVeto": "DRS_Board7_Group1_Channel6",
    "NC": "DRS_Board7_Group1_Channel7", "T3": "DRS_Board7_Group2_Channel0",
    "T4": "DRS_Board7_Group2_Channel1", "KT1": "DRS_Board7_Group2_Channel2",
    "KT2": "DRS_Board7_Group2_Channel3", "TTUMuonVeto": "DRS_Board7_Group2_Channel4",
    "Cer474": "DRS_Board7_Group2_Channel5", "Cer519": "DRS_Board7_Group2_Channel6",
    "Cer537": "DRS_Board7_Group2_Channel7",
}

def get_service_drs_cut(service_drs: str) -> tuple:
    cuts = {
        "HoleVeto": (100, 350, -2e3, "Sum"), "PSD": (100, 400, -3500.0, "Sum"),
        "TTUMuonVeto": (200, 400, -2e3, "Sum"), "Cer474": (800, 900, -2000.0, "Sum"),
        "Cer519": (450, 550, -1000.0, "Sum"), "Cer537": (400, 500, -500.0, "Sum"),
    }
    return cuts.get(service_drs, (0, 1000, -5e4, "Sum"))

def get_particle_selection(particle_type: str) -> dict:
    selections = {
        "muon": {"TTUMuonVeto": True, "PSD": False},
        "pion": {"TTUMuonVeto": False, "PSD": False, "Cer474": True, "Cer519": True, "Cer537": True},
        "electron": {"TTUMuonVeto": False, "PSD": True, "Cer474": True, "Cer519": True, "Cer537": True},
        "proton": {"TTUMuonVeto": False, "PSD": False, "Cer474": False, "Cer519": False, "Cer537": False},
    }
    return selections.get(particle_type.lower(), {}) if particle_type else {}

# =========================================================
# GRIDS & TIMING LIMITS
# =========================================================
QUARTZ_GRID = [
    [None,  None,  "617", "616", "615", "614", None,  None],
    [None,  None,  "625", "624", "623", "622", None,  None],
    [None,  "637", "631", "630", "627", "626", "636", None],
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
    [None,  "437", "407", "406", "405", "404", "436", None],
    [None,  None,  "413", "412", "411", "410", None,  None],
    [None,  None,  "417", "416", "415", "414", None,  None],
]

SCI_GRID = [
    [None,None,"605","604",None,None],
    [None,None,None,None,None,None],
    [None,None,"621","620",None,None],
    [None,None,None,None,None,None],
    [None,"003","001","203","201",None],
    [None,"007","005","207","205",None],
    [None,"013","011","213","211",None],
    [None,"017","015","217","215",None],
    [None,"023","021","223","221",None],
    [None,"027","025","227","225",None],
    [None,"033","031","233","231",None],
    [None,'531',"035",'535',"235",None],
    [None,"103","101","303","301",None],
    [None,"107","105","307","305",None],
    [None,"113","111","313","311",None],
    [None,"117","115","317","315",None],
    [None,"123","121","323","321",None],
    [None,"127","125","327","325",None],
    [None,"133","131","333","331",None],
    [None,"533","135","537","335",None],
    [None,None,None,None,None,None],
    [None,None,"421","420",None,None],
    [None,None,None,None,None,None],
    [None,None,"425","434",None,None],
]

PLASTIC_GRID = [
    [None,None,"603","602","601","600",None,None],
    [None,None,None,"607","606",None,None,None],
    [None,None,"613","612","611","610",None,None],
    [None,None,None,None,None,None,None,None],
    [None,None,None,None,None,None,None,None],
    [None,None,None,"000","202","200",None,None],
    [None,None,"012","010","212","210",None,None],
    [None,None,"022","020","222","220",None,None],
    [None,None,"032",None,"232","230",None,None],
    [None,None,"102","100","302","300",None,None],
    [None,None,"112","110","312","310",None,None],
    [None,None,"122","120","322","320",None,None],
    [None,None,"132","130","332","330",None,None],
    [None,None,None,None,None,None,None,None],
    [None,None,None,None,None,None,None,None],
    [None,None,"425","424","423","422",None,None],
    [None,None,None,"427","426",None,None,None],
    [None,None,"433","432","431","430",None,None],
]

FAMILIES = {"CER-Quartz": QUARTZ_GRID, "SCI": SCI_GRID, "CER-Plastic": PLASTIC_GRID}
MASTER_CHANNELS = {fam: [c for row in grid for c in row if c is not None] for fam, grid in FAMILIES.items()}

# Ensure correct channel targets for trigger subtraction
THREEMM_ALL = set(MASTER_CHANNELS["CER-Quartz"] + MASTER_CHANNELS["SCI"] + MASTER_CHANNELS["CER-Plastic"])

# Bounds mapping based on global reference y1065
Y_CONFIGS = {
    "y1065": {"ref": "run1501_250928105227", "SCI": [-13.5, -7.5], "CER-Plastic": [-14.5, -11.5], "CER-Quartz": [-15.0, -11.5]},
    "y1000": {"ref": "run1502_250928113749", "SCI": [-11.0, -8.0], "CER-Plastic": [-12.5, -10.5], "CER-Quartz": [-13.5, -10.0]},
    "y936":  {"ref": "run1504_250928133854", "SCI": [-11.0, -7.5], "CER-Plastic": [-12.5, -11.0], "CER-Quartz": [-12.6, -11.0]},
    "y1028": {"ref": "run1506_250928143030", "SCI": [-10.5, -7.5], "CER-Plastic": [-12.5, -10.5], "CER-Quartz": [-12.5, -11.0]}
}

MASTER_Y = "y1065"
ANCHORS = { "CER-Quartz": "104", "CER-Plastic": "010", "SCI": "107" }


# =========================================================
# HELPER FUNCTIONS & MASKS
# =========================================================
def _parse_code(code_str): return int(code_str[0]), int(code_str[1]), int(code_str[2])

def get_hit_times_vectorized(events):
    if events.ndim != 2: return np.zeros(len(events))
    baselines = np.mean(events[:, :20], axis=1, keepdims=True)
    return np.argmin(events - baselines, axis=1)

def compute_wc_mask(tree, limit=WC_X_CUT):
    if WC_CHANNELS["L1"] not in tree.keys() or WC_CHANNELS["R1"] not in tree.keys():
        return np.ones(tree.num_entries, dtype=bool)
    L1 = ak.to_numpy(tree[WC_CHANNELS["L1"]].array(library="ak"))
    R1 = ak.to_numpy(tree[WC_CHANNELS["R1"]].array(library="ak"))
    L1_t, R1_t = get_hit_times_vectorized(L1), get_hit_times_vectorized(R1)
    return np.abs(L1_t - R1_t) < limit

def compute_pid_mask(tree, particle_type):
    requirements = get_particle_selection(particle_type)
    final_mask = np.ones(tree.num_entries, dtype=bool)
    if not requirements: return final_mask
    
    available_keys = set(tree.keys())
    for det, must_fire in requirements.items():
        branch = PID_BRANCH_MAP.get(det)
        if branch not in available_keys: continue
        ts_min, ts_max, val_cut, method = get_service_drs_cut(det)
        try:
            waves = tree[branch].array(library="ak")
            baseline = ak.mean(waves[:, :30], axis=1)
            window_sum = ak.sum((waves - baseline)[:, int(ts_min):int(ts_max)], axis=1)
            is_fired = ak.to_numpy(window_sum) < val_cut
            final_mask = final_mask & is_fired if must_fire else final_mask & (~is_fired)
        except Exception: continue
    return final_mask

def compute_adc_mask(tree, b, g, c):
    drs_br = f"DRS_Board{b}_Group{g}_Channel{c}"
    if drs_br not in tree: return np.ones(tree.num_entries, dtype=bool)
    waves = tree[drs_br].array(library="ak")
    baseline = ak.mean(waves[:, :30], axis=1)
    waves_blsub = waves - baseline
    return ak.to_numpy((ak.max(waves_blsub, axis=1) >= AMP_THRESHOLD) & (ak.min(waves_blsub, axis=1) >= MIN_ADC_CUT))

def compute_tfinal(tree, code_str, suffix="_LP2_50"):
    b, g, c = _parse_code(code_str)
    trg_b = b if code_str in THREEMM_ALL else 0
    names = [
        f"DRS_Board{b}_Group{g}_Channel{c}{suffix}", f"DRS_Board{b}_Group{g}_Channel8{suffix}",
        f"DRS_Board{trg_b}_Group3_Channel7{suffix}", f"DRS_Board{trg_b}_Group3_Channel8{suffix}"
    ]
    if not all(n in tree.keys() for n in names): return None
    arrs = [tree[n].array(library="np") for n in names]
    return -np.abs((arrs[0] - arrs[1]) - (arrs[2] - arrs[3]))

def _gauss(x, A, mu, sig): return A * np.exp(-0.5 * ((x - mu) / sig) ** 2)

def fit_mode(arr, bins, window=0.5):
    h, edges = np.histogram(arr, bins=bins)
    if h.sum() < 25: return np.nan
    centers = 0.5 * (edges[1:] + edges[:-1])
    x0 = float(centers[np.argmax(h)])
    m = (centers >= x0 - window) & (centers <= x0 + window)
    if h[m].max() < 5: return x0
    try:
        popt, _ = curve_fit(_gauss, centers[m], h[m], p0=[h.max(), x0, 0.15], bounds=([0, x0-window, 0.02], [np.inf, x0+window, 2.0]))
        return float(popt[1])
    except: return x0

def find_neighbor_shift(fam, target_ch, final_shifts_dict):
    grid = FAMILIES.get(fam, [])
    r_t, c_t = next(((r, c) for r, row in enumerate(grid) for c, val in enumerate(row) if val == target_ch), (-1, -1))
    if r_t == -1: return None, 0.0
    
    max_dist = max(len(grid), max(len(row) for row in grid))
    for d in range(1, max_dist):
        for dr in range(-d, d + 1):
            for dc in range(-d, d + 1):
                if max(abs(dr), abs(dc)) == d:
                    nr, nc = r_t + dr, c_t + dc
                    if 0 <= nr < len(grid) and 0 <= nc < len(grid[nr]):
                        neighbor_code = grid[nr][nc]
                        if neighbor_code and neighbor_code in final_shifts_dict[fam]:
                            return neighbor_code, final_shifts_dict[fam][neighbor_code]
    return None, 0.0

# =========================================================
# MULTIPROCESSING WORKER
# =========================================================
def worker_fit_channel(fpath, ch, fam, tmin, tmax, particle_type):
    """Worker function to process a single channel. Opens file, applies masks, fits mode."""
    try:
        with uproot.open(fpath) as uf:
            tree = uf[TREE_NAME]
            t_ch = compute_tfinal(tree, ch)
            if t_ch is None: return ch, np.nan
            
            # Global Masks per file
            pid_mask = compute_pid_mask(tree, particle_type)
            #wc_mask = compute_wc_mask(tree)
            
            # Local ADC Mask
            adc_mask = compute_adc_mask(tree, *_parse_code(ch))
            
            # Combine all cuts cleanly
            combined_mask = pid_mask  & adc_mask
            
            t_ch_cut = t_ch[combined_mask]
            t_ch_cut = t_ch_cut[(t_ch_cut >= tmin) & (t_ch_cut <= tmax)]
            
            bins = np.linspace(tmin, tmax, NBINS + 1)
            mode_ch = fit_mode(t_ch_cut, bins)
            return ch, mode_ch
    except Exception as e:
        print(f"Error processing channel {ch}: {e}")
        return ch, np.nan
    

# ================= PLOTTING CORE =================
# ================= PLOTTING CORE =================
def plot_family_mosaic(file_list, shifts_dict, outdir, particle_type="electron"):
    run_names = [re.search(r"(run\d+)", os.path.basename(f)).group(1) for f in file_list if re.search(r"(run\d+)", f)]
    runs_str = ", ".join(sorted(set(run_names)))
    
    pdf_path = os.path.join(outdir, "Mosaic_Calibration_Validation.pdf")
    
    # Pre-open trees to avoid excessive IO for every channel loop
    trees = [uproot.open(f)[TREE_NAME] for f in file_list]
    
    with PdfPages(pdf_path) as pdf:
        for fam, grid in FAMILIES.items():
            print(f"  -> Generating mosaic for {fam}...")
            
            # Use the MASTER_Y bounds for validation to keep plots standardized
            tmin, tmax = Y_CONFIGS[MASTER_Y][fam]
            
            xlim = (tmin, tmax)
            bins = np.linspace(tmin, tmax, NBINS + 1)
            centers = 0.5 * (bins[1:] + bins[:-1])

            nrows = len(grid)
            ncols = max(len(row) for row in grid)
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 2.2), sharex=True)
            if nrows == 1: axes = np.atleast_2d(axes)

            fig.suptitle(f"{fam} Family | Validation Runs: {runs_str}\nBounds: [{tmin}, {tmax}] | Blue = Pre-Calib | Orange = Post-Calib", 
                         fontsize=24, fontweight='bold', y=0.98)
            
            # Arrays to store modes for the heatmap
            heatmap_pre = np.full((nrows, ncols), np.nan)
            heatmap_post = np.full((nrows, ncols), np.nan)
            
            for r, row in enumerate(grid):
                for c in range(ncols):
                    ax = axes[r, c]
                    code = row[c] if c < len(row) else None
                    
                    if code is None:
                        ax.axis('off')
                        continue

                    b_drs, g_drs, c_drs = _parse_code(code)
                    combined_raw = []
                    
                    # Extract and concatenate data across all trees for this specific channel
                    for tree in trees:
                        raw_arr = compute_tfinal(tree, code)
                        if raw_arr is None: continue
                        
                        adc_mask = compute_adc_mask(tree, b_drs, g_drs, c_drs)
                        pid_mask = compute_pid_mask(tree, particle_type)
                        
                        valid_raw = raw_arr[adc_mask & pid_mask]
                        valid_raw = valid_raw[~np.isnan(valid_raw)]
                        combined_raw.append(valid_raw)
                        
                    if not combined_raw:
                        ax.text(0.5, 0.5, f"{code}\n(No Data)", ha='center', va='center', fontsize=10, alpha=0.5)
                        ax.axis('off')
                        continue
                        
                    raw_arr = np.concatenate(combined_raw)
                    raw_arr_cut = raw_arr[(raw_arr >= tmin) & (raw_arr <= tmax)]

                    if len(raw_arr_cut) < 25:
                        ax.text(0.5, 0.5, f"{code}\n(Low Stats)", ha='center', va='center', fontsize=10, alpha=0.5)
                        ax.axis('off')
                        continue
                    
                    # Ready to plot
                    raw_arr = raw_arr_cut
                    shift_val = shifts_dict[fam].get(code, 0.0)
                    shifted_arr = raw_arr + shift_val

                    h_raw, _ = np.histogram(raw_arr, bins=bins)
                    h_shifted, _ = np.histogram(shifted_arr, bins=bins)
                    
                    mode_raw = fit_mode(raw_arr, bins)
                    mode_shifted = fit_mode(shifted_arr, bins)
                    
                    # Store for heatmap
                    heatmap_pre[r, c] = mode_raw
                    heatmap_post[r, c] = mode_shifted
                    
                    # Plotting
                    ax.step(centers, h_raw, where='mid', lw=1.5, alpha=0.7, color='tab:blue', label=f'Pre: {mode_raw:.2f}')
                    ax.step(centers, h_shifted, where='mid', lw=2.0, alpha=0.85, color='tab:orange', label=f'Post: {mode_shifted:.2f}')
                    
                    ax.axvline(mode_raw, color='tab:blue', ls=':', lw=1.5, alpha=0.6)
                    ax.axvline(mode_shifted, color='tab:orange', ls='--', lw=1.5, alpha=0.9)

                    ax.set_title(code, fontsize=14, fontweight='bold', pad=3)
                    
                    # Clean Legend
                    ax.legend(loc='upper right', fontsize=8, title=f"Shift: {shift_val:+.2f} ns", title_fontsize=9, framealpha=0.8, edgecolor='gray')
                    
                    ax.set_xlim(*xlim)
                    
                    # Dynamic y-axis
                    max_h = max(h_raw.max(), h_shifted.max())
                    if max_h > 0:
                        ax.set_ylim(0, max_h * 1.4)
                        
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    if c == 0: ax.set_ylabel("Events", fontsize=10)
                    if r == nrows - 1: ax.set_xlabel("Time [ns]", fontsize=10)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fig.subplots_adjust(hspace=0.4, wspace=0.3)
            pdf.savefig(fig)
            plt.close(fig)

            # --- 2. HEATMAP PLOT ---
            fig_heat, axes_heat = plt.subplots(1, 2, figsize=(14, max(8, nrows * 0.4)))
            fig_heat.suptitle(f"{fam} T.O.A. Modes Heatmap | Validation", fontsize=20, fontweight='bold', y=0.98)
            
            cmap = plt.cm.viridis.copy()
            cmap.set_bad(color='white')

            im0 = axes_heat[0].imshow(heatmap_pre, cmap=cmap)
            axes_heat[0].set_title("Pre-Calib Mode [ns]", fontsize=16)
            fig_heat.colorbar(im0, ax=axes_heat[0], fraction=0.046, pad=0.04)
            
            im1 = axes_heat[1].imshow(heatmap_post, cmap=cmap)
            axes_heat[1].set_title("Post-Calib Mode [ns]", fontsize=16)
            fig_heat.colorbar(im1, ax=axes_heat[1], fraction=0.046, pad=0.04)

            txt_path_effect = [pe.withStroke(linewidth=1.5, foreground="black")]
            for r_idx in range(nrows):
                for c_idx in range(ncols):
                    code = grid[r_idx][c_idx] if c_idx < len(grid[r_idx]) else None
                    if code is not None:
                        val_pre = heatmap_pre[r_idx, c_idx]
                        val_post = heatmap_post[r_idx, c_idx]
                        
                        txt_pre = f"{code}\n{val_pre:.2f}" if not np.isnan(val_pre) else f"{code}\nN/A"
                        txt_post = f"{code}\n{val_post:.2f}" if not np.isnan(val_post) else f"{code}\nN/A"
                        
                        axes_heat[0].text(c_idx, r_idx, txt_pre, ha='center', va='center', fontsize=9, color='white', path_effects=txt_path_effect)
                        axes_heat[1].text(c_idx, r_idx, txt_post, ha='center', va='center', fontsize=9, color='white', path_effects=txt_path_effect)

            axes_heat[0].axis('off')
            axes_heat[1].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig_heat)
            plt.close(fig_heat)
            
            # --- 3. 1D HISTOGRAM OF ALL SHIFTS ---
            fig_shifts, ax_shifts = plt.subplots(figsize=(10, 6))
            ax_shifts.set_xlabel("Applied Shift [ns]", fontsize=14)
            ax_shifts.set_ylabel("Number of Channels", fontsize=14)
            hep.cms.label(ax=ax_shifts, exp="CaloX", data=False, llabel="Validation", rlabel=f"{particle_type.capitalize()}")

            shifts = []
            for ch in MASTER_CHANNELS[fam]:
                if ch in MANUAL_OVERRIDES.get(fam, {}): continue
                if ch in shifts_dict[fam] and shifts_dict[fam][ch] != 0.0:
                    shifts.append(shifts_dict[fam][ch])

            if shifts:
                counts, bins, patches = ax_shifts.hist(shifts, bins=25, color='tab:green', alpha=0.7, edgecolor='black', linewidth=1.2)
                try:
                    bin_centers = 0.5 * (bins[1:] + bins[:-1])
                    p0 = [np.max(counts), np.mean(shifts), np.std(shifts)]
                    popt, _ = curve_fit(_gauss, bin_centers, counts, p0=p0)
                    mu_shift, sig_shift = popt[1], abs(popt[2])
                    
                    x_fit = np.linspace(min(shifts), max(shifts), 200)
                    y_fit = _gauss(x_fit, *popt)
                    ax_shifts.plot(x_fit, y_fit, color='darkgreen', lw=2.5, label=f"Fit: $\mu={mu_shift:.2f}$, $\sigma={sig_shift:.2f}$")
                    ax_shifts.legend(loc="upper right", fontsize=12)
                except:
                    pass

                mean_val = np.mean(shifts)
                ax_shifts.axvline(mean_val, color='red', linestyle='--', lw=2, label=f"Mean: {mean_val:.2f} ns")
                ax_shifts.legend(loc="upper right", fontsize=12)
                
                ax_shifts.set_title(f"{fam} Shift Distribution (Overrides Excluded)", fontsize=16, pad=10)
                plt.tight_layout()
                pdf.savefig(fig_shifts)
            plt.close(fig_shifts)

    # Clean up ROOT files
    for tree in trees:
        tree.file.close()
        
    print(f"Saved Validation Plots to: {pdf_path}")


def get_file_path(directory, run_label):
    """Finds the actual filename in the directory based on the run label."""
    for f in os.listdir(directory):
        if run_label in f and f.endswith(".root"): 
            return os.path.join(directory, f)
    return None

# =========================================================
# MAIN PIPELINE
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir",  default="/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples")
    ap.add_argument("--outdir", default="./Calib_output_newstrat", help="Output directory")
    ap.add_argument("--pid", default="electron", help="Particle type for PID (e.g. electron, pion)")
    ap.add_argument("--cores", type=int, default=8, help="Number of CPU cores for multiprocessing")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    final_shifts = {"CER-Quartz": {}, "CER-Plastic": FIXED_PLASTIC, "SCI": {}}
    master_modes = {}
    failed_channels = []

    # 1. ESTABLISH MASTER MODES
    print(f"\n--- [1] Establishing Master Global Reference Anchors ---")
    y1065_path = get_file_path(args.dir, Y_CONFIGS['y1065']['ref'])
    
    if not y1065_path:
        print(f"[FATAL ERROR] Could not find the y1065 master reference file in {args.dir}")
        return

    for fam, anchor in ANCHORS.items():
        if fam == "CER-Plastic": continue 
        tmin, tmax = Y_CONFIGS[MASTER_Y][fam]
        _, mode = worker_fit_channel(y1065_path, anchor, fam, tmin, tmax, args.pid)
        master_modes[fam] = mode
        final_shifts[fam][anchor] = 0.0  # Force anchor shift to exactly 0.0
        print(f"  [{fam}] Master Anchor ({anchor}) Mode: {mode:.3f} ns (Shift locked to 0.00 ns)")

    # 2. MULTIPROCESSING CHANNEL FLUSHING
    print(f"\n--- [2] Flushing Channels to Anchor Modes (Multiprocessing) ---")
    for yg in ["y1000", "y1065", "y936", "y1028"]:
        fpath = get_file_path(args.dir, Y_CONFIGS[yg]['ref'])
        if not fpath: 
            print(f"  [WARN] Could not find file for {yg}. Skipping...")
            continue
        
        for fam in ["CER-Quartz", "SCI"]:
            tmin, tmax = Y_CONFIGS[yg][fam]
            
            # Use TARGETS dict, or sweep up the rest if we are on the master file
            if yg == MASTER_Y:
                targets = [ch for ch in MASTER_CHANNELS[fam] if ch not in final_shifts[fam]]
            else:
                targets = TARGETS.get(yg, {}).get(fam, [])
                targets = [ch for ch in targets if ch not in final_shifts[fam]]

            if not targets: continue

            with ProcessPoolExecutor(max_workers=args.cores) as executor:
                futures = {executor.submit(worker_fit_channel, fpath, ch, fam, tmin, tmax, args.pid): ch for ch in targets}
                
                for future in as_completed(futures):
                    ch, mode_ch = future.result()
                    
                    # Manual Override check
                    if ch in MANUAL_OVERRIDES.get(fam, {}):
                        shift = MANUAL_OVERRIDES[fam][ch]
                        final_shifts[fam][ch] = shift
                        
                        # Handle case where mode is NaN but we forced a shift anyway
                        if np.isnan(mode_ch):
                            print(f"  [{fam}] Ch {ch}: OVERRIDE -> Shift = {shift:+.2f} ns (Pre: NaN, Post: NaN)")
                        else:
                            post_mode = mode_ch + shift
                            print(f"  [{fam}] Ch {ch}: OVERRIDE -> Shift = {shift:+.2f} ns (Pre: {mode_ch:.2f}, Post: {post_mode:.2f})")
                        continue
                    
                    # Normal Calculation
                    if np.isnan(mode_ch):
                        failed_channels.append((fam, ch))
                    else:
                        shift = float(master_modes[fam] - mode_ch)
                        final_shifts[fam][ch] = shift
                        post_mode = mode_ch + shift
                        
                        print(f"  [{fam}] Ch {ch}: Shift = {shift:+.2f} ns (Pre: {mode_ch:.2f}, Post: {post_mode:.2f})")

    # 3. NEAREST NEIGHBOR FALLBACK
    print(f"\n--- [3] Nearest Neighbor Fallback for {len(failed_channels)} noisy/missing channels ---")
    for fam, ch in failed_channels:
        if ch in final_shifts[fam]: continue
        neighbor_ch, neighbor_shift = find_neighbor_shift(fam, ch, final_shifts)
        if neighbor_ch:
            final_shifts[fam][ch] = neighbor_shift
            print(f"  [{fam}] Ch {ch} -> Copied Shift {neighbor_shift:+.2f} ns from neighbor '{neighbor_ch}'")
        else:
            final_shifts[fam][ch] = 0.0
            print(f"  [{fam}] Ch {ch} -> No neighbors. Set to +0.00 ns")

    # 4. EXPORT
    out_json = os.path.join(args.outdir, "master_calibration_shifts.json")
    with open(out_json, "w") as f:
        json.dump({"shifts_by_family": final_shifts}, f, indent=2, sort_keys=True)
    print(f"\n[DONE] Calibration JSON saved to {out_json}")

    # 5. VALIDATION PLOTS
    # 5. VALIDATION PLOTS
    print(f"\n--- [4] Generating Validation Mosaics & Heatmaps ---")
    test_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith(".root")][:3] 
    if test_files:
        plot_family_mosaic(test_files, final_shifts, args.outdir, args.pid)
if __name__ == "__main__": 
    main()