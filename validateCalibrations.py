#!/usr/bin/env python3
import os
import re
import json
import argparse
import numpy as np
import uproot
import awkward as ak
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mplhep as hep
import matplotlib.patheffects as pe


# Apply CMS style for publication-quality plots
plt.style.use(hep.style.CMS)

# ================= CONFIGURATION =================
TREE_NAME = "EventTree"
AMP_THRESHOLD = 20.0     # Loose cuts
MIN_ADC_CUT = -500.0     # Loose cuts
NBINS = 100

# ================= 3MM EXACT CHANNELS =================
THREEMM_ALL = set([
    "002", "006", "004", "206", "204", "016", "014", "216", "214", "026", "024", "226", "224", 
    "030", "034", "106", "104", "306", "304", "116", "114", "316", "314", "126", "124", "326", 
    "324", "134", "334", "000", "202", "200", "012", "010", "212", "210", "022", "020", "222", 
    "220", "032", "232", "230", "102", "100", "302", "300", "112", "110", "312", "310", "122", 
    "120", "322", "320", "132", "130", "332", "330", "003", "001", "203", "201", "007", "005", 
    "207", "205", "013", "011", "213", "211", "017", "015", "217", "215", "023", "021", "223", 
    "221", "027", "025", "227", "225", "033", "031", "233", "231", "035", "235", "103", "101", 
    "303", "301", "107", "105", "307", "305", "113", "111", "313", "311", "117", "115", "317", 
    "315", "123", "121", "323", "321", "127", "125", "327", "325", "133", "131", "333", "331", 
    "135", "335"
])

# ================= RUN MAPPING =================
RUN_MAP = {
    "run1502_250928113749": "y1000",
    "run1508_250928161049": "y1000",
    "run1512_250928183645": "y1000",
    "run1501_250928105227": "y1065",
    "run1511_250928180741": "y1065",
    "run1507_250928160030": "y1065",
    "run1513_250928192918": "y1065",
    "run1513_250928194230": "y1065",
    "run1504_250928133854": "y936",
    "run1509_250928164817": "y936",
    "run1512_250928185722": "y936",
    "run1506_250928143030": "y1028",
    "run1506_250928145724": "y1028",
    "run1510_250928172949": "y1028"
}

# ================= DYNAMIC TIMING LIMITS =================
Y_CONFIGS = {
    "y1000": {
        "SCI":         {"tmin": -11.0, "tmax": -8.0},
        "CER-Plastic": {"tmin": -12.5, "tmax": -10.5},
        "CER-Quartz":  {"tmin": -13.5, "tmax": -10.0}
    },
    "y1065": {
        "CER-Plastic": {"tmin": -14.5, "tmax": -11.5},
        "CER-Quartz":  {"tmin": -15.0, "tmax": -11.5},
        "SCI":         {"tmin": -13.5, "tmax": -9.5}
    },
    "y936": {
        "SCI":         {"tmin": -11.0, "tmax": -8.0},
        "CER-Plastic": {"tmin": -12.5, "tmax": -11.0},
        "CER-Quartz":  {"tmin": -12.6, "tmax": -11.0}
    },
    "y1028": {
        "SCI":         {"tmin": -10.5, "tmax": -7.0},
        "CER-Plastic": {"tmin": -12.5, "tmax": -10.5},
        "CER-Quartz":  {"tmin": -12.5, "tmax": -11.0}
    }
}

# ================= GRIDS =================
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
    [None,None,None,None,None,None],
    [None,None,"421","420",None,None],
    [None,None,None,None,None,None],
    [None,None,"425","434",None,None],
]

PLASTIC_GRID = [
    [None,  None,  "603", "602", "601", "600", None,  None],
    [None,  None,  None,  "607", "606", None,  None,  None],
    [None,  None,  "613", "612", "611", "610", None,  None],
    [None,  None,  None,  None,  None,  None,  None,  None],
    [None,  None,  None,  None,  None,  None,  None,  None],
    [None,  None,  None,  None,  None,  None,  None,  None],
    [None,  "000", "202", "200"],
    ["012", "010", "212", "210"],
    ["022", "020", "222", "220"],
    ["032", None,  "232", "230"],
    ["102", "100", "302", "300"],
    ["112", "110", "312", "310"],
    ["122", "120", "322", "320"],
    ["132", "130", "332", "330"],
    [None,  None,  None,  None,  None,  None,  None,  None],
    [None,  None,  None,  None,  None,  None,  None,  None],
    [None,  None,  None,  None,  None,  None,  None,  None],
    [None,  None,  "425", "424", "423", "422", None,  None],
    [None,  None,  None,  "427", "426", None,  None,  None],
    [None,  None,  "433", "432", "431", "430", None,  None],
]

FAMILIES = {
    "CER-Quartz": QUARTZ_GRID,
    "SCI": SCI_GRID,
    "CER-Plastic": PLASTIC_GRID
}

# ================= HELPER FUNCTIONS =================
def _parse_code(code_str):
    return int(code_str[0]), int(code_str[1]), int(code_str[2])

def get_run_group(path: str) -> str:
    base = os.path.basename(path)
    for run_key, y_group in RUN_MAP.items():
        if run_key in base:
            return y_group
    raise ValueError(f"Run {base} is not in RUN_MAP!")

def get_z_position(run_label):
    if "run1513" in run_label:
        if "192918" in run_label: return -54.5
        if "194230" in run_label: return -400.3
    match = re.search(r"run(\d+)", run_label)
    run_num = int(match.group(1)) if match else None
    
    z_map = {
        # y1065 runs
        1501: -168.0, 1507: -218.0, 1511: -268.0,
        
        # y936 runs
        1504: -168.0, 1509: -218.0, 1512: -268.0,
        
        # y1000 runs
        1502: -168.0,  
        1508: -218.0,  
        
        # y1028 runs
        1506: -168.0,  
        1510: -218.0   
    }
    return z_map.get(run_num, -999.0)

def compute_adc_mask(tree, b, g, c):
    drs_br = f"DRS_Board{b}_Group{g}_Channel{c}"
    if drs_br not in tree:
        return np.ones(tree.num_entries, dtype=bool)
    waves = tree[drs_br].array(library="ak")
    baseline = ak.mean(waves[:, :30], axis=1)
    waves_blsub = waves - baseline
    peak = ak.max(waves_blsub, axis=1)
    min_adc = ak.min(waves_blsub, axis=1)
    return ak.to_numpy((peak >= AMP_THRESHOLD) & (min_adc >= MIN_ADC_CUT))

def compute_tfinal(tree, code_str, suffix="_LP2_50"):
    b, g, c = _parse_code(code_str)
    br_sig     = f"DRS_Board{b}_Group{g}_Channel{c}{suffix}"
    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8{suffix}"
    
    trg_b = b if code_str in THREEMM_ALL else 0
    br_trg     = f"DRS_Board{trg_b}_Group3_Channel7{suffix}"
    br_trg_ref = f"DRS_Board{trg_b}_Group3_Channel8{suffix}"
    
    if not all(br in tree.keys() for br in [br_sig, br_sig_ref, br_trg, br_trg_ref]):
        return None
            
    arr_sig     = tree[br_sig].array(library="np")
    arr_sig_ref = tree[br_sig_ref].array(library="np")
    arr_trg     = tree[br_trg].array(library="np")
    arr_trg_ref = tree[br_trg_ref].array(library="np")
        
    t_final = (arr_sig - arr_sig_ref) - (arr_trg - arr_trg_ref)
    
    # Force negative domain, matching the calibration
    return -np.abs(t_final)

def _gauss(x, A, mu, sig):
    return A * np.exp(-0.5 * ((x - mu) / sig) ** 2)

def fit_mode(arr, bins, window=0.5):
    h, edges = np.histogram(arr, bins=bins)
    if h.sum() == 0: return np.nan
    centers = 0.5 * (edges[1:] + edges[:-1])
    imax = int(np.argmax(h))
    x0 = float(centers[imax])

    m = (centers >= x0 - window) & (centers <= x0 + window)
    x = centers[m]
    y = h[m]

    if x.size < 6 or y.max() < 5: return x0

    p0 = [float(y.max()), x0, 0.15]
    bounds = ([0.0, x0 - window, 0.02], [np.inf, x0 + window, 2.0])

    try:
        popt, _ = curve_fit(_gauss, x, y, p0=p0, bounds=bounds, maxfev=5000)
        return float(popt[1])
    except Exception:
        return x0

# ================= PLOTTING CORE =================
def plot_family_mosaic_by_z(file_list, shifts_dict, z_pos, outdir):
    # Prepare titles and dynamic bounds based on all files being mapped
    run_names = [re.search(r"(run\d+)", os.path.basename(f)).group(1) for f in file_list if re.search(r"(run\d+)", f)]
    runs_str = ", ".join(sorted(set(run_names)))
    
    pdf_path = os.path.join(outdir, f"Mosaic_Z_{z_pos}mm.pdf")
    y_groups = [get_run_group(f) for f in file_list]
    
    # Pre-open trees to avoid excessive IO for every channel loop
    trees = [uproot.open(f)[TREE_NAME] for f in file_list]
    
    with PdfPages(pdf_path) as pdf:
        for fam, grid in FAMILIES.items():
            print(f"  -> Generating mosaic for {fam}...")
            
            # Since files might have different Y configurations, combine the bounds dynamically
            tmin = min(Y_CONFIGS[yg][fam]["tmin"] for yg in y_groups)
            tmax = max(Y_CONFIGS[yg][fam]["tmax"] for yg in y_groups)
            
            xlim = (tmin, tmax)
            bins = np.linspace(tmin, tmax, NBINS + 1)
            centers = 0.5 * (bins[1:] + bins[:-1])

            nrows = len(grid)
            ncols = max(len(row) for row in grid)
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 2.2), sharex=True)
            if nrows == 1: axes = np.atleast_2d(axes)

            fig.suptitle(f"{fam} Family | Z = {z_pos} mm | Runs: {runs_str}\nBounds: [{tmin}, {tmax}] | Blue = Pre-Calib | Orange = Post-Calib", 
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
                        valid_raw = raw_arr[adc_mask]
                        valid_raw = valid_raw[~np.isnan(valid_raw)]
                        combined_raw.append(valid_raw)
                        
                    if not combined_raw:
                        ax.text(0.5, 0.5, f"{code}\n(No Data)", ha='center', va='center', fontsize=10, alpha=0.5)
                        ax.axis('off')
                        continue
                        
                    raw_arr = np.concatenate(combined_raw)
                    
                    # Apply bounds limits for plotting
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
                    ax.step(centers, h_raw, where='mid', lw=1.5, alpha=0.7, color='tab:blue', label='Pre')
                    ax.step(centers, h_shifted, where='mid', lw=2.0, alpha=0.85, color='tab:orange', label='Post')
                    
                    ax.axvline(mode_raw, color='tab:blue', ls=':', lw=1.5, alpha=0.6)
                    ax.axvline(mode_shifted, color='tab:orange', ls='--', lw=1.5, alpha=0.9)

                    ax.set_title(code, fontsize=14, fontweight='bold', pad=3)
                    ax.text(0.05, 0.85, f"Shift: {shift_val:+.2f}ns", transform=ax.transAxes, fontsize=10, 
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                    
                    ax.set_xlim(*xlim)
                    
                    # Dynamic y-axis to handle both huge peaks and tiny ones without breaking
                    max_h = max(h_raw.max(), h_shifted.max())
                    if max_h > 0:
                        ax.set_ylim(0, max_h * 1.3)
                        
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    if c == 0: ax.set_ylabel("Events", fontsize=10)
                    if r == nrows - 1: ax.set_xlabel("Time [ns]", fontsize=10)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fig.subplots_adjust(hspace=0.4, wspace=0.3)
            pdf.savefig(fig)
            plt.close(fig)


            # --- 2. HEATMAP PLOT ---
            fig_heat, axes_heat = plt.subplots(1, 2, figsize=(14, max(8, nrows * 0.4)))
            fig_heat.suptitle(f"{fam} T.O.A. Modes Heatmap | Z = {z_pos} mm", fontsize=20, fontweight='bold', y=0.98)
            
            # Using viridis colormap, treating NaNs as transparent/background
            cmap = plt.cm.viridis.copy()
            cmap.set_bad(color='white')

            im0 = axes_heat[0].imshow(heatmap_pre, cmap=cmap)
            axes_heat[0].set_title("Pre-Calib Mode [ns]", fontsize=16)
            fig_heat.colorbar(im0, ax=axes_heat[0], fraction=0.046, pad=0.04)
            
            im1 = axes_heat[1].imshow(heatmap_post, cmap=cmap)
            axes_heat[1].set_title("Post-Calib Mode [ns]", fontsize=16)
            fig_heat.colorbar(im1, ax=axes_heat[1], fraction=0.046, pad=0.04)

            # Overlay channel names and exact values
            txt_path_effect = [pe.withStroke(linewidth=1.5, foreground="black")]
            for r_idx in range(nrows):
                for c_idx in range(ncols):
                    code = grid[r_idx][c_idx] if c_idx < len(grid[r_idx]) else None
                    if code is not None:
                        val_pre = heatmap_pre[r_idx, c_idx]
                        val_post = heatmap_post[r_idx, c_idx]
                        
                        txt_pre = f"{code}\n{val_pre:.2f}" if not np.isnan(val_pre) else f"{code}\nN/A"
                        txt_post = f"{code}\n{val_post:.2f}" if not np.isnan(val_post) else f"{code}\nN/A"
                        
                        axes_heat[0].text(c_idx, r_idx, txt_pre, ha='center', va='center', fontsize=9, 
                                          color='white', path_effects=txt_path_effect)
                        axes_heat[1].text(c_idx, r_idx, txt_post, ha='center', va='center', fontsize=9, 
                                          color='white', path_effects=txt_path_effect)

            axes_heat[0].axis('off')
            axes_heat[1].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig_heat)
            plt.close(fig_heat)
            
    # Clean up ROOT files
    for tree in trees:
        tree.file.close()

    print(f"Saved mosaic PDF to: {pdf_path}")

# ================= MAIN EXECUTION =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", default="/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples", help="Directory with ROOT files")
    ap.add_argument("--json", default="/lustre/research/hep/akshriva/Dream-Timing/Calib_output/master_3mm_6mm_shifts.json", help="Path to JSON")
    ap.add_argument("--outdir", default="/lustre/research/hep/akshriva/Dream-Timing/Calib_output", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    with open(args.json, "r") as f:
        master_calib = json.load(f)
    shifts = master_calib["shifts_by_family"]

    # Read the directory, find ROOT files, and keep only the ones in RUN_MAP
    valid_files = []
    if os.path.isdir(args.test_dir):
        for fname in os.listdir(args.test_dir):
            if not fname.endswith(".root"): 
                continue
            for run_key in RUN_MAP.keys():
                if run_key in fname:
                    valid_files.append(os.path.join(args.test_dir, fname))
                    break
    else:
        print(f"[ERROR] {args.test_dir} is not a valid directory.")
        return

    if not valid_files:
        print(f"[ERROR] Found no .root files matching the runs in RUN_MAP inside {args.test_dir}")
        return

    print(f"Found {len(valid_files)} matching runs in {args.test_dir}.")

    # Group files by their Z position
    files_by_z = {}
    for fpath in valid_files:
        run_label = os.path.basename(fpath)
        z_pos = get_z_position(run_label)
        if z_pos not in files_by_z:
            files_by_z[z_pos] = []
        files_by_z[z_pos].append(fpath)

    # Process files grouped by Z position
    for z_pos, file_list in files_by_z.items():
        print(f"\nProcessing Z Position: {z_pos} mm (Contains {len(file_list)} runs)")
        try:
            plot_family_mosaic_by_z(file_list, shifts, z_pos, args.outdir)
        except Exception as e:
            print(f"[ERROR] Failed plotting for Z = {z_pos}: {e}")

if __name__ == "__main__":
    main()