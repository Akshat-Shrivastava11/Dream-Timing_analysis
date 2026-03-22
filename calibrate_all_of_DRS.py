#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import uproot
import awkward as ak
from scipy.optimize import curve_fit

# ================= CONFIGURATION =================
TREE_NAME = "EventTree"
AMP_THRESHOLD = 20.0  
MIN_ADC_CUT = -500.0
NBINS = 100

# ================= 3MM EXACT CHANNELS =================
THREEMM_QUARTZ = [
    "002", "006", "004", "206", "204", "016", "014", "216", "214", "026", "024", "226", "224", 
    "030", "034", "106", "104", "306", "304", "116", "114", "316", "314", "126", "124", "326", 
    "324", "134", "334"
]
THREEMM_PLASTIC = [
    "000", "202", "200", "012", "010", "212", "210", "022", "020", "222", "220", "032", "232", 
    "230", "102", "100", "302", "300", "112", "110", "312", "310", "122", "120", "322", "320", 
    "132", "130", "332", "330"
]
THREEMM_SCI = [
    "003", "001", "203", "201", "007", "005", "207", "205", "013", "011", "213", "211", "017", 
    "015", "217", "215", "023", "021", "223", "221", "027", "025", "227", "225", "033", "031", 
    "233", "231", "035", "235", "103", "101", "303", "301", "107", "105", "307", "305", "113", 
    "111", "313", "311", "117", "115", "317", "315", "123", "121", "323", "321", "127", "125", 
    "327", "325", "133", "131", "333", "331", "135", "335"
]
THREEMM_ALL = set(THREEMM_QUARTZ + THREEMM_PLASTIC + THREEMM_SCI)

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

# ================= CALIBRATION MAP =================
CALIB_GROUPS = {
    "y1000": {
        "ref_run": "run1502_250928113749",
        "families": {
            "CER-Quartz": {
                "anchor": "002",
                "targets": ["617", "616", "615", "614", "625", "624", "623", "622", "637", "631", "630", "627",
                             "626", "636", "515", "514", "635", "634", "633", "632", "501", "500", "002", "517", 
                             "516", "006", "004", "206", "204", "503", "502", "016", "014", "216", "214", "521",
                               "520", "026", "024", "226", "224", "505", "504"]
            },
            "CER-Plastic": {
                "anchor": "000",
                "targets": ["613", "612", "611", "610", "000", "202", "200", "012", "010", "212", "210"]
            },
            "SCI": {
                "anchor": "003",
                "targets": ["621", "620", "003", "001", "203", "201", "007", "005", "207", "205", "013",
                             "011", "213", "211", "017", "015", "217", "215"]
            }
        }
    },
    "y936": {
        "ref_run": "run1509_250928164817",
        "families": {
            "CER-Plastic": {
                "anchor": "607",
                "targets": ["603", "602", "601", "600", "607", "606"]
            },
            "SCI": {
                "anchor": "605",
                "targets": ["605", "604"]
            }
        }
    },
    "y1028": {
        "ref_run": "run1510_250928172949",
        "families": {
            "CER-Quartz": {
                "anchor": "413",
                "targets": ["437", "407", "406", "405", "404", "436", "413", "412", "411", "410", "417", "416", "415", "414"]
            },
            "SCI": {
                "anchor": "421",
                "targets": ["421", "420", "425", "434"]
            },
            "CER-Plastic": {
                "anchor": "425",
                "targets": ["425", "424", "423", "422", "427", "426", "433", "432", "431", "430"]
            }
        }
    }
}

# Cleanup Run Configuration
Y1065_RUN = "run1501_250928105227"
Y1065_ANCHORS = {
    "CER-Quartz": "030",
    "CER-Plastic": "010",
    "SCI": "107"
}

# FULL MASTER GRIDS (Used to find "the rest" of the channels for y1065)
FULL_QUARTZ = ["617", "616", "615", "614", "625", "624", "623", "622", "637", "631", "630", "627", "626", "636", "515", "514", "635", "634", "633", "632", "501", "500", "002", "517", "516", "006", "004", "206", "204", "503", "502", "016", "014", "216", "214", "521", "520", "026", "024", "226", "224", "505", "504", "030", "530", "034", "534", "234", "523", "522", "106", "104", "306", "304", "507", "506", "116", "114", "316", "314", "525", "524", "126", "124", "326", "324", "511", "510", "532", "134", "536", "334", "527", "526", "403", "402", "401", "400", "513", "512", "437", "407", "406", "405", "404", "436", "413", "412", "411", "410", "417", "416", "415", "414"]
FULL_SCI = ["605", "604", "621", "620", "003", "001", "203", "201", "007", "005", "207", "205", "013", "011", "213", "211", "017", "015", "217", "215", "023", "021", "223", "221", "027", "025", "227", "225", "033", "031", "233", "231", "531", "035", "535", "235", "103", "101", "303", "301", "107", "105", "307", "305", "113", "111", "313", "311", "117", "115", "317", "315", "123", "121", "323", "321", "127", "125", "327", "325", "133", "131", "333", "331", "533", "135", "537", "335", "421", "420", "425", "434"]
FULL_PLASTIC = ["603", "602", "601", "600", "607", "606", "613", "612", "611", "610", "000", "202", "200", "012", "010", "212", "210", "022", "020", "222", "220", "032", "232", "230", "102", "100", "302", "300", "112", "110", "312", "310", "122", "120", "322", "320", "132", "130", "332", "330", "425", "424", "423", "422", "427", "426", "433", "432", "431", "430"]

MASTER_CHANNELS = {
    "CER-Quartz": FULL_QUARTZ,
    "SCI": FULL_SCI,
    "CER-Plastic": FULL_PLASTIC
}

# ================= HELPER FUNCTIONS =================
def _parse_code(code_str):
    return int(code_str[0]), int(code_str[1]), int(code_str[2])

def compute_adc_mask(tree, b, g, c):
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

def compute_tfinal(tree, code_str, suffix="_LP2_50"):
    b, g, c = _parse_code(code_str)
    br_sig     = f"DRS_Board{b}_Group{g}_Channel{c}{suffix}"
    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8{suffix}"
    
    # Check 3mm vs 6mm definition
    trg_b = b if code_str in THREEMM_ALL else 0
    br_trg     = f"DRS_Board{trg_b}_Group3_Channel7{suffix}"
    br_trg_ref = f"DRS_Board{trg_b}_Group3_Channel8{suffix}"
    
    keys = tree.keys()
    if not all(br in keys for br in [br_sig, br_sig_ref, br_trg, br_trg_ref]):
        return None
            
    arr_sig     = tree[br_sig].array(library="np")
    arr_sig_ref = tree[br_sig_ref].array(library="np")
    arr_trg     = tree[br_trg].array(library="np")
    arr_trg_ref = tree[br_trg_ref].array(library="np")
        
    # Strictly Raw Values (No np.abs)
    t_final = (arr_sig - arr_sig_ref) - (arr_trg - arr_trg_ref)
    return t_final

def compute_tfinal(tree, code_str, suffix="_LP2_50"):
    b, g, c = _parse_code(code_str)
    br_sig     = f"DRS_Board{b}_Group{g}_Channel{c}{suffix}"
    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8{suffix}"
    
    # Check 3mm vs 6mm definition
    trg_b = b if code_str in THREEMM_ALL else 0
    br_trg     = f"DRS_Board{trg_b}_Group3_Channel7{suffix}"
    br_trg_ref = f"DRS_Board{trg_b}_Group3_Channel8{suffix}"
    
    keys = tree.keys()
    if not all(br in keys for br in [br_sig, br_sig_ref, br_trg, br_trg_ref]):
        return None
            
    arr_sig     = tree[br_sig].array(library="np")
    arr_sig_ref = tree[br_sig_ref].array(library="np")
    arr_trg     = tree[br_trg].array(library="np")
    arr_trg_ref = tree[br_trg_ref].array(library="np")
        
    t_final = (arr_sig - arr_sig_ref) - (arr_trg - arr_trg_ref)
    
    # ---------------------------------------------------------
    # FIX: Force everything into the negative domain.
    # This aligns the positive 1000/936/1028 raw runs with your 
    # strictly negative bound configurations.
    # ---------------------------------------------------------
    return -np.abs(t_final)

def get_clean_array(tree, code_str, tmin, tmax):
    """Fetches tfinal array, applies ADC/NAN bounds cuts dynamically"""
    b, g, c = _parse_code(code_str)
    
    t_arr = compute_tfinal(tree, code_str)
    if t_arr is None: return None
    
    adc_mask = compute_adc_mask(tree, b, g, c)
    t_arr = t_arr[adc_mask]
    t_arr = t_arr[~np.isnan(t_arr)]
    
    # Check if data survives basic cuts before strict timing cuts
    if len(t_arr) < 25: 
        return None
        
    t_arr_cut = t_arr[(t_arr >= tmin) & (t_arr <= tmax)]
    
    # If cutting by the bounds destroys the stats, return the uncut array
    # so the main loop can print a helpful diagnostic warning about where 
    # the median actually is vs where the bounds are.
    if len(t_arr_cut) < 25:
        return t_arr  
        
    return t_arr_cut
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

# def get_clean_array(tree, code_str, tmin, tmax):
#     """Fetches tfinal array, applies ADC/NAN bounds cuts dynamically"""
#     b, g, c = _parse_code(code_str)
#     t_arr = compute_tfinal(tree, code_str)
#     if t_arr is None: return None
    
#     adc_mask = compute_adc_mask(tree, b, g, c)
#     t_arr = t_arr[adc_mask]
#     t_arr = t_arr[~np.isnan(t_arr)]
#     t_arr = t_arr[(t_arr >= tmin) & (t_arr <= tmax)]
    
#     return t_arr if len(t_arr) >= 25 else None

def get_file_path(directory, run_label):
    for f in os.listdir(directory):
        if run_label in f and f.endswith(".root"):
            return os.path.join(directory, f)
    raise FileNotFoundError(f"Could not find a ROOT file matching {run_label} in {directory}")

# ================= MAIN EXECUTION =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples")
    ap.add_argument("--outdir", default="./Calib_output", help="Directory to save JSON")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    final_shifts = {"CER-Quartz": {}, "CER-Plastic": {}, "SCI": {}}

    # ================= PHASE 1: Targeted Y-Scans =================
    for y_group, calib_info in CALIB_GROUPS.items():
        print(f"\n--- Processing Targeted Group: {y_group} ---")
        ref_path = get_file_path(args.dir, calib_info["ref_run"])
        
        with uproot.open(ref_path) as uf:
            tree = uf[TREE_NAME]
            
            for fam, fam_data in calib_info["families"].items():
                anchor_code = fam_data["anchor"]
                targets = fam_data["targets"]
                
                tmin = Y_CONFIGS[y_group][fam]["tmin"]
                tmax = Y_CONFIGS[y_group][fam]["tmax"]
                bins = np.linspace(tmin, tmax, NBINS + 1)
                
                anchor_arr = get_clean_array(tree, anchor_code, tmin, tmax)
                if anchor_arr is None:
                    print(f"[WARN] Failed to get stats for {fam} anchor {anchor_code} in {y_group}. Skipping family.")
                    continue
                
                anchor_mode = fit_mode(anchor_arr, bins)
                print(f"[{fam}] Anchor {anchor_code} Mode: {anchor_mode:.3f} ns (Bounds: {tmin} to {tmax})")
                
                for target_code in targets:
                    target_arr = get_clean_array(tree, target_code, tmin, tmax)
                    if target_arr is None: continue
                    
                    target_mode = fit_mode(target_arr, bins)
                    shift = float(anchor_mode - target_mode)
                    final_shifts[fam][target_code] = shift

    # ================= PHASE 2: Y1065 Cleanup =================
    print("\n--- Processing Remaining Channels with Y1065 ---")
    y1065_path = get_file_path(args.dir, Y1065_RUN)
    
    with uproot.open(y1065_path) as uf:
        tree = uf[TREE_NAME]
        
        y1065_modes = {}
        for fam, anchor_code in Y1065_ANCHORS.items():
            tmin = Y_CONFIGS["y1065"][fam]["tmin"]
            tmax = Y_CONFIGS["y1065"][fam]["tmax"]
            bins = np.linspace(tmin, tmax, NBINS + 1)
            
            anchor_arr = get_clean_array(tree, anchor_code, tmin, tmax)
            if anchor_arr is not None:
                y1065_modes[fam] = fit_mode(anchor_arr, bins)
                print(f"[{fam}] Cleanup Anchor {anchor_code} Mode: {y1065_modes[fam]:.3f} ns (Bounds: {tmin} to {tmax})")
            else:
                print(f"[ERROR] Cleanup anchor {anchor_code} failed for {fam}. Remaining channels will be skipped.")

        for fam, all_channels in MASTER_CHANNELS.items():
            if fam not in y1065_modes: continue
            
            tmin = Y_CONFIGS["y1065"][fam]["tmin"]
            tmax = Y_CONFIGS["y1065"][fam]["tmax"]
            bins = np.linspace(tmin, tmax, NBINS + 1)
            
            anchor_mode = y1065_modes[fam]
            remaining = [ch for ch in all_channels if ch not in final_shifts[fam]]
            
            for ch in remaining:
                target_arr = get_clean_array(tree, ch, tmin, tmax)
                if target_arr is None: continue
                
                target_mode = fit_mode(target_arr, bins)
                shift = float(anchor_mode - target_mode)
                final_shifts[fam][ch] = shift
                print(f"  -> Cleaned up {ch} with shift {shift:+.2f} ns")

    # ================= EXPORT =================
    out_json = os.path.join(args.outdir, "master_3mm_6mm_shifts.json")
    payload = {
        "meta": {
            "description": "Targeted mapping with negative bounds and y1065 remainder cleanup",
            "y1000_ref": CALIB_GROUPS["y1000"]["ref_run"],
            "y936_ref": CALIB_GROUPS["y936"]["ref_run"],
            "y1028_ref": CALIB_GROUPS["y1028"]["ref_run"],
            "y1065_cleanup_ref": Y1065_RUN
        },
        "shifts_by_family": final_shifts
    }
    
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        
    print(f"\nAll operations complete. Final shifts exported to: {out_json}")

if __name__ == "__main__":
    main()