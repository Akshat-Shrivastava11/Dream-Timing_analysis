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
NBINS = 250 # Finer binning for tighter fits

# ================= MANUAL OVERRIDES =================
# Forcing rogue channels to match the average shifts of their physical neighbors
MANUAL_OVERRIDES = {
    "CER-Quartz": {
        "002": +0.71,  "204": +1.56,  "026": +1.42,
        "216": -1.06,  "226": -1.66,
        "502": -3.25,  "503": -3.25,  "504": -3.25,  "505": -3.25,
        "306": -2.62,  "316": +0.78,
        "532": -0.61,  "536": -2.80,  "334": -2.98,
        "402": -3.00,
        "522": +0.10,  "523": +0.10,  "506": +0.10,  "507": +0.10,
        "524": +0.10,  "525": +0.10,  "510": +0.10,  "511": +0.10,
        "520": -0.40,  "521": -0.40,
        "016": -0.45,  "206": -0.40,
        "526": -1.30,  "416": -1.44
    },
    "CER-Plastic": {
        "012": -0.10,  "010": -0.10,  "212": -0.10,  "210": -0.10,
        "202": -2.44,  "132": -1.41,
        "000": -1.75,  "302": -0.15,  "312": -0.25,  "424": -1.85,
        "603": -1.55,  "602": -1.60
    },
    "SCI": {
        "621": -3.09,  "620": -4.07,
        "221": +1.80,
        "531": -3.73,  "535": -3.73,  "533": -3.90,
        "135": -2.66,  "537": -3.78,  "335": +1.96,
        "425": -4.34,  "434": -4.34,
        "133": -0.10,  "333": -0.15,  "315": -0.10
    }
}

# ================= 3MM EXACT CHANNELS =================
THREEMM_QUARTZ = ["002", "006", "004", "206", "204", "016", "014", "216", "214", "026", "024", "226", "224", "030", "034", "106", "104", "306", "304", "116", "114", "316", "314", "126", "124", "326", "324", "134", "334"]
THREEMM_PLASTIC = ["000", "202", "200", "012", "010", "212", "210", "022", "020", "222", "220", "032", "232", "230", "102", "100", "302", "300", "112", "110", "312", "310", "122", "120", "322", "320", "132", "130", "332", "330"]
THREEMM_SCI = ["003", "001", "203", "201", "007", "005", "207", "205", "013", "011", "213", "211", "017", "015", "217", "215", "023", "021", "223", "221", "027", "025", "227", "225", "033", "031", "233", "231", "035", "235", "103", "101", "303", "301", "107", "105", "307", "305", "113", "111", "313", "311", "117", "115", "317", "315", "123", "121", "323", "321", "127", "125", "327", "325", "133", "131", "333", "331", "135", "335"]
THREEMM_ALL = set(THREEMM_QUARTZ + THREEMM_PLASTIC + THREEMM_SCI)

# ================= RUN MAPPING & BOUNDS =================
# ALL REFERENCES FORCED TO Z = -168.0 mm
Y_CONFIGS = {
    "y1065": {"ref": "run1501_250928105227", "SCI": [-13.5, -7.5], "CER-Plastic": [-14.5, -11.5], "CER-Quartz": [-15.0, -11.5]},
    "y1000": {"ref": "run1502_250928113749", "SCI": [-11.0, -7.5], "CER-Plastic": [-12.5, -10.5], "CER-Quartz": [-13.5, -10.0]},
    "y936":  {"ref": "run1504_250928133854", "SCI": [-11.0, -7.5], "CER-Plastic": [-12.5, -11.0], "CER-Quartz": [-12.6, -11.0]},
    "y1028": {"ref": "run1506_250928143030", "SCI": [-10.5, -7.0], "CER-Plastic": [-12.5, -10.5], "CER-Quartz": [-12.5, -11.0]}
}

# MASTER ANCHORS (y1065 is the Global Reference)
MASTER_Y = "y1065"
ANCHORS = { "CER-Quartz": "030", "CER-Plastic": "010", "SCI": "107" }

# CHANNEL TARGETS PER GROUP
TARGETS = {
    "y1000": {
        "CER-Quartz": ["617", "616", "615", "614", "625", "624", "623", "622", "637", "631", "630", "627", "626", "636", "515", "514", "635", "634", "633", "632", "501", "500", "002", "517", "516", "006", "004", "206", "204", "503", "502", "016", "014", "216", "214", "521", "520", "026", "024", "226", "224", "505", "504"],
        "CER-Plastic": ["613", "612", "611", "610", "000", "202", "200", "012", "010", "212", "210"],
        "SCI": ["621", "620", "003", "001", "203", "201", "007", "005", "207", "205", "013", "011", "213", "211", "017", "015", "217", "215"]
    },
    "y936": {
        "CER-Plastic": ["603", "602", "601", "600", "607", "606"],
        "SCI": ["605", "604"]
    },
    "y1028": {
        "CER-Quartz": ["437", "407", "406", "405", "404", "436", "413", "412", "411", "410", "417", "416", "415", "414"],
        "SCI": ["421", "420", "425", "434"],
        "CER-Plastic": ["425", "424", "423", "422", "427", "426", "433", "432", "431", "430"]
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
    [None,  None,  "603", "602", "601", "600", None,  None],
    [None,  None,  None,  "607", "606", None,  None,  None],
    [None,  None,  "613", "612", "611", "610", None,  None],
    [None,  None,  None,  None,  None,  None,  None,  None],
    [None,  None,  None,  None,  None,  None,  None,  None],
    [None,  None,  None,  "000", "202", "200", None,  None,],
    [None,  None,  "012", "010", "212", "210", None,  None,],
    [None,  None,  "022", "020", "222", "220", None,  None,],
    [None,  None, "032", None,  "232", "230" ,None,  None,],
    [None,  None, "102", "100", "302", "300", None,  None,],
    [None,  None, "112", "110", "312", "310", None,  None,],
    [None,  None, "122", "120", "322", "320", None,  None,],
    [None,  None, "132", "130", "332", "330", None,  None,],
    [None,  None,  None,  None,  None,  None,  None,  None],
    [None,  None,  None,  None,  None,  None,  None,  None],
    [None,  None,  "425", "424", "423", "422", None,  None],
    [None,  None,  None,  "427", "426", None,  None,  None],
    [None,  None,  "433", "432", "431", "430", None,  None],
]


FAMILIES = {"CER-Quartz": QUARTZ_GRID, "SCI": SCI_GRID, "CER-Plastic": PLASTIC_GRID}

# MASTER LISTS FOR Y1065 REMAINDER
MASTER_CHANNELS = {
    "CER-Quartz": [c for row in QUARTZ_GRID for c in row if c is not None],
    "SCI": [c for row in SCI_GRID for c in row if c is not None],
    "CER-Plastic": [c for row in PLASTIC_GRID for c in row if c is not None]
}

# ================= HELPER FUNCTIONS =================
def _parse_code(code_str): return int(code_str[0]), int(code_str[1]), int(code_str[2])

def compute_adc_mask(tree, b, g, c):
    drs_br = f"DRS_Board{b}_Group{g}_Channel{c}"
    if drs_br not in tree: return np.ones(tree.num_entries, dtype=bool)
    waves = tree[drs_br].array(library="ak")
    baseline = ak.mean(waves[:, :30], axis=1)
    waves_blsub = waves - baseline
    mask = (ak.max(waves_blsub, axis=1) >= AMP_THRESHOLD) & (ak.min(waves_blsub, axis=1) >= MIN_ADC_CUT)
    return ak.to_numpy(mask)

def compute_tfinal(tree, code_str):
    b, g, c = _parse_code(code_str)
    trg_b = b if code_str in THREEMM_ALL else 0
    names = [f"DRS_Board{b}_Group{g}_Channel{c}_LP2_50", f"DRS_Board{b}_Group{g}_Channel8_LP2_50",
             f"DRS_Board{trg_b}_Group3_Channel7_LP2_50", f"DRS_Board{trg_b}_Group3_Channel8_LP2_50"]
    if not all(n in tree.keys() for n in names): return None
    arrs = [tree[n].array(library="np") for n in names]
    return -np.abs((arrs[0] - arrs[1]) - (arrs[2] - arrs[3]))

def fit_mode(arr, bins, window=0.3):
    h, edges = np.histogram(arr, bins=bins)
    if h.sum() < 25: return np.nan
    centers = 0.5 * (edges[1:] + edges[:-1])
    x0 = float(centers[np.argmax(h)])
    try:
        m = (centers >= x0 - window) & (centers <= x0 + window)
        popt, _ = curve_fit(lambda x, A, mu, sig: A * np.exp(-0.5 * ((x - mu) / sig) ** 2), 
                            centers[m], h[m], p0=[h.max(), x0, 0.10], bounds=([0, x0-window, 0.01], [np.inf, x0+window, 1.0]))
        return popt[1]
    except: return x0

def get_file_path(directory, run_label):
    for f in os.listdir(directory):
        if run_label in f and f.endswith(".root"): return os.path.join(directory, f)
    raise FileNotFoundError(f"Missing {run_label}")

def find_neighbor_shift(fam, target_ch, final_shifts_dict):
    """Searches outwards radially to find the closest valid calibrated channel shift"""
    grid = FAMILIES.get(fam, [])
    r_t, c_t = -1, -1
    
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val == target_ch:
                r_t, c_t = r, c
                break
        if r_t != -1: break
        
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


def format_grid_txt(grid, shift_map, family_name, cell_width=14):
    """
    Build a text block that preserves the detector grid layout.
    Each occupied cell is shown as:
        107
        -1.23 ns
    Blank cells remain empty.
    """
    lines = []
    lines.append("=" * (cell_width * len(grid[0])))
    lines.append(f"{family_name} SHIFT GRID")
    lines.append("=" * (cell_width * len(grid[0])))

    for row in grid:
        # first line: channel IDs
        ch_line = []
        # second line: shifts
        sh_line = []

        for ch in row:
            if ch is None:
                ch_line.append("".center(cell_width))
                sh_line.append("".center(cell_width))
            else:
                shift = shift_map.get(ch, None)
                ch_line.append(f"{ch}".center(cell_width))
                if shift is None:
                    sh_line.append("N/A".center(cell_width))
                else:
                    sh_line.append(f"{shift:+.2f} ns".center(cell_width))

        lines.append("".join(ch_line).rstrip())
        lines.append("".join(sh_line).rstrip())
        lines.append("")  # blank spacer between rows

    lines.append("")
    return "\n".join(lines)


def export_shift_grids_txt(outpath, final_shifts):
    """
    Write all families into one text file using the same grid geometry.
    """
    blocks = []
    for fam, grid in FAMILIES.items():
        blocks.append(format_grid_txt(grid, final_shifts.get(fam, {}), fam))
    with open(outpath, "w") as f:
        f.write("\n\n".join(blocks))
# ================= MAIN =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples")
    ap.add_argument("--outdir", default="./Calib_output")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    final_shifts = {"CER-Quartz": {}, "CER-Plastic": {}, "SCI": {}}
    failed_channels = [] 
    master_modes = {} 

    # ---------------------------------------------------------
    # PHASE 1: Establish y1065 Master Anchor Modes at Z=-168
    # ---------------------------------------------------------
    print(f"--- Establishing Master Global Reference (from Z = -168.0 mm) ---")
    y1065_path = get_file_path(args.dir, Y_CONFIGS["y1065"]["ref"])
    with uproot.open(y1065_path) as uf:
        tree = uf[TREE_NAME]
        for fam, anchor in ANCHORS.items():
            tmin, tmax = Y_CONFIGS["y1065"][fam]
            t_arr = compute_tfinal(tree, anchor)
            if t_arr is None: continue
            mask = compute_adc_mask(tree, *_parse_code(anchor))
            t_arr = t_arr[mask & (t_arr >= tmin) & (t_arr <= tmax)]
            master_modes[fam] = fit_mode(t_arr, np.linspace(tmin, tmax, NBINS+1))
            print(f"  [{fam}] Master Mode ({anchor}): {master_modes[fam]:.3f} ns")

    # ---------------------------------------------------------
    # PHASE 2: Direct Flush to Master Anchors (with Overrides)
    # ---------------------------------------------------------
    for yg in ["y1000", "y1065", "y936", "y1028"]:
        print(f"\n--- Processing {yg} (Z = -168.0 mm) and Directly Flushing to y1065 ---")
        path = get_file_path(args.dir, Y_CONFIGS[yg]["ref"])
        with uproot.open(path) as uf:
            tree = uf[TREE_NAME]
            for fam in ["CER-Quartz", "CER-Plastic", "SCI"]:
                if fam not in Y_CONFIGS[yg]: continue
                tmin, tmax = Y_CONFIGS[yg][fam]
                bins = np.linspace(tmin, tmax, NBINS+1)

                targets = TARGETS.get(yg, {}).get(fam, [])
                if yg == MASTER_Y: 
                    targets = [ch for ch in MASTER_CHANNELS[fam] if ch not in final_shifts[fam]]

                for ch in targets:
                    # Attempt to calculate the mode normally for ALL channels first
                    t_ch = compute_tfinal(tree, ch)
                    mode_ch = np.nan
                    
                    if t_ch is not None:
                        m_ch = compute_adc_mask(tree, *_parse_code(ch))
                        t_ch_cut = t_ch[m_ch & (t_ch >= tmin) & (t_ch <= tmax)]
                        mode_ch = fit_mode(t_ch_cut, bins)

                    # 1. Check for manual override first
                    if ch in MANUAL_OVERRIDES.get(fam, {}):
                        final_shift = MANUAL_OVERRIDES[fam][ch]
                        final_shifts[fam][ch] = final_shift
                        mode_str = f"{mode_ch:.2f}" if not np.isnan(mode_ch) else "FAILED/NAN"
                        print(f"  [{fam}] Channel {ch}: MANUAL OVERRIDE Shift = {final_shift:+.2f} ns  (Raw Mode: {mode_str})")
                        continue

                    # 2. Otherwise calculate normally
                    if np.isnan(mode_ch): 
                        failed_channels.append((fam, ch))
                        continue
                    
                    final_shift = float(master_modes[fam] - mode_ch)
                    final_shifts[fam][ch] = final_shift
                    print(f"  [{fam}] Channel {ch}: Shift = {final_shift:+.2f} ns  (Mode: {mode_ch:.2f})")

    # ---------------------------------------------------------
    # PHASE 3: Fallback Imputation (Nearest Neighbor)
    # ---------------------------------------------------------
    if failed_channels:
        print(f"\n--- Processing Fallback (Nearest Neighbor) for {len(failed_channels)} missing channels ---")
        for fam, ch in failed_channels:
            if ch in final_shifts[fam]: continue 
            
            neighbor_ch, neighbor_shift = find_neighbor_shift(fam, ch, final_shifts)
            
            if neighbor_ch:
                final_shifts[fam][ch] = neighbor_shift
                print(f"  [{fam}] Channel {ch} [LOW STATS] -> Copied Shift {neighbor_shift:+.2f} ns from nearest neighbor '{neighbor_ch}'")
            else:
                final_shifts[fam][ch] = 0.0
                print(f"  [{fam}] Channel {ch} [LOW STATS] -> No valid neighbors found. Shift set to +0.00 ns")

    # ---------------------------------------------------------
    # EXPORT
    # ---------------------------------------------------------
    out_json = os.path.join(args.outdir, "master_3mm_6mm_shifts_global_best2.json")
    out_txt = os.path.join(args.outdir, "master_3mm_6mm_shifts_grid.txt")
    export_shift_grids_txt(out_txt, final_shifts)
    print(f"Grid text file saved to {out_txt}")
    with open(out_json, "w") as f:
        json.dump({"shifts_by_family": final_shifts}, f, indent=2, sort_keys=True)
    print(f"\nGlobal Flushing Complete. JSON saved to {out_json}")

if __name__ == "__main__": main()