#!/usr/bin/env python3
import os
import re
import argparse
import json
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit

# ================= CALIBRATION & PATHS =================
BASE_DIR   = "/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples"
PED_FILE   = "/lustre/research/hep/akshriva/CaloXDataAnalysis/data/fers/FERS_pedestals_run1425.json"
HG2LG_FILE = "/lustre/research/hep/akshriva/CaloXDataAnalysis/data/fers/FERS_HG2LG_Sep.json"
RESP_FILE  = "/lustre/research/hep/akshriva/CaloXDataAnalysis/data/fers/FERS_response_Sep.json"

HG_SATURATION_THRESHOLD = 4000.0
TREE_NAME = "EventTree"

# ================= DYNAMIC RUN CONFIGURATIONS =================
Y_CONFIGS = {
    "y1000": {
        "SCI":         {"channels": ["620","621"], "tmin": 8.0,  "tmax": 11.0},
        "Plastic-CER": {"channels": ["612","611","610","613"], "tmin": 10.5, "tmax": 12.5},
        "Quartz-CER":  {"channels": ["631","630","627","637"], "tmin": 10.0, "tmax": 13.5}
    },
    "y1065": {
        "Quartz-CER":  {"channels": ["523","522","521","520"], "tmin": 10.0, "tmax": 13.5}
    },
    "y936": {
        "SCI":         {"channels": ["604","605"], "tmin": 8.0,  "tmax": 11.0},
        "Plastic-CER": {"channels": ["607","606"], "tmin": 11.0, "tmax": 12.5},
        "Quartz-CER":  {"channels": ["617","616","615","614"], "tmin": 11.0, "tmax": 12.6}
    },
    "y1028": {
        "SCI":         {"channels": ["421","420"], "tmin": 7.0, "tmax": 10.5},
        "Plastic-CER": {"channels": ["425","423","422","424"], "tmin": 10.5, "tmax": 12.5},
        "Quartz-CER":  {"channels": ["413","412","411","410"], "tmin": 11.0, "tmax": 12.5}
    }
}


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

def get_z_position(run_label):
    if "run1513" in run_label:
        if "192918" in run_label: return -54.5
        if "194230" in run_label: return -400.3
    match = re.search(r"run(\d+)", run_label)
    run_num = int(match.group(1)) if match else None
    
    z_map = {
        1501: -168.0, 1507: -218.0, 1511: -268.0,
        1504: -168.0, 1509: -218.0, 1512: -268.0,
        1502: -168.0, 1508: -218.0,
        1506: -168.0, 1510: -218.0
    }
    return z_map.get(run_num, -999.0)

def _resolve_files():
    """Automatically finds the input files based on RUN_MAP."""
    files = []
    for run_label in RUN_MAP.keys():
        fpath = os.path.join(BASE_DIR, f"{run_label}_converted_timingskim.root")
        if os.path.exists(fpath):
            files.append(fpath)
        else:
            print(f"[WARN] File missing on disk: {fpath}")
    return files

# ================= PAIRED GRIDS FOR LOOKUP (CORES IGNORED) =================
DRS_QUARTZ_GRID = [
    [None, None, None, "617", "616", "615", "614", None, None, None],
    [None, None, None, "625", "624", "623", "622", None, None, None],
    [None, None, "637", "631", "630", "627", "626", "636", None, None],
    ["515", "514", None, None, None, None, None, None, "501", "500"],
    ["517", "516", None, None, None, None, None, None, "503", "502"],
    ["521", "520", None, None, None, None, None, None, "505", "504"],
    ["523", "522", None, None, None, None, None, None, "507", "506"],
    ["525", "524", None, None, None, None, None, None, "511", "510"],
    ["527", "526", None, None, None, None, None, None, "513", "512"],
    [None]*10,
    [None, None, "437", "407", "406", "405", "404", "436", None, None],
    [None, None, None, "413", "412", "411", "410", None, None, None],
    [None, None, None, "417", "416", "415", "414", None, None, None],
]

FERS_QUARTZ_GRID = [
    [None, None, None, "419", "532", "534", "533", None, None, None],
    [None, None, None, "425", "542", "540", "543", None, None, None],
    [None, None, "432", "434", "548", "550", "549", "551", None, None],
    ["343", "341", None, None, None, None, None, None, "642", "640"],
    ["349", "351", None, None, None, None, None, None, "648", "650"],
    ["359", "357", None, None, None, None, None, None, "658", "656"],
    ["1101", "1103", None, None, None, None, None, None, "1400", "1402"],
    ["1111", "1109", None, None, None, None, None, None, "1410", "1408"],
    ["1117", "1119", None, None, None, None, None, None, "1416", "1418"],
    [None]*10,
    [None, None, "1226", "1224", "1310", "1308", "1311", "1309", None, None],
    [None, None, None, "1234", "1233", "1316", "1318", None, None, None],
    [None, None, None, "1241", "1326", "1324", "1327", None, None, None],
]

DRS_PLASTIC_GRID = [
    [None,  None,  "603", "602", "601", "600", None,  None],
    [None,  None,  None,  "607", "606", None,  None,  None],
    [None,  None,  "613", "612", "611", "610", None,  None],
    [None]*8, [None]*8, [None]*8, [None]*8, [None]*8, 
    [None]*8, [None]*8, [None]*8, [None]*8, [None]*8, [None]*8, [None]*8,
    [None,  None,  "425", "424", "423", "422", None,  None],
    [None,  None,  None,  "427", "426", None,  None,  None],
    [None,  None,  "433", "432", "431", "430", None,  None],
]

FERS_PLASTIC_GRID = [
    [None,  None,  "510", "508", "511", "509", None,  None], 
    [None,  None,  None,  "518", "517", None,  None,  None], 
    [None,  None,  "526", "524", "527", "525", None,  None], 
    [None]*8, [None]*8, [None]*8, [None]*8, [None]*8, 
    [None]*8, [None]*8, [None]*8, [None]*8, [None]*8, [None]*8, [None]*8,
    [None,  None,  "1332", "1334", "1333", "1335", None,  None], 
    [None,  None,  None,   "1340", "1343", None,   None,  None], 
    [None,  None,  "1348", "1350", "1349", "1351", None,  None], 
]

DRS_SCI_GRID = [
    [None, None, "605", "604", None, None], 
    [None]*6,
    [None, None, "621", "620", None, None], 
    [None]*6, [None]*6,
    [None]*6, [None]*6, [None]*6, [None]*6, [None]*6, [None]*6, [None]*6, [None]*6,
    [None]*6, [None]*6, [None]*6, [None]*6, [None]*6, [None]*6, [None]*6, [None]*6,
    [None]*6, [None]*6,
    [None, None, "421", "420", None, None], 
    [None]*6,
    [None, None, "435", "434", None, None], 
]

FERS_SCI_GRID = [
    [None, None, "512", "515", None, None], 
    [None]*6,
    [None, None, "538", "537", None, None], 
    [None]*6, [None]*6,
    [None]*6, [None]*6, [None]*6, [None]*6, [None]*6, [None]*6, [None]*6, [None]*6,
    [None]*6, [None]*6, [None]*6, [None]*6, [None]*6, [None]*6, [None]*6, [None]*6,
    [None]*6, [None]*6,
    [None, None, "1328", "1331", None, None], 
    [None]*6,
    [None, None, "1354", "1353", None, None], 
]

def exp_func(t, A, lam):
    return A * np.exp(t / lam)
    #return A * np.exp(-x / lam)

def build_channel_map():
    drs_to_fers = {}
    drs_grids = [DRS_QUARTZ_GRID, DRS_PLASTIC_GRID, DRS_SCI_GRID]
    fers_grids = [FERS_QUARTZ_GRID, FERS_PLASTIC_GRID, FERS_SCI_GRID]
    
    for d_grid, f_grid in zip(drs_grids, fers_grids):
        for r in range(len(d_grid)):
            for c in range(len(d_grid[r])):
                d_code, f_code = d_grid[r][c], f_grid[r][c]
                if d_code is not None and f_code is not None:
                    drs_to_fers[d_code] = f_code
    return drs_to_fers

# ================= PID LOGIC =================
PID_BRANCH_MAP = {
    "PSD": "DRS_Board7_Group1_Channel1", "TTUMuonVeto": "DRS_Board7_Group2_Channel4",
    "Cer474": "DRS_Board7_Group2_Channel5", "Cer519": "DRS_Board7_Group2_Channel6", "Cer537": "DRS_Board7_Group2_Channel7",
}

def get_service_drs_cut(service_drs):
    cuts = {"PSD": (100, 400, -3500.0), "TTUMuonVeto": (200, 400, -2e3),
            "Cer474": (800, 900, -2000.0), "Cer519": (450, 550, -1000.0), "Cer537": (400, 500, -500.0)}
    return cuts.get(service_drs, (0, 1000, -5e4))

def compute_pid_mask(tree, particle_type):
    n_entries = tree.num_entries
    selections = {
        "muon": {"TTUMuonVeto": True, "PSD": False}, 
        "pion": {"TTUMuonVeto": False, "PSD": False, "Cer474": True, "Cer519": True, "Cer537": True},
        "electron": {"TTUMuonVeto": False, "PSD": True, "Cer474": True, "Cer519": True, "Cer537": True}, 
        "proton": {"TTUMuonVeto": False, "PSD": False, "Cer474": False, "Cer519": False, "Cer537": False},
        "all": {}
    }
    reqs = selections.get(particle_type.lower(), {})
    mask = np.ones(n_entries, dtype=bool)
    
    if not reqs: 
        return mask
        
    for det, must_fire in reqs.items():
        if PID_BRANCH_MAP.get(det) not in tree: 
            continue
            
        ts_min, ts_max, val_cut = get_service_drs_cut(det)
        waves = tree[PID_BRANCH_MAP[det]].array(library="ak")
        baseline = ak.mean(waves[:, :30], axis=1)
        
        window_sum = ak.to_numpy(ak.sum((waves - baseline)[:, int(ts_min):int(ts_max)], axis=1))
        is_fired = window_sum < val_cut
        mask = mask & (is_fired if must_fire else ~is_fired)
        
    return mask

# ================= CALIBRATION LOAD =================
def load_calibrations():
    def load_json(path):
        if not os.path.exists(path): return {}
        with open(path, 'r') as f: return json.load(f)
    return load_json(PED_FILE), load_json(HG2LG_FILE), load_json(RESP_FILE)

PEDS, HG2LG, RESP = load_calibrations()

def reconstruct_energy_1d(b_str, c_str, hg_array, lg_array):
    ped_hg = PEDS.get(b_str, {}).get(c_str, {}).get("HG", 0.0)
    ped_lg = PEDS.get(b_str, {}).get(c_str, {}).get("LG", 0.0)
    slope = HG2LG.get(b_str, {}).get(c_str, {}).get("slope", 1.0)
    intercept = HG2LG.get(b_str, {}).get(c_str, {}).get("intercept", 0.0)
    if slope == 0: slope = 1.0 
    
    hg_sub = hg_array - ped_hg
    lg_sub = lg_array - ped_lg
    is_sat = hg_sub > HG_SATURATION_THRESHOLD
    mix_energy = np.where(is_sat, (lg_sub - intercept) / slope, hg_sub)
    return np.maximum(0, mix_energy)

# ================= TIMING LOGIC =================
def compute_tfinal_6mm(tree, b, g, c, suffix="_LP2_50"):
    br_sig     = f"DRS_Board{b}_Group{g}_Channel{c}{suffix}"
    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8{suffix}"
    br_trg     = f"DRS_Board0_Group3_Channel7{suffix}"
    br_trg_ref = f"DRS_Board0_Group3_Channel8{suffix}"
    
    keys = tree.keys()
    for br in [br_sig, br_sig_ref, br_trg, br_trg_ref]:
        if br not in keys: return None
            
    arr_sig     = tree[br_sig].array(library="np")
    arr_sig_ref = tree[br_sig_ref].array(library="np")
    arr_trg     = tree[br_trg].array(library="np")
    arr_trg_ref = tree[br_trg_ref].array(library="np")
    
    if not (arr_sig.shape == arr_sig_ref.shape == arr_trg.shape == arr_trg_ref.shape):
        return None
        
    t_final = (arr_sig - arr_sig_ref) - (arr_trg - arr_trg_ref)
    
    # Cast safely to float to prevent np.isnan() object errors
    return np.asarray(np.abs(t_final), dtype=float)

# ================= HELPERS =================
def _parse_code(code_str): return int(code_str[0]), int(code_str[1]), int(code_str[2])
def _run_label(path): 
    m = re.search(r"(run\d+_\d{11,12})", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)

# ================= PLOTTING HELPER =================
def create_joint_plot(t_arr, e_arr, tmin, tmax, emin, emax, nbins, title, pdf):
    """Creates a 2D histogram with 1D marginals on the top and right."""
    fig, ax_main = plt.subplots(figsize=(10, 10))
    
    h2 = ax_main.hist2d(t_arr, e_arr, bins=[nbins, 100], range=[[tmin, tmax], [emin, emax]], cmap='turbo', cmin=1)
    
    divider = make_axes_locatable(ax_main)
    ax_top = divider.append_axes("top", size="20%", pad=0.1, sharex=ax_main)
    ax_right = divider.append_axes("right", size="20%", pad=0.1, sharey=ax_main)
    ax_cbar = divider.append_axes("right", size="5%", pad=0.3)

    ax_top.hist(t_arr, bins=nbins, range=[tmin, tmax], color='#2b5b84', histtype='stepfilled')
    ax_top.xaxis.set_tick_params(labelbottom=False)
    ax_top.set_ylabel("Events")

    ax_right.hist(e_arr, bins=100, range=[emin, emax], orientation='horizontal', color='#2b5b84', histtype='stepfilled')
    ax_right.yaxis.set_tick_params(labelleft=False)
    ax_right.set_xlabel("Events")

    fig.colorbar(h2[3], cax=ax_cbar, label='Total Events')
    
    corr_val = np.corrcoef(t_arr, e_arr)[0, 1] if len(t_arr) > 1 else 0.0
    ax_main.legend(handles=[mpatches.Patch(color='none', label=f"Correlation (r): {corr_val:.3f}")], loc='upper right', fontsize=12)
    
    fig.suptitle(title, fontsize=14, y=0.95)
    ax_main.set_xlabel(r"$|t_{\mathrm{final}}|$ [ns]", fontsize=12)
    ax_main.set_ylabel("Energy [A.U.]", fontsize=12)
    ax_main.grid(True, alpha=0.3)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

# ================= AUTOMATED ENGINE (COMBINED CHANNELS / NO ADC CUTS) =================
def run_y_config_correlations(files, outdir, emin, emax, nbins, particle_type):
    channel_map = build_channel_map()

    # 1. Group files by y_group matching their run label
    grouped_files = {grp: [] for grp in Y_CONFIGS.keys()}
    for f in files:
        rl = _run_label(f)
        for r_key, r_grp in RUN_MAP.items():
            if r_key in rl:
                grouped_files[r_grp].append(f)
                break

    # 2. Iterate through configs, setup directories, and process COMBINED channels
    for grp, grp_files in grouped_files.items():
        if not grp_files: continue
        
        print(f"\n" + "="*60)
        print(f" PROCESSING GROUP: {grp} ({len(grp_files)} files mapped)")
        print("="*60)

        cfg = Y_CONFIGS[grp]
        for family, fam_cfg in cfg.items():
            channels = fam_cfg.get("channels", [])
            if not channels: continue
            
            # Setup clean nested output directory: Outdir / y_{y} / {family}
            fam_dir = os.path.join(outdir, grp, family)
            os.makedirs(fam_dir, exist_ok=True)
            
            tmin = fam_cfg.get("tmin", 4.0)
            tmax = fam_cfg.get("tmax", 25.0)
            pid_tag = particle_type if particle_type else "NoPID"
            
            pdf_path = os.path.join(fam_dir, f"{family}_CombinedChannels_PID_zoomedin{pid_tag}.pdf")
            print(f"  -> Extracting & Combining {len(channels)} Channels for {family}...")
            print(f"  -> Target PDF: {pdf_path}")

            all_t_family = []
            all_e_family = []

            with PdfPages(pdf_path) as pdf:
                for fpath in grp_files:
                    rl = _run_label(fpath)
                    run_t = []
                    run_e = []
                    
                    try:
                        with uproot.open(fpath) as uf:
                            tree = uf[TREE_NAME]
                            keys = set(tree.keys())
                            total_events = tree.num_entries
                            
                            pid_mask = compute_pid_mask(tree, particle_type) if particle_type else np.ones(total_events, dtype=bool)

                            # Extract data for EVERY channel in this family
                            for drs_code in channels:
                                fers_code = channel_map.get(drs_code)
                                if not fers_code: continue

                                db, dg, dc = _parse_code(drs_code)
                                fb, fc = int(fers_code[:-2]), int(fers_code[-2:])

                                f_hg_branch = f"FERS_Board{fb}_energyHG"
                                f_lg_branch = f"FERS_Board{fb}_energyLG"

                                if not all(k in keys for k in (f_hg_branch, f_lg_branch)):
                                    continue

                                t_abs = compute_tfinal_6mm(tree, db, dg, dc)
                                if t_abs is None: continue

                                # ONLY apply PID and Timing cuts (No ADC Waveform cuts applied)
                                time_mask = (~np.isnan(t_abs)) & (t_abs >= tmin) & (t_abs <= tmax)
                                final_mask = pid_mask & time_mask
                                
                                if np.sum(final_mask) == 0: continue

                                t_valid = t_abs[final_mask]
                                hg_ak = tree[f_hg_branch].array(library="ak")[final_mask]
                                lg_ak = tree[f_lg_branch].array(library="ak")[final_mask]

                                hg_valid = ak.to_numpy(hg_ak[:, fc])
                                lg_valid = ak.to_numpy(lg_ak[:, fc])

                                e_valid = reconstruct_energy_1d(str(fb), str(fc), hg_valid, lg_valid)

                                run_t.extend(t_valid)
                                run_e.extend(e_valid)

                    except Exception as e:
                        print(f"     [ERROR] Failed processing {rl}: {e}")

                    # 1. Joint plot for this single Z-position run (All Channels Merged)
                    # 1. Joint plot for this single Z-position run (All Channels Merged)
                    if len(run_t) > 10:
                        z_pos = get_z_position(rl)
                        title = f"{family} | Run: {rl} (Z = {z_pos} mm)\nAll Channels Combined (PID: {pid_tag})"
                        create_joint_plot(np.array(run_t), np.array(run_e), tmin, tmax, emin, emax, nbins, title, pdf)

                        # =================================================================
                        # ---> PASTE THIS NEW BLOCK: Individual Run Energy Profile Plot
                        # =================================================================
                        fig_run_prof, ax_run_prof = plt.subplots(figsize=(10, 8))
                        n_run_profile_bins = 20
                        run_energy_bins = np.linspace(emin, emax, n_run_profile_bins + 1)
                        run_e_centers, run_t_medians, run_t_errs = [], [], []
                        
                        run_e_arr = np.array(run_e)
                        run_t_arr = np.array(run_t)
                        
                        for i in range(len(run_energy_bins)-1):
                            mask = (run_e_arr >= run_energy_bins[i]) & (run_e_arr < run_energy_bins[i+1])
                            if np.sum(mask) >= 5: 
                                t_subset = run_t_arr[mask]
                                run_e_centers.append(0.5 * (run_energy_bins[i] + run_energy_bins[i+1]))
                                run_t_medians.append(np.median(t_subset))
                                run_t_errs.append(1.253 * (np.std(t_subset) / np.sqrt(len(t_subset))))
                                
                        run_e_centers = np.array(run_e_centers)
                        run_t_medians = np.array(run_t_medians)
                        run_t_errs = np.array(run_t_errs)
                        
                        if len(run_e_centers) > 0:
                            ax_run_prof.errorbar(run_t_medians, run_e_centers, xerr=run_t_errs, fmt='o', color='black', label="Median ToA per Energy Bin", markersize=8, capsize=4, zorder=5)
                            if len(run_e_centers) > 2:
                                try:
                                    # Robust initial guess using log transform
                                    valid = run_e_centers > 0
                                    p_log = np.polyfit(run_t_medians[valid], np.log(run_e_centers[valid]), 1)
                                    lam_guess = 1.0 / p_log[0] if p_log[0] != 0 else 1.0
                                    A_guess = np.exp(p_log[1])

                                    # Fit E = A * exp(t/lam)
                                    popt_run, _ = curve_fit(exp_func, run_t_medians, run_e_centers, p0=[A_guess, lam_guess], maxfev=10000)
                                    A_opt_r, lam_opt_r = popt_run
                                    
                                    t_smooth_r = np.linspace(min(run_t_medians), max(run_t_medians), 500)
                                    e_smooth_r = exp_func(t_smooth_r, A_opt_r, lam_opt_r)
                                    ax_run_prof.plot(t_smooth_r, e_smooth_r, color='red', lw=2.5, label=f"Fit: $E = {A_opt_r:.2e} e^{{t / {lam_opt_r:.2f}}}$", zorder=4)
                                except Exception: pass
                            
                            ax_run_prof.set_title(f"Profile (Energy Binned): {family} | Run: {rl} (Z = {z_pos} mm)", fontsize=14)
                            ax_run_prof.set_xlabel(r"Median $|t_{\mathrm{final}}|$ [ns]", fontsize=12)
                            ax_run_prof.set_ylabel("Energy [A.U.]", fontsize=12)
                            ax_run_prof.set_xlim(tmin, tmax)
                            ax_run_prof.set_ylim(emin, emax)
                            ax_run_prof.grid(True, alpha=0.3)
                            ax_run_prof.legend(fontsize=12, framealpha=1)
                            pdf.savefig(fig_run_prof)
                            plt.close(fig_run_prof)
                        

                        all_t_family.extend(run_t)
                        all_e_family.extend(run_e)

                # 2. Master plots combining All Runs AND All Channels
                if len(all_t_family) > 0:
                    all_t_arr = np.array(all_t_family)
                    all_e_arr = np.array(all_e_family)
                    
                    master_title = f"MASTER COMBINED {family} | {grp}\nAll Z-Pos & All Channels Combined (PID: {pid_tag})"
                    create_joint_plot(all_t_arr, all_e_arr, tmin, tmax, emin, emax, nbins, master_title, pdf)

                    # Generate Master Profiles
                    # Generate Master Profiles
                    n_profile_bins = 20
                    
                    # Profile 1: Energy Binned
                    fig2, ax2 = plt.subplots(figsize=(10, 8))
                    energy_bins = np.linspace(emin, emax, n_profile_bins + 1)
                    e_centers, t_medians, t_errs = [], [], []
                    for i in range(len(energy_bins)-1):
                        mask = (all_e_arr >= energy_bins[i]) & (all_e_arr < energy_bins[i+1])
                        if np.sum(mask) >= 5: 
                            t_subset = all_t_arr[mask]
                            e_centers.append(0.5 * (energy_bins[i] + energy_bins[i+1]))
                            t_medians.append(np.median(t_subset))
                            t_errs.append(1.253 * (np.std(t_subset) / np.sqrt(len(t_subset))))
                            
                    e_centers, t_medians, t_errs = np.array(e_centers), np.array(t_medians), np.array(t_errs)
                    
                    if len(e_centers) > 0:
                        ax2.errorbar(t_medians, e_centers, xerr=t_errs, fmt='o', color='black', label="Median ToA per Energy Bin", markersize=8, capsize=4, zorder=5)
                        if len(e_centers) > 2:
                            try:
                                # Robust initial guess using log transform: ln(E) = ln(A) + (1/lam)*t
                                valid = e_centers > 0
                                p_log = np.polyfit(t_medians[valid], np.log(e_centers[valid]), 1)
                                lam_guess = 1.0 / p_log[0] if p_log[0] != 0 else 1.0
                                A_guess = np.exp(p_log[1])

                                # Fit E = A * exp(t/lam)
                                popt, _ = curve_fit(exp_func, t_medians, e_centers, p0=[A_guess, lam_guess], maxfev=10000)
                                A_opt, lam_opt = popt
                                
                                t_smooth = np.linspace(min(t_medians), max(t_medians), 500)
                                e_smooth = exp_func(t_smooth, A_opt, lam_opt)
                                ax2.plot(t_smooth, e_smooth, color='red', lw=2.5, label=f"Fit: $E = {A_opt:.2e} e^{{t / {lam_opt:.2f}}}$", zorder=4)
                            except Exception: pass
                        
                        ax2.set_title(f"Master Profile (Energy Binned): {family} (All Channels)", fontsize=14)
                        ax2.set_xlabel(r"Median $|t_{\mathrm{final}}|$ [ns]", fontsize=12)
                        ax2.set_ylabel("Energy [A.U.]", fontsize=12)
                        ax2.set_xlim(tmin, tmax)
                        ax2.set_ylim(emin, emax)
                        ax2.grid(True, alpha=0.3)
                        ax2.legend(fontsize=12, framealpha=1)
                        pdf.savefig(fig2)
                        plt.close(fig2)

                    # Profile 2: Time Binned
                    fig3, ax3 = plt.subplots(figsize=(10, 8))
                    time_bins = np.linspace(tmin, tmax, n_profile_bins + 1)
                    t_centers, e_medians, e_errs = [], [], []
                    for i in range(len(time_bins)-1):
                        mask = (all_t_arr >= time_bins[i]) & (all_t_arr < time_bins[i+1])
                        if np.sum(mask) >= 5:
                            e_subset = all_e_arr[mask]
                            t_centers.append(0.5 * (time_bins[i] + time_bins[i+1]))
                            e_medians.append(np.median(e_subset))
                            e_errs.append(1.253 * (np.std(e_subset) / np.sqrt(len(e_subset))))
                            
                    t_centers, e_medians, e_errs = np.array(t_centers), np.array(e_medians), np.array(e_errs)
                    
                    if len(t_centers) > 0:
                        ax3.errorbar(t_centers, e_medians, yerr=e_errs, fmt='o', color='blue', label="Median Energy per Time Bin", markersize=8, capsize=4, zorder=5)
                        if len(t_centers) > 2:
                            try:
                                # Robust initial guess using log transform
                                valid = e_medians > 0
                                p_log = np.polyfit(t_centers[valid], np.log(e_medians[valid]), 1)
                                lam_guess = 1.0 / p_log[0] if p_log[0] != 0 else 1.0
                                A_guess = np.exp(p_log[1])

                                # Fit E = A * exp(t/lam)
                                popt3, _ = curve_fit(exp_func, t_centers, e_medians, p0=[A_guess, lam_guess], maxfev=10000)
                                A_opt3, lam_opt3 = popt3
                                
                                t_smooth3 = np.linspace(min(t_centers), max(t_centers), 500)
                                e_smooth3 = exp_func(t_smooth3, A_opt3, lam_opt3)
                                ax3.plot(t_smooth3, e_smooth3, color='orange', lw=2.5, label=f"Fit: $E = {A_opt3:.2e} e^{{t / {lam_opt3:.2f}}}$", zorder=4)
                            except Exception: pass
                        
                        ax3.set_title(f"Master Profile (Time Binned): {family} (All Channels)", fontsize=14)
                        ax3.set_xlabel(r"$|t_{\mathrm{final}}|$ [ns]", fontsize=12)
                        ax3.set_ylabel("Median Energy [A.U.]", fontsize=12)
                        ax3.set_xlim(tmin, tmax)
                        ax3.set_ylim(emin, emax)
                        ax3.grid(True, alpha=0.3)
                        ax3.legend(fontsize=12, framealpha=1)
                        pdf.savefig(fig3)
                        plt.close(fig3)
                else:
                    fig = plt.figure()
                    plt.text(0.5, 0.5, f"No valid events passed cuts across all files & channels.", ha='center')
                    pdf.savefig(fig)
                    plt.close(fig)

# ================= MAIN =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="./Attenuation/6mmFERS_Correlations", help="Base output directory")
    parser.add_argument("--pid", default="electron", choices=["muon", "pion", "electron", "proton", "all"])
    parser.add_argument("--emin", type=float, default=0.0, help="Min Energy for Y-axis")
    parser.add_argument("--emax", type=float, default=3000.0, help="Max Energy for Y-axis")
    parser.add_argument("--nbins", type=int, default=100)
    args = parser.parse_args()

    files = _resolve_files()
    if not files:
        print("[ERROR] No files located. Exiting.")
        return
        
    print(f"Located {len(files)} files via mapping dictionary.")
    
    run_y_config_correlations(files, args.outdir, args.emin, args.emax, args.nbins, args.pid)

if __name__ == "__main__":
    main()