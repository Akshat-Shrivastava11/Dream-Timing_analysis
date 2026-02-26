#!/usr/bin/env python3
import os
import re
import glob
import argparse
import json
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# ================= CALIBRATION PATHS =================
PED_FILE   = "/lustre/research/hep/akshriva/CaloXDataAnalysis/data/fers/FERS_pedestals_run1425.json"
HG2LG_FILE = "/lustre/research/hep/akshriva/CaloXDataAnalysis/data/fers/FERS_HG2LG_Sep.json"
RESP_FILE  = "/lustre/research/hep/akshriva/CaloXDataAnalysis/data/fers/FERS_response_Sep.json"

HG_SATURATION_THRESHOLD = 4000.0

# ================= CUT DEFAULTS =================
TREE_NAME = "EventTree"
AMP_THRESHOLD = 100.0  
MIN_ADC_CUT = -100.0

# ================= PAIRED GRIDS FOR LOOKUP =================
DRS_QUARTZ_GRID = [
    [None,  "002", None,  None], ["006", "004", "206", "204"],
    ["016", "014", "216", "214"], ["026", "024", "226", "224"],
    [None,  "030", None,  None], [None,  "034", None,  None],
    ["106", "104", "306", "304"], ["116", "114", "316", "314"],
    ["126", "124", "326", "324"], [None,  "134", None,  "334"],
]
FERS_QUARTZ_GRID = [
    ["902", None,  None,   None], ["906", "905", "1006", "1005"],
    ["914", "913", "1014", "1013"], ["920", "923", "1020", "1023"],
    [None,  "927", None,   None], ["928", "931", "1028", "1031"],
    ["936", "939", "1036", "1039"], ["944", "947", "1044", "1047"],
    ["952", "955", "1052", "1055"], ["960", "963", "1060", "1063"],
]

DRS_PLASTIC_GRID = [
    [None,  "000", "202", "200"], ["012", "010", "212", "210"],
    ["022", "020", "222", "220"], ["032", None,  "232", "230"],
    ["102", "100", "302", "300"], ["112", "110", "312", "310"],
    ["122", "120", "322", "320"], ["132", "130", "332", "330"],
]
FERS_PLASTIC_GRID = [
    [None,  "901", "1002", "1001"], ["910", "909", "1010", "1009"],
    ["916", "919", "1016", "1019"], ["924", None,  "1024", "1027"],
    ["932", "935", "1032", "1035"], ["940", "943", "1040", "1043"],
    ["948", "951", "1048", "1051"], ["956", "959", "1056", "1059"],
]

DRS_SCI_ALL_GRID = [
    ["003", "001", "203", "201"], ["007", "005", "207", "205"],
    ["013", "011", "213", "211"], ["017", "015", "217", "215"],
    ["023", "021", "223", "221"], ["027", "025", "227", "225"],
    ["033", "031", "233", "231"], [None,  "035", None,  "235"],
    ["103", "101", "303", "301"], ["107", "105", "307", "305"],
    ["113", "111", "313", "311"], ["117", "115", "317", "315"],
    ["123", "121", "323", "321"], ["127", "125", "327", "325"],
    ["133", "131", "333", "331"], [None,  "135", None,  "335"],
]
FERS_SCI_GRID = [
    ["900", "903", "1000", "1003"], ["904", "907", "1004", "1007"],
    ["908", "911", "1008", "1011"], ["912", "915", "1012", "1015"],
    ["918", "917", "1018", "1017"], ["922", "921", "1022", "1021"],
    ["926", "925", "1026", "1025"], ["930", "929", "1030", "1029"],
    ["934", "933", "1034", "1033"], ["938", "937", "1038", "1037"],
    ["942", "941", "1042", "1041"], ["946", "945", "1046", "1045"],
    ["950", "949", "1050", "1049"], ["954", "953", "1054", "1053"],
    ["958", "957", "1058", "1057"], ["962", "961", "1062", "1061"],
]

def build_channel_map():
    drs_to_fers = {}
    drs_grids = [DRS_QUARTZ_GRID, DRS_PLASTIC_GRID, DRS_SCI_ALL_GRID]
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
        print(f"      [PID] No PID requested ('{particle_type}'). All {n_entries} events pass.")
        return mask
        
    print(f"      [PID] Starting PID filtering for '{particle_type}' on {n_entries} events...")
    
    for det, must_fire in reqs.items():
        if PID_BRANCH_MAP.get(det) not in tree: 
            print(f"      [PID] WARNING: Branch for {det} not found in tree! Skipping this cut.")
            continue
            
        ts_min, ts_max, val_cut = get_service_drs_cut(det)
        waves = tree[PID_BRANCH_MAP[det]].array(library="ak")
        baseline = ak.mean(waves[:, :30], axis=1)
        
        # Calculate sum in window
        window_sum = ak.to_numpy(ak.sum((waves - baseline)[:, int(ts_min):int(ts_max)], axis=1))
        is_fired = window_sum < val_cut
        
        initial_count = np.sum(mask)
        mask = mask & (is_fired if must_fire else ~is_fired)
        removed = initial_count - np.sum(mask)
        
        status = "FIRED" if must_fire else "VETOED"
        print(f"      [PID] {det:<12} ({status}): Cut < {val_cut:.1f} | Thrown away: {removed:<5} | Remaining: {np.sum(mask)}")
        
    print(f"      [PID] Final events passing '{particle_type}' PID: {np.sum(mask)} / {n_entries}")
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
    calib = RESP.get(b_str, {}).get(c_str, 1.0)
    
    hg_sub = hg_array - ped_hg
    lg_sub = lg_array - ped_lg
    is_sat = hg_sub > HG_SATURATION_THRESHOLD
    mix_energy = np.where(is_sat, (lg_sub - intercept) / slope, hg_sub)
    
    return np.maximum(0, mix_energy * calib)
# GLOBAL_GEV_SCALE = 1.0 

# def reconstruct_energy_1d(b_str, c_str, hg_array, lg_array):
#     """Vectorized reconstruction converting raw FERS ADC to pure GeV."""
#     # 1. Load Pedestals
#     ped_hg = PEDS.get(b_str, {}).get(c_str, {}).get("HG", 0.0)
#     ped_lg = PEDS.get(b_str, {}).get(c_str, {}).get("LG", 0.0)
    
#     # 2. Load HG2LG Conversion
#     slope = HG2LG.get(b_str, {}).get(c_str, {}).get("slope", 1.0)
#     intercept = HG2LG.get(b_str, {}).get(c_str, {}).get("intercept", 0.0)
#     if slope == 0: slope = 1.0 
    
#     # 3. Load Channel-wise Response (Equalization)
#     calib = RESP.get(b_str, {}).get(c_str, 1.0)
    
#     # 4. Process Arrays
#     hg_sub = hg_array - ped_hg
#     lg_sub = lg_array - ped_lg
#     is_sat = hg_sub > HG_SATURATION_THRESHOLD
    
#     # "Mix": use LG converted to HG scale if saturated
#     mix_energy = np.where(is_sat, (lg_sub - intercept) / slope, hg_sub)
    
#     # 5. Apply Calibration & Absolute GeV Scale
#     energy_gev = mix_energy * calib * GLOBAL_GEV_SCALE
    
#     # Zero-suppress negative noise fluctuations
#     return np.maximum(0, energy_gev)
# ================= HELPERS =================
def _parse_code(code_str): return int(code_str[0]), int(code_str[1]), int(code_str[2])
def _run_label(path): return re.search(r"(run\d+_\d{11,12})", os.path.basename(path)).group(1)

# ================= MAIN CORRELATION ENGINE =================
def make_channel_correlation(files, drs_code, outdir, tmin, tmax, emin, emax, nbins, particle_type=None):
    os.makedirs(outdir, exist_ok=True)
    
    channel_map = build_channel_map()
    fers_code = channel_map.get(drs_code)
    
    if not fers_code:
        print(f"Error: DRS code '{drs_code}' not found in any grid mapping.")
        return

    db, dg, dc = _parse_code(drs_code)
    fb, fc = int(fers_code[:-2]), int(fers_code[-2:])
    
    t_branch = f"tfinal_Board{db}_Group{dg}_Channel{dc}"
    w_branch = f"DRS_Board{db}_Group{dg}_Channel{dc}"
    f_hg_branch = f"FERS_Board{fb}_energyHG"
    f_lg_branch = f"FERS_Board{fb}_energyLG"

    pid_tag = particle_type if particle_type else "NoPID"
    out_pdf = os.path.join(outdir, f"CORRELATION_D{drs_code}_F{fers_code}_{pid_tag}.pdf")

    all_t_valid, all_e_valid = [], []
    print(f"\n=========================================================")
    print(f" PROCESSING CORRELATION: DRS {drs_code} <---> FERS {fers_code}")
    print(f"=========================================================")

    with PdfPages(out_pdf) as pdf:
        for fpath in files:
            rl = _run_label(fpath)
            print(f"\n---> Opening File: {rl}")
            try:
                with uproot.open(fpath) as uf:
                    tree = uf[TREE_NAME]
                    keys = set(tree.keys())
                    total_events = tree.num_entries
                    print(f"     Total Events in Tree: {total_events}")
                    
                    if not all(k in keys for k in (t_branch, w_branch, f_hg_branch, f_lg_branch)):
                        print(f"     [WARN] Missing required branches. Skipping file.")
                        continue

                    # 1. PID Mask
                    pid_mask = compute_pid_mask(tree, particle_type) if particle_type else np.ones(total_events, dtype=bool)
                    events_post_pid = np.sum(pid_mask)

                    if events_post_pid == 0:
                        print("     [CUTFLOW] 0 events passed PID. Skipping to next file.")
                        continue

                    # 2. Timing Mask
                    t_arr = tree[t_branch].array(library="np")
                    t_abs = np.abs(t_arr)
                    time_mask = (~np.isnan(t_abs)) & (t_abs >= tmin) & (t_abs <= tmax)
                    
                    events_post_time = np.sum(pid_mask & time_mask)
                    thrown_by_time = events_post_pid - events_post_time
                    print(f"      [CUTFLOW] Timing cut [{tmin}, {tmax}] ns: Thrown away {thrown_by_time} | Remaining: {events_post_time}")

                    if events_post_time == 0:
                        continue

                    # 3. ADC Mask (DRS Waveform)
                    waves = tree[w_branch].array(library="ak")
                    baseline = ak.mean(waves[:, :30], axis=1)
                    waves_blsub = waves - baseline
                    peak = ak.to_numpy(ak.max(waves_blsub, axis=1))
                    min_adc = ak.to_numpy(ak.min(waves_blsub, axis=1))
                    adc_mask = (peak >= AMP_THRESHOLD) & (min_adc >= MIN_ADC_CUT)

                    # Combine Masks
                    final_mask = pid_mask & time_mask & adc_mask
                    final_events = np.sum(final_mask)
                    thrown_by_adc = events_post_time - final_events
                    print(f"      [CUTFLOW] ADC cut (Amp>{AMP_THRESHOLD}, Min>{MIN_ADC_CUT}): Thrown away {thrown_by_adc} | FINAL PLOTTED: {final_events}")

                    if final_events == 0: 
                        continue

                    # 4. Extract arrays safely using awkward
                    t_valid = t_abs[final_mask]
                    hg_ak = tree[f_hg_branch].array(library="ak")[final_mask]
                    lg_ak = tree[f_lg_branch].array(library="ak")[final_mask]
                    
                    # Select the specific channel index and convert to pure numpy 1D array
                    hg_valid = ak.to_numpy(hg_ak[:, fc])
                    lg_valid = ak.to_numpy(lg_ak[:, fc])

                    # 5. Calibrate Energy
                    e_valid = reconstruct_energy_1d(str(fb), str(fc), hg_valid, lg_valid)

                    # Append to globals
                    all_t_valid.extend(t_valid)
                    all_e_valid.extend(e_valid)

                    # --- PLOT INDIVIDUAL RUN ---
                    if len(t_valid) > 10:
                        fig, ax = plt.subplots(figsize=(9, 7))
                        h2 = ax.hist2d(t_valid, e_valid, bins=[nbins, 100], range=[[tmin, tmax], [emin, emax]], cmap='viridis', cmin=1)
                        fig.colorbar(h2[3], ax=ax, label='Events')
                        
                        corr_val = np.corrcoef(t_valid, e_valid)[0, 1] if len(t_valid) > 1 else 0.0
                        ax.legend(handles=[mpatches.Patch(color='none', label=f"Correlation (r): {corr_val:.3f}")], loc='upper right')
                        
                        ax.set_title(f"Energy vs ToA: DRS {drs_code} $\\rightarrow$ FERS {fers_code}\nRun: {rl} (PID: {pid_tag})")
                        ax.set_xlabel(r"$|t_{\mathrm{final}}|$ [ns]")
                        ax.set_ylabel("A. U. ")
                        ax.grid(True, alpha=0.3)
                        
                        fig.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)

            except Exception as e:
                print(f"     [ERROR] Failed processing {rl}: {e}")

        # --- PLOT COMBINED RUNS ---
        if len(all_t_valid) > 0:
            print(f"\n=========================================================")
            print(f" DONE! Generating Combined Plot with {len(all_t_valid)} total events.")
            print(f"=========================================================\n")
            fig, ax = plt.subplots(figsize=(10, 8))
            h2 = ax.hist2d(all_t_valid, all_e_valid, bins=[nbins, 100], range=[[tmin, tmax], [emin, emax]], cmap='turbo', cmin=1)
            fig.colorbar(h2[3], ax=ax, label='Total Events')
            
            corr_val = np.corrcoef(all_t_valid, all_e_valid)[0, 1] if len(all_t_valid) > 1 else 0.0
            ax.legend(handles=[mpatches.Patch(color='none', label=f"Overall Correlation (r): {corr_val:.3f}")], loc='upper right', fontsize=12)
            
            ax.set_title(f"COMBINED Energy vs ToA: DRS {drs_code} $\\rightarrow$ FERS {fers_code}\nAll Runs (PID: {pid_tag})", fontsize=14)
            ax.set_xlabel(r"$|t_{\mathrm{final}}|$ [ns]", fontsize=12)
            ax.set_ylabel("Calibrated Energy [MIP/MeV]", fontsize=12)
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        else:
            fig = plt.figure()
            plt.text(0.5, 0.5, "No valid events passed cuts across all files.", ha='center')
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved Correlation Plots to: {out_pdf}")

# ================= MAIN CORRELATION ENGINE =================
# ================= MAIN CORRELATION ENGINE =================
def make_channel_correlation(files, drs_code, outdir, tmin, tmax, emin, emax, nbins, particle_type=None):
    os.makedirs(outdir, exist_ok=True)
    
    channel_map = build_channel_map()
    fers_code = channel_map.get(drs_code)
    
    if not fers_code:
        print(f"Error: DRS code '{drs_code}' not found in any grid mapping.")
        return

    db, dg, dc = _parse_code(drs_code)
    fb, fc = int(fers_code[:-2]), int(fers_code[-2:])
    
    t_branch = f"tfinal_Board{db}_Group{dg}_Channel{dc}"
    w_branch = f"DRS_Board{db}_Group{dg}_Channel{dc}"
    f_hg_branch = f"FERS_Board{fb}_energyHG"
    f_lg_branch = f"FERS_Board{fb}_energyLG"

    pid_tag = particle_type if particle_type else "NoPID"
    out_pdf = os.path.join(outdir, f"CORRELATION_D{drs_code}_F{fers_code}_{pid_tag}.pdf")

    all_t_valid, all_e_valid = [], []
    print(f"\n=========================================================")
    print(f" PROCESSING CORRELATION: DRS {drs_code} <---> FERS {fers_code}")
    print(f"=========================================================")

    with PdfPages(out_pdf) as pdf:
        for fpath in files:
            rl = _run_label(fpath)
            print(f"\n---> Opening File: {rl}")
            try:
                with uproot.open(fpath) as uf:
                    tree = uf[TREE_NAME]
                    keys = set(tree.keys())
                    total_events = tree.num_entries
                    print(f"     Total Events in Tree: {total_events}")
                    
                    if not all(k in keys for k in (t_branch, w_branch, f_hg_branch, f_lg_branch)):
                        print(f"     [WARN] Missing required branches. Skipping file.")
                        continue

                    # 1. PID Mask
                    pid_mask = compute_pid_mask(tree, particle_type) if particle_type else np.ones(total_events, dtype=bool)
                    events_post_pid = np.sum(pid_mask)

                    if events_post_pid == 0:
                        continue

                    # 2. Timing Mask
                    t_arr = tree[t_branch].array(library="np")
                    t_abs = np.abs(t_arr)
                    time_mask = (~np.isnan(t_abs)) & (t_abs >= tmin) & (t_abs <= tmax)
                    
                    # 3. ADC Mask (DRS Waveform)
                    waves = tree[w_branch].array(library="ak")
                    baseline = ak.mean(waves[:, :30], axis=1)
                    waves_blsub = waves - baseline
                    peak = ak.to_numpy(ak.max(waves_blsub, axis=1))
                    min_adc = ak.to_numpy(ak.min(waves_blsub, axis=1))
                    adc_mask = (peak >= AMP_THRESHOLD) & (min_adc >= MIN_ADC_CUT)

                    # Combine Masks
                    final_mask = pid_mask & time_mask & adc_mask
                    final_events = np.sum(final_mask)
                    print(f"      [CUTFLOW] Final Events Plotted: {final_events} / {total_events}")

                    if final_events == 0: 
                        continue

                    # 4. Extract arrays
                    t_valid = t_abs[final_mask]
                    hg_ak = tree[f_hg_branch].array(library="ak")[final_mask]
                    lg_ak = tree[f_lg_branch].array(library="ak")[final_mask]
                    
                    hg_valid = ak.to_numpy(hg_ak[:, fc])
                    lg_valid = ak.to_numpy(lg_ak[:, fc])

                    # 5. Calibrate Energy
                    e_valid = reconstruct_energy_1d(str(fb), str(fc), hg_valid, lg_valid)

                    all_t_valid.extend(t_valid)
                    all_e_valid.extend(e_valid)

                    # --- PLOT INDIVIDUAL RUN ---
                    if len(t_valid) > 10:
                        fig, ax = plt.subplots(figsize=(9, 7))
                        h2 = ax.hist2d(t_valid, e_valid, bins=[nbins, 100], range=[[tmin, tmax], [emin, emax]], cmap='viridis', cmin=1)
                        fig.colorbar(h2[3], ax=ax, label='Events')
                        
                        corr_val = np.corrcoef(t_valid, e_valid)[0, 1] if len(t_valid) > 1 else 0.0
                        ax.legend(handles=[mpatches.Patch(color='none', label=f"Correlation (r): {corr_val:.3f}")], loc='upper right')
                        
                        ax.set_title(f"Energy vs ToA: DRS {drs_code} $\\rightarrow$ FERS {fers_code}\nRun: {rl} (PID: {pid_tag})")
                        ax.set_xlabel(r"$|t_{\mathrm{final}}|$ [ns]")
                        ax.set_ylabel("Energy [A.U.]")
                        ax.grid(True, alpha=0.3)
                        
                        fig.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)

            except Exception as e:
                print(f"     [ERROR] Failed processing {rl}: {e}")

        # --- PLOT COMBINED RUNS ---
        if len(all_t_valid) > 0:
            print(f"\n=========================================================")
            print(f" DONE! Generating Combined Plots with {len(all_t_valid)} total events.")
            print(f"=========================================================\n")
            
            all_e_arr = np.array(all_e_valid)
            all_t_arr = np.array(all_t_valid)

            # --- 1. Combined 2D Histogram ---
            fig, ax = plt.subplots(figsize=(10, 8))
            h2 = ax.hist2d(all_t_valid, all_e_valid, bins=[nbins, 100], range=[[tmin, tmax], [emin, emax]], cmap='turbo', cmin=1)
            fig.colorbar(h2[3], ax=ax, label='Total Events')
            
            corr_val = np.corrcoef(all_t_valid, all_e_valid)[0, 1] if len(all_t_valid) > 1 else 0.0
            ax.legend(handles=[mpatches.Patch(color='none', label=f"Overall Correlation (r): {corr_val:.3f}")], loc='upper right', fontsize=12)
            
            ax.set_title(f"COMBINED Energy vs ToA: DRS {drs_code} $\\rightarrow$ FERS {fers_code}\nAll Runs (PID: {pid_tag})", fontsize=14)
            ax.set_xlabel(r"$|t_{\mathrm{final}}|$ [ns]", fontsize=12)
            ax.set_ylabel("Energy [A.U.]", fontsize=12)
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Define the number of condensed points for the profile plots
            n_profile_bins = 10

            # --- 2. THE PROFILE PLOT (Energy Bins -> Median Time) ---
            print(f" Generating Profile Plot 1 (Energy Bins: {n_profile_bins} points)...")
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            
            # Create 10 energy segments
            energy_bins = np.linspace(emin, emax, n_profile_bins + 1)
            e_centers, t_medians, t_errs = [], [], []
            
            for i in range(len(energy_bins)-1):
                mask = (all_e_arr >= energy_bins[i]) & (all_e_arr < energy_bins[i+1])
                if np.sum(mask) >= 5: # Lowered requirement slightly since bins are wider
                    t_subset = all_t_arr[mask]
                    e_centers.append(0.5 * (energy_bins[i] + energy_bins[i+1]))
                    t_medians.append(np.median(t_subset))
                    t_errs.append(1.253 * (np.std(t_subset) / np.sqrt(len(t_subset))))
                    
            e_centers, t_medians, t_errs = np.array(e_centers), np.array(t_medians), np.array(t_errs)
            
            if len(e_centers) > 0:
                ax2.errorbar(t_medians, e_centers, xerr=t_errs, fmt='o', color='black', 
                             label="Median ToA per Energy Bin", markersize=8, capsize=4, zorder=5)
                
                # --- NEW LINEAR FIT ---
                if len(e_centers) > 1: # Only need 2 points for a line
                    # Fit a 1st degree polynomial: t = m*E + b
                    popt = np.polyfit(e_centers, t_medians, 1)
                    e_smooth = np.linspace(min(e_centers), max(e_centers), 500)
                    t_smooth = np.polyval(popt, e_smooth)
                    
                    m, b = popt[0], popt[1]
                    sign = "+" if b >= 0 else "-"
                    fit_eq = f"Linear Fit: $t = {m:.2e}E {sign} {abs(b):.2f}$"
                    
                    ax2.plot(t_smooth, e_smooth, color='red', lw=2.5, label=fit_eq, zorder=4)
                

                
                ax2.set_title(f"Profile (Energy Binned - {n_profile_bins} pts): DRS {drs_code} $\\rightarrow$ FERS {fers_code}", fontsize=14)
                ax2.set_xlabel(r"Median $|t_{\mathrm{final}}|$ [ns]", fontsize=12)
                ax2.set_ylabel("Energy [A.U.]", fontsize=12)
                ax2.set_xlim(tmin, tmax)
                ax2.set_ylim(emin, emax)
                ax2.grid(True, alpha=0.3)
                ax2.legend(fontsize=12, framealpha=1)
                
                fig2.tight_layout()
                pdf.savefig(fig2)
                plt.close(fig2)

            # --- 3. THE REVERSE PROFILE PLOT (Time Bins -> Median Energy) ---
            print(f" Generating Profile Plot 2 (Time Bins: {n_profile_bins} points)...")
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            
            # Create 10 time segments
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
                ax3.errorbar(t_centers, e_medians, yerr=e_errs, fmt='o', color='blue', 
                             label="Median Energy per Time Bin", markersize=8, capsize=4, zorder=5)
                
                # --- NEW LINEAR FIT ---
                if len(t_centers) > 1: # Only need 2 points for a line
                    # Fit a 1st degree polynomial: E = m*t + b
                    popt3 = np.polyfit(t_centers, e_medians, 1)
                    t_smooth3 = np.linspace(min(t_centers), max(t_centers), 500)
                    e_smooth3 = np.polyval(popt3, t_smooth3)
                    
                    m, b = popt3[0], popt3[1]
                    sign = "+" if b >= 0 else "-"
                    fit_eq3 = f"Linear Fit: $E = {m:.2e}t {sign} {abs(b):.2f}$"
                    
                    ax3.plot(t_smooth3, e_smooth3, color='orange', lw=2.5, label=fit_eq3, zorder=4)
                
                ax3.set_title(f"Profile (Time Binned - {n_profile_bins} pts): DRS {drs_code} $\\rightarrow$ FERS {fers_code}", fontsize=14)
                ax3.set_xlabel(r"$|t_{\mathrm{final}}|$ [ns]", fontsize=12)
                ax3.set_ylabel("Median Energy [A.U.]", fontsize=12)
                ax3.set_xlim(tmin, tmax)
                ax3.set_ylim(emin, emax)
                ax3.grid(True, alpha=0.3)
                ax3.legend(fontsize=12, framealpha=1)
                
                fig3.tight_layout()
                pdf.savefig(fig3)
                plt.close(fig3)
            else:
                print("     [WARN] Not enough valid time bins to create Profile 2.")
                
        else:
            fig = plt.figure()
            plt.text(0.5, 0.5, "No valid events passed cuts across all files.", ha='center')
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved Correlation Plots to: {out_pdf}")
# ================= MAIN =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ana-files", nargs="+", required=True, help="List of input ROOT files.")
    parser.add_argument("--outdir", default="./Attenuation/FERS_Correlations", help="Output directory")
    parser.add_argument("--single-channel", required=True, help="3-digit DRS code (e.g., '104')")
    parser.add_argument("--pid", default="electron", choices=["muon", "pion", "electron", "proton", "all"])
    parser.add_argument("--tmin", type=float, default=4.0, help="Min |tfinal|")
    parser.add_argument("--tmax", type=float, default=25.0, help="Max |tfinal|")
    parser.add_argument("--emin", type=float, default=0.0, help="Min Energy for Y-axis")
    parser.add_argument("--emax", type=float, default=4000.0, help="Max Energy for Y-axis")
    parser.add_argument("--nbins", type=int, default=100)
    args = parser.parse_args()

    files = args.ana_files
    make_channel_correlation(files, args.single_channel, args.outdir, args.tmin, args.tmax, args.emin, args.emax, args.nbins, args.pid)

if __name__ == "__main__":
    main()