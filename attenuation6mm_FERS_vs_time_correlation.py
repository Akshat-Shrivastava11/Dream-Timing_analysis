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

# ================= CALIBRATION PATHS =================
PED_FILE   = "/lustre/research/hep/akshriva/CaloXDataAnalysis/data/fers/FERS_pedestals_run1425.json"
HG2LG_FILE = "/lustre/research/hep/akshriva/CaloXDataAnalysis/data/fers/FERS_HG2LG_Sep.json"
RESP_FILE  = "/lustre/research/hep/akshriva/CaloXDataAnalysis/data/fers/FERS_response_Sep.json"

HG_SATURATION_THRESHOLD = 4000.0

# ================= CUT DEFAULTS =================
TREE_NAME = "EventTree"
AMP_THRESHOLD = 100.0  
MIN_ADC_CUT = -100.0

# ================= PAIRED GRIDS FOR LOOKUP (CORES IGNORED) =================

# ==========================================
# QUARTZ: Top, Bottom & Sides (Core Ignored)
# ==========================================
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

# ==========================================
# PLASTIC: Top & Bottom (Core Ignored)
# ==========================================
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

# ==============================================
# SCINTILLATOR: Top & Bottom (Core Ignored)
# ==============================================
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

def exp_func(x, A, lam):
    return A * np.exp(-x / lam)

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
    
    hg_sub = hg_array - ped_hg
    lg_sub = lg_array - ped_lg
    is_sat = hg_sub > HG_SATURATION_THRESHOLD
    mix_energy = np.where(is_sat, (lg_sub - intercept) / slope, hg_sub)
    
    return np.maximum(0, mix_energy)

# ================= TIMING LOGIC =================
def compute_tfinal_6mm(tree, b, g, c, suffix=""):
    """Dynamically computes ToA for 6mm setup sensors."""
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
                    
                    if not all(k in keys for k in (w_branch, f_hg_branch, f_lg_branch)):
                        print(f"     [WARN] Missing waveform/FERS branches. Skipping file.")
                        continue

                    t_abs = compute_tfinal_6mm(tree, db, dg, dc)
                    if t_abs is None:
                        print(f"     [WARN] Missing required timing references. Skipping file.")
                        continue

                    pid_mask = compute_pid_mask(tree, particle_type) if particle_type else np.ones(total_events, dtype=bool)
                    events_post_pid = np.sum(pid_mask)

                    if events_post_pid == 0:
                        continue

                    time_mask = (~np.isnan(t_abs)) & (t_abs >= tmin) & (t_abs <= tmax)
                    
                    waves = tree[w_branch].array(library="ak")
                    baseline = ak.mean(waves[:, :30], axis=1)
                    waves_blsub = waves - baseline
                    peak = ak.to_numpy(ak.max(waves_blsub, axis=1))
                    min_adc = ak.to_numpy(ak.min(waves_blsub, axis=1))
                    adc_mask = (peak >= AMP_THRESHOLD) & (min_adc >= MIN_ADC_CUT)

                    final_mask = pid_mask & time_mask & adc_mask
                    final_events = np.sum(final_mask)
                    print(f"      [CUTFLOW] Final Events Plotted: {final_events} / {total_events}")

                    if final_events == 0: 
                        continue

                    t_valid = t_abs[final_mask]
                    hg_ak = tree[f_hg_branch].array(library="ak")[final_mask]
                    lg_ak = tree[f_lg_branch].array(library="ak")[final_mask]
                    
                    hg_valid = ak.to_numpy(hg_ak[:, fc])
                    lg_valid = ak.to_numpy(lg_ak[:, fc])

                    e_valid = reconstruct_energy_1d(str(fb), str(fc), hg_valid, lg_valid)

                    all_t_valid.extend(t_valid)
                    all_e_valid.extend(e_valid)

            except Exception as e:
                print(f"     [ERROR] Failed processing {rl}: {e}")

        if len(all_t_valid) > 0:
            print(f"\n=========================================================")
            print(f" DONE! Generating Combined Plots with {len(all_t_valid)} total events.")
            print(f"=========================================================\n")
            
            all_e_arr = np.array(all_e_valid)
            all_t_arr = np.array(all_t_valid)

            # --- 1. JOINT PLOT: 2D Histogram + Marginal 1D Histograms ---
            fig, ax_main = plt.subplots(figsize=(10, 10))
            
            h2 = ax_main.hist2d(all_t_valid, all_e_valid, bins=[nbins, 100], range=[[tmin, tmax], [emin, emax]], cmap='turbo', cmin=1)
            
            # Setup dividers for marginal histograms
            divider = make_axes_locatable(ax_main)
            ax_top = divider.append_axes("top", size="20%", pad=0.1, sharex=ax_main)
            ax_right = divider.append_axes("right", size="20%", pad=0.1, sharey=ax_main)
            ax_cbar = divider.append_axes("right", size="5%", pad=0.3)

            # Top Marginal (Time)
            ax_top.hist(all_t_valid, bins=nbins, range=[tmin, tmax], color='#2b5b84', histtype='stepfilled')
            ax_top.xaxis.set_tick_params(labelbottom=False)
            ax_top.set_ylabel("Events")

            # Right Marginal (Energy)
            ax_right.hist(all_e_valid, bins=100, range=[emin, emax], orientation='horizontal', color='#2b5b84', histtype='stepfilled')
            ax_right.yaxis.set_tick_params(labelleft=False)
            ax_right.set_xlabel("Events")

            # Add colorbar
            fig.colorbar(h2[3], cax=ax_cbar, label='Total Events')
            
            corr_val = np.corrcoef(all_t_valid, all_e_valid)[0, 1] if len(all_t_valid) > 1 else 0.0
            ax_main.legend(handles=[mpatches.Patch(color='none', label=f"Correlation (r): {corr_val:.3f}")], loc='upper right', fontsize=12)
            
            fig.suptitle(f"COMBINED Energy vs ToA: DRS {drs_code} $\\rightarrow$ FERS {fers_code}\nAll Runs (PID: {pid_tag})", fontsize=14, y=0.95)
            ax_main.set_xlabel(r"$|t_{\mathrm{final}}|$ [ns]", fontsize=12)
            ax_main.set_ylabel("Energy [A.U.]", fontsize=12)
            ax_main.grid(True, alpha=0.3)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            n_profile_bins = 20

            print(f" Generating Profile Plot 1 (Energy Bins: {n_profile_bins} points)...")
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
                ax2.errorbar(t_medians, e_centers, xerr=t_errs, fmt='o', color='black', 
                             label="Median ToA per Energy Bin", markersize=8, capsize=4, zorder=5)
                
                if len(e_centers) > 2: 
                    try:
                        p0 = [np.max(t_medians), np.mean(e_centers) if np.mean(e_centers) != 0 else 1.0]
                        popt, _ = curve_fit(exp_func, e_centers, t_medians, p0=p0, maxfev=10000)
                        
                        A_opt, lam_opt = popt
                        e_smooth = np.linspace(min(e_centers), max(e_centers), 500)
                        t_smooth = exp_func(e_smooth, A_opt, lam_opt)
                        
                        fit_eq = f"Fit: $t = {A_opt:.2e} e^{{-E / {lam_opt:.2f}}}$"
                        
                        ax2.plot(t_smooth, e_smooth, color='red', lw=2.5, label=fit_eq, zorder=4)
                    except Exception as e:
                        print(f"     [WARN] Exponential fit failed for Profile 1: {e}")
                
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

            print(f" Generating Profile Plot 2 (Time Bins: {n_profile_bins} points)...")
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
                ax3.errorbar(t_centers, e_medians, yerr=e_errs, fmt='o', color='blue', 
                             label="Median Energy per Time Bin", markersize=8, capsize=4, zorder=5)
                
                if len(t_centers) > 2: 
                    try:
                        p0 = [np.max(e_medians), np.mean(t_centers) if np.mean(t_centers) != 0 else 1.0]
                        popt3, _ = curve_fit(exp_func, t_centers, e_medians, p0=p0, maxfev=10000)
                        
                        A_opt3, lam_opt3 = popt3
                        t_smooth3 = np.linspace(min(t_centers), max(t_centers), 500)
                        e_smooth3 = exp_func(t_smooth3, A_opt3, lam_opt3)
                        
                        fit_eq3 = f"Fit: $E = {A_opt3:.2e} e^{{-t / {lam_opt3:.2f}}}$"
                        
                        ax3.plot(t_smooth3, e_smooth3, color='orange', lw=2.5, label=fit_eq3, zorder=4)
                    except Exception as e:
                        print(f"     [WARN] Exponential fit failed for Profile 2: {e}")
                
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

def process_and_plot_channel(files, drs_code, fers_code, family_name, pdf, tmin, tmax, emin, emax, nbins, particle_type, pid_tag):
    db, dg, dc = _parse_code(drs_code)
    fb, fc = int(fers_code[:-2]), int(fers_code[-2:])
    
    w_branch = f"DRS_Board{db}_Group{dg}_Channel{dc}"
    f_hg_branch = f"FERS_Board{fb}_energyHG"
    f_lg_branch = f"FERS_Board{fb}_energyLG"

    all_t_valid, all_e_valid = [], []
    print(f"  -> Extracting DRS {drs_code} <---> FERS {fers_code} ...", end=" ")

    for fpath in files:
        try:
            with uproot.open(fpath) as uf:
                tree = uf[TREE_NAME]
                keys = set(tree.keys())
                
                if not all(k in keys for k in (w_branch, f_hg_branch, f_lg_branch)):
                    continue

                t_abs = compute_tfinal_6mm(tree, db, dg, dc)
                if t_abs is None:
                    continue

                pid_mask = compute_pid_mask(tree, particle_type) if particle_type else np.ones(tree.num_entries, dtype=bool)
                if np.sum(pid_mask) == 0: continue

                time_mask = (~np.isnan(t_abs)) & (t_abs >= tmin) & (t_abs <= tmax)

                waves = tree[w_branch].array(library="ak")
                baseline = ak.mean(waves[:, :30], axis=1)
                waves_blsub = waves - baseline
                peak = ak.to_numpy(ak.max(waves_blsub, axis=1))
                min_adc = ak.to_numpy(ak.min(waves_blsub, axis=1))
                adc_mask = (peak >= AMP_THRESHOLD) & (min_adc >= MIN_ADC_CUT)

                final_mask = pid_mask & time_mask & adc_mask
                if np.sum(final_mask) == 0: continue

                t_valid = t_abs[final_mask]
                hg_ak = tree[f_hg_branch].array(library="ak")[final_mask]
                lg_ak = tree[f_lg_branch].array(library="ak")[final_mask]
                
                hg_valid = ak.to_numpy(hg_ak[:, fc])
                lg_valid = ak.to_numpy(lg_ak[:, fc])

                e_valid = reconstruct_energy_1d(str(fb), str(fc), hg_valid, lg_valid)

                all_t_valid.extend(t_valid)
                all_e_valid.extend(e_valid)

        except Exception as e:
            pass

    print(f"Found {len(all_t_valid)} events.")

    if len(all_t_valid) > 0:
        all_e_arr = np.array(all_e_valid)
        all_t_arr = np.array(all_t_valid)
        n_profile_bins = 20

        # --- 1. JOINT PLOT: 2D Histogram + Marginal 1D Histograms ---
        fig, ax_main = plt.subplots(figsize=(10, 10))
        
        h2 = ax_main.hist2d(all_t_valid, all_e_valid, bins=[nbins, 100], range=[[tmin, tmax], [emin, emax]], cmap='turbo', cmin=1)
        
        divider = make_axes_locatable(ax_main)
        ax_top = divider.append_axes("top", size="20%", pad=0.1, sharex=ax_main)
        ax_right = divider.append_axes("right", size="20%", pad=0.1, sharey=ax_main)
        ax_cbar = divider.append_axes("right", size="5%", pad=0.3)

        # Top Marginal (Time)
        ax_top.hist(all_t_valid, bins=nbins, range=[tmin, tmax], color='#2b5b84', histtype='stepfilled')
        ax_top.xaxis.set_tick_params(labelbottom=False)
        ax_top.set_ylabel("Events")

        # Right Marginal (Energy)
        ax_right.hist(all_e_valid, bins=100, range=[emin, emax], orientation='horizontal', color='#2b5b84', histtype='stepfilled')
        ax_right.yaxis.set_tick_params(labelleft=False)
        ax_right.set_xlabel("Events")

        fig.colorbar(h2[3], cax=ax_cbar, label='Total Events')
        
        corr_val = np.corrcoef(all_t_valid, all_e_valid)[0, 1] if len(all_t_valid) > 1 else 0.0
        ax_main.legend(handles=[mpatches.Patch(color='none', label=f"Correlation (r): {corr_val:.3f}")], loc='upper right', fontsize=12)
        
        fig.suptitle(f"[{family_name}] Energy vs ToA: DRS {drs_code} $\\rightarrow$ FERS {fers_code}\nAll Runs Combined (PID: {pid_tag})", fontsize=14, y=0.95)
        ax_main.set_xlabel(r"$|t_{\mathrm{final}}|$ [ns]", fontsize=12)
        ax_main.set_ylabel("Energy [A.U.]", fontsize=12)
        ax_main.grid(True, alpha=0.3)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

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
                    p0 = [np.max(t_medians), np.mean(e_centers) if np.mean(e_centers) != 0 else 1.0]
                    popt, _ = curve_fit(exp_func, e_centers, t_medians, p0=p0, maxfev=10000)
                    
                    A_opt, lam_opt = popt
                    e_smooth = np.linspace(min(e_centers), max(e_centers), 500)
                    t_smooth = exp_func(e_smooth, A_opt, lam_opt)
                    
                    fit_eq = f"Fit: $t = {A_opt:.2e} e^{{-E / {lam_opt:.2f}}}$"
                    
                    ax2.plot(t_smooth, e_smooth, color='red', lw=2.5, label=fit_eq, zorder=4)
                except Exception as e:
                    print(f"     [WARN] Exponential fit failed for Profile 1: {e}")
            
            ax2.set_title(f"[{family_name}] Profile (Energy Binned - {n_profile_bins} pts): DRS {drs_code} $\\rightarrow$ FERS {fers_code}", fontsize=14)
            ax2.set_xlabel(r"Median $|t_{\mathrm{final}}|$ [ns]", fontsize=12)
            ax2.set_ylabel("Energy [A.U.]", fontsize=12)
            ax2.set_xlim(tmin, tmax)
            ax2.set_ylim(emin, emax)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=12, framealpha=1)
            fig2.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)

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
                    p0 = [np.max(e_medians), np.mean(t_centers) if np.mean(t_centers) != 0 else 1.0]
                    popt3, _ = curve_fit(exp_func, t_centers, e_medians, p0=p0, maxfev=10000)
                    
                    A_opt3, lam_opt3 = popt3
                    t_smooth3 = np.linspace(min(t_centers), max(t_centers), 500)
                    e_smooth3 = exp_func(t_smooth3, A_opt3, lam_opt3)
                    
                    fit_eq3 = f"Fit: $E = {A_opt3:.2e} e^{{-t / {lam_opt3:.2f}}}$"
                    
                    ax3.plot(t_smooth3, e_smooth3, color='orange', lw=2.5, label=fit_eq3, zorder=4)
                except Exception as e:
                    print(f"     [WARN] Exponential fit failed for Profile 2: {e}")
            
            ax3.set_title(f"[{family_name}] Profile (Time Binned - {n_profile_bins} pts): DRS {drs_code} $\\rightarrow$ FERS {fers_code}", fontsize=14)
            ax3.set_xlabel(r"$|t_{\mathrm{final}}|$ [ns]", fontsize=12)
            ax3.set_ylabel("Median Energy [A.U.]", fontsize=12)
            ax3.set_xlim(tmin, tmax)
            ax3.set_ylim(emin, emax)
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=12, framealpha=1)
            fig3.tight_layout()
            pdf.savefig(fig3)
            plt.close(fig3)

def make_master_correlation_pdf(files, outdir, tmin, tmax, emin, emax, nbins, particle_type=None):
    os.makedirs(outdir, exist_ok=True)
    drs_to_fers = build_channel_map()
    pid_tag = particle_type if particle_type else "NoPID"
    
    out_pdf = os.path.join(outdir, f"MASTER_CORRELATIONS_ALL_CHANNELS_{pid_tag}.pdf")
    
    families = [
        ("CER-Quartz", DRS_QUARTZ_GRID),
        ("CER-Plastic", DRS_PLASTIC_GRID),
        ("Scintillator", DRS_SCI_GRID)
    ]

    print(f"\n=========================================================")
    print(f" GENERATING MASTER PDF FOR ALL CHANNELS (PID: {pid_tag})")
    print(f" Output File: {out_pdf}")
    print(f"=========================================================\n")

    with PdfPages(out_pdf) as pdf:
        for family_name, grid in families:
            print(f"\n--- Processing Family: {family_name} ---")
            
            if family_name == "Scintillator":
                active_tmin, active_tmax = 5.0, 20.0
                active_emax, active_emin = 2000.0, 0.0
                print(f"  [INFO] Scintillator detected. Forcing ToA limits to [{active_tmin}, {active_tmax}] ns")
            else:
                active_tmin, active_tmax = tmin, tmax
                active_emax, active_emin = emax, emin
                
            for row in grid:
                for drs_code in row:
                    if drs_code is None: continue
                    fers_code = drs_to_fers.get(drs_code)
                    if not fers_code: continue
                    
                    process_and_plot_channel(
                        files, drs_code, fers_code, family_name, pdf, 
                        active_tmin, active_tmax, active_emin, active_emax, nbins, particle_type, pid_tag
                    )

    print(f"\n[DONE] Saved Master PDF to: {out_pdf}")

# ================= MAIN =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ana-files", nargs="+", required=True, help="List of input ROOT files.")
    parser.add_argument("--outdir", default="./Attenuation/FERS_Correlations", help="Output directory")
    parser.add_argument("--pid", default="electron", choices=["muon", "pion", "electron", "proton", "all"])
    parser.add_argument("--single-channel", required=False, help="3-digit DRS code (e.g., '104')")
    parser.add_argument("--tmin", type=float, default=4.0, help="Min |tfinal|")
    parser.add_argument("--tmax", type=float, default=25.0, help="Max |tfinal|")
    parser.add_argument("--emin", type=float, default=0.0, help="Min Energy for Y-axis")
    parser.add_argument("--emax", type=float, default=5000.0, help="Max Energy for Y-axis")
    parser.add_argument("--nbins", type=int, default=100)
    args = parser.parse_args()

    files = args.ana_files
    
    if args.single_channel:
        make_channel_correlation(files, args.single_channel, args.outdir, args.tmin, args.tmax, args.emin, args.emax, args.nbins, args.pid)
    else:
        make_master_correlation_pdf(files, args.outdir, args.tmin, args.tmax, args.emin, args.emax, args.nbins, args.pid)

if __name__ == "__main__":
    main()