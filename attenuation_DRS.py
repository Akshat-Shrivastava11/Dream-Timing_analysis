#!/usr/bin/env python3
import os
import re
import random
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ================= USER CONFIGURATION =================
FAMILIES = {
    "CER-Quartz": [
        "DRS_Board1_Group0_Channel4", 
        "DRS_Board1_Group0_Channel6"
    ],
    "CER-Plastic": [
        "DRS_Board1_Group0_Channel0",
        "DRS_Board1_Group1_Channel0",
        "DRS_Board1_Group0_Channel2",
        "DRS_Board1_Group1_Channel2"
    ],
    "SCI": [
        "DRS_Board1_Group0_Channel7",
        "DRS_Board1_Group1_Channel3",
        "DRS_Board1_Group1_Channel1"
    ]
}

PID_BRANCH_MAP = {
    "PSD": "DRS_Board7_Group1_Channel1",
    "HoleVeto": "DRS_Board7_Group1_Channel6",
    "TTUMuonVeto": "DRS_Board7_Group2_Channel4",
    "Cer474": "DRS_Board7_Group2_Channel5",
    "Cer519": "DRS_Board7_Group2_Channel6",
    "Cer537": "DRS_Board7_Group2_Channel7",
}

FILE_LIST = [
    "/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1501_250928105227_converted_timingskim.root",
    "/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1513_250928194230_converted_timingskim.root",
    "/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1507_250928160030_converted_timingskim.root",
    "/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1511_250928180741_converted_timingskim.root",
    "/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1513_250928192918_converted_timingskim.root"
]

TREE_NAME = "EventTree"
OUT_DIR = "./Attenuation/AmpvsZ_Plots"
DIAG_DIR = "/lustre/research/hep/akshriva/Dream-Timing/Attenuation/Diagnostics"

# --- ROBUST SETTINGS ---
BASELINE_SAMPLES = 200
AMP_THRESHOLD = 100 
SEARCH_WINDOW = (400, 600) 
PARTICLE_TYPE = "electron"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)

# ================= PID HELPERS =================
def get_service_drs_cut(service_drs):
    cuts = {
        "HoleVeto": (100, 350, -2e3),
        "PSD": (100, 400, -3500.0),
        "TTUMuonVeto": (200, 400, -2e3),
        "Cer474": (800, 900, -2000.0),
        "Cer519": (450, 550, -1000.0),
        "Cer537": (400, 500, -500.0),
    }
    return cuts.get(service_drs, (0, 1000, -5e4))

def get_particle_selection(particle_type):
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
    for det, must_fire in requirements.items():
        branch = PID_BRANCH_MAP.get(det)
        if branch not in tree.keys(): continue
        ts_min, ts_max, val_cut = get_service_drs_cut(det)
        wf = tree[branch].array(library="ak")
        bl = ak.mean(wf[:, :30], axis=1)
        integral = ak.sum(wf[:, int(ts_min):int(ts_max)] - bl, axis=1)
        is_fired = ak.to_numpy(integral < val_cut)
        final_mask = final_mask & (is_fired if must_fire else ~is_fired)
    return final_mask

# ================= GEOMETRY HELPER =================
def get_z_position(run_label):
    if "run1513" in run_label:
        if "192918" in run_label: return -54.5
        if "194230" in run_label: return -400.3
    match = re.search(r"run(\d+)", run_label)
    run_num = int(match.group(1)) if match else None
    z_map = {1501: -168.0, 1507: -218.0, 1511: -268.0}
    return z_map.get(run_num, -999)

# ================= MAIN LOGIC =================
def main():
    plot_data = {fam: {ch: {"z": [], "amp": []} for ch in channels} for fam, channels in FAMILIES.items()}

    for fpath in FILE_LIST:
        run_label = os.path.basename(fpath)
        z_val = get_z_position(run_label)
        if z_val == -999: continue 
        
        print(f"--- Processing {run_label} (Z={z_val}) ---")
        try:
            tree = uproot.open(f"{fpath}:{TREE_NAME}")
        except: continue
            
        pid_mask = compute_pid_mask(tree, PARTICLE_TYPE)

        for fam, channels in FAMILIES.items():
            for ch in channels:
                if ch not in tree.keys(): continue
                
                wf_ak = tree[ch].array(library="ak")[pid_mask]
                if len(wf_ak) == 0: continue

                baseline_ak = ak.mean(wf_ak[:, :BASELINE_SAMPLES], axis=1)
                baseline_np = ak.to_numpy(baseline_ak)
                
                max_len = ak.max(ak.num(wf_ak))
                wf_padded = ak.fill_none(ak.pad_none(wf_ak, max_len, axis=1), 0)
                wf_np = ak.to_numpy(wf_padded)
                
                wf_bs = wf_np - baseline_np[:, np.newaxis]
                window_wf = wf_bs[:, SEARCH_WINDOW[0]:SEARCH_WINDOW[1]]
                
                local_max_idx = np.argmax(window_wf, axis=1)
                global_max_idx = local_max_idx + SEARCH_WINDOW[0]
                amps_bs = np.max(window_wf, axis=1)
                
                cut_mask = (amps_bs > AMP_THRESHOLD)
                final_amps = amps_bs[cut_mask]
                final_indices = global_max_idx[cut_mask]
                final_wf_np = wf_bs[cut_mask]

                if len(final_amps) > 0:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(final_amps, bins=50, range=(0, 2000), histtype='step', color='navy')
                    ax.set_title(f"ADC Max (>150): {fam} - {ch}\nRun: {run_label}")
                    ax.set_xlabel("Amplitude [ADC]")
                    ax.set_yscale('log')
                    ax.minorticks_on()
                    ax.tick_params(axis='both', which='major', length=7, width=1.2)
                    ax.tick_params(axis='both', which='minor', length=4, width=1.0)
                    plt.savefig(f"{DIAG_DIR}/Hist_{fam}_{ch}_{run_label}.png")
                    plt.close()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    samples = random.sample(range(len(final_wf_np)), min(10, len(final_wf_np)))
                    for s_idx in samples:
                        ax.plot(final_wf_np[s_idx], alpha=0.5)
                        ax.scatter(final_indices[s_idx], final_amps[s_idx], color='red', s=30)
                    
                    ax.axhline(0, color='black', lw=1, ls='--')
                    ax.axvline(SEARCH_WINDOW[0], color='green', alpha=0.2)
                    ax.axvline(SEARCH_WINDOW[1], color='green', alpha=0.2)
                    ax.set_title(f"Fixed Waveforms: {fam} - {ch}\nRun: {run_label}")
                    ax.minorticks_on()
                    ax.tick_params(axis='both', which='major', length=7, width=1.2)
                    ax.tick_params(axis='both', which='minor', length=4, width=1.0)
                    plt.savefig(f"{DIAG_DIR}/Waves_{fam}_{ch}_{run_label}.png")
                    plt.close()

                avg_amp = np.nanmean(final_amps) if len(final_amps) > 0 else np.nan
                if not np.isnan(avg_amp):
                    plot_data[fam][ch]["z"].append(z_val)
                    plot_data[fam][ch]["amp"].append(avg_amp)

    # ================= PLOTTING INDIVIDUAL Z-SCANS =================
    print("\nGenerating Individual Attenuation Plots...")
    for fam, channels_data in plot_data.items():
        fig, ax = plt.subplots(figsize=(10, 7))
        
        all_z = []
        all_amp = []
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(channels_data)))
        
        has_data = False
        for (ch, data), color in zip(channels_data.items(), colors):
            if not data["z"]: continue
            has_data = True
            
            short_ch = ch.replace("DRS_Board", "B").replace("_Group", "G").replace("_Channel", "C")
            
            ax.plot(data["z"], data["amp"], marker='o', linestyle='', color=color, markersize=8, alpha=0.7, label=short_ch)
            all_z.extend(data["z"])
            all_amp.extend(data["amp"])
            
        if not has_data: continue

        if len(all_z) > 2:
            coeffs = np.polyfit(all_z, all_amp, 1)
            poly = np.poly1d(coeffs)
            
            z_fit = np.linspace(min(all_z), max(all_z), 100)
            ax.plot(z_fit, poly(z_fit), 'r--', linewidth=2, label=f"Linear Fit: {coeffs[0]:.2f}x + {coeffs[1]:.2f}")
        
        ax.set_title(f"Attenuation Scan: {fam} (All Channels)\nSelection: {PARTICLE_TYPE}")
        ax.set_xlabel("Z Position [mm]")
        ax.set_ylabel("Average Amplitude [ADC]")
        
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', length=8, width=1.2, labelsize=10)
        ax.tick_params(axis='both', which='minor', length=4, width=1.0)
        ax.grid(True, which='major', alpha=0.4)
        ax.grid(True, which='minor', alpha=0.1, linestyle=':')
        
        ax.legend()
        plt.savefig(f"{OUT_DIR}/ZScan_{fam}_{PARTICLE_TYPE}.pdf", bbox_inches='tight')
        plt.close()

    # ================= COMBINED Z-SCAN PLOT =================
    print("\nGenerating Combined Attenuation Plot...")
    fig_comb, ax_comb = plt.subplots(figsize=(10, 7))
    
    # Assign one distinct color per family
    family_colors = plt.cm.Set1(np.linspace(0, 1, len(plot_data)))
    
    for (fam, channels_data), color in zip(plot_data.items(), family_colors):
        fam_all_z = []
        fam_all_amp = []
        
        # Aggregate all points across channels for this specific family
        for ch, data in channels_data.items():
            fam_all_z.extend(data["z"])
            fam_all_amp.extend(data["amp"])
            
        if not fam_all_z: continue
        
        # Scatter all points for the family
        ax_comb.plot(fam_all_z, fam_all_amp, marker='o', linestyle='', color=color, markersize=7, alpha=0.6, label=f"{fam} Data")
        
        # Fit and plot a single line for the family
        if len(fam_all_z) > 2:
            coeffs = np.polyfit(fam_all_z, fam_all_amp, 1)
            poly = np.poly1d(coeffs)
            
            z_fit = np.linspace(min(fam_all_z), max(fam_all_z), 100)
            ax_comb.plot(z_fit, poly(z_fit), linestyle='--', color=color, linewidth=2, label=f"{fam} Fit: {coeffs[0]:.2f}x + {coeffs[1]:.2f}")

    ax_comb.set_title(f"Combined Attenuation Scan (All Families)\nSelection: {PARTICLE_TYPE}")
    ax_comb.set_xlabel("Z Position [mm]")
    ax_comb.set_ylabel("Average Amplitude [ADC]")
    
    ax_comb.minorticks_on()
    ax_comb.tick_params(axis='both', which='major', length=8, width=1.2, labelsize=10)
    ax_comb.tick_params(axis='both', which='minor', length=4, width=1.0)
    ax_comb.grid(True, which='major', alpha=0.4)
    ax_comb.grid(True, which='minor', alpha=0.1, linestyle=':')
    
    # Put legend outside the plot so it doesn't cover up the data
    ax_comb.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.savefig(f"{OUT_DIR}/ZScan_COMBINED_{PARTICLE_TYPE}.pdf", bbox_inches='tight')
    plt.close()

    print("Done!")

if __name__ == "__main__":
    main()