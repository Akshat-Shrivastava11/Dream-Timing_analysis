#!/usr/bin/env python3
import os
import re
import glob
import argparse
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import concurrent.futures

# =========================================================
# CMS STYLING VIA MPLHEP
# =========================================================
try:
    import mplhep as hep
    plt.style.use(hep.style.CMS)
except ImportError:
    pass

# =========================================================
# BASIC CONFIG
# =========================================================
TREE_NAME = "EventTree"
TIME_PER_BIN_NS = 0.2
N_EVENTS_PER_COMBO = 10
BASELINE_BINS = 30
TIMING_SUFFIX = "_LP2_50"

# =========================================================
# PID & CHANNEL CONFIGURATION
# =========================================================
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

def get_service_drs_cut(service_drs: str):
    cuts = {
        "HoleVeto":    (100, 350, -2e3,   "Sum"),
        "PSD":         (100, 400, -3500., "Sum"),
        "TTUMuonVeto": (200, 400, -2e3,   "Sum"),
        "Cer474":      (800, 900, -2000., "Sum"),
        "Cer519":      (450, 550, -1000., "Sum"),
        "Cer537":      (400, 500, -500.,  "Sum"),
    }
    return cuts.get(service_drs, (0, 1000, -5e4, "Sum"))

def get_particle_selection(particle_type: str):
    selections = {
        "muon": {"TTUMuonVeto": True, "PSD": False},
        "pion": {"TTUMuonVeto": False, "PSD": False, "Cer474": True, "Cer519": True, "Cer537": True},
        "electron": {"TTUMuonVeto": False, "PSD": True, "Cer474": True, "Cer519": True, "Cer537": True},
        "electron_90deg": {"TTUMuonVeto": False, "PSD": True, "Cer474": True, "Cer519": True, "Cer537": True},
    }
    return selections.get(particle_type.lower(), {})

def get_display_name(particle):
    if particle.lower() == "electron": return "Positron"
    if particle.lower() == "electron_90deg": return "Positron (90°)"
    return particle.capitalize()

# CHANNELS
CHANNELS_3MM = {"Quartz": "104", "Plastic": "010", "Scintillator": "107"}
CHANNELS_6MM = {"Quartz": "604", "Plastic": "606", "Scintillator": "615"}
MCP1_CHANNEL = "037" # B0G3C7

RUN_FILES = {
    "3mm": {
        "pion": ["/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1429_250926183919_converted_timingskim.root"],
        "muon": ["/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1480_250928004120_converted_timingskim.root"],
        "electron": ["/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1355_250924165834_converted_timingskim.root"],
        "electron_90deg": ["/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1501_250928105227_converted_timingskim.root"],
    },
    "6mm": {
        "pion": ["/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1474_250927193729_converted_timingskim.root"],
        "muon": [],
        "electron": ["/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1509_250928164817_converted_timingskim.root"],
    },
}

# =========================================================
# HELPERS
# =========================================================
def expand_files(file_patterns):
    out = []
    for item in file_patterns:
        matches = sorted(glob.glob(item))
        if matches: out.extend(matches)
        elif os.path.exists(item): out.append(item)
    return list(dict.fromkeys(out))

def parse_code(code_str):
    code = re.sub(r"[^0-9]", "", code_str)[:3]
    return int(code[0]), int(code[1]), int(code[2]), code

def get_branch_name(code):
    b, g, c, _ = parse_code(code)
    return f"DRS_Board{b}_Group{g}_Channel{c}"

def run_label(path):
    m = re.search(r"(run\d+)", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]

def compute_pid_mask(tree, particle_type):
    requirements = get_particle_selection(particle_type)
    if not requirements: return np.ones(tree.num_entries, dtype=bool)

    final_mask = np.ones(tree.num_entries, dtype=bool)
    for det, must_fire in requirements.items():
        branch_name = PID_BRANCH_MAP.get(det)
        if branch_name not in tree.keys(): continue

        ts_min, ts_max, val_cut, method = get_service_drs_cut(det)
        if method != "Sum": continue

        try:
            waveforms = tree[branch_name].array(library="ak")
            baseline = ak.mean(waveforms[:, :BASELINE_BINS], axis=1)
            window_sum = ak.sum((waveforms - baseline)[:, int(ts_min):int(ts_max)], axis=1)
            is_fired = ak.to_numpy(window_sum) < val_cut
            final_mask &= is_fired if must_fire else (~is_fired)
        except Exception:
            continue
    return final_mask

# =========================================================
# PLOTTING & HISTOGRAMS
# =========================================================
def style_paper_axes(ax, xlabel, ylabel, run_name, ev, source_name, particle_type, thickness):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", labelsize=14, length=8, direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor", length=4, direction="in", top=True, right=True)
    
    display_name = get_display_name(particle_type)
    r_label = f"40 GeV {display_name} | {thickness}"
    l_label = f"Run {run_name} | Ev {ev} | {source_name}"
    
    try:
        hep.cms.label(ax=ax, exp="CaloX", data=False, llabel=l_label, rlabel=r_label)
    except:
        ax.set_title(f"{l_label}  ---  {r_label}", fontsize=12)

def plot_single_panel(ax, tree, ev, branch_code, source_name, run_name, color, particle, thickness):
    raw_branch = get_branch_name(branch_code)
    timing_branch = f"{raw_branch}{TIMING_SUFFIX}"
    
    if raw_branch not in tree.keys():
        ax.text(0.5, 0.5, f"Missing {source_name}", ha="center", va="center")
        return
        
    arr = tree[raw_branch].array(entry_start=ev, entry_stop=ev + 1, library="np")
    if len(arr) == 0:
        return
        
    w = arr[0]
    # MULTIPLY ALL WAVEFORMS BY -1
    w_sub = -(w - np.mean(w[:BASELINE_BINS]))
    t = np.arange(len(w)) * TIME_PER_BIN_NS
    
    t_50 = np.nan
    if timing_branch in tree.keys():
        t_arr = tree[timing_branch].array(entry_start=ev, entry_stop=ev + 1, library="np")
        if len(t_arr) > 0 and np.isfinite(t_arr[0]):
            t_50 = t_arr[0]
            
    # Shift time axis so t_50 is at 0
    if np.isfinite(t_50):
        t_shifted = t - t_50
        ax.plot(t_shifted, w_sub, lw=2.5, color=color)
        ax.axvline(0, color="black", ls="--", lw=2.0, alpha=0.8)
        ax.set_xlim(-15, 25)
    else:
        ax.plot(t, w_sub, lw=2.5, color=color)

    style_paper_axes(ax, r"Time $t - t_{50}$ [ns]", "Amplitude [ADC counts]", run_name, ev, source_name, particle, thickness)

def plot_t50_histograms(thickness, particle, all_t50s, panels_config, outdir):
    hist_outname = os.path.join(outdir, f"T50_Histograms_{thickness}_{particle}.pdf")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    modes = {}
    for ax, (source_name, _, color) in zip(axes, panels_config):
        data = np.array(all_t50s[source_name])
        
        display_name = get_display_name(particle)
        try:
            hep.cms.label(ax=ax, exp="CaloX", data=False, llabel=f"{source_name} $t_{{50}}$", rlabel=f"40 GeV {display_name} | {thickness}")
        except:
            ax.set_title(f"{source_name} $t_{{50}}$ | {thickness} {display_name}")
            
        if len(data) == 0:
            modes[source_name] = 100.0 # fallback
            continue
            
        hist, bin_edges = np.histogram(data, bins=100)
        mode_val = bin_edges[np.argmax(hist)] + (bin_edges[1] - bin_edges[0]) / 2.0
        modes[source_name] = mode_val
        
        ax.hist(data, bins=100, color=color, alpha=0.7, edgecolor='black', linewidth=1.2)
        ax.axvline(mode_val, color='black', linestyle='dashed', linewidth=2.5, label=f'Mode = {mode_val:.2f} ns')
        ax.set_xlabel(r"$t_{50}$ [ns]", fontsize=16)
        ax.set_ylabel("Counts", fontsize=16)
        ax.legend(fontsize=14, frameon=False)

    fig.tight_layout()
    fig.savefig(hist_outname)
    plt.close(fig)
    print(f"[OK] Wrote $t_{{50}}$ histograms: {hist_outname}")
    return modes

# =========================================================
# CORE PROCESSING
# =========================================================
def process_combination(thickness, particle, files, outdir, n_events):
    if not files: return f"[SKIP] No files for {thickness} | {particle}"

    channel_map = CHANNELS_3MM if thickness == "3mm" else CHANNELS_6MM
    os.makedirs(outdir, exist_ok=True)
    
    panels_config = [
        ("MCP1", MCP1_CHANNEL, "tab:green"),
        ("Plastic", channel_map["Plastic"], "tab:blue"),
        ("Quartz", channel_map["Quartz"], "tab:orange"),
        ("Scintillator", channel_map["Scintillator"], "tab:red")
    ]

    # --- Pass 1: Build independent histograms to find the true Mode per channel ---
    all_t50s = {src: [] for src, _, _ in panels_config}
    for fpath in files:
        try:
            with uproot.open(fpath) as f:
                tree = f[TREE_NAME]
                pid_mask = compute_pid_mask(tree, particle)
                for source_name, branch_code, _ in panels_config:
                    br = get_branch_name(branch_code)
                    if br not in tree.keys(): continue
                    
                    waves = tree[br].array(library="ak")
                    baseline = ak.mean(waves[:, :BASELINE_BINS], axis=1)
                    
                    # ALL WFS TIMES -1
                    w_sub = -(waves - baseline)
                    
                    # Apply ADC limits to inverted pulse (Min > -100, Max > 100)
                    w_min = ak.min(w_sub, axis=1)
                    w_max = ak.max(w_sub, axis=1)
                    adc_mask = ak.to_numpy((w_min > -100) & (w_max > 100))
                    
                    t50_br = f"{br}{TIMING_SUFFIX}"
                    if t50_br in tree.keys():
                        t50_arr = tree[t50_br].array(library="np")
                        valid_mask = pid_mask & adc_mask & np.isfinite(t50_arr) & (t50_arr > 0)
                        all_t50s[source_name].extend(t50_arr[valid_mask])
        except Exception as exc:
            pass

    modes = plot_t50_histograms(thickness, particle, all_t50s, panels_config, outdir)
    
    # --- Pass 2: Select events sitting directly on the modes ---
    candidate_events = [] 
    for fpath in files:
        try:
            with uproot.open(fpath) as f:
                tree = f[TREE_NAME]
                pid_mask = compute_pid_mask(tree, particle)
                combined_mask = pid_mask.copy()
                n_entries = tree.num_entries
                diff_sum = np.zeros(n_entries)
                
                # Demand that ALL 4 channels pass ADC cut and have t50s near the mode
                for source_name, branch_code, _ in panels_config:
                    br = get_branch_name(branch_code)
                    if br not in tree.keys() or f"{br}{TIMING_SUFFIX}" not in tree.keys():
                        combined_mask[:] = False; break
                        
                    waves = tree[br].array(library="ak")
                    baseline = ak.mean(waves[:, :BASELINE_BINS], axis=1)
                    
                    # ALL WFS TIMES -1
                    w_sub = -(waves - baseline)
                    
                    # Enforce ADC cuts
                    w_min = ak.min(w_sub, axis=1)
                    w_max = ak.max(w_sub, axis=1)
                    combined_mask &= ak.to_numpy((w_min > -100) & (w_max > 100))
                    
                    t50_arr = tree[f"{br}{TIMING_SUFFIX}"].array(library="np")
                    combined_mask &= np.isfinite(t50_arr) & (t50_arr > 0)
                    
                    # Add to metric: absolute difference from this channel's mode
                    diff_sum += np.abs(t50_arr - modes[source_name])
                    
                valid_idx = np.where(combined_mask)[0]
                for idx in valid_idx:
                    candidate_events.append((fpath, idx, diff_sum[idx]))
        except Exception:
            pass

    # Sort events by how closely ALL 4 channels match their modes simultaneously
    candidate_events.sort(key=lambda x: x[2])
    events_to_plot = candidate_events[:n_events]

    if not events_to_plot:
        return f"[WARN] No overlapping events found for {thickness} | {particle}."

    # --- Pass 3: Plot the highly optimized waveforms ---
    wave_outname = os.path.join(outdir, f"Waveforms_AllSensors_{thickness}_{particle}_{n_events}events.pdf")
    with PdfPages(wave_outname) as pdf:
        for fpath, ev, _ in events_to_plot:
            rl = run_label(fpath)
            try:
                with uproot.open(fpath) as f:
                    tree = f[TREE_NAME]
                    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                    axes = axes.flatten()
                    
                    for ax, (source_name, branch_code, color) in zip(axes, panels_config):
                        plot_single_panel(ax, tree, ev, branch_code, source_name, rl, color, particle, thickness)
                    
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
            except Exception as exc:
                print(f"[WARN] Failed plotting event {ev} from {fpath}: {exc}")

    return f"[OK] Completed Waveforms for {thickness} {particle} -> {wave_outname}"

# =========================================================
# DRIVER
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./paper_waveform_plots", help="Output directory")
    ap.add_argument("--n-events", type=int, default=N_EVENTS_PER_COMBO, help="Number of events per combo")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for thickness in RUN_FILES.keys():
            for particle in RUN_FILES[thickness].keys():
                files = expand_files(RUN_FILES[thickness][particle])
                if len(files) == 0: continue
                
                subdir = os.path.join(args.outdir, thickness, particle)
                future = executor.submit(process_combination, thickness, particle, files, subdir, args.n_events)
                futures.append(future)

        for f in concurrent.futures.as_completed(futures):
            print(f.result())

    print("\nAll requested waveform PDFs and Histograms are done.")

if __name__ == "__main__":
    main()