#!/usr/bin/env python3


#working on energy now
import os
import re
import glob
import argparse
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit



'''
python attenuation_time_correlation.py --ana-files  /lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1501_250928105227_converted_timingskim.root  /lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1513_250928194230_converted_timingskim.root  /lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1507_250928160030_converted_timingskim.root   /lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1511_250928180741_converted_timingskim.root   /lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1513_250928192918_converted_timingskim.root  --outdir /lustre/research/hep/akshriva/Dream-Timing/Attenuation/Correlation/CER-Plastic/ --single-channel 110 --xmin 11.5 --xmax 14.5 --pid electron
'''
# ================= DEFAULTS =================
TREE_NAME = "EventTree"

NBINS = 200
CUT_MIN = 1.0
MIN_ENTRIES = 200
MIN_RAW = 500

HSPACE = 0.10
WSPACE = 0.05

# how many runs to print (mu,sigma) inside each cell
CELL_STATS_MAXLINES = 3

# --- WAVEFORM ROBUST SETTINGS ---
BASELINE_SAMPLES = 200
AMP_THRESHOLD = 50
SEARCH_WINDOW = (200, 800) 

# ================= PID CONFIGURATION =================
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

def get_particle_selection(particle_type: str) -> dict:
    selections = {
        "muon": {
            "TTUMuonVeto": True,
            "PSD": False,
        },
        "pion": {
            "TTUMuonVeto": False,
            "PSD": False,
            "Cer474": True, "Cer519": True, "Cer537": True
        },
        "electron": {
            "TTUMuonVeto": False,
            "PSD": True,
            "Cer474": True, "Cer519": True, "Cer537": True
        },
        "proton": {
            "TTUMuonVeto": False,
            "PSD": False,
            "Cer474": False, "Cer519": False, "Cer537": False
        },
    }
    return selections.get(particle_type.lower(), {})

def compute_pid_mask(tree, particle_type):
    requirements = get_particle_selection(particle_type)
    if not requirements:
        return None

    n_entries = tree.num_entries
    final_mask = np.ones(n_entries, dtype=bool)
    available_keys = set(tree.keys())

    print(f"  [PID] Applying selection for {particle_type} (with baseline subtraction)...")

    for det, must_fire in requirements.items():
        branch_name = PID_BRANCH_MAP.get(det)
        if not branch_name or branch_name not in available_keys:
            continue

        ts_min, ts_max, val_cut, method = get_service_drs_cut(det)

        try:
            waveforms = tree[branch_name].array(library="ak")
            
            if method == "Sum":
                baseline = ak.mean(waveforms[:, :30], axis=1)
                waveforms_blsub = waveforms - baseline
                window = waveforms_blsub[:, int(ts_min):int(ts_max)]
                window_sum = ak.sum(window, axis=1)
                window_sum_np = ak.to_numpy(window_sum)
                is_fired = window_sum_np < val_cut
            else:
                continue

            initial_count = np.sum(final_mask)
            if must_fire:
                final_mask = final_mask & is_fired
            else:
                final_mask = final_mask & (~is_fired)
            
            removed = initial_count - np.sum(final_mask)
            status = "FIRED" if must_fire else "VETOED"
            print(f"    [PID] {det:<12} ({status}): cut {val_cut:.1f} in [{ts_min}:{ts_max}] -> Removed {removed} events")

        except Exception as e:
            print(f"    [PID] CRITICAL ERROR processing {det}: {e}")
            continue

    return final_mask

# ================= HELPERS =================
def _xlabel():
    return r"$|t_{\mathrm{final}}|$ [ns]"

def _global_ylabel(fig, text="Events"):
    fig.text(0.010, 0.5, text, va="center", rotation=90)

def _tighten(fig, left=0.05, right=0.98, top=0.985, bottom=0.035, hspace=HSPACE, wspace=WSPACE):
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace)

def _parse_code(code_str):
    b = int(code_str[0])
    g = int(code_str[1])
    c = int(code_str[2])
    return b, g, c

def _branch(b, g, c):
    return f"tfinal_Board{b}_Group{g}_Channel{c}"

def _run_label(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"(run\d+_\d{11,12})", base)
    if m:
        return m.group(1)
    return os.path.splitext(base)[0]

def _resolve_files(args):
    if args.ana_files:
        files = list(args.ana_files)
    else:
        files = sorted(glob.glob(args.ana_glob))

    if args.run_min is not None and args.run_max is not None:
        keep = []
        for f in files:
            m = re.search(r"run(\d+)", os.path.basename(f))
            if not m: continue
            r = int(m.group(1))
            if args.run_min <= r <= args.run_max:
                keep.append(f)
        files = keep

    def _sort_key(p):
        b = os.path.basename(p)
        mrun = re.search(r"run(\d+)", b)
        r = int(mrun.group(1)) if mrun else 10**9
        mts = re.search(r"_(\d{11,12})(?:_|\.|$)", b)
        ts = int(mts.group(1)) if mts else 10**18
        return (r, ts, b)

    return sorted(files, key=_sort_key)


# ================= CORRELATION PLOT =================
def make_time_vs_adc_correlation(files, code_str, label, xlim, outdir,
                                 tree_name, nbins, cut_min, particle_type=None):
    os.makedirs(outdir, exist_ok=True)
    
    pid_tag = particle_type if particle_type else "NoPID"
    base_tag = f"files_n{len(files)}"
    out_pdf = os.path.join(outdir, f"CORRELATION_{code_str}_{label}_{base_tag}_{pid_tag}.pdf")

    b, g, ch = _parse_code(code_str)
    t_branch = _branch(b, g, ch)
    wf_branch = f"DRS_Board{b}_Group{g}_Channel{ch}"

    print(f"\n--- Processing Correlation Plot for Channel {code_str} ({pid_tag}) ---")
    
    all_t_valid = []
    all_amp_valid = []
    
    with PdfPages(out_pdf) as pdf:
        for fpath in files:
            rl = _run_label(fpath)
            try:
                uf = uproot.open(fpath)
                tree = uf[tree_name]
                keys = set(tree.keys())
            except Exception as e:
                print(f"[WARN] failed to open {fpath}: {e}")
                continue
                
            if t_branch not in keys or wf_branch not in keys:
                print(f"[WARN] Branches missing in {rl}. Skipping.")
                continue

            # 1. PID Mask
            pid_mask = None
            if particle_type:
                pid_mask = compute_pid_mask(tree, particle_type)

            # 2. Extract Data
            t_arr = tree[t_branch].array(library="np")
            wf_ak = tree[wf_branch].array(library="ak")
            
            # 3. Apply PID Mask to both
            if pid_mask is not None:
                if len(t_arr) == len(pid_mask):
                    t_arr = t_arr[pid_mask]
                    wf_ak = wf_ak[pid_mask]
                else:
                    print(f"[WARN] Shape mismatch in {rl}.")
                    continue
            
            if len(wf_ak) == 0: continue

            # 4. Calculate Max ADC (Baseline subtracted)
            baseline_ak = ak.mean(wf_ak[:, :BASELINE_SAMPLES], axis=1)
            baseline_np = ak.to_numpy(baseline_ak)
            
            max_len = ak.max(ak.num(wf_ak))
            wf_padded = ak.fill_none(ak.pad_none(wf_ak, max_len, axis=1), 0)
            wf_np = ak.to_numpy(wf_padded)
            
            wf_bs = wf_np - baseline_np[:, np.newaxis]
            window_wf = wf_bs[:, SEARCH_WINDOW[0]:SEARCH_WINDOW[1]]
            amps_bs = np.max(window_wf, axis=1)
            
            # 5. Apply event-by-event filters (time limits + amplitude threshold)
            t_abs = np.abs(t_arr)
            valid_mask = (
                ~np.isnan(t_abs) & 
                (t_abs >= cut_min) & 
                (t_abs >= xlim[0]) & 
                (t_abs <= xlim[1]) & 
                (amps_bs > AMP_THRESHOLD)
            )
            
            t_valid = t_abs[valid_mask]
            amp_valid = amps_bs[valid_mask]
            
            # Append to master lists for the combined plot
            all_t_valid.extend(t_valid)
            all_amp_valid.extend(amp_valid)

            # --- Plot Individual Run ---
            if len(t_valid) > 0:
                fig, ax = plt.subplots(figsize=(9, 7))
                
                # 2D Histogram for density
                h2 = ax.hist2d(t_valid, amp_valid, bins=[nbins, 100], 
                               range=[[xlim[0], xlim[1]], [0, 2000]], cmap='viridis', cmin=1)
                fig.colorbar(h2[3], ax=ax, label='Events')
                
                # Calculate Correlation
                if len(t_valid) > 1:
                    corr_matrix = np.corrcoef(t_valid, amp_valid)
                    corr_val = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
                else:
                    corr_val = 0.0
                
                # Add Correlation to Legend
                corr_patch = mpatches.Patch(color='none', label=f"Correlation (r): {corr_val:.3f}")
                ax.legend(handles=[corr_patch], loc='upper right', frameon=True, fontsize=10)
                
                ax.set_title(f"Time vs ADC Max: Channel {code_str}\nRun: {rl} (PID: {pid_tag})")
                ax.set_xlabel(_xlabel())
                ax.set_ylabel("Max Amplitude [ADC]")
                
                ax.minorticks_on()
                ax.tick_params(axis='both', which='major', length=8, width=1.2)
                ax.tick_params(axis='both', which='minor', length=4, width=1.0)
                ax.grid(True, which='major', alpha=0.3)
                
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                print(f"  Mapped {len(t_valid)} valid events for {rl}")

        # --- Plot ALL RUNS COMBINED ---
        if len(all_t_valid) > 0:
            print(f"\nGenerating Combined Correlation Plot with {len(all_t_valid)} total events...")
            fig, ax = plt.subplots(figsize=(10, 8))
            
            h2 = ax.hist2d(all_t_valid, all_amp_valid, bins=[nbins, 100], 
                           range=[[xlim[0], xlim[1]], [0, 2000]], cmap='turbo', cmin=1)
            fig.colorbar(h2[3], ax=ax, label='Total Events')
            
            # Calculate Correlation for Combined Data
            if len(all_t_valid) > 1:
                corr_matrix = np.corrcoef(all_t_valid, all_amp_valid)
                corr_val = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
            else:
                corr_val = 0.0
            
            # Add Correlation to Legend
            corr_patch = mpatches.Patch(color='none', label=f"Overall Correlation (r): {corr_val:.3f}")
            ax.legend(handles=[corr_patch], loc='upper right', frameon=True, fontsize=12)
            
            ax.set_title(f"COMBINED Time vs ADC Max: Channel {code_str}\nAll Runs (PID: {pid_tag})", fontsize=14)
            ax.set_xlabel(_xlabel(), fontsize=12)
            ax.set_ylabel("Max Amplitude [ADC]", fontsize=12)
            
            ax.minorticks_on()
            ax.tick_params(axis='both', which='major', length=8, width=1.2)
            ax.tick_params(axis='both', which='minor', length=4, width=1.0)
            ax.grid(True, which='major', alpha=0.3)
            
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved Correlation Plots to: {out_pdf}")

# ================= MAIN =================
def main():
    global NBINS, CUT_MIN, MIN_ENTRIES, MIN_RAW, HSPACE, WSPACE, CELL_STATS_MAXLINES

    ap = argparse.ArgumentParser()
    ap.add_argument("--ana-files", nargs="+", default=None,
                    help="Explicit list of input ROOT files.")
    ap.add_argument("--ana-glob", default=None,
                    help="Glob for input ROOT files.")

    ap.add_argument("--run-min", type=int, default=None, help="Keep only runs >= run-min")
    ap.add_argument("--run-max", type=int, default=None, help="Keep only runs <= run-max")

    ap.add_argument("--tree", default=TREE_NAME, help="Tree name")
    ap.add_argument("--outdir", default="./TRUE-HGtiming/calibration_studiesZ/overlay_runs_1499_1501",
                    help="Output directory")

    ap.add_argument("--xmin", type=float, default=4.0, help="Min |tfinal|")
    ap.add_argument("--xmax", type=float, default=25.0, help="Max |tfinal|")
    ap.add_argument("--nbins", type=int, default=NBINS, help="Histogram bins")
    ap.add_argument("--cut-min", type=float, default=CUT_MIN, help="Ignore |tfinal| < cut-min")
    ap.add_argument("--min-entries", type=int, default=MIN_ENTRIES, help="Min entries after cuts")
    ap.add_argument("--min-raw", type=int, default=MIN_RAW, help="Min raw entries before cuts")

    ap.add_argument("--single-channel", default="104",
                    help="3-digit code bgc to make a standalone overlay plot for (default: 104).")

    # PID ARGUMENT
    ap.add_argument("--pid", default=None, choices=["muon", "pion", "electron", "proton"],
                    help="Apply PID selection (muon, pion, electron, proton). Default: None.")

    args = ap.parse_args()

    if args.ana_files is None and args.ana_glob is None:
        raise SystemExit("ERROR: provide either --ana-files or --ana-glob")

    NBINS = args.nbins
    CUT_MIN = args.cut_min
    MIN_ENTRIES = args.min_entries
    MIN_RAW = args.min_raw

    files = _resolve_files(args)
    if len(files) == 0:
        raise SystemExit("ERROR: no files matched your selection")

    print(f"Found {len(files)} files.")
    for f in files:
        print("  ", os.path.basename(f))

    xlim = (args.xmin, args.xmax)
    os.makedirs(args.outdir, exist_ok=True)
    
    pid_label = f"PID_{args.pid}" if args.pid else "AllParticles"

    # --- CALL THE NEW CORRELATION FUNCTION ---
    make_time_vs_adc_correlation(files, args.single_channel, f"ALL-RUNS_{pid_label}", xlim, args.outdir,
                                 args.tree, NBINS, CUT_MIN, args.pid)

    print("All done.")

if __name__ == "__main__":
    main()
