#!/usr/bin/env python3
import os
import re
import glob
import argparse
import numpy as np
import uproot
import awkward as ak
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mplhep as hep

# Apply the CMS style globally
plt.style.use(hep.style.CMS)

# ================= CONFIGURATION =================
TREE_NAME = "EventTree"
AMP_THRESHOLD = 100.0  
MIN_ADC_CUT = -100.0

FAMILIES = {
    "Plastic": {"channels": ["100","102","112", "110"], "tmin": -14.5, "tmax": -11.5, "legend": "Cherenkov-Plastic", "color": "red"},
    "Quartz":  {"channels": ["104","106", "304","114"], "tmin": -15.0, "tmax": -11.5, "legend": "Cherenkov-Quartz",  "color": "blue"},
    "SCI":     {"channels": ["105", "107","111","117"], "tmin": -13.5, "tmax":  -9.5, "legend": "Scintillating",     "color": "green"}
}

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

# ================= PID & ADC MASKS =================
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
    available_keys = set(tree.keys())

    for det, must_fire in requirements.items():
        branch_name = PID_BRANCH_MAP.get(det)
        if not branch_name or branch_name not in available_keys: continue
        ts_min, ts_max, val_cut, method = get_service_drs_cut(det)

        try:
            waveforms = tree[branch_name].array(library="ak")
            if method == "Sum":
                baseline = ak.mean(waveforms[:, :30], axis=1)
                waveforms_blsub = waveforms - baseline
                window_sum = ak.sum(waveforms_blsub[:, int(ts_min):int(ts_max)], axis=1)
                is_fired = ak.to_numpy(window_sum) < val_cut
            else:
                continue

            final_mask = final_mask & is_fired if must_fire else final_mask & (~is_fired)
        except Exception:
            continue
    return final_mask

def compute_adc_mask(tree, code_str):
    b, g, c = int(code_str[0]), int(code_str[1]), int(code_str[2])
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

# ================= CORE TIMING STRATEGIES =================
def get_tfinal_3mm_baseline(tree, b, g, c, suffix):
    """
    BASELINE Strategy: Single MCP Trigger (Channel 7)
    """
    br_sig     = f"DRS_Board{b}_Group{g}_Channel{c}{suffix}"
    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8{suffix}"
    br_trg     = f"DRS_Board0_Group3_Channel7{suffix}"
    br_trg_ref = f"DRS_Board0_Group3_Channel8{suffix}"
    
    keys = tree.keys()
    if any(br not in keys for br in [br_sig, br_sig_ref, br_trg, br_trg_ref]):
        return None
            
    arr_sig     = tree[br_sig].array(library="np")
    arr_sig_ref = tree[br_sig_ref].array(library="np")
    arr_trg     = tree[br_trg].array(library="np")
    arr_trg_ref = tree[br_trg_ref].array(library="np")
        
    return (arr_sig - arr_sig_ref) - (arr_trg - arr_trg_ref)

def get_tfinal_3mm_mcp6(tree, b, g, c, suffix):
    """
    SECONDARY Strategy: Single MCP Trigger (Channel 6)
    """
    br_sig     = f"DRS_Board{b}_Group{g}_Channel{c}{suffix}"
    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8{suffix}"
    br_trg     = f"DRS_Board0_Group3_Channel6{suffix}"
    br_trg_ref = f"DRS_Board0_Group3_Channel8{suffix}"
    
    keys = tree.keys()
    if any(br not in keys for br in [br_sig, br_sig_ref, br_trg, br_trg_ref]):
        return None
            
    arr_sig     = tree[br_sig].array(library="np")
    arr_sig_ref = tree[br_sig_ref].array(library="np")
    arr_trg     = tree[br_trg].array(library="np")
    arr_trg_ref = tree[br_trg_ref].array(library="np")
        
    return (arr_sig - arr_sig_ref) - (arr_trg - arr_trg_ref)

def get_tfinal_3mm_avg(tree, b, g, c, suffix):
    """
    NEW STRATEGY: Average of two MCP Triggers (Channel 6 and Channel 7)
    """
    br_sig     = f"DRS_Board{b}_Group{g}_Channel{c}{suffix}"
    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8{suffix}"
    
    br_mcp1    = f"DRS_Board0_Group3_Channel6{suffix}" 
    br_mcp2    = f"DRS_Board0_Group3_Channel7{suffix}"
    
    br_trg_ref = f"DRS_Board0_Group3_Channel8{suffix}"
    
    keys = tree.keys()
    if any(br not in keys for br in [br_sig, br_sig_ref, br_mcp1, br_mcp2, br_trg_ref]):
        return None
            
    arr_sig     = tree[br_sig].array(library="np")
    arr_sig_ref = tree[br_sig_ref].array(library="np")
    arr_mcp1    = tree[br_mcp1].array(library="np")
    arr_mcp2    = tree[br_mcp2].array(library="np")
    arr_trg_ref = tree[br_trg_ref].array(library="np")
    
    mcp_avg = (arr_mcp1 + arr_mcp2) / 2.0
        
    return (arr_sig - arr_sig_ref) - (mcp_avg - arr_trg_ref)

# ================= MATH HELPERS =================
def gaussian_peak_1(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def _mode_from_hist(arr, bins):
    h, _ = np.histogram(arr, bins=bins)
    if h.sum() == 0: return (np.nan, 0, h)
    idx = int(np.argmax(h))
    centers = 0.5 * (bins[1:] + bins[:-1])
    return float(centers[idx])

def _run_label(path: str) -> str:
    m = re.search(r"(run\d+_\d{11,12})", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]

# ================= STRATEGY TEST PLOTTER =================
def run_strategy_test_104(files, outdir, tree_name, particle_type):
    os.makedirs(outdir, exist_ok=True)
    out_pdf = os.path.join(outdir, f"STRATEGY_TEST_CH104_{particle_type}.pdf")
    
    # Channel 104 is Quartz -> Pulling Quartz limits from FAMILIES config
    xlim = [FAMILIES["Quartz"]["tmin"], FAMILIES["Quartz"]["tmax"]]
    b, g, ch = 1, 0, 4 # Board 1, Group 0, Channel 4 -> 104
    suffix = "_LP2_50"

    print(f"\n[TEST] -----------------------------------------------------------")
    print(f"[TEST] Running MCP Strategy Comparisons on Channel 104")
    print(f"[TEST] Using Window: {xlim}")

    with PdfPages(out_pdf) as pdf:
        for fpath in files:
            rl = _run_label(fpath)
            print(f"  -> Processing Run {rl}...")
            
            try:
                uf = uproot.open(fpath)
                tree = uf[tree_name]
                pid_mask = compute_pid_mask(tree, particle_type) if particle_type else None
                adc_mask = compute_adc_mask(tree, "104")
                
                total_mask = pid_mask & adc_mask if pid_mask is not None else adc_mask
            except Exception as e:
                print(f"     [ERROR] Could not load or mask tree for {rl}: {e}")
                continue

            # =================================================================
            # PAGE 1: RAW & RELATIVE MCP TIMING HISTOGRAMS
            # =================================================================
            fig_mcp, axes_mcp = plt.subplots(2, 2, figsize=(16, 12))
            br_mcp1 = f"DRS_Board0_Group3_Channel6{suffix}"
            br_mcp2 = f"DRS_Board0_Group3_Channel7{suffix}"
            br_ref  = f"DRS_Board0_Group3_Channel8{suffix}"
            
            mcp_branches = [(br_mcp1, "MCP 1 (C6)"), (br_mcp2, "MCP 2 (C7)")]
            
            # --- ROW 1: Raw Histograms (Range 105 to 125) ---
            for idx, (br, mcp_name) in enumerate(mcp_branches):
                ax = axes_mcp[0, idx]
                if br in tree.keys():
                    arr_mcp = tree[br].array(library="np")
                    arr_mcp = arr_mcp[total_mask]
                    arr_mcp = arr_mcp[~np.isnan(arr_mcp)]
                    
                    # Filter strictly to the 105-125 range
                    arr_mcp_cut = arr_mcp[(arr_mcp >= 105) & (arr_mcp <= 125)]
                    
                    if len(arr_mcp_cut) > 0:
                        ax.hist(arr_mcp_cut, bins=100, range=(105, 125), color='steelblue', alpha=0.8)
                        ax.set_title(f"Raw {mcp_name}: {br}\nMean: {arr_mcp_cut.mean():.2f} ns, Std: {arr_mcp_cut.std():.2f} ns")
                        ax.set_xlabel(f"Raw {suffix} Time [ns]")
                        ax.set_ylabel("Events")
                        ax.set_xlim(105, 125)
                    else:
                        ax.set_title(f"Raw {mcp_name}\n(No valid events in [105, 125])")
                        ax.axis('off')
                else:
                    ax.set_title(f"Raw {mcp_name}\n(BRANCH NOT FOUND)")
                    ax.axis('off')

            # --- ROW 2: Relative Histograms (MCP - C8 Ref) ---
            for idx, (br, mcp_name) in enumerate(mcp_branches):
                ax = axes_mcp[1, idx]
                if br in tree.keys() and br_ref in tree.keys():
                    arr_mcp = tree[br].array(library="np")
                    arr_ref_data = tree[br_ref].array(library="np")
                    
                    # Apply masks
                    arr_mcp = arr_mcp[total_mask]
                    arr_ref_data = arr_ref_data[total_mask]
                    
                    # Subtract Ref
                    arr_rel = arr_mcp - arr_ref_data
                    arr_rel = arr_rel[~np.isnan(arr_rel)]
                    
                    if len(arr_rel) > 0:
                        # Auto-center range around the mean to ensure we see the peak
                        mean_rel = arr_rel.mean()
                        rel_min, rel_max = mean_rel - 5, mean_rel + 5
                        
                        # Apply local window cut for clean stats
                        arr_rel_cut = arr_rel[(arr_rel >= rel_min) & (arr_rel <= rel_max)]
                        
                        if len(arr_rel_cut) > 0:
                            ax.hist(arr_rel_cut, bins=100, range=(rel_min, rel_max), color='darkorange', alpha=0.8)
                            ax.set_title(f"Relative {mcp_name} - Ref (C8)\nMean: {arr_rel_cut.mean():.2f} ns, Std: {arr_rel_cut.std():.2f} ns")
                            ax.set_xlabel(f"Relative Time (MCP - C8) [ns]")
                            ax.set_ylabel("Events")
                            ax.set_xlim(rel_min, rel_max)
                        else:
                            ax.set_title(f"Relative {mcp_name} - Ref (C8)\n(No events near mean)")
                            ax.axis('off')
                    else:
                        ax.set_title(f"Relative {mcp_name} - Ref (C8)\n(No valid events)")
                        ax.axis('off')
                else:
                    ax.set_title(f"Relative {mcp_name} - Ref (C8)\n(BRANCHES NOT FOUND)")
                    ax.axis('off')

            display_name = "Positron" if particle_type == "electron" else particle_type.capitalize()
            hep.cms.label(ax=axes_mcp[0, 0], exp="CaloX", data=True, rlabel=f"40 GeV {display_name} | {rl}")
            
            fig_mcp.tight_layout()
            pdf.savefig(fig_mcp)
            plt.close(fig_mcp)

            # =================================================================
            # PAGE 2: STRATEGY COMPARISON OVERLAY
            # =================================================================
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_xlim(*xlim)
            ax.set_ylim(0, 1.4)
            
            strategies = [
                ("Baseline (C7)", get_tfinal_3mm_baseline, "black"),
                ("MCP 1 (C6)", get_tfinal_3mm_mcp6, "blue"),
                ("MCP Avg (C6+C7)", get_tfinal_3mm_avg, "red")
            ]

            handles = []
            labels_list = []

            for label, func, color in strategies:
                arr = func(tree, b, g, ch, suffix)
                if arr is None: 
                    print(f"     [WARN] Missing branches for {label} strategy.")
                    continue
                
                # Apply Masks
                arr = arr[total_mask]
                arr = arr[~np.isnan(arr)]
                arr = arr[(arr >= xlim[0]) & (arr <= xlim[1])]
                
                if len(arr) < 25:
                    print(f"     [WARN] Not enough stats ({len(arr)}) for {label}.")
                    continue

                # Histogram and Fitting
                bins = np.linspace(xlim[0], xlim[1], 100)
                h, _ = np.histogram(arr, bins=bins)
                if h.max() == 0: continue
                
                h_norm = h / h.max()
                centers = 0.5 * (bins[1:] + bins[:-1])
                mode = _mode_from_hist(arr, bins)

                try:
                    popt, _ = curve_fit(gaussian_peak_1, centers, h_norm, p0=[mode, arr.std()])
                    fit_mu, fit_sig = popt[0], abs(popt[1])
                except:
                    fit_mu, fit_sig = mode, float(arr.std())
                    
                # Calculate FWHM
                fwhm = 2.355 * fit_sig

                # Plotting
                ax.step(centers, h_norm, where="mid", lw=1.5, alpha=0.3, color=color)
                x_smooth = np.linspace(xlim[0], xlim[1], 500)
                line, = ax.plot(x_smooth, gaussian_peak_1(x_smooth, fit_mu, fit_sig), color=color, lw=3)
                
                handles.append(line)
                labels_list.append(f"{label}: Mean = {fit_mu} FWHM = {fwhm:.3f}ns sig = {fit_sig}ns (N={len(arr)})")

            if not handles:
                plt.close(fig)
                continue

            # Dress plot
            hep.cms.label(ax=ax, exp="CaloX", data=True, rlabel=f"40 GeV {display_name} | Ch 104 | {rl}")
            
            ax.set_xlabel("Time of Arrival [ns]")
            ax.set_ylabel("Normalized Events")
            ax.legend(handles, labels_list, loc="upper right", frameon=False, fontsize=12)
            
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            uf.close()
            
    print(f"\n[TEST] Finished. PDF saved to: {out_pdf}")

# ================= EXECUTION =================
def _resolve_files(args):
    if args.ana_files: files = list(args.ana_files)
    else: files = sorted(glob.glob(args.ana_glob))
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ana-files", nargs="+", default=None, help="Explicit list of input ROOT files.")
    ap.add_argument("--ana-glob", default=None, help="Glob for input ROOT files.")
    ap.add_argument("--outdir", default="./PreciseTiming/MCPtest", help="Output directory")
    ap.add_argument("--pid", default='electron', choices=["muon", "pion", "electron", "proton"], help="Apply PID selection")

    args = ap.parse_args()

    if args.ana_files is None and args.ana_glob is None:
        raise SystemExit("[FATAL ERROR] Provide either --ana-files or --ana-glob")

    files = _resolve_files(args)
    if not files:
        raise SystemExit("[FATAL ERROR] No files matched your selection")

    print(f"[INIT] Resolved {len(files)} files.")
    print(f"[INIT] Output directory: {args.outdir}")
    print(f"[INIT] Particle Type: {args.pid}")
    
    # Run the isolated Channel 104 test
    run_strategy_test_104(files, args.outdir, TREE_NAME, args.pid)

if __name__ == "__main__":
    main()