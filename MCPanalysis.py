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

# ================= PID MASKS =================
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

# ================= MATH HELPERS =================
def gaussian_peak_1(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def _run_label(path: str) -> str:
    m = re.search(r"(run\d+_\d{11,12})", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]

# ================= MINI MCP STUDY CORE =================
def run_mini_mcp_study(files, outdir, tree_name, particle_type):
    os.makedirs(outdir, exist_ok=True)
    out_pdf = os.path.join(outdir, f"MINI_MCP_STUDY_{particle_type if particle_type else 'AllParticles'}.pdf")
    suffix = "_LP2_50"

    print(f"\n[MCP STUDY] ------------------------------------------------------")
    print(f"[MCP STUDY] Running Mini MCP Analysis (Channel 6 vs Channel 7)")

    with PdfPages(out_pdf) as pdf:
        for fpath in files:
            rl = _run_label(fpath)
            print(f"  -> Processing Run {rl}...")
            
            try:
                uf = uproot.open(fpath)
                tree = uf[tree_name]
                
                # Check for branches
                br_t6 = f"DRS_Board0_Group3_Channel6{suffix}"
                br_t7 = f"DRS_Board0_Group3_Channel7{suffix}"
                br_w6 = f"DRS_Board0_Group3_Channel6"
                br_w7 = f"DRS_Board0_Group3_Channel7"
                
                if not all(b in tree.keys() for b in [br_t6, br_t7, br_w6, br_w7]):
                    print(f"     [SKIP] Missing MCP branches in {rl}")
                    continue
                
                # Get PID mask
                pid_mask = compute_pid_mask(tree, particle_type) if particle_type else np.ones(tree.num_entries, dtype=bool)
                if pid_mask is None: pid_mask = np.ones(tree.num_entries, dtype=bool)

                # Get Times
                t6 = tree[br_t6].array(library="np")
                t7 = tree[br_t7].array(library="np")
                
                # Get Waveforms and calculate Peak ADC
                w6 = tree[br_w6].array(library="ak")
                w7 = tree[br_w7].array(library="ak")
                
                # Baseline subtraction (first 30 bins)
                bl6 = ak.mean(w6[:, :30], axis=1)
                bl7 = ak.mean(w7[:, :30], axis=1)
                w6_sub = w6 - bl6
                w7_sub = w7 - bl7
                
                p6 = ak.to_numpy(ak.max(w6_sub, axis=1))
                p7 = ak.to_numpy(ak.max(w7_sub, axis=1))
                
                # -------------------------------------------------------------
                # INDEPENDENT FIRING LOGIC
                # -------------------------------------------------------------
                fire6 = (p6 > AMP_THRESHOLD) & (~np.isnan(t6))
                fire7 = (p7 > AMP_THRESHOLD) & (~np.isnan(t7))
                
                both_fire = fire6 & fire7 & pid_mask
                only_6_fire = fire6 & (~fire7) & pid_mask
                only_7_fire = (~fire6) & fire7 & pid_mask
                
                count_both = np.sum(both_fire)
                count_only6 = np.sum(only_6_fire)
                count_only7 = np.sum(only_7_fire)
                
                # -------------------------------------------------------------
                # Build a common mask for clean scatter/correlation plots
                # -------------------------------------------------------------
                valid_mask = both_fire
                
                t6_c, t7_c = t6[valid_mask], t7[valid_mask]
                p6_c, p7_c = p6[valid_mask], p7[valid_mask]
                w6_c, w7_c = w6_sub[valid_mask], w7_sub[valid_mask]
                
                if len(t6_c) < 50:
                    print(f"     [SKIP] Not enough valid events ({len(t6_c)})")
                    continue
                    
            except Exception as e:
                print(f"     [ERROR] Processing {rl}: {e}")
                continue

            # =================================================================
            # PAGE 1: CORRELATIONS & TIME DIFFERENCE
            # =================================================================
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            
            display_name = "Positron" if particle_type == "electron" else (particle_type.capitalize() if particle_type else "All Particles")
            fig.suptitle(f"Mini MCP Study | Run {rl} | 40 GeV {display_name}", fontsize=20, fontweight='bold')
            
            # -------------------------------------------------
            # Plot 1: Time Correlation (105 to 125 ns range)
            # -------------------------------------------------
            ax = axes[0, 0]
            h, xedges, yedges, im = ax.hist2d(t6_c, t7_c, bins=100, range=[[105, 125], [105, 125]], cmap='viridis', cmin=1)
            ax.plot([105, 125], [105, 125], 'r--', alpha=0.5, label="y = x")
            fig.colorbar(im, ax=ax)
            ax.set_title("Time of Arrival Correlation (Both Fired)")
            ax.set_xlabel(f"MCP 6 {suffix} [ns]")
            ax.set_ylabel(f"MCP 7 {suffix} [ns]")
            ax.legend(loc="upper left")

            # -------------------------------------------------
            # Plot 2: Peak ADC Correlation
            # -------------------------------------------------
            ax = axes[0, 1]
            max_adc = max(np.percentile(p6_c, 99), np.percentile(p7_c, 99))
            h, xedges, yedges, im = ax.hist2d(p6_c, p7_c, bins=100, range=[[0, max_adc], [0, max_adc]], cmap='plasma', cmin=1)
            ax.plot([0, max_adc], [0, max_adc], 'r--', alpha=0.5, label="y = x")
            fig.colorbar(im, ax=ax)
            ax.set_title("Peak ADC Correlation (Both Fired)")
            ax.set_xlabel("MCP 6 Peak ADC [mV]")
            ax.set_ylabel("MCP 7 Peak ADC [mV]")
            ax.legend(loc="upper left")

            # -------------------------------------------------
            # Plot 3: Delta T (MCP 7 - MCP 6)
            # -------------------------------------------------
            ax = axes[1, 0]
            dt = t7_c - t6_c
            
            dt_mean = np.mean(dt)
            dt_min, dt_max = dt_mean - 0.5, dt_mean + 0.5
            dt_cut = dt[(dt >= dt_min) & (dt <= dt_max)]
            
            if len(dt_cut) > 10:
                bins = np.linspace(dt_min, dt_max, 100)
                centers = 0.5 * (bins[1:] + bins[:-1])
                counts, _ = np.histogram(dt_cut, bins=bins)
                
                if counts.max() > 0:
                    counts_norm = counts / counts.max()
                    ax.step(centers, counts_norm, where="mid", color='darkorange', alpha=0.6)
                    
                    try:
                        mode_idx = np.argmax(counts)
                        p0 = [centers[mode_idx], dt_cut.std()]
                        bounds = ([dt_min, 0.001], [dt_max, 5.0]) 
                        
                        popt, _ = curve_fit(gaussian_peak_1, centers, counts_norm, p0=p0, bounds=bounds)
                        fit_mu, fit_sig = popt[0], abs(popt[1])
                        
                        x_fine = np.linspace(dt_min, dt_max, 500)
                        ax.plot(x_fine, gaussian_peak_1(x_fine, fit_mu, fit_sig), 'r-', lw=3, 
                                label=f"Fit $\mu$: {fit_mu:.3f} ns\nFit $\sigma$: {fit_sig:.3f} ns\nFWHM: {2.355*fit_sig:.3f} ns")
                    except Exception as e:
                        print(f"     [FIT WARN] Gaussian fit failed for Delta T: {e}")
                        ax.plot([], [], ' ', label="Fit Failed")
            
            ax.set_title(f"$\Delta t$ (MCP 7 - MCP 6)")
            ax.set_xlabel("$\Delta t$ [ns]")
            ax.set_ylabel("Normalized Events")
            ax.legend(loc="upper right", frameon=False, fontsize=12)

            # -------------------------------------------------
            # Plot 4: Firing Correlation Bar Chart
            # -------------------------------------------------
            ax = axes[1, 1]
            categories = ['Both Fire', 'Only MCP 6', 'Only MCP 7']
            counts_bar = [count_both, count_only6, count_only7]
            
            bars = ax.bar(categories, counts_bar, color=['mediumseagreen', 'steelblue', 'indianred'], alpha=0.8)
            ax.set_title("MCP Firing Correlation (PID Applied)")
            ax.set_ylabel("Number of Events")
            
            # Add value labels on top of the bars
            max_count = max(counts_bar)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + (max_count * 0.02), 
                        int(yval), ha='center', va='bottom', fontsize=14, fontweight='bold')
                
            ax.set_ylim(0, max_count * 1.15 if max_count > 0 else 10)

            hep.cms.label(ax=axes[0, 0], exp="CaloX", data=True, rlabel=f"Run {rl}")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

            # =================================================================
            # PAGE 2: WAVEFORMS
            # =================================================================
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f"Average MCP Waveforms (Both Fired) | Run {rl}", fontsize=18, fontweight='bold')
            
            w6_np = ak.to_numpy(w6_c)
            w7_np = ak.to_numpy(w7_c)
            
            time_bins = np.arange(w6_np.shape[1]) * 0.2 
            
            for idx, (w_np, name) in enumerate([(w6_np, "MCP 6"), (w7_np, "MCP 7")]):
                ax = axes[idx]
                w_mean = np.mean(w_np, axis=0)
                w_std = np.std(w_np, axis=0)
                
                ax.plot(time_bins, w_mean, color='black', lw=2, label="Mean Waveform")
                ax.fill_between(time_bins, w_mean - w_std, w_mean + w_std, color='steelblue', alpha=0.4, label="$\pm 1\sigma$ Spread")
                
                ax.set_title(name)
                ax.set_xlabel("Time [ns]")
                ax.set_ylabel("Amplitude [mV]")
                ax.legend(loc="upper right", frameon=False)
                ax.set_xlim(0, max(time_bins))
            
            hep.cms.label(ax=axes[0], exp="CaloX", data=True, rlabel=f"Run {rl}")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[MCP STUDY] Saved Mini MCP Study to: {out_pdf}")

# ================= EXECUTION =================
def _resolve_files(args):
    if args.ana_files: files = list(args.ana_files)
    else: files = sorted(glob.glob(args.ana_glob))
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ana-files", nargs="+", default=None, help="Explicit list of input ROOT files.")
    ap.add_argument("--ana-glob", default=None, help="Glob for input ROOT files.")
    ap.add_argument("--outdir", default="./PreciseTiming/MCP_Study", help="Output directory")
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
    
    # Run the Mini MCP Study
    run_mini_mcp_study(files, args.outdir, TREE_NAME, args.pid)

if __name__ == "__main__":
    main()