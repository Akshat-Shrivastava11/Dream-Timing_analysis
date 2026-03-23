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

# ================= PAPER PLOTTING =================
def plot_combined_delta_mcp(pdf, all_dt, particle_type):
    """Generates a high-quality, paper-ready plot for the Delta T between MCPs."""
    if len(all_dt) < 10:
        print("[WARN] Not enough data for combined plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    
    
    # Calculate Histogram
    
    # Range and Binning for paper quality
    xmin, xmax = -0.75-1.0, 0.75
    bins = np.linspace(xmin, xmax, 120)
    counts, bin_edges = np.histogram(all_dt, bins=bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    # Normalize to Peak (A.U.)
    max_val = np.max(counts)
    counts_norm = counts / max_val if max_val > 0 else counts
    errors_norm = np.sqrt(counts) / max_val if max_val > 0 else np.zeros_like(counts)

    # Plot data points with hep style
    hep.histplot(counts_norm, bins=bin_edges, yerr=errors_norm, histtype='errorbar', 
                 color='black', marker='o', markersize=4, label='Data', ax=ax)

    # Gaussian Fit
    try:
        # Initial guess: mode, and a reasonable sigma
        p0 = [bin_centers[np.argmax(counts_norm)], 0.1]
        popt, pcov = curve_fit(gaussian_peak_1, bin_centers, counts_norm, p0=p0)
        fit_mu, fit_sig = popt[0], abs(popt[1])
        
        x_fit = np.linspace(xmin, xmax, 1000)
        y_fit = gaussian_peak_1(x_fit, *popt)
        
        ax.plot(x_fit, y_fit, color='red', lw=2.5, 
                label=f'Gaussian Fit\n$\mu = {fit_mu:.3f}$ ns\n$\sigma = {fit_sig*1000:.1f}$ ps')
    except Exception as e:
        print(f"[ERROR] Fit failed: {e}")
    
    # Labels and Aesthetics
    ax.set_xlabel(r"$\Delta t_{MCP7-MCP6}$ [ns]", fontsize=22)
    ax.set_ylabel(" [A.U.]", fontsize=22)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 1.2)
    
    # CMS Labeling
    display_name = particle_type.capitalize() if particle_type else "All Particles"
    hep.cms.label(ax=ax, exp="CaloX", data=True, llabel="Preliminary", rlabel="2026 Test Beam")
    
    ax.text(0.05, 0.9, f"Particle: {display_name}", 
            transform=ax.transAxes, fontsize=16, verticalalignment='top')

    ax.legend(loc="upper right", frameon=True, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    print(f"[MCP STUDY] Combined Delta T plot added to PDF.")

# ================= MINI MCP STUDY CORE =================
def run_mini_mcp_study(files, outdir, tree_name, particle_type):
    os.makedirs(outdir, exist_ok=True)
    out_pdf = os.path.join(outdir, f"MCP_STUDY_COMBINED_{particle_type if particle_type else 'All'}.pdf")
    suffix = "_LP2_50"

    all_delta_t = []

    with PdfPages(out_pdf) as pdf:
        for fpath in files:
            rl = _run_label(fpath)
            try:
                uf = uproot.open(fpath)
                tree = uf[tree_name]
                
                br_t6, br_t7 = f"DRS_Board0_Group3_Channel6{suffix}", f"DRS_Board0_Group3_Channel7{suffix}"
                br_w6, br_w7 = "DRS_Board0_Group3_Channel6", "DRS_Board0_Group3_Channel7"
                
                if not all(b in tree.keys() for b in [br_t6, br_t7, br_w6, br_w7]): continue
                
                pid_mask = compute_pid_mask(tree, particle_type) if particle_type else np.ones(tree.num_entries, dtype=bool)
                
                t6, t7 = tree[br_t6].array(library="np"), tree[br_t7].array(library="np")
                w6, w7 = tree[br_w6].array(library="ak"), tree[br_w7].array(library="ak")
                
                p6 = ak.to_numpy(ak.max(w6 - ak.mean(w6[:, :30], axis=1), axis=1))
                p7 = ak.to_numpy(ak.max(w7 - ak.mean(w7[:, :30], axis=1), axis=1))
                
                valid_mask = (p6 > AMP_THRESHOLD) & (p7 > AMP_THRESHOLD) & (~np.isnan(t6)) & (~np.isnan(t7)) & pid_mask
                
                dt = t7[valid_mask] - t6[valid_mask]
                all_delta_t.extend(dt)
                
            except Exception as e:
                print(f"     [ERROR] Skipping {rl}: {e}")
                continue

        # Convert to numpy for plotting
        all_delta_t = np.array(all_delta_t)
        
        # Call the new dedicated plotting function for a separate page
        plot_combined_delta_mcp(pdf, all_delta_t, particle_type)

    print(f"[MCP STUDY] Saved Combined MCP Study to: {out_pdf}")

# ================= EXECUTION =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ana-files", nargs="+", default=None)
    ap.add_argument("--ana-glob", default=None)
    ap.add_argument("--outdir", default="./PreciseTiming/MCP_Study")
    ap.add_argument("--pid", default='electron', choices=["muon", "pion", "electron", "proton"])

    args = ap.parse_args()
    files = list(args.ana_files) if args.ana_files else sorted(glob.glob(args.ana_glob))
    
    if not files:
        raise SystemExit("[FATAL] No files found.")

    run_mini_mcp_study(files, args.outdir, TREE_NAME, args.pid)

if __name__ == "__main__":
    main()