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
from matplotlib.colors import LogNorm

# ================= CONFIGURATION =================
TREE_NAME = "EventTree"

# The specific channels you asked for
TARGET_CHANNELS = ["100", "110", "105", "104"]

# Mapping for Wire Chamber Raw Data
WC_CHANNELS = {
    "L1": "DRS_Board7_Group0_Channel0",
    "R1": "DRS_Board7_Group0_Channel1",
    "U1": "DRS_Board7_Group0_Channel2",
    "D1": "DRS_Board7_Group0_Channel3",
}

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
        "muon": { "TTUMuonVeto": True, "PSD": False },
        "pion": { "TTUMuonVeto": False, "PSD": False, "Cer474": True, "Cer519": True, "Cer537": True },
        "electron": { "TTUMuonVeto": False, "PSD": True, "Cer474": True, "Cer519": True, "Cer537": True },
        "proton": { "TTUMuonVeto": False, "PSD": False, "Cer474": False, "Cer519": False, "Cer537": False },
    }
    return selections.get(particle_type.lower(), {})

def compute_pid_mask(tree, particle_type):
    requirements = get_particle_selection(particle_type)
    if not requirements:
        return None

    n_entries = tree.num_entries
    try:
        final_mask = np.ones(n_entries, dtype=bool)
    except:
        return None
        
    available_keys = set(tree.keys())

    for det, must_fire in requirements.items():
        branch_name = PID_BRANCH_MAP.get(det)
        if not branch_name or branch_name not in available_keys:
            continue

        ts_min, ts_max, val_cut, method = get_service_drs_cut(det)

        try:
            waveforms = tree[branch_name].array(library="ak")
            # Basic processing
            if method == "Sum":
                baseline = ak.mean(waveforms[:, :30], axis=1)
                waveforms_blsub = waveforms - baseline
                window = waveforms_blsub[:, int(ts_min):int(ts_max)]
                window_sum = ak.sum(window, axis=1)
                window_sum_np = ak.to_numpy(window_sum)
                is_fired = window_sum_np < val_cut
            else:
                continue

            if must_fire:
                final_mask = final_mask & is_fired
            else:
                final_mask = final_mask & (~is_fired)

        except Exception as e:
            print(f"    [PID] Error processing {det}: {e}")
            continue

    return final_mask

# ================= WIRE CHAMBER CALCULATIONS =================

def get_hit_times_vectorized(events):
    """
    Given 2D array of waveforms (N_events, 1024),
    subtract baseline (mean of first 20 samples) and find index of minimum.
    """
    if events.ndim != 2:
        return np.zeros(len(events))

    baselines = np.mean(events[:, :20], axis=1, keepdims=True)
    corrected = events - baselines
    hit_indices = np.argmin(corrected, axis=1)
    return hit_indices

def get_wc_positions(tree):
    """
    Loads raw WC waveforms, computes hit times, returns X and Y arrays.
    """
    try:
        keys = set(tree.keys())
        required = [WC_CHANNELS["L1"], WC_CHANNELS["R1"], WC_CHANNELS["U1"], WC_CHANNELS["D1"]]
        if not all(k in keys for k in required):
            return None, None

        L1 = ak.to_numpy(tree[WC_CHANNELS["L1"]].array(library="ak"))
        R1 = ak.to_numpy(tree[WC_CHANNELS["R1"]].array(library="ak"))
        U1 = ak.to_numpy(tree[WC_CHANNELS["U1"]].array(library="ak"))
        D1 = ak.to_numpy(tree[WC_CHANNELS["D1"]].array(library="ak"))

        # Get times
        L1_t = get_hit_times_vectorized(L1)
        R1_t = get_hit_times_vectorized(R1)
        U1_t = get_hit_times_vectorized(U1)
        D1_t = get_hit_times_vectorized(D1)

        # Calculate X and Y positions (simple difference method)
        x_positions = L1_t - R1_t
        y_positions = U1_t - D1_t
        
        return x_positions, y_positions
    except Exception as e:
        print(f"  [WC] Error calculating positions: {e}")
        return None, None

# ================= HELPER FUNCTIONS =================

def _parse_code(code_str):
    b = int(code_str[0])
    g = int(code_str[1])
    c = int(code_str[2])
    return b, g, c

def _branch(b, g, c):
    return f"tfinal_Board{b}_Group{g}_Channel{c}"

def _run_label(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"run(\d+)", base)
    if m: return m.group(1)
    return base

def _sort_files(files):
    def key(p):
        m = re.search(r"run(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0
    return sorted(files, key=key)

# ================= MAIN LOGIC =================

def process_channel(ch_code, files, args):
    b, g, c = _parse_code(ch_code)
    tfinal_branch = _branch(b, g, c)
    
    pid_tag = args.pid if args.pid else "NoPID"
    out_dir = args.outdir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    pdf_filename = f"WirechamberCorr_Chan{ch_code}_{pid_tag}.pdf"
    out_path = os.path.join(out_dir, pdf_filename)
    
    print(f"\n--- Generating PDF for Channel {ch_code} -> {out_path} ---")

    with PdfPages(out_path) as pdf:
        
        for fpath in files:
            run_id = _run_label(fpath)
            print(f"  Processing Run {run_id}...")
            
            try:
                with uproot.open(fpath) as uf:
                    if TREE_NAME not in uf:
                        print(f"    [SKIP] Tree {TREE_NAME} missing.")
                        continue
                    
                    tree = uf[TREE_NAME]
                    keys = set(tree.keys())

                    if tfinal_branch not in keys:
                        print(f"    [SKIP] Branch {tfinal_branch} not found.")
                        continue

                    # 1. Get tfinal
                    t_ak = tree[tfinal_branch].array(library="ak")
                    t_data = ak.to_numpy(t_ak)
                    
                    # 2. Get Wire Chamber
                    wc_x, wc_y = get_wc_positions(tree)
                    if wc_x is None: 
                        print("    [SKIP] WC data missing/error.")
                        continue

                    # 3. Align Lengths
                    n_events = min(len(t_data), len(wc_x), len(wc_y))
                    
                    # 4. Apply Cuts & PID
                    mask = np.ones(n_events, dtype=bool)

                    if args.pid:
                        pid_mask = compute_pid_mask(tree, args.pid)
                        if pid_mask is not None:
                            mask = mask & pid_mask[:n_events]

                    # Slice data
                    t_final = np.abs(t_data[:n_events][mask])
                    x_final = wc_x[:n_events][mask]
                    y_final = wc_y[:n_events][mask]

                    # 5. Range Mask (Zoom)
                    range_mask = (t_final >= args.tmin) & (t_final <= args.tmax) & \
                                 (x_final >= -args.wc_range) & (x_final <= args.wc_range) & \
                                 (y_final >= -args.wc_range) & (y_final <= args.wc_range)
                    
                    t_plot = t_final[range_mask]
                    x_plot = x_final[range_mask]
                    y_plot = y_final[range_mask]

                    if len(t_plot) < 10:
                        print("    [SKIP] Not enough events after cuts.")
                        continue

                    # ==========================================
                    # CALCULATE CORRELATION
                    # ==========================================
                    try:
                        corr_x = np.corrcoef(t_plot, x_plot)[0, 1]
                    except:
                        corr_x = 0.0
                    
                    try:
                        corr_y = np.corrcoef(t_plot, y_plot)[0, 1]
                    except:
                        corr_y = 0.0

                    # 6. Plotting
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11))
                    
                    # --- Plot 1: tfinal vs X ---
                    h1 = ax1.hist2d(t_plot, x_plot, bins=100, 
                                    range=[[args.tmin, args.tmax], [-args.wc_range, args.wc_range]],
                                    cmap="viridis", norm=LogNorm())
                    fig.colorbar(h1[3], ax=ax1, label="Counts (Log)")
                    ax1.set_xlabel(f"|t_final| Ch{ch_code} [ns]")
                    ax1.set_ylabel("Wire Chamber X (L - R)")
                    ax1.set_title(f"Run {run_id} | Ch {ch_code} | Time vs X | {pid_tag}")
                    
                    # Add Correlation Score Text
                    ax1.text(0.95, 0.95, f"Corr: {corr_x:.4f}", 
                             transform=ax1.transAxes, ha='right', va='top', 
                             fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

                    # --- Plot 2: tfinal vs Y ---
                    h2 = ax2.hist2d(t_plot, y_plot, bins=100, 
                                    range=[[args.tmin, args.tmax], [-args.wc_range, args.wc_range]],
                                    cmap="viridis", norm=LogNorm())
                    fig.colorbar(h2[3], ax=ax2, label="Counts (Log)")
                    ax2.set_xlabel(f"|t_final| Ch{ch_code} [ns]")
                    ax2.set_ylabel("Wire Chamber Y (U - D)")
                    ax2.set_title(f"Run {run_id} | Ch {ch_code} | Time vs Y | {pid_tag}")
                    
                    # Add Correlation Score Text
                    ax2.text(0.95, 0.95, f"Corr: {corr_y:.4f}", 
                             transform=ax2.transAxes, ha='right', va='top', 
                             fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

            except Exception as e:
                print(f"    [ERR] Failed to process run {run_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ana_files", nargs="*", help="Input ROOT files") # Positional is easier usually
    ap.add_argument("--ana-glob", help="Glob pattern for ROOT files")
    ap.add_argument("--outdir", default="./WirechamberCorr", help="Output folder")
    ap.add_argument("--pid", default=None, choices=["muon", "pion", "electron", "proton"], 
                    help="Filter by particle type")
    
    # Defaults
    ap.add_argument("--tmin", type=float, default=11.0, help="Min tfinal (default 11.0)")
    ap.add_argument("--tmax", type=float, default=14.0, help="Max tfinal (default 14.0)")
    ap.add_argument("--wc-range", type=float, default=250.0, help="Wire chamber range (+/-)")

    args = ap.parse_args()

    files = []
    if args.ana_files:
        files.extend(args.ana_files)
    if args.ana_glob:
        files.extend(glob.glob(args.ana_glob))
    
    # Remove duplicates and sort
    files = _sort_files(list(set(files)))
    
    if not files:
        print("Error: No files found! Use arguments to provide files or --ana-glob.")
        return

    print(f"Found {len(files)} files.")
    
    # Process each requested channel
    for ch in TARGET_CHANNELS:
        process_channel(ch, files, args)

    print("\nAll done.")

if __name__ == "__main__":
    main()