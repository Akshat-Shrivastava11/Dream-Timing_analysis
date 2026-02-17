#!/usr/bin/env python3
import os
import re
import glob
import argparse
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import warnings  # <--- Added standard warnings module
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

# ================= CONFIGURATION =================
TREE_NAME = "EventTree"

# The specific channels you asked for
TARGET_CHANNELS = ["100", "110", "105", "104"]

# Hodoscope Configuration
FERS_X_BRANCH = "FERS_Board1_energyHG"
FERS_Y_BRANCH = "FERS_Board0_energyHG"
HG_THRESHOLD = 4000
PITCH = 0.64

# Mapping Arrays
X_mapping = [
    63,55,47,39,31,23,15,7,
    3,11,19,27,35,43,51,59,
    61,53,45,37,29,21,13,5,
    1,9,17,25,33,41,49,57,
    62,54,46,38,30,22,14,6,
    2,10,18,26,34,42,50,58,
    60,52,44,36,28,20,12,4,
    0,8,16,24,32,40,48,56
]

Y_mapping = [
    7,15,23,31,39,47,55,63,
    59,51,43,35,27,19,11,3,
    5,13,21,29,37,45,53,61,
    57,49,41,33,25,17,9,1,
    6,14,22,30,38,46,54,62,
    58,50,42,34,26,18,10,2,
    4,12,20,28,36,44,52,60,
    56,48,40,32,24,16,8,0
]

MAP_X = np.asarray(X_mapping, dtype=np.int64)
MAP_Y = np.asarray(Y_mapping, dtype=np.int64)

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

# ================= HODOSCOPE CALCULATIONS =================

def get_hodo_positions(tree):
    """
    Decodes FERS data, applies mapping, clustering, and returns X, Y arrays.
    """
    try:
        # 1. Load FERS Data using Awkward to handle jagged arrays safely
        raw_x_ak = tree[FERS_X_BRANCH].array(library="ak")
        raw_y_ak = tree[FERS_Y_BRANCH].array(library="ak")

        # 2. Convert to rectangular NumPy arrays (N_events, 64)
        raw_x = ak.to_numpy(raw_x_ak)
        raw_y = ak.to_numpy(raw_y_ak)
        
        # Verify shape
        if raw_x.ndim != 2 or raw_y.ndim != 2:
            print(f"  [HODO] Shape mismatch: {raw_x.shape} (Expected N, 64)")
            return None, None, None

        # 3. Apply Mapping
        HGx = raw_x[:, MAP_X]
        HGy = raw_y[:, MAP_Y]

        # 4. Define Hits
        hit_x = HGx > HG_THRESHOLD
        hit_y = HGy > HG_THRESHOLD

        # 5. Multiplicity Check
        n_hit_x = hit_x.sum(axis=1)
        n_hit_y = hit_y.sum(axis=1)
        
        good_mult = (
            (n_hit_x >= 1) & (n_hit_x <= 2) &
            (n_hit_y >= 1) & (n_hit_y <= 2)
        )

        # 6. Localization/Adjacency Check
        idx = np.arange(64)
        
        hit_x_idx = np.where(hit_x, idx, np.nan)
        hit_y_idx = np.where(hit_y, idx, np.nan)
        
        # Suppress warnings for empty slices (handled by good_mult mask)
        # FIX: Using standard python warnings instead of np.warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN slice encountered')
            span_x = np.nanmax(hit_x_idx, axis=1) - np.nanmin(hit_x_idx, axis=1)
            span_y = np.nanmax(hit_y_idx, axis=1) - np.nanmin(hit_y_idx, axis=1)

        adjacent = (span_x <= 1) & (span_y <= 1)

        # 7. Combine Filter (Good Hodoscope Hit)
        good_hodo = good_mult & adjacent

        # 8. Reconstruct Position (Max Bar)
        HGx_hit = np.where(hit_x, HGx, 0.0)
        HGy_hit = np.where(hit_y, HGy, 0.0)
        
        bar_pos = PITCH * (idx - 31.5)
        
        x_idx = np.argmax(HGx_hit, axis=1)
        x_rec = bar_pos[x_idx]
        
        y_idx = np.argmax(HGy_hit, axis=1)
        y_rec = bar_pos[y_idx]

        return x_rec, y_rec, good_hodo

    except Exception as e:
        print(f"  [HODO] Error calculating positions: {e}")
        return None, None, None

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
    pdf_filename = f"HodoscopeCorr_Chan{ch_code}_{pid_tag}.pdf"
    out_path = os.path.join(args.outdir, pdf_filename)
    
    print(f"\n--- Generating PDF for Channel {ch_code} -> {out_path} ---")

    with PdfPages(out_path) as pdf:
        
        for fpath in files:
            run_id = _run_label(fpath)
            print(f"  Processing Run {run_id}...")
            
            try:
                uf = uproot.open(fpath)
                tree = uf[TREE_NAME]
                keys = set(tree.keys())

                if tfinal_branch not in keys:
                    print(f"    [SKIP] Branch {tfinal_branch} not found.")
                    continue
                
                # Check for FERS branches
                if FERS_X_BRANCH not in keys or FERS_Y_BRANCH not in keys:
                    print(f"    [SKIP] FERS branches not found in this run.")
                    continue

                # 1. Get tfinal
                t_ak = tree[tfinal_branch].array(library="ak")
                t_data = ak.to_numpy(t_ak)
                
                # 2. Get Hodoscope Data
                hodo_x, hodo_y, good_hodo_mask = get_hodo_positions(tree)
                if hodo_x is None: continue

                # 3. Align Lengths
                n_events = min(len(t_data), len(hodo_x))
                
                # 4. Apply Masks (Good Hodo + PID)
                final_mask = good_hodo_mask[:n_events]

                # Apply PID if requested
                if args.pid:
                    pid_mask = compute_pid_mask(tree, args.pid)
                    if pid_mask is not None:
                        final_mask = final_mask & pid_mask[:n_events]

                # Apply Mask to Data
                t_final = np.abs(t_data[:n_events][final_mask])
                x_final = hodo_x[:n_events][final_mask]
                y_final = hodo_y[:n_events][final_mask]

                # Range Mask (Zoom)
                range_mask = (t_final >= args.tmin) & (t_final <= args.tmax) & \
                             (x_final >= -args.pos_range) & (x_final <= args.pos_range) & \
                             (y_final >= -args.pos_range) & (y_final <= args.pos_range)
                
                t_plot = t_final[range_mask]
                x_plot = x_final[range_mask]
                y_plot = y_final[range_mask]

                if len(t_plot) < 10:
                    print("    [SKIP] Not enough events after cuts.")
                    continue

                # 5. Plotting
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11))
                
                # Plot 1: tfinal vs X
                h1 = ax1.hist2d(t_plot, x_plot, bins=64, 
                                range=[[args.tmin, args.tmax], [-args.pos_range, args.pos_range]],
                                cmap="viridis", norm=LogNorm())
                fig.colorbar(h1[3], ax=ax1, label="Counts (Log)")
                ax1.set_xlabel(f"|t_final| Ch{ch_code} [ns]")
                ax1.set_ylabel("Hodoscope X [mm]")
                ax1.set_title(f"Run {run_id} - Channel {ch_code} - Timing vs Hodoscope X ({pid_tag})")
                
                # Plot 2: tfinal vs Y
                h2 = ax2.hist2d(t_plot, y_plot, bins=64, 
                                range=[[args.tmin, args.tmax], [-args.pos_range, args.pos_range]],
                                cmap="viridis", norm=LogNorm())
                fig.colorbar(h2[3], ax=ax2, label="Counts (Log)")
                ax2.set_xlabel(f"|t_final| Ch{ch_code} [ns]")
                ax2.set_ylabel("Hodoscope Y [mm]")
                ax2.set_title(f"Run {run_id} - Channel {ch_code} - Timing vs Hodoscope Y ({pid_tag})")
            
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            except Exception as e:
                print(f"    [ERR] Failed to process run {run_id}: {e}")
                continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ana-files", nargs="+", help="Input ROOT files")
    ap.add_argument("--ana-glob", help="Glob pattern for ROOT files")
    ap.add_argument("--outdir", default="./HodoscopeCorr", help="Output folder")
    ap.add_argument("--pid", default=None, choices=["muon", "pion", "electron", "proton"], 
                    help="Filter by particle type (includes Veto logic)")
    
    # Defaults
    ap.add_argument("--tmin", type=float, default=11.0, help="Min tfinal (default 11.0)")
    ap.add_argument("--tmax", type=float, default=14.0, help="Max tfinal (default 14.0)")
    ap.add_argument("--pos-range", type=float, default=25.0, help="Hodoscope position range (+/- mm). Default 25.")

    args = ap.parse_args()

    if not args.ana_files and not args.ana_glob:
        print("Error: Please provide files using --ana-files or --ana-glob")
        return

    # Resolve files
    if args.ana_files:
        files = args.ana_files
    else:
        files = glob.glob(args.ana_glob)
    
    files = _sort_files(files)
    
    if not files:
        print("No files found!")
        return

    print(f"Found {len(files)} files.")
    os.makedirs(args.outdir, exist_ok=True)

    # Process each requested channel
    for ch in TARGET_CHANNELS:
        process_channel(ch, files, args)

    print("\nAll done.")

if __name__ == "__main__":
    main()