#!/usr/bin/env python3
import os
import argparse
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt

# PID Config matched to your scripts
PID_BRANCH_MAP = {
    "PSD": "DRS_Board7_Group1_Channel1",
    "HoleVeto": "DRS_Board7_Group1_Channel6",
    "TTUMuonVeto": "DRS_Board7_Group2_Channel4",
    "Cer474": "DRS_Board7_Group2_Channel5",
    "Cer519": "DRS_Board7_Group2_Channel6",
    "Cer537": "DRS_Board7_Group2_Channel7",
}

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

def compute_pid_mask(tree, particle_type):
    if particle_type is None or particle_type.lower() == "none":
        return np.ones(tree.num_entries, dtype=bool)
        
    selections = {
        "muon": {"TTUMuonVeto": True, "PSD": False},
        "pion": {"TTUMuonVeto": False, "PSD": False, "Cer474": True, "Cer519": True, "Cer537": True},
        "electron": {"TTUMuonVeto": False, "PSD": True, "Cer474": True, "Cer519": True, "Cer537": True},
        "proton": {"TTUMuonVeto": False, "PSD": False, "Cer474": False, "Cer519": False, "Cer537": False},
    }
    reqs = selections.get(particle_type.lower(), {})
    mask = np.ones(tree.num_entries, dtype=bool)
    
    for det, must_fire in reqs.items():
        branch = PID_BRANCH_MAP.get(det)
        if branch not in tree: continue
        ts_min, ts_max, val_cut = get_service_drs_cut(det)
        waves = tree[branch].array(library="ak")
        baseline = ak.mean(waves[:, :30], axis=1)
        window = (waves - baseline)[:, int(ts_min):int(ts_max)]
        is_fired = ak.to_numpy(ak.sum(window, axis=1)) < val_cut
        mask = mask & (is_fired if must_fire else ~is_fired)
    return mask

def main():
    parser = argparse.ArgumentParser(description="Investigate waveforms making up tfinal with Main Channel ADC cuts.")
    parser.add_argument("-i", "--input", required=True, help="Input ROOT file")
    parser.add_argument("-c", "--channel", required=True, help="3-digit channel code (e.g., 104)")
    parser.add_argument("--pid", default="electron", choices=["muon", "pion", "electron", "proton", "none"])
    
    # Target times
    parser.add_argument("--t-left", type=float, default=13.5, help="Target |tfinal| for the left-side event")
    parser.add_argument("--t-peak", type=float, default=13.7, help="Target |tfinal| for the peak event")
    parser.add_argument("--t-right", type=float, default=13.9, help="Target |tfinal| for the right-side event")
    parser.add_argument("--n-events", type=int, default=3, help="Number of events to plot PER target region")
    
    # ADC Cuts ONLY for the main channel
    parser.add_argument("--amp", type=float, default=100.0, help="Main channel positive peak threshold (must be >= this)")
    parser.add_argument("--min-adc", type=float, default=-50.0, help="Main channel negative peak threshold (must be >= this)")
    
    parser.add_argument("--outdir", default="WaveformInvestigations", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    chan_str = str(args.channel).zfill(3)
    b, g, c = int(chan_str[0]), int(chan_str[1]), int(chan_str[2])
    
    # The 4 branches that make up the formula
    main_br  = f"DRS_Board{b}_Group{g}_Channel{c}"
    trig_br  = f"DRS_Board{b}_Group{g}_Channel8"
    mcp_br   = f"DRS_Board{b}_Group3_Channel7"
    trig3_br = f"DRS_Board{b}_Group3_Channel8"
    tfinal_br= f"tfinal_Board{b}_Group{g}_Channel{c}"
    
    req_branches = [main_br, trig_br, mcp_br, trig3_br, tfinal_br]

    with uproot.open(args.input) as f:
        tree = f["EventTree"]
        keys = tree.keys()
        
        for br in req_branches:
            if br not in keys:
                raise ValueError(f"Required branch {br} not found in tree!")

        print(f"Loading data for channel {chan_str}...")
        
        # 1. Apply PID
        pid_mask = compute_pid_mask(tree, args.pid)
        
        # 2. Apply ADC Cuts ONLY on the MAIN channel (RESTORED AND LOGIC)
        print(f"Applying ADC cut on {main_br}: Amp >= {args.amp} AND Min >= {args.min_adc}")
        waves_main = tree[main_br].array(library="ak")
        baseline = ak.mean(waves_main[:, :30], axis=1)
        waves_blsub = waves_main - baseline
        peak = ak.max(waves_blsub, axis=1)
        min_adc = ak.min(waves_blsub, axis=1)
        
        # Ensures a positive peak AND no massive negative dip
        adc_mask = ak.to_numpy((peak >= args.amp) & (min_adc >= args.min_adc))
        
        # 3. Extract and format tfinal array, APPLYING ABSOLUTE VALUE
        tf_raw = tree[tfinal_br].array(library="ak")
        if tf_raw.ndim > 1:
            tf_raw = ak.flatten(tf_raw, axis=-1)
        
        tf_array = np.abs(ak.to_numpy(tf_raw))
        
        # 4. Combine PID mask, Main Channel ADC cut, and valid timing bounds
        valid_mask = pid_mask & adc_mask & (tf_array > 0) & (tf_array < 40)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            print("\n[!] No events passed the combined PID and Main Channel ADC cuts.")
            return
            
        print(f"Total events available after PID + ADC cuts: {len(valid_indices)}")

        tfinals = tf_array[valid_indices]

        # 5. Find multiple unique closest events
        used_events = set()
        
        def get_closest_unique_events(target_t, count):
            selected = []
            for _ in range(count):
                available_mask = ~np.isin(valid_indices, list(used_events))
                if not np.any(available_mask):
                    break # Ran out of events
                    
                available_idx = valid_indices[available_mask]
                available_tf = tfinals[available_mask]
                
                idx_min = np.argmin(np.abs(available_tf - target_t))
                best_ev = available_idx[idx_min]
                best_tf = available_tf[idx_min]
                
                used_events.add(best_ev)
                selected.append((best_ev, best_tf))
            return selected

        left_events = get_closest_unique_events(args.t_left, args.n_events)
        peak_events = get_closest_unique_events(args.t_peak, args.n_events)
        right_events = get_closest_unique_events(args.t_right, args.n_events)

        print("\nFound Events:")
        all_events = []
        for i, (ev, tf) in enumerate(left_events):
            print(f"  Left target  ({args.t_left} ns) [{i+1}/{args.n_events}] -> Event {ev} at {tf:.3f} ns")
            all_events.append((f"Left {i+1}\n(~{args.t_left}ns)", ev, tf))
            
        for i, (ev, tf) in enumerate(peak_events):
            print(f"  Peak target  ({args.t_peak} ns) [{i+1}/{args.n_events}] -> Event {ev} at {tf:.3f} ns")
            all_events.append((f"Peak {i+1}\n(~{args.t_peak}ns)", ev, tf))
            
        for i, (ev, tf) in enumerate(right_events):
            print(f"  Right target ({args.t_right} ns) [{i+1}/{args.n_events}] -> Event {ev} at {tf:.3f} ns")
            all_events.append((f"Right {i+1}\n(~{args.t_right}ns)", ev, tf))

        if not all_events:
            return

        # 6. Plotting
        nrows = len(all_events)
        fig, axes = plt.subplots(nrows, 4, figsize=(16, 2.5 * nrows), squeeze=False)
        fig.suptitle(f"Equation Waveforms | Channel {chan_str} | PID: {args.pid}\nFormula: (Main - TrigG) - (MCP - Trig3)", fontsize=16)

        branches_to_plot = [
            (main_br, "Main Channel", "tab:blue"),
            (trig_br, "Group Trigger", "tab:orange"),
            (mcp_br,  "MCP Ref", "tab:green"),
            (trig3_br,"Board Trigger", "tab:red")
        ]

        for row_idx, (region_name, ev, tf_val) in enumerate(all_events):
            for col_idx, (br_name, label, color) in enumerate(branches_to_plot):
                ax = axes[row_idx, col_idx]
                
                # Load waveform
                w = tree[br_name].array(entry_start=ev, entry_stop=ev+1, library="np")[0]
                bl = np.mean(w[:30])
                
                ax.plot(w - bl, color=color, lw=1.5)
                ax.grid(True, alpha=0.3)
                
                # Formatting
                if row_idx == 0:
                    ax.set_title(label, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(f"[{region_name}]\nEv: {ev}\n|tf|: {tf_val:.2f}ns", fontweight='bold')
                    
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        run_label = os.path.basename(args.input).split('_')[0]
        save_path = os.path.join(args.outdir, f"TargetedWaveforms_{run_label}_Ch{chan_str}_{args.pid}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"\nSaved targeted plots to: {save_path}")

if __name__ == "__main__":
    main()