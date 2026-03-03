#!/usr/bin/env python3
import os
import argparse
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt

# ================= PID CONFIG =================
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
    parser = argparse.ArgumentParser(description="Waveform Investigation: L50 vertical lines correctly aligned to TS.")
    parser.add_argument("-i", "--input", required=True, help="Input ROOT file")
    parser.add_argument("-c", "--channel", required=True, help="3-digit channel code (e.g., 104)")
    parser.add_argument("--pid", default="electron", choices=["muon", "pion", "electron", "proton", "none"])
    parser.add_argument("--t-left", type=float, default=13.5)
    parser.add_argument("--t-peak", type=float, default=13.7)
    parser.add_argument("--t-right", type=float, default=13.9)
    parser.add_argument("--n-events", type=int, default=3)
    parser.add_argument("--amp", type=float, default=100.0)
    parser.add_argument("--min-adc", type=float, default=-50.0)
    parser.add_argument("--outdir", default="WaveformInvestigations")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    chan_str = str(args.channel).zfill(3)
    b, g, c = int(chan_str[0]), int(chan_str[1]), int(chan_str[2])
    
    main_br  = f"DRS_Board{b}_Group{g}_Channel{c}"
    trig_br  = f"DRS_Board{b}_Group{g}_Channel8"
    mcp_br   = f"DRS_Board{b}_Group3_Channel7"
    trig3_br = f"DRS_Board{b}_Group3_Channel8"
    tfinal_br= f"tfinal_Board{b}_Group{g}_Channel{c}"

    with uproot.open(args.input) as f:
        tree = f["EventTree"]
        
        pid_mask = compute_pid_mask(tree, args.pid)
        waves_main = tree[main_br].array(library="ak")
        baseline_main = ak.mean(waves_main[:, :30], axis=1)
        adc_mask = ak.to_numpy((ak.max(waves_main - baseline_main, axis=1) >= args.amp))
        
        tf_raw = tree[tfinal_br].array(library="ak")
        tf_array = np.abs(ak.to_numpy(ak.flatten(tf_raw, axis=-1)))
        valid_mask = pid_mask & adc_mask & (tf_array > 0) & (tf_array < 40)
        valid_indices = np.where(valid_mask)[0]

        def get_events(target, count, used):
            selected = []
            avail = valid_indices[~np.isin(valid_indices, list(used))]
            if len(avail) == 0: return []
            sorted_idx = np.argsort(np.abs(tf_array[avail] - target))
            for i in range(min(count, len(sorted_idx))):
                ev = avail[sorted_idx[i]]
                used.add(ev)
                selected.append((ev, tf_array[ev]))
            return selected

        used = set()
        all_events = []
        for label, t in [("Left", args.t_left), ("Peak", args.t_peak), ("Right", args.t_right)]:
            for i, (ev, tf) in enumerate(get_events(t, args.n_events, used)):
                all_events.append((f"{label} {i+1}", ev, tf))

        nrows = len(all_events)
        fig, axes = plt.subplots(nrows, 4, figsize=(20, 3.5 * nrows), squeeze=False)
        
        plot_configs = [
            (main_br, f"{main_br}_LP2_50", "Main Channel", "tab:blue"),
            (trig_br, f"{trig_br}_LP2_50", "Group Trigger", "tab:orange"),
            (mcp_br,  f"{mcp_br}_LP2_50",  "MCP Ref", "tab:green"),
            (trig3_br,f"{trig3_br}_LP2_50","Board Trigger", "tab:red")
        ]

        for row_idx, (region, ev, tf_val) in enumerate(all_events):
            for col_idx, (br, l50_br, label, color) in enumerate(plot_configs):
                ax = axes[row_idx, col_idx]
                
                w = tree[br].array(entry_start=ev, entry_stop=ev+1, library="np")[0]
                l50_raw = tree[l50_br].array(entry_start=ev, entry_stop=ev+1, library="np")[0]
                
                # Convert the raw L50 value into Time Slices (bins)
                l50_ts = l50_raw / 0.2
                
                # X-axis remains in raw bins
                bins_axis = np.arange(len(w))
                
                ax.plot(bins_axis, w - np.mean(w[:30]), color=color, lw=1.5)
                
                # The vertical line now correctly plots at the calculated TS value
                ax.axvline(x=l50_ts, color='black', linestyle='--', lw=2, 
                           label=f"L50: {l50_ts:.2f} TS")
                
                # XLIM specifically ONLY for the "Main Channel" (column 0)
                if col_idx == 0:
                    ax.set_xlim(400, 600)
                
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize=8)
                
                if row_idx == 0: ax.set_title(label, fontweight='bold')
                if col_idx == 0: ax.set_ylabel(f"{region}\nEv: {ev}\n|tf|: {tf_val:.2f}ns", fontweight='bold')
                if row_idx == nrows-1: ax.set_xlabel("Time Slice (Bin Index)")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(args.outdir, f"Waveform_L50_Reverted_Zoom_{chan_str}.png"), dpi=200)
        plt.close()

if __name__ == "__main__":
    main()