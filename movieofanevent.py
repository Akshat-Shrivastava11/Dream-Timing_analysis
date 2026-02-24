#!/usr/bin/env python3
"""
make_3d_pid_movie_subplots.py

Interactive 3D Event Slider for HGCal Timing data.
- 1x3 Subplots: CER Quartz | CER Plastic | SCI 
- Y axis (Horizontal Depth) = |tfinal| (Time of Arrival) POST-FLUSHING.
- Z axis (Vertical Height) = Physical grid Y (Rows).
- X axis (Horizontal Width) = Physical grid X (Columns).
- Event-by-event Particle ID (PID) evaluation using DRS baseline subtraction.
"""

import os
import re
import json
import argparse
import numpy as np
import uproot
import awkward as ak

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("CRITICAL ERROR: 'plotly' is required. Run: pip install plotly")
    exit(1)

# ================= CONFIG =================
TREE_NAME = "EventTree"
XLIM = (0.0, 20.0)   # range for |tfinal|
GRID_PADDING = 1.0   # How far the grid extends past active channels

DEFAULT_SHIFT_JSON = "/lustre/research/hep/akshriva/Dream-Timing/TRUE-HGtiming/calibration_studiesZ/MODE_CALIB_OUTPUT/shifts_run1501_250928105227_calib_mode.json"

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

    print(f"\n  [PID] Applying selection for {particle_type.upper()}...")

    for det, must_fire in requirements.items():
        branch_name = PID_BRANCH_MAP.get(det)
        if not branch_name or branch_name not in available_keys: continue
        ts_min, ts_max, val_cut, method = get_service_drs_cut(det)

        try:
            waveforms = tree[branch_name].array(library="ak")
            if method == "Sum":
                baseline = ak.mean(waveforms[:, :30], axis=1)
                waveforms_blsub = waveforms - baseline
                window = waveforms_blsub[:, int(ts_min):int(ts_max)]
                window_sum_np = ak.to_numpy(ak.sum(window, axis=1))
                is_fired = window_sum_np < val_cut
            else: continue

            initial_count = np.sum(final_mask)
            if must_fire: final_mask = final_mask & is_fired
            else: final_mask = final_mask & (~is_fired)
            
            removed = initial_count - np.sum(final_mask)
            status = "FIRED" if must_fire else "VETOED"
            print(f"    [PID] {det:<12} ({status}): cut {val_cut:.1f} -> Removed {removed} events")

        except Exception as e:
            print(f"    [PID] CRITICAL ERROR processing {det}: {e}")
            continue
            
    print(f"  [PID] Total {particle_type.upper()} surviving: {np.sum(final_mask)} / {n_entries}")
    return final_mask

# ================= GRIDS =================
QUARTZ_GRID = [
    [None,  "002", None,  None],
    ["006", "004", "206", "204"],
    ["016", "014", "216", "214"],
    ["026", "024", "226", "224"],
    [None,  "030", None,  None],
    [None,  "034", None,  None],
    ["106", "104", None, "304"],
    ["116", "114", None, "314"],
    ["126", "124", "326", "324"],
    [None,  "134", None,  "334"],
]

PLASTIC_GRID = [
    [None,  "000", "202", "200"],
    ["012", "010", "212", "210"],
    ["022", "020", "222", "220"],
    ["032", None,  "232", "230"],
    ["102", "100", "302", "300"],
    ["112", "110", "312", "310"],
    ["122", "120", "322", "320"],
    ["132", "130", "332", "330"],
]

SCI_GRID = [
    ["003", "001", "203", "201"],
    ["007", "005", "207", "205"],
    ["013", "011", "213", "211"],
    ["017", "015", "217", "215"],
    ["023", "021", "223", "221"],
    ["027", "025", "227", "225"],
    ["033", "031", "233", "231"],
    [None,  "035", None,  "235"],
    ["103", "101", "303", "301"],
    ["107", "105", "307", "305"],
    ["113", "111", "313", "311"],
    ["117", "115", "317", "315"],
    ["123", "121", "323", "321"],
    ["127", "125", "327", "325"],
    ["133", "131", "333", "331"],
    [None,  "135", None,  "335"],
]

FAMILIES = {
    "CER-Quartz": QUARTZ_GRID,
    "CER-Plastic": PLASTIC_GRID,
    "SCI": SCI_GRID,
}

# ================= HELPERS =================
def _infer_run_label(path: str) -> str:
    m = re.search(r"(run\d+_\d{11,12})", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]

def parse_code(code: str):
    return int(code[0]), int(code[1]), int(code[2])

def branch_name(b, g, c):
    return f"tfinal_Board{b}_Group{g}_Channel{c}"

def positions_from_grid_xy(grid):
    pos = {}
    for row_idx, row in enumerate(grid):
        for col_idx, code in enumerate(row):
            if code is None: continue
            b, g, ch = parse_code(code)
            pos[(b, g, ch)] = (float(col_idx), float(row_idx), code)
    return pos

def load_calibration_shifts(json_path):
    if not json_path or not os.path.exists(json_path):
        print(f"\nWARNING: Calibration JSON not found at {json_path}. Proceeding uncalibrated.")
        return {}
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    shifts = {}
    for fam, fam_shifts in data.get("shifts_by_family", {}).items():
        for k_str, shift_val in fam_shifts.items():
            b, g, ch = map(int, k_str.split("_"))
            shifts[(b, g, ch)] = float(shift_val)
    return shifts

def build_segments_for_event(ev_idx_local: int, arrays: dict, branch_by_key: dict, pos_map: dict, shifts: dict):
    xs, ys, zs, hover = [], [], [], []
    for key, (mapx, mapy, code) in pos_map.items():
        if key not in branch_by_key: continue
        br = branch_by_key[key]
        arr = arrays.get(br, None)
        if arr is None: continue

        v = arr[ev_idx_local]
        
        # Protect against awkward arrays returning lists instead of scalars
        if isinstance(v, (list, np.ndarray, ak.Array)):
            if len(v) == 0: continue
            v = v[0]
            
        if not np.isfinite(v): continue
        
        shift = shifts.get(key, 0.0)
        t_shifted = float(abs(v)) + shift
        
        if t_shifted < XLIM[0] or t_shifted > XLIM[1]: continue

        # ================= AXIS MAPPING =================
        # X: Physical Grid Column
        # Z: Physical Grid Row (Height/Vertical)
        # Y: TOA (Horizontal Depth)
        xs += [mapx, mapx, None]
        zs += [mapy, mapy, None]
        ys += [XLIM[0], t_shifted, None] 
        
        hover += [f"<b>{code}</b><br>TOA (Calibrated): {t_shifted:.3f} ns<br>Raw: {abs(v):.3f} | Shift: {shift:.3f}"] * 3

    if not xs:
        xs, ys, zs, hover = [], [], [], []
        
    return xs, ys, zs, hover

# ================= MAIN =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1429_250926183919_converted_timingskim.root")
    ap.add_argument("--shifts-json", default=DEFAULT_SHIFT_JSON)
    ap.add_argument("--outdir", default="./4Dplots_pid_movie")
    ap.add_argument("--event-start", type=int, default=0)
    ap.add_argument("--max-events", type=int, default=500)
    ap.add_argument("--filter-pid", default='pion', choices=["muon", "pion", "electron", "proton"])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    run_label = _infer_run_label(args.input)
    
    pos_maps = {
        "CER-Quartz": positions_from_grid_xy(FAMILIES["CER-Quartz"]),
        "CER-Plastic": positions_from_grid_xy(FAMILIES["CER-Plastic"]),
        "SCI": positions_from_grid_xy(FAMILIES["SCI"])
    }
    
    all_xs, all_ys = [], []
    for pmap in pos_maps.values():
        all_xs.extend([p[0] for p in pmap.values()])
        all_ys.extend([p[1] for p in pmap.values()])
    x_min, x_max = min(all_xs), max(all_xs)
    z_min, z_max = min(all_ys), max(all_ys) 

    wanted_branches = {key: branch_name(*key) for pmap in pos_maps.values() for key in pmap.keys()}
    shifts = load_calibration_shifts(args.shifts_json)

    with uproot.open(args.input) as f:
        tree = f[TREE_NAME]
        tree_keys = set(tree.keys())
        n_entries = int(tree.num_entries)

        print("\n=== Evaluating PID Masks ===")
        master_pid_masks = {}
        for ptype in ["muon", "pion", "electron", "proton"]:
            mask = compute_pid_mask(tree, ptype)
            if mask is not None:
                master_pid_masks[ptype] = mask
        print("============================\n")

        all_ev_indices = np.arange(n_entries)
        if args.filter_pid and args.filter_pid in master_pid_masks:
            valid_indices = all_ev_indices[master_pid_masks[args.filter_pid]]
            valid_indices = valid_indices[valid_indices >= args.event_start][:args.max_events]
            ev_indices = list(valid_indices)
            print(f"Plotting {len(ev_indices)} events strictly matching PID: {args.filter_pid.upper()}")
        else:
            ev_indices = list(range(args.event_start, n_entries))[:args.max_events]

        if not ev_indices:
            print("No events found matching your criteria.")
            return

        branch_by_key = {k: br for k, br in wanted_branches.items() if br in tree_keys}
        branch_list = list(branch_by_key.values())

        fig = make_subplots(
            rows=1, cols=3, 
            specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=("CER Quartz", "CER Plastic", "SCI"),
            horizontal_spacing=0.02
        )
        
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(width=8.0, color="red"), name="Quartz"), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(width=8.0, color="blue"), name="Plastic"), row=1, col=2)
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(width=8.0, color="green"), name="SCI"), row=1, col=3)

        frames = []
        chunk_size = 5000
        chunk_start = (ev_indices[0] // chunk_size) * chunk_size

        print("Building 3D Subplot Frames...")
        while chunk_start <= ev_indices[-1]:
            chunk_stop = min(n_entries, chunk_start + chunk_size)
            in_chunk = [e for e in ev_indices if chunk_start <= e < chunk_stop]
            if not in_chunk:
                chunk_start += chunk_size
                continue

            arrays = tree.arrays(branch_list, entry_start=chunk_start, entry_stop=chunk_stop, library="np")

            for ev_global in in_chunk:
                ev_local = int(ev_global - chunk_start)
                
                event_pids_found = [ptype for ptype, mask in master_pid_masks.items() if mask[ev_global]]
                pid_display = ", ".join(event_pids_found) if event_pids_found else "Unclassified"

                q_x, q_y, q_z, q_h = build_segments_for_event(ev_local, arrays, branch_by_key, pos_maps["CER-Quartz"], shifts)
                p_x, p_y, p_z, p_h = build_segments_for_event(ev_local, arrays, branch_by_key, pos_maps["CER-Plastic"], shifts)
                s_x, s_y, s_z, s_h = build_segments_for_event(ev_local, arrays, branch_by_key, pos_maps["SCI"], shifts)

                frame_data = [
                    go.Scatter3d(x=q_x, y=q_y, z=q_z, text=q_h),
                    go.Scatter3d(x=p_x, y=p_y, z=p_z, text=p_h),
                    go.Scatter3d(x=s_x, y=s_y, z=s_z, text=s_h)
                ]

                frame_name = f"Ev {ev_global} | PID: {pid_display.upper()}"
                frames.append(go.Frame(name=frame_name, data=frame_data, traces=[0, 1, 2]))

            chunk_start += chunk_size

        fig.frames = frames

        slider_steps = []
        for fr in frames:
            slider_steps.append(dict(
                method="animate",
                label=fr.name,
                args=[[fr.name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))]
            ))

        # Z is up (Plotly Default)
        axis_cfg = dict(
            xaxis=dict(title="Grid X", range=[x_min - GRID_PADDING, x_max + GRID_PADDING], showgrid=True),
            yaxis=dict(title="TOA [ns] (Depth)", range=[XLIM[0], XLIM[1]*1.05], showgrid=True),
            zaxis=dict(title="Grid Y (Height)", range=[z_min - GRID_PADDING, z_max + GRID_PADDING], showgrid=True),
            camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=-1.5, y=-1.5, z=0.8)), 
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=2.0, z=1.2)
        )

        fig.update_layout(
            title=f"Tri-Module 3D Event Movie | Post-Flushing | Run {run_label}",
            scene=axis_cfg,
            scene2=axis_cfg,
            scene3=axis_cfg,
            sliders=[dict(active=0, currentvalue=dict(prefix=""), pad=dict(t=30), steps=slider_steps)],
            updatemenus=[dict(
                type="buttons", showactive=False, x=0.02, y=0.95,
                buttons=[
                    dict(label="Play Movie", method="animate", args=[None, dict(frame=dict(duration=150, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
                    dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate")])
                ]
            )]
        )

        filter_str = f"_Filtered_{args.filter_pid}" if args.filter_pid else ""
        outname = os.path.join(args.outdir, f"3D_MOVIE_ALL_FAMILIES_{run_label}{filter_str}.html")
        fig.write_html(outname, include_plotlyjs="cdn", full_html=True)
        print(f"Movie Saved: {outname}")

if __name__ == "__main__":
    main()