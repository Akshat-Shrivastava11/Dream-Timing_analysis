#!/usr/bin/env python3
"""
make_3d_grid_towers_event_slider.py

Option A: Plotly HTML with an EVENT SLIDER (animation) for a full run (NO calibration).

DISPLAY (HARD-CODED)  [UPDATED: X <-> Y SWAPPED]:
  - display X = mapping y
  - display Y = |tfinal|  (tower length along Y)
  - display Z = mapping x

Per event:
  - For each channel: tower is a 3D LINE segment from Y=0 to Y=|tfinal| at (X=mapy, Z=mapx).

Missing / bad values:
  - If tfinal is non-finite or outside XLIM:
      --missing skip  -> no segment
      --missing zero  -> segment of length 0 (a point at y=0)

Performance notes:
  - HTML size grows ~ linearly with number of frames (events).
  - Use --max-events and/or --stride.
"""

import os
import re
import json
import argparse
import numpy as np
import uproot

try:
    import plotly.graph_objects as go
except Exception:
    go = None


# ================= CONFIG =================
TREE_NAME = "EventTree"
XLIM = (4.0, 25.0)   # range for |tfinal|
Z_EPS = 1e-3         # floor for log axis (must be > 0)


# ================= GRIDS =================
QUARTZ_GRID = [
    [None,"603","602","601","600",None],
    [None,"697","606",None,None,None],
    [None,"613","612","611","610",None],
    [None,"617","616","615","614",None],
    [None,"625","624","623","622",None],
    ["637","631","630","627","626","636"],
    [None,"635","634","633","632",None],
    [None,None,"002",None,None,None],
    [None,"006","004","206","204",None],
    [None,"016","014","216","214",None],
    [None,"026","024","226","224",None],
    [None,None,"030",None,None,None],
    [None,None,"034",None,None,None],
    [None,"106","104","306","304",None],
    [None,"116","114","316","314",None],
    [None,"126","124","326","324",None],
    [None,"532","134","536","334",None],
    [None,"403","402","401","400",None],
    ["437","407","406","405","404","436"],
    [None,"413","412","411","410",None],
    [None,"417","416","415","414",None],
    [None,"425","424","423","422",None],
    [None,None,"427","426",None,None],
    [None,"433","432","431","430",None],
]

PLASTIC_GRID = [
    [None,"603","602","601","600",None],
    [None,"697","606",None,None,None],
    [None,"613","612","611","610",None],
    [None,None,None,None,None,None],
    [None,None,None,None,None,None],
    [None,"000","202","200",None,None],
    [None,"012","010","212","210",None],
    [None,"022","020","222","220",None],
    [None,"032",None,"232","230",None],
    [None,"102","100","302","300",None],
    [None,"112","110","312","310",None],
    [None,"122","120","322","320",None],
    [None,"132","130","332","330",None],
    [None,None,None,None,None,None],
    [None,None,None,None,None,None],
    [None,"425","424","423","422",None],
    [None,None,"427","426",None,None],
    [None,"433","432","431","430",None],
]

SCI_GRID = [
    [None,None,"605","604",None,None],
    [None,None,None,None,None,None],
    [None,None,"621","620",None,None],
    [None,None,None,None,None,None],
    [None,None,None,None,None,None],
    [None,"003","001","203","201",None],
    [None,"007","005","207","205",None],
    [None,"013","011","213","211",None],
    [None,"017","015","217","215",None],
    [None,"023","021","223","221",None],
    [None,"027","025","227","225",None],
    [None,"033","031","233","231",None],
    [None,None,"035",None,"235",None],
    [None,"103","101","303","301",None],
    [None,"107","105","307","305",None],
    [None,"113","111","313","311",None],
    [None,"117","115","317","315",None],
    [None,"123","121","323","321",None],
    [None,"127","125","327","325",None],
    [None,"133","131","333","331",None],
    [None,"533","135","537","335",None],
    [None,None,None,None,None,None],
    [None,None,None,None,None,None],
    [None,None,"421","420",None,None],
    [None,None,None,None,None,None],
    [None,None,"425","434",None,None],
]

CER_ALL_GRID = [
    ["603","602","601","600"],
    [None,"697","606",None],
    ["613","612","611","610"],
    ["617","616","615","614"],
    ["625","624","623","622"],
    ["637","631","630","627","626","636"],
    ["635","634","633","632"],
    ["002","000","202","200"],
    ["006","004","206","204"],
    ["012","010","212","210"],
    ["016","014","216","214"],
    ["022","020","222","220"],
    ["026","024","226","224"],
    ["032","030","232","230"],
    [None,"034",None,"234"],
    ["102","100","302","300"],
    ["106","104","306","304"],
    ["112","110","312","310"],
    ["116","114","316","314"],
    ["122","120","322","320"],
    ["126","124","326","324"],
    ["132","130","332","330"],
    ["532","134","536","334"],
    ["403","402","401","400"],
    ["437","407","406","405","404","436"],
    ["413","412","411","410"],
    ["417","416","415","414"],
    ["425","424","423","422"],
    [None,"427","426",None],
    ["433","432","431","430"],
]

FAMILIES = {
    "CER-Quartz": QUARTZ_GRID,
    "CER-Plastic": PLASTIC_GRID,
    "SCI": SCI_GRID,
    "CER-All": CER_ALL_GRID,
}


# ================= HELPERS =================
def _infer_run_label(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"(run\d+_\d{11,12})", base)
    return m.group(1) if m else os.path.splitext(base)[0]

def parse_code(code: str):
    return int(code[0]), int(code[1]), int(code[2])

def branch_name(b, g, c):
    return f"tfinal_Board{b}_Group{g}_Channel{c}"

def positions_from_grid_xy(grid):
    pos = {}
    for row_idx, row in enumerate(grid):
        for col_idx, code in enumerate(row):
            if code is None:
                continue
            b, g, ch = parse_code(code)
            pos[(b, g, ch)] = (float(row_idx), float(col_idx), code)  # mapx=row, mapy=col
    return pos

def positions_from_xy_map(grid, xy_map_path):
    with open(xy_map_path, "r") as f:
        mapping = json.load(f)
    pos = {}
    for row in grid:
        for code in row:
            if code is None or code not in mapping:
                continue
            b, g, ch = parse_code(code)
            mapx, mapy = mapping[code]
            pos[(b, g, ch)] = (float(mapx), float(mapy), code)
    return pos

def value_for_event(v, xlim, missing="skip"):
    if not np.isfinite(v):
        return None if missing == "skip" else 0.0
    a = float(abs(v))
    if a < xlim[0] or a > xlim[1]:
        return None if missing == "skip" else 0.0
    return a

def build_segments_for_event(
    ev_idx_local: int,
    arrays: dict,
    branch_by_key: dict,
    pos_map: dict,
    keys_in_group: set,
    missing: str,
):
    """
    UPDATED DISPLAY:
      X = mapy
      Y = time
      Z = mapx
    """
    xs, ys, zs = [], [], []
    hover = []

    for key in keys_in_group:
        if key not in pos_map:
            continue

        mapx, mapy, code = pos_map[key]

        if key not in branch_by_key:
            if missing == "zero":
                # zero-length at Y=0
                xs += [mapy, mapy, None]
                ys += [0.0, 0.0, None]
                zs += [mapx, mapx, None]
                hover += [f"{code} {key}<br>|tfinal|=0 (missing)"] * 3
            continue

        br = branch_by_key[key]
        arr = arrays.get(br, None)
        if arr is None:
            continue

        v = arr[ev_idx_local]
        t = value_for_event(v, XLIM, missing=missing)
        if t is None:
            continue

        # segment from Y=0 -> Y=t at fixed X=mapy, Z=mapx
        xs += [mapy, mapy, None]
        ys += [0.0, max(t, Z_EPS), None]
        zs += [mapx, mapx, None]
        hover += [f"<b>{code}</b><br>(b,g,ch)={key}<br>|tfinal|={t:.4f} ns"] * 3

    return xs, ys, zs, hover


# ================= MAIN =================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True, help="Input ROOT file containing EventTree and tfinal branches.")
    ap.add_argument("--outdir", default="./4Dplots_event_slider", help="Output directory.")
    ap.add_argument("--family", choices=list(FAMILIES.keys()), default="CER-All")
    ap.add_argument("--xy-map", default=None, help="Optional JSON mapping {code: [x,y], ...}")

    ap.add_argument("--interactive", action="store_true", help="Write interactive HTML (plotly).")
    ap.add_argument("--zscale", choices=["linear", "log"], default="log",
                    help="Axis scaling for TIME axis (now display-Y).")

    ap.add_argument("--event-start", type=int, default=0, help="Starting event index (0-based).")
    ap.add_argument("--stride", type=int, default=10, help="Take every Nth event.")
    ap.add_argument("--max-events", type=int, default=300, help="Max frames to include in slider.")

    ap.add_argument("--chunk-size", type=int, default=5000,
                    help="How many consecutive entries to read per uproot chunk.")

    ap.add_argument("--missing", choices=["skip", "zero"], default="skip",
                    help="If tfinal is non-finite/outside XLIM: skip segment or draw 0-length.")

    ap.add_argument("--line-width", type=float, default=8.0, help="3D line width (screen-space).")
    ap.add_argument("--no-labels", action="store_true", help="Disable code labels (static).")
    ap.add_argument("--label-x", type=float, default=0.0,
                    help="Place static labels at fixed TIME value (now display-Y). Default 0.0.")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if not args.interactive:
        raise SystemExit("This script is Option A (slider). Please pass --interactive.")

    if go is None:
        raise SystemExit("plotly not available in this environment.")

    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    if args.max_events < 1:
        raise SystemExit("--max-events must be >= 1")
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be >= 1")

    run_label = _infer_run_label(args.input)
    grid = FAMILIES[args.family]

    # mapping
    pos_map = positions_from_xy_map(grid, args.xy_map) if args.xy_map else positions_from_grid_xy(grid)

    # wanted branches
    wanted_keys = list(pos_map.keys())
    wanted_branches = {key: branch_name(*key) for key in wanted_keys}

    # Color grouping logic
    quartz_codes = {c for row in QUARTZ_GRID for c in row if c is not None}
    plastic_codes = {c for row in PLASTIC_GRID for c in row if c is not None}

    quartz_keys, plastic_keys, other_keys, sci_keys = set(), set(), set(), set()
    for key, (_mx, _my, code) in pos_map.items():
        if args.family == "SCI":
            sci_keys.add(key)
        else:
            if code in quartz_codes:
                quartz_keys.add(key)
            elif code in plastic_codes:
                plastic_keys.add(key)
            else:
                other_keys.add(key)

    with uproot.open(args.input) as f:
        tree = f[TREE_NAME]
        tree_keys = set(tree.keys())
        n_entries = int(tree.num_entries)

        start = max(0, int(args.event_start))
        if start >= n_entries:
            raise SystemExit(f"--event-start {start} >= num_entries {n_entries}")

        ev_indices = list(range(start, n_entries, int(args.stride)))
        ev_indices = ev_indices[: int(args.max_events)]
        if len(ev_indices) == 0:
            raise SystemExit("No events selected (check --event-start/--stride/--max-events).")

        branch_by_key = {}
        branch_list = []
        for key, br in wanted_branches.items():
            if br in tree_keys:
                branch_by_key[key] = br
                branch_list.append(br)

        if len(branch_list) == 0:
            raise SystemExit("None of the tfinal branches were found in the file for this family/grid.")

        # Axis extents from mapping:
        #   display X = mapy  -> use mapy range
        #   display Z = mapx  -> use mapx range
        mapx_vals = [pos_map[k][0] for k in pos_map]
        mapy_vals = [pos_map[k][1] for k in pos_map]
        x_min, x_max = min(mapy_vals), max(mapy_vals)  # display-x is mapy
        z_min, z_max = min(mapx_vals), max(mapx_vals)  # display-z is mapx

        fig = go.Figure()
        trace_ids = {}

        if args.family == "SCI":
            fig.add_trace(
                go.Scatter3d(
                    x=[], y=[], z=[],
                    mode="lines",
                    line=dict(width=float(args.line_width), color="green"),
                    hoverinfo="text",
                    text=[],
                    name="SCI",
                    showlegend=True,
                )
            )
            trace_ids["SCI"] = 0
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=[], y=[], z=[],
                    mode="lines",
                    line=dict(width=float(args.line_width), color="red"),
                    hoverinfo="text",
                    text=[],
                    name="Quartz",
                    showlegend=True,
                )
            )
            trace_ids["Quartz"] = 0

            fig.add_trace(
                go.Scatter3d(
                    x=[], y=[], z=[],
                    mode="lines",
                    line=dict(width=float(args.line_width), color="blue"),
                    hoverinfo="text",
                    text=[],
                    name="Plastic",
                    showlegend=True,
                )
            )
            trace_ids["Plastic"] = 1

            if len(other_keys) > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=[], y=[], z=[],
                        mode="lines",
                        line=dict(width=float(args.line_width), color="black"),
                        hoverinfo="text",
                        text=[],
                        name="Other",
                        showlegend=True,
                    )
                )
                trace_ids["Other"] = 2

        # Static labels at fixed TIME value -> that's display-Y now
        if not args.no_labels:
            codes = []
            lx, ly, lz = [], [], []
            for key, (mapx, mapy, code) in pos_map.items():
                codes.append(code)
                lx.append(float(mapy))            # X = mapy
                ly.append(float(args.label_x))    # Y = time (fixed)
                lz.append(float(mapx))            # Z = mapx
            fig.add_trace(
                go.Scatter3d(
                    x=lx, y=ly, z=lz,
                    mode="text",
                    text=codes,
                    textfont=dict(size=11, color="white"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        frames = []

        first_ev = ev_indices[0]
        last_ev = ev_indices[-1]
        chunk_start = (first_ev // args.chunk_size) * args.chunk_size

        while chunk_start <= last_ev:
            chunk_stop = min(n_entries, chunk_start + int(args.chunk_size))
            in_chunk = [e for e in ev_indices if (e >= chunk_start and e < chunk_stop)]
            if len(in_chunk) == 0:
                chunk_start += int(args.chunk_size)
                continue

            arrays = tree.arrays(
                branch_list,
                entry_start=int(chunk_start),
                entry_stop=int(chunk_stop),
                library="np",
            )

            for ev_global in in_chunk:
                ev_local = int(ev_global - chunk_start)

                if args.family == "SCI":
                    xg, yg, zg, hg = build_segments_for_event(
                        ev_local, arrays, branch_by_key, pos_map, sci_keys, args.missing
                    )
                    frames.append(
                        go.Frame(
                            name=str(ev_global),
                            data=[go.Scatter3d(x=xg, y=yg, z=zg, text=hg)],
                            traces=[trace_ids["SCI"]],
                        )
                    )
                else:
                    xq, yq, zq, hq = build_segments_for_event(
                        ev_local, arrays, branch_by_key, pos_map, quartz_keys, args.missing
                    )
                    xp, yp, zp, hp = build_segments_for_event(
                        ev_local, arrays, branch_by_key, pos_map, plastic_keys, args.missing
                    )

                    data_list = [
                        go.Scatter3d(x=xq, y=yq, z=zq, text=hq),
                        go.Scatter3d(x=xp, y=yp, z=zp, text=hp),
                    ]
                    trace_list = [trace_ids["Quartz"], trace_ids["Plastic"]]

                    if "Other" in trace_ids:
                        xo, yo, zo, ho = build_segments_for_event(
                            ev_local, arrays, branch_by_key, pos_map, other_keys, args.missing
                        )
                        data_list.append(go.Scatter3d(x=xo, y=yo, z=zo, text=ho))
                        trace_list.append(trace_ids["Other"])

                    frames.append(go.Frame(name=str(ev_global), data=data_list, traces=trace_list))

            chunk_start += int(args.chunk_size)

        if len(frames) == 0:
            raise SystemExit("No frames built (unexpected). Check branches/XLIM/missing settings.")

        fig.frames = frames

        # ---- Layout axes (UPDATED: X=mapy, Y=time, Z=mapx) ----
        PAD_X = 2.0
        PAD_Z = 2.0

        xaxis = dict(
            title="mapping y",
            range=[x_min - PAD_X, x_max + PAD_X],
            showgrid=True,
            zeroline=False,
        )
        yaxis = dict(
            title="|tfinal| [ns]" + (" (log axis)" if args.zscale == "log" else ""),
            showgrid=True,
            zeroline=False,
        )
        zaxis = dict(
            title="mapping x",
            range=[z_min - PAD_Z, z_max + PAD_Z],
            showgrid=True,
            zeroline=False,
        )

        # log/linear scaling applies to TIME axis -> display-Y
        if args.zscale == "log":
            yaxis.update(type="log", range=[np.log10(max(XLIM[0], Z_EPS)), np.log10(max(XLIM[1], Z_EPS))])
        else:
            yaxis.update(range=[0.0, float(XLIM[1]) * 1.05])

        steps = []
        for fr in frames:
            steps.append(
                dict(
                    method="animate",
                    label=f"ev {fr.name}",
                    args=[
                        [fr.name],
                        dict(mode="immediate",
                             frame=dict(duration=0, redraw=True),
                             transition=dict(duration=0)),
                    ],
                )
            )

        fig.update_layout(
            title=f"{args.family} {run_label} EVENT SLIDER |tfinal| (start={args.event_start}, stride={args.stride}, max={args.max_events})",
            scene=dict(
                xaxis=xaxis,
                yaxis=yaxis,
                zaxis=zaxis,
                aspectmode="manual",
                aspectratio=dict(x=1.2, y=2.2, z=2.0),
                camera=dict(eye=dict(x=1.7, y=1.4, z=1.5), up=dict(x=0, y=0, z=1)),
            ),
            margin=dict(l=0, r=0, t=60, b=0),
            sliders=[dict(active=0, currentvalue=dict(prefix="Event: "), pad=dict(t=30), steps=steps)],
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.02, y=0.95,
                    xanchor="left", yanchor="top",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(frame=dict(duration=60, redraw=True),
                                     transition=dict(duration=0),
                                     fromcurrent=True,
                                     mode="immediate"),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(frame=dict(duration=0, redraw=False),
                                     transition=dict(duration=0),
                                     mode="immediate"),
                            ],
                        ),
                    ],
                )
            ],
        )

        outname = (
            f"3D_TOWERS_SLIDER_XYswapped_{args.family}_{run_label}"
            f"_start{args.event_start}_stride{args.stride}_max{args.max_events}"
            f"_missing{args.missing}_xlim{XLIM[0]}to{XLIM[1]}.html"
        )
        outhtml = os.path.join(args.outdir, outname)
        fig.write_html(outhtml, include_plotlyjs="cdn", full_html=True)
        print("Saved:", outhtml)
        print(f"[INFO] Frames: {len(frames)}  (events shown: {frames[0].name} ... {frames[-1].name})")


if __name__ == "__main__":
    main()
