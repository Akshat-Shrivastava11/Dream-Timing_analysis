#!/usr/bin/env python3
"""
Professional 3D detector-map towers (Plotly) + optional PNG (matplotlib).

Goal:
  - mapping plane looks like the real rectangular channel layout
  - each channel is a SOLID 3D tower (rectangular prism), not a line
  - hover + labels show the *channel code* at its true location
  - CER-All: Quartz (red), Plastic (blue)
  - SCI: black

Coordinates:
  x = mapping x  (default: row index in grid; or from --xy-map JSON)
  y = mapping y  (default: col index in grid; or from --xy-map JSON)
  z = mean(|tfinal| + shift) in ns  (post calibration)

Mapping override:
  --xy-map mapping.json with format:
    { "002": [x,y], "000": [x,y], ... }

Usage:
  python3 make_3d_grid_towers.py --interactive
  python3 make_3d_grid_towers.py --interactive --calib-stat mode
  python3 make_3d_grid_towers.py --interactive --xy-map cer_all_map.json --cell-size 0.95

Notes:
  - Plotly must be installed (pip install plotly).
  - Keeps your calibration logic (anchors + gaussian-fit anchor peak) intact.
"""

import os
import re
import json
import argparse
from typing import Dict, Tuple, Optional

import numpy as np
import uproot
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import plotly.graph_objects as go


# ================= CONFIG =================
TREE_NAME = "EventTree"
NBINS = 200
XLIM = (8.0, 15.0)

MIN_RAW = 500
MIN_ENTRIES = 200


# ================= GRIDS =================
QUARTZ_GRID = [
    [None, "002", None, None],
    ["006", "004", "206", "204"],
    ["016", "014", "216", "214"],
    ["026", "024", "226", "224"],
    [None, "030", None, None],
    [None, "034", None, None],
    ["106", "104", "306", "304"],
    ["116", "114", "316", "314"],
    ["126", "124", "326", "324"],
    [None, "134", None, "334"],
]

PLASTIC_GRID = [
    [None, "000", "202", "200"],
    ["012", "010", "212", "210"],
    ["022", "020", "222", "220"],
    ["032", None, "232", "230"],
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
    [None, "035", None, "235"],
    ["103", "101", "303", "301"],
    ["107", "105", "307", "305"],
    ["113", "111", "313", "311"],
    ["117", "115", "317", "315"],
    ["123", "121", "323", "321"],
    ["127", "125", "327", "325"],
    ["133", "131", "333", "331"],
    [None, "135", None, "335"],
]

CER_ALL_GRID = [
    ["002", "000", "202", "200"],
    ["006", "004", "206", "204"],
    ["012", "010", "212", "210"],
    ["016", "014", "216", "214"],
    ["022", "020", "222", "220"],
    ["026", "024", "226", "224"],
    ["032", "030", "232", "230"],
    [None, "034", None, "234"],
    ["102", "100", "302", "300"],
    ["106", "104", "306", "304"],
    ["112", "110", "312", "310"],
    ["116", "114", "316", "314"],
    ["122", "120", "322", "320"],
    ["126", "124", "326", "324"],
    ["132", "130", "332", "330"],
    [None, "134", None, "334"],
]

FAMILIES = {
    "CER-Quartz": QUARTZ_GRID,
    "CER-Plastic": PLASTIC_GRID,
    "SCI": SCI_GRID,
    "CER-All": CER_ALL_GRID,
}

# Fixed anchors (b,g,ch)
ANCHORS: Dict[str, Tuple[int, int, int]] = {
    "SCI": (1, 0, 7),
    "CER-Quartz": (1, 0, 4),
    "CER-Plastic": (1, 0, 0),
}


# ================= UTILITIES =================
def _infer_run_label(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"(run\d+_\d{11,12})", base)
    return m.group(1) if m else os.path.splitext(base)[0]


def parse_code(code: str) -> Tuple[int, int, int]:
    return int(code[0]), int(code[1]), int(code[2])


def branch_name(b: int, g: int, c: int) -> str:
    return f"tfinal_Board{b}_Group{g}_Channel{c}"


def prep_array(arr: np.ndarray) -> Optional[np.ndarray]:
    arr = np.abs(arr)
    arr = arr[np.isfinite(arr)]
    arr = arr[(arr >= XLIM[0]) & (arr <= XLIM[1])]
    if arr.size < MIN_ENTRIES:
        return None
    return arr


def hist_stats(arr: np.ndarray, bins: np.ndarray):
    h, edges = np.histogram(arr, bins=bins)
    if h.sum() == 0:
        return None
    mu = float(arr.mean())
    mode = float(0.5 * (edges[np.argmax(h)] + edges[np.argmax(h) + 1]))
    return mu, mode, h


def _gauss(x, A, mu, sig):
    return A * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def fit_gaussian_to_peak(arr_abs: np.ndarray, bins: np.ndarray, window: float = 0.5):
    h, edges = np.histogram(arr_abs, bins=bins)
    centers = 0.5 * (edges[1:] + edges[:-1])
    if h.sum() == 0:
        return False, np.nan, np.nan, np.nan

    imax = int(np.argmax(h))
    x0 = float(centers[imax])

    m = (centers >= x0 - window) & (centers <= x0 + window)
    x = centers[m]
    y = h[m]

    if x.size < 6 or y.max() < 5:
        return False, np.nan, np.nan, np.nan

    p0 = [float(y.max()), x0, 0.15]
    bounds = ([0.0, x0 - window, 0.02], [np.inf, x0 + window, 2.0])

    try:
        popt, _ = curve_fit(_gauss, x, y, p0=p0, bounds=bounds, maxfev=10000)
        A, mu, sig = map(float, popt)
        return True, mu, sig, A
    except Exception:
        return False, np.nan, np.nan, np.nan


def derive_family_calibration_fixed_anchor(root_file: str, grid, anchor_key, calib_stat: str = "mean"):
    """
    calib_stat:
      - "mean": shift = anchor_mu - mean(channel)
      - "mode": shift = anchor_mu - mode(channel)

    anchor_mu is Gaussian-fit peak (fallback to anchor mean).
    """
    if calib_stat not in ("mean", "mode"):
        raise ValueError(f"--calib-stat must be 'mean' or 'mode' (got {calib_stat})")

    bins = np.linspace(*XLIM, NBINS + 1)
    stats = {}
    arrays = {}

    with uproot.open(root_file) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        for row in grid:
            for code in row:
                if code is None:
                    continue
                b, g, c = parse_code(code)
                k = branch_name(b, g, c)
                if k not in keys:
                    continue

                raw = tree[k].array(library="np")
                if raw.size < MIN_RAW:
                    continue

                arr = prep_array(raw)
                if arr is None:
                    continue

                arrays[(b, g, c)] = arr
                st = hist_stats(arr, bins)
                if st is None:
                    continue
                mu, mode, _ = st
                stats[(b, g, c)] = {"N": int(arr.size), "mu": float(mu), "mode": float(mode)}

    if anchor_key not in arrays or anchor_key not in stats:
        raise RuntimeError(f"Anchor {anchor_key} not usable/found in this family for {root_file}")

    anchor_arr = arrays[anchor_key]
    fit_ok, mu_fit, sig_fit, _A = fit_gaussian_to_peak(anchor_arr, bins, window=0.5)
    anchor_mu = float(mu_fit) if (fit_ok and np.isfinite(mu_fit)) else float(stats[anchor_key]["mu"])

    shifts = {}
    for key, st in stats.items():
        loc = st["mu"] if calib_stat == "mean" else st["mode"]
        shifts[key] = float(anchor_mu - float(loc))

    anchor_info = {
        "mu": anchor_mu,
        "N": int(stats[anchor_key]["N"]),
        "fit_ok": bool(fit_ok),
        "sig_fit": float(sig_fit) if np.isfinite(sig_fit) else np.nan,
        "calib_stat": calib_stat,
    }
    return shifts, (anchor_key, anchor_info)


def codes_in_grid(grid):
    s = set()
    for row in grid:
        for code in row:
            if code is not None:
                s.add(code)
    return s


def positions_from_grid_xy(grid):
    """
    Default mapping:
      x = row index
      y = col index
    pos_map[key] = (x_map, y_map, code)
    """
    pos = {}
    for row_idx, row in enumerate(grid):
        for col_idx, code in enumerate(row):
            if code is None:
                continue
            b, g, ch = parse_code(code)
            pos[(b, g, ch)] = (float(row_idx), float(col_idx), code)
    return pos


def positions_from_xy_map(grid, xy_map_path: str):
    """
    JSON mapping: { "CODE": [x, y], ... }
    pos_map[key] = (x_map, y_map, code)
    """
    with open(xy_map_path, "r") as f:
        mapping = json.load(f)

    pos = {}
    for row in grid:
        for code in row:
            if code is None:
                continue
            if code not in mapping:
                continue
            b, g, ch = parse_code(code)
            x_map, y_map = mapping[code]
            pos[(b, g, ch)] = (float(x_map), float(y_map), code)
    return pos


def compute_post_mean_map(root_file: str, grid, shifts: Dict[Tuple[int, int, int], float]):
    """
    mean(|tfinal| + shift) per channel (post-calibration)
    """
    out = {}
    with uproot.open(root_file) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        for row in grid:
            for code in row:
                if code is None:
                    continue
                b, g, ch = parse_code(code)
                k = branch_name(b, g, ch)
                if k not in keys:
                    continue
                arr = prep_array(tree[k].array(library="np"))
                if arr is None:
                    continue
                out[(b, g, ch)] = float((arr + shifts.get((b, g, ch), 0.0)).mean())

    return out


# ================= OPTIONAL PNG (legacy) =================
def plot_3d_lines_matplotlib(outpng, title, pos_map, t_map, color_fn):
    """
    Simple spike PNG (optional). Interactive view is Plotly towers.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    xs, ys, ts = [], [], []
    for key, (x_map, y_map, _code) in pos_map.items():
        if key not in t_map:
            continue
        t = float(t_map[key])
        if np.isfinite(t):
            xs.append(x_map); ys.append(y_map); ts.append(t)

    if not ts:
        print(f"[WARN] No channels to plot for {title}; skipping {outpng}")
        plt.close(fig)
        return

    for key, (x_map, y_map, code) in pos_map.items():
        if key not in t_map:
            continue
        t = float(t_map[key])
        if not np.isfinite(t):
            continue
        colr = color_fn(code)
        ax.plot([x_map, x_map], [y_map, y_map], [0.0, t], color=colr, linewidth=2.0, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("mapping x")
    ax.set_ylabel("mapping y")
    ax.set_zlabel("time [ns]")
    ax.view_init(elev=25, azim=-55)
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close(fig)
    print("Saved:", outpng)


# ================= PLOTLY TOWERS (SOLID) =================
def _cuboid_mesh(xc, yc, z0, z1, dx, dy, color, hovertext=""):
    """
    One cuboid centered at (xc,yc), spanning z0->z1, footprint dx x dy.
    Implemented as 12 triangles via Mesh3d (solid-looking tower).
    """
    x0, x1 = xc - dx / 2.0, xc + dx / 2.0
    y0, y1 = yc - dy / 2.0, yc + dy / 2.0

    # 8 vertices
    X = [x0, x1, x1, x0,  x0, x1, x1, x0]
    Y = [y0, y0, y1, y1,  y0, y0, y1, y1]
    Z = [z0, z0, z0, z0,  z1, z1, z1, z1]

    # 12 triangles (2 per face)
    I = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    J = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    K = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]

    return go.Mesh3d(
        x=X, y=Y, z=Z,
        i=I, j=J, k=K,
        color=color,
        opacity=0.95,
        flatshading=True,
        hovertext=hovertext,
        hoverinfo="text",
        showscale=False,
    )


def plot_3d_towers_plotly(
    outhtml,
    title,
    pos_map,
    t_map,
    color_fn,
    tower_half=0.30,          # <-- smaller towers (was ~0.49)
    z_floor=0.0,
    pad_xy=1.2,               # <-- grid exceeds towers
    show_text=True,           # turn off if labels clutter
    text_size=11,
):
    """
    Interactive Plotly:
      x = mapping x
      y = mapping y
      z = time
    Towers are solid prisms from z=0 to z=time.

    Camera is rotated so Z axis appears more horizontal in the view.
    """
    if go is None:
        print("[WARN] plotly not available; skipping interactive HTML:", outhtml)
        return

    # helper: add one rectangular prism tower as Mesh3d (8 vertices, 12 triangles)
    def add_prism(fig, x, y, z_top, color, hovertext):
        # 8 vertices of a box centered at (x,y), spanning z=[0,z_top]
        x0, x1 = x - tower_half, x + tower_half
        y0, y1 = y - tower_half, y + tower_half
        z0, z1 = z_floor, z_top

        xs = [x0, x1, x1, x0,  x0, x1, x1, x0]
        ys = [y0, y0, y1, y1,  y0, y0, y1, y1]
        zs = [z0, z0, z0, z0,  z1, z1, z1, z1]

        # 12 triangles (two per face)
        i = [0,0,4,4, 0,0,1,1, 2,2,3,3]
        j = [1,2,5,6, 1,5,2,6, 3,7,0,4]
        k = [2,3,6,7, 5,4,6,5, 7,6,4,7]

        fig.add_trace(go.Mesh3d(
            x=xs, y=ys, z=zs,
            i=i, j=j, k=k,
            color=color,
            opacity=1.0,              # <-- filled/solid
            flatshading=True,
            hoverinfo="text",
            hovertext=hovertext,
            showscale=False,
            lighting=dict(
                ambient=0.35,
                diffuse=0.75,
                specular=0.20,
                roughness=0.90,
                fresnel=0.05,
            ),
            lightposition=dict(x=100, y=50, z=200),
            name="",
            showlegend=False,
        ))

    fig = go.Figure()

    # collect bounds
    xvals, yvals, zvals = [], [], []
    labels_x, labels_y, labels_z, labels_txt = [], [], [], []

    for key, (x_map, y_map, code) in pos_map.items():
        if key not in t_map:
            continue
        t = float(t_map[key])
        if not np.isfinite(t):
            continue

        x = float(x_map)
        y = float(y_map)
        z = float(t)

        xvals.append(x); yvals.append(y); zvals.append(z)

        hover = (
            f"<b>{code}</b><br>"
            f"(b,g,ch)={key}<br>"
            f"x={x:.2f}, y={y:.2f}<br>"
            f"time={z:.4f} ns"
        )
        add_prism(fig, x, y, z, color_fn(code), hover)

        # optional text at the top
        labels_x.append(x)
        labels_y.append(y)
        labels_z.append(z + 0.10)
        labels_txt.append(code)

    if not zvals:
        print("[WARN] No channels to plot for", title, "; skipping", outhtml)
        return

    xmin, xmax = min(xvals), max(xvals)
    ymin, ymax = min(yvals), max(yvals)
    zmax = max(zvals)

    # ---- floor plane (grid exceeds towers)
    # a simple surface at z=0 spanning x/y bounds with padding
    X = np.array([[xmin - pad_xy, xmax + pad_xy],
                  [xmin - pad_xy, xmax + pad_xy]])
    Y = np.array([[ymin - pad_xy, ymin - pad_xy],
                  [ymax + pad_xy, ymax + pad_xy]])
    Z = np.zeros_like(X) + z_floor

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        showscale=False,
        opacity=0.25,
        hoverinfo="skip",
        name="",
        colorscale=[[0, "rgb(80,80,80)"], [1, "rgb(80,80,80)"]],
    ))

    # ---- optional code labels (can clutter)
    if show_text:
        fig.add_trace(go.Scatter3d(
            x=labels_x, y=labels_y, z=labels_z,
            mode="text",
            text=labels_txt,
            textposition="top center",
            textfont=dict(size=text_size, color="white"),
            hoverinfo="skip",
            showlegend=False,
        ))

    # ---- KEY: rotate camera so z “looks” horizontal
    # Make Y the vertical direction in the VIEW => Z appears horizontal.
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="mapping x",
            yaxis_title="mapping y",
            zaxis_title="mean(post) time [ns]",
            xaxis=dict(range=[xmin - pad_xy, xmax + pad_xy], zeroline=False),
            yaxis=dict(range=[ymin - pad_xy, ymax + pad_xy], zeroline=False),
            zaxis=dict(range=[0.0, zmax * 1.10], zeroline=False),
            aspectmode="data",
            camera=dict(
                up=dict(x=0, y=1, z=0),     # <-- THIS rotates so z looks horizontal
                eye=dict(x=2.2, y=1.6, z=0.8),
            ),
        ),
        margin=dict(l=0, r=0, t=45, b=0),
    )

    fig.write_html(outhtml, include_plotlyjs=True, full_html=True)  # offline-safe
    print("Saved:", outhtml)



# ================= MAIN =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1501_250928105227_converted_timingskim.root")
    ap.add_argument("--test", default="/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1511_250928180741_converted_timingskim.root")
    ap.add_argument("--outdir", default="/lustre/research/hep/akshriva/Dream-Timing/TRUE-HGtiming/4Dplots")

    ap.add_argument("--calib-stat", choices=["mean", "mode"], default="mean",
                    help="Use channel mean or histogram mode when computing shifts (anchor uses Gaussian-fit peak).")

    ap.add_argument("--xy-map", default=None,
                    help="Optional JSON mapping {code: [x,y], ...}. If omitted, uses default (x=row, y=col).")

    ap.add_argument("--interactive", action="store_true",
                    help="Write interactive Plotly HTML with solid towers.")

    ap.add_argument("--cell-size", type=float, default=0.95,
                    help="Tower footprint size in mapping units (use ~0.95 for tight rectangle look).")

    ap.add_argument("--labels", action="store_true",
                    help="Draw channel codes on top of towers in the interactive view.")

    ap.add_argument("--camera", choices=["iso", "top", "side"], default="iso",
                    help="Initial camera preset for the interactive HTML.")

    ap.add_argument("--png", action="store_true",
                    help="Also write legacy PNG spikes using matplotlib (optional).")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ref_label = _infer_run_label(args.reference)
    test_label = _infer_run_label(args.test)
    suffix = f"calib_{args.calib_stat}"

    # ---- derive shifts from REFERENCE
    shifts_by_family = {}
    for fam in ["CER-Quartz", "CER-Plastic", "SCI"]:
        grid = FAMILIES[fam]
        anchor_key = ANCHORS[fam]
        shifts, (akey, ainfo) = derive_family_calibration_fixed_anchor(
            args.reference, grid, anchor_key, calib_stat=args.calib_stat
        )
        shifts_by_family[fam] = shifts
        print(f"[{fam}] anchor={akey} mu={ainfo['mu']:.4f} N={ainfo['N']} fit_ok={ainfo['fit_ok']} sig_fit={ainfo['sig_fit']:.3f}")

    # CER-All = Quartz + Plastic
    shifts_by_family["CER-All"] = {**shifts_by_family["CER-Quartz"], **shifts_by_family["CER-Plastic"]}

    quartz_codes = codes_in_grid(QUARTZ_GRID)
    plastic_codes = codes_in_grid(PLASTIC_GRID)

    def cher_color(code: str) -> str:
        if code in quartz_codes:
            return "red"
        if code in plastic_codes:
            return "blue"
        return "black"

    def sci_color(_code: str) -> str:
        return "green"

    def make_pos(grid):
        return positions_from_xy_map(grid, args.xy_map) if args.xy_map else positions_from_grid_xy(grid)

    # ---------- CER-All ----------
    fam = "CER-All"
    grid = FAMILIES[fam]
    shifts = shifts_by_family[fam]
    pos = make_pos(grid)

    z_ref = compute_post_mean_map(args.reference, grid, shifts)
    z_test = compute_post_mean_map(args.test, grid, shifts)

    if args.png:
        outpng = os.path.join(args.outdir, f"3D_LINES_{fam}_REF_{ref_label}_{suffix}.png")
        plot_3d_lines_matplotlib(outpng, f"{fam} REF post-calib mean ({suffix})", pos, z_ref, cher_color)
        outpng = os.path.join(args.outdir, f"3D_LINES_{fam}_TEST_{test_label}_calibfrom_{ref_label}_{suffix}.png")
        plot_3d_lines_matplotlib(outpng, f"{fam} TEST post-calib mean (calib from {ref_label}; {suffix})", pos, z_test, cher_color)

    if args.interactive:
        outhtml = os.path.join(args.outdir, f"3D_TOWERS_{fam}_REF_{ref_label}_{suffix}.html")


        plot_3d_towers_plotly(
            outhtml,
            f"{fam} REF post-calib mean ({suffix})",
            pos, z_ref, cher_color,
            tower_half=0.25,      # try 0.25–0.32
            pad_xy=1.5,
            show_text=False,      # <-- I strongly recommend False to avoid label pile-up
        )

        
        outhtml = os.path.join(args.outdir, f"3D_TOWERS_{fam}_TEST_{test_label}_calibfrom_{ref_label}_{suffix}.html")

        plot_3d_towers_plotly(
            outhtml,
            f"{fam} TEST post-calib mean (calib from {ref_label}; {suffix})",
            pos, z_ref, cher_color,
            tower_half=0.25,      # try 0.25–0.32
            pad_xy=1.5,
            show_text=False,      # <-- I strongly recommend False to avoid label pile-up
        )

    # ---------- SCI ----------
    fam = "SCI"
    grid = FAMILIES[fam]
    shifts = shifts_by_family[fam]
    pos = make_pos(grid)

    z_ref = compute_post_mean_map(args.reference, grid, shifts)
    z_test = compute_post_mean_map(args.test, grid, shifts)

    if args.png:
        outpng = os.path.join(args.outdir, f"3D_LINES_{fam}_REF_{ref_label}_{suffix}.png")
        plot_3d_lines_matplotlib(outpng, f"{fam} REF post-calib mean ({suffix})", pos, z_ref, sci_color)
        outpng = os.path.join(args.outdir, f"3D_LINES_{fam}_TEST_{test_label}_calibfrom_{ref_label}_{suffix}.png")
        plot_3d_lines_matplotlib(outpng, f"{fam} TEST post-calib mean (calib from {ref_label}; {suffix})", pos, z_test, sci_color)

    if args.interactive:
        outhtml = os.path.join(args.outdir, f"3D_TOWERS_{fam}_REF_{ref_label}_{suffix}.html")
        plot_3d_towers_plotly(
            outhtml,
            f"{fam} REF post-calib mean ({suffix})",
            pos, z_ref, cher_color,
            tower_half=0.25,      # try 0.25–0.32
            pad_xy=1.5,
            show_text=False,      # <-- I strongly recommend False to avoid label pile-up
        )
        
        
        outhtml = os.path.join(args.outdir, f"3D_TOWERS_{fam}_TEST_{test_label}_calibfrom_{ref_label}_{suffix}.html")


        plot_3d_towers_plotly(
            outhtml,
            f"{fam} TEST post-calib mean (calib from {ref_label}; {suffix})",
            pos, z_ref, cher_color,
            tower_half=0.25,      # try 0.25–0.32
            pad_xy=1.5,
            show_text=False,      # <-- I strongly recommend False to avoid label pile-up
        )
        
        

if __name__ == "__main__":
    main()
