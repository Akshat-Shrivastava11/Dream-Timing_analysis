#!/usr/bin/env python3
"""
make_3d_grid_towers.py

Interactive 3D "tower" plots (Plotly HTML + optional Matplotlib PNG) for timing channels.

Conventions:
  - x = mapping x (default: row index from the grid; can be overridden via --xy-map JSON)
  - y = mapping y (default: col index from the grid; can be overridden via --xy-map JSON)
  - z = mean(|tfinal| + shift) in ns AFTER applying calibration shifts derived from REFERENCE (fixed anchors)

Calibration:
  - For each family (SCI, CER-Quartz, CER-Plastic), derive shifts from REFERENCE only:
        shift_i = anchor_mu_ref - loc_i_ref
    where loc_i_ref is either mean or histogram mode in REF, and anchor_mu_ref comes from a Gaussian peak fit
    (with fallback to anchor mean if fit fails).
  - Apply those frozen shifts to TEST (and any future run you feed as --test).

Colors:
  - CER-All: Quartz = red, Plastic = blue
  - SCI: green (as requested)

Plot styles:
  - Towers are filled rectangular prisms (Mesh3d cubes stretched in z).
  - Optional Z log-scale in Plotly: --zscale log
      * Plotly log axis requires strictly positive z, so we apply a safety floor Z_EPS.

Outputs:
  - PNG (matplotlib): 3D_TOWERS_<FAM>_REF_...png and 3D_TOWERS_<FAM>_TEST_...png
  - HTML (plotly, if --interactive): same base names with .html

Examples:
  # Linear z
  python3 make_3d_grid_towers.py --interactive

  # Log z (Plotly)
  python3 make_3d_grid_towers.py --interactive --zscale log

  # Use external mapping file
  python3 make_3d_grid_towers.py --xy-map mapping.json --interactive --zscale log
"""

import os
import re
import json
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

try:
    import plotly.graph_objects as go
except Exception:
    go = None


# ================= CONFIG =================
TREE_NAME = "EventTree"
NBINS = 200
XLIM = (8.0, 15.0)         # range for |tfinal|
MIN_RAW = 500              # require at least this many raw samples before filtering
MIN_ENTRIES = 200          # require at least this many after filtering
Z_EPS = 1e-3               # ns floor for log axes (must be > 0)


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
ANCHORS = {
    "SCI": (1, 0, 7),
    "CER-Quartz": (1, 0, 4),
    "CER-Plastic": (1, 0, 0),
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


def prep_array(arr: np.ndarray):
    """Return filtered |tfinal| array in [XLIM[0], XLIM[1]] with finite values."""
    arr = np.abs(arr)
    arr = arr[np.isfinite(arr)]
    arr = arr[(arr >= XLIM[0]) & (arr <= XLIM[1])]
    if arr.size < MIN_ENTRIES:
        return None
    return arr


def hist_stats(arr, bins):
    h, edges = np.histogram(arr, bins=bins)
    if h.sum() == 0:
        return None
    mu = float(arr.mean())
    mode = float(0.5 * (edges[np.argmax(h)] + edges[np.argmax(h) + 1]))
    return mu, mode, h


def _gauss(x, A, mu, sig):
    return A * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def fit_gaussian_to_peak(arr_abs, bins, window=0.5):
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


def derive_family_calibration_fixed_anchor(root_file, grid, anchor_key, calib_stat="mean"):
    """
    Derive per-channel shifts from REFERENCE only.

    calib_stat:
      - "mean": shift = anchor_mu - mean(channel)
      - "mode": shift = anchor_mu - mode(channel)

    anchor_mu from Gaussian fit to anchor peak (fallback to anchor mean).
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


def positions_from_xy_map(grid, xy_map_path):
    """
    JSON mapping: { "CODE": [x, y], ... }
    pos_map[key] = (x_map, y_map, code)
    """
    with open(xy_map_path, "r") as f:
        mapping = json.load(f)

    pos = {}
    for row in grid:
        for code in row:
            if code is None or code not in mapping:
                continue
            b, g, ch = parse_code(code)
            x_map, y_map = mapping[code]
            pos[(b, g, ch)] = (float(x_map), float(y_map), code)
    return pos


def compute_post_mean_map(root_file, grid, shifts):
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
                arr_raw = tree[k].array(library="np")
                if arr_raw.size < MIN_RAW:
                    continue
                arr = prep_array(arr_raw)
                if arr is None:
                    continue
                out[(b, g, ch)] = float((arr + shifts.get((b, g, ch), 0.0)).mean())

    return out


# ================= PLOTTING: TOWERS =================
def _cube_mesh(xc, yc, z0, z1, dx, dy):
    """
    Return vertices and triangulation for a rectangular prism (cube stretched in z).
    Centered at (xc, yc), spanning [z0, z1], with half-widths dx/2, dy/2.
    """
    x0, x1 = xc - dx / 2.0, xc + dx / 2.0
    y0, y1 = yc - dy / 2.0, yc + dy / 2.0

    # 8 vertices
    xs = [x0, x1, x1, x0, x0, x1, x1, x0]
    ys = [y0, y0, y1, y1, y0, y0, y1, y1]
    zs = [z0, z0, z0, z0, z1, z1, z1, z1]

    # 12 triangles (two per face)
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]

    return xs, ys, zs, i, j, k


def plot_3d_towers_plotly(
    outhtml,
    title,
    pos_map,
    t_map,
    color_fn,
    zscale="linear",
    dx=0.82,
    dy=0.82,
    show_labels=True,
):
    """
    Interactive Plotly HTML:
      x = mapping x
      y = mapping y
      z = time

    zscale: "linear" or "log"
    For log:
      - z values must be > 0
      - scene.zaxis.range is in log10 space
    """
    if go is None:
        print("[WARN] plotly not available; skipping interactive HTML:", outhtml)
        return

    fig = go.Figure()

    xvals, yvals, zvals = [], [], []

    # Add each tower as a Mesh3d prism
    for key, (x_map, y_map, code) in pos_map.items():
        if key not in t_map:
            continue

        t = float(t_map[key])
        if not np.isfinite(t):
            continue

        z_top = max(t, Z_EPS)  # ensure positive for log
        col = color_fn(code)

        xs, ys, zs, ii, jj, kk = _cube_mesh(x_map, y_map, 0.0, z_top, dx, dy)

        hover = (
            f"<b>{code}</b><br>"
            f"(b,g,ch)={key}<br>"
            f"x={x_map:.2f}, y={y_map:.2f}<br>"
            f"time={t:.4f} ns"
        )

        fig.add_trace(
            go.Mesh3d(
                x=xs,
                y=ys,
                z=zs,
                i=ii,
                j=jj,
                k=kk,
                color=col,
                opacity=0.95,
                flatshading=True,
                hoverinfo="text",
                hovertext=hover,
                showscale=False,
            )
        )

        xvals.append(float(x_map))
        yvals.append(float(y_map))
        zvals.append(float(z_top))

    if not zvals:
        print("[WARN] No channels to plot for", title, "; skipping", outhtml)
        return

    xmin, xmax = min(xvals), max(xvals)
    ymin, ymax = min(yvals), max(yvals)
    zmin, zmax = min(zvals), max(zvals)

    # Optional code labels (positioned at the top of each tower)
    if show_labels:
        codes = []
        tx, ty, tz = [], [], []
        for key, (x_map, y_map, code) in pos_map.items():
            if key not in t_map:
                continue
            t = float(t_map[key])
            if not np.isfinite(t):
                continue
            codes.append(code)
            tx.append(float(x_map))
            ty.append(float(y_map))
            tz.append(max(float(t), Z_EPS))

        fig.add_trace(
            go.Scatter3d(
                x=tx,
                y=ty,
                z=tz,
                mode="text",
                text=codes,
                textposition="top center",
                textfont=dict(size=11, color="white"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Z axis configuration
    if zscale == "log":
        zaxis = dict(
            title="mean(post) time [ns] (log)",
            type="log",
            range=[np.log10(zmin * 0.98), np.log10(zmax * 1.02)],
            showgrid=True,
            zeroline=False,
        )
    else:
        zaxis = dict(
            title="mean(post) time [ns]",
            range=[0.0, zmax * 1.10],
            showgrid=True,
            zeroline=False,
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title="mapping x",
                range=[xmin - 1.2, xmax + 1.2],
                showgrid=True,
                zeroline=False,
            ),
            yaxis=dict(
                title="mapping y",
                range=[ymin - 1.2, ymax + 1.2],
                showgrid=True,
                zeroline=False,
            ),
            zaxis=zaxis,
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=45, b=0),
    )

    fig.write_html(outhtml, include_plotlyjs="cdn", full_html=True)
    print("Saved:", outhtml)


def plot_3d_towers_matplotlib(outpng, title, pos_map, t_map, color_fn, dx=0.8, dy=0.8):
    """
    Optional PNG output (matplotlib). This is linear only.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def prism_faces(xc, yc, z0, z1, dx, dy):
        x0, x1 = xc - dx / 2.0, xc + dx / 2.0
        y0, y1 = yc - dy / 2.0, yc + dy / 2.0
        # 8 corners
        p = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ])
        # faces as quads
        faces = [
            [p[0], p[1], p[2], p[3]],  # bottom
            [p[4], p[5], p[6], p[7]],  # top
            [p[0], p[1], p[5], p[4]],  # side
            [p[1], p[2], p[6], p[5]],
            [p[2], p[3], p[7], p[6]],
            [p[3], p[0], p[4], p[7]],
        ]
        return faces

    xs, ys, zs = [], [], []
    for key, (x_map, y_map, code) in pos_map.items():
        if key not in t_map:
            continue
        t = float(t_map[key])
        if not np.isfinite(t):
            continue
        xs.append(x_map); ys.append(y_map); zs.append(t)

        faces = prism_faces(x_map, y_map, 0.0, t, dx, dy)
        poly = Poly3DCollection(faces, alpha=0.95)
        poly.set_facecolor(color_fn(code))
        poly.set_edgecolor("k")
        poly.set_linewidth(0.3)
        ax.add_collection3d(poly)

    if not zs:
        print(f"[WARN] No channels to plot for {title}; skipping {outpng}")
        plt.close(fig)
        return

    ax.set_title(title)
    ax.set_xlabel("mapping x")
    ax.set_ylabel("mapping y")
    ax.set_zlabel("mean(post) time [ns]")

    ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)
    ax.set_zlim(0.0, max(zs) * 1.10)

    ax.view_init(elev=25, azim=-55)
    plt.tight_layout()
    plt.savefig(outpng, dpi=220)
    plt.close(fig)
    print("Saved:", outpng)


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
                    help="Write interactive HTML (plotly). Requires plotly.")
    ap.add_argument("--zscale", choices=["linear", "log"], default="linear",
                    help="Plotly z-axis scaling (linear or log).")
    ap.add_argument("--tower-dx", type=float, default=0.82, help="Tower width in x (mapping units).")
    ap.add_argument("--tower-dy", type=float, default=0.82, help="Tower width in y (mapping units).")
    ap.add_argument("--no-labels", action="store_true", help="Disable code labels on towers (HTML only).")
    ap.add_argument("--png", action="store_true", help="Also write PNG towers (matplotlib).")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ref_label = _infer_run_label(args.reference)
    test_label = _infer_run_label(args.test)
    suffix = f"calib_{args.calib_stat}"

    # ---- derive shifts from REFERENCE ONLY (frozen constants)
    shifts_by_family = {}
    for fam in ["CER-Quartz", "CER-Plastic", "SCI"]:
        grid = FAMILIES[fam]
        anchor_key = ANCHORS[fam]
        shifts, (akey, ainfo) = derive_family_calibration_fixed_anchor(
            args.reference, grid, anchor_key, calib_stat=args.calib_stat
        )
        shifts_by_family[fam] = shifts
        print(
            f"[{fam}] anchor={akey} mu={ainfo['mu']:.4f} "
            f"N={ainfo['N']} fit_ok={ainfo['fit_ok']} sig_fit={ainfo['sig_fit']:.3f}"
        )

    # CER-All = Quartz + Plastic
    shifts_cer_all = {}
    shifts_cer_all.update(shifts_by_family["CER-Quartz"])
    shifts_cer_all.update(shifts_by_family["CER-Plastic"])
    shifts_by_family["CER-All"] = shifts_cer_all

    quartz_codes = codes_in_grid(QUARTZ_GRID)
    plastic_codes = codes_in_grid(PLASTIC_GRID)

    def cher_color(code):
        if code in quartz_codes:
            return "red"
        if code in plastic_codes:
            return "blue"
        return "black"

    def sci_color(_code):
        return "green"

    def make_pos(grid):
        if args.xy_map:
            return positions_from_xy_map(grid, args.xy_map)
        return positions_from_grid_xy(grid)

    # ================= CER-All =================
    fam = "CER-All"
    grid = FAMILIES[fam]
    shifts = shifts_by_family[fam]
    pos = make_pos(grid)

    z_ref = compute_post_mean_map(args.reference, grid, shifts)
    z_test = compute_post_mean_map(args.test, grid, shifts)

    if args.png:
        outpng = os.path.join(args.outdir, f"3D_TOWERS_{fam}_REF_{ref_label}_{suffix}.png")
        plot_3d_towers_matplotlib(outpng, f"{fam} REF post-calib mean ({suffix})", pos, z_ref, cher_color,
                                  dx=args.tower_dx, dy=args.tower_dy)
        outpng = os.path.join(args.outdir, f"3D_TOWERS_{fam}_TEST_{test_label}_calibfrom_{ref_label}_{suffix}.png")
        plot_3d_towers_matplotlib(outpng, f"{fam} TEST post-calib mean (calib from {ref_label}; {suffix})",
                                  pos, z_test, cher_color, dx=args.tower_dx, dy=args.tower_dy)

    if args.interactive:
        outhtml = os.path.join(args.outdir, f"3D_TOWERS_{fam}_REF_{ref_label}_{suffix}.html")
        plot_3d_towers_plotly(
            outhtml,
            f"{fam} REF post-calib mean ({suffix})",
            pos, z_ref, cher_color,
            zscale=args.zscale,
            dx=args.tower_dx,
            dy=args.tower_dy,
            show_labels=not args.no_labels,
        )
        outhtml = os.path.join(args.outdir, f"3D_TOWERS_{fam}_TEST_{test_label}_calibfrom_{ref_label}_{suffix}.html")
        plot_3d_towers_plotly(
            outhtml,
            f"{fam} TEST post-calib mean (calib from {ref_label}; {suffix})",
            pos, z_test, cher_color,
            zscale=args.zscale,
            dx=args.tower_dx,
            dy=args.tower_dy,
            show_labels=not args.no_labels,
        )

    # ================= SCI =================
    fam = "SCI"
    grid = FAMILIES[fam]
    shifts = shifts_by_family[fam]
    pos = make_pos(grid)

    z_ref = compute_post_mean_map(args.reference, grid, shifts)
    z_test = compute_post_mean_map(args.test, grid, shifts)

    if args.png:
        outpng = os.path.join(args.outdir, f"3D_TOWERS_{fam}_REF_{ref_label}_{suffix}.png")
        plot_3d_towers_matplotlib(outpng, f"{fam} REF post-calib mean ({suffix})", pos, z_ref, sci_color,
                                  dx=args.tower_dx, dy=args.tower_dy)
        outpng = os.path.join(args.outdir, f"3D_TOWERS_{fam}_TEST_{test_label}_calibfrom_{ref_label}_{suffix}.png")
        plot_3d_towers_matplotlib(outpng, f"{fam} TEST post-calib mean (calib from {ref_label}; {suffix})",
                                  pos, z_test, sci_color, dx=args.tower_dx, dy=args.tower_dy)

    if args.interactive:
        outhtml = os.path.join(args.outdir, f"3D_TOWERS_{fam}_REF_{ref_label}_{suffix}.html")
        plot_3d_towers_plotly(
            outhtml,
            f"{fam} REF post-calib mean ({suffix})",
            pos, z_ref, sci_color,
            zscale=args.zscale,
            dx=args.tower_dx,
            dy=args.tower_dy,
            show_labels=not args.no_labels,
        )
        outhtml = os.path.join(args.outdir, f"3D_TOWERS_{fam}_TEST_{test_label}_calibfrom_{ref_label}_{suffix}.html")
        plot_3d_towers_plotly(
            outhtml,
            f"{fam} TEST post-calib mean (calib from {ref_label}; {suffix})",
            pos, z_test, sci_color,
            zscale=args.zscale,
            dx=args.tower_dx,
            dy=args.tower_dy,
            show_labels=not args.no_labels,
        )


if __name__ == "__main__":
    main()
