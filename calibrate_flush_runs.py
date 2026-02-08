#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

# ================= CONFIG =================
TREE_NAME = "EventTree"
NBINS = 200

# requested
XLIM = (8.0, 15.0)

MIN_RAW = 500
MIN_ENTRIES = 200

# ================= GRIDS =================
QUARTZ_GRID = [
    [None,"002",None,None],
    ["006","004","206","204"],
    ["016","014","216","214"],
    ["026","024","226","224"],
    [None,"030",None,None],
    [None,"034",None,None],
    ["106","104","306","304"],
    ["116","114","316","314"],
    ["126","124","326","324"],
    [None,"134",None,"334"],
]

PLASTIC_GRID = [
    [None,"000","202","200"],
    ["012","010","212","210"],
    ["022","020","222","220"],
    ["032",None,"232","230"],
    ["102","100","302","300"],
    ["112","110","312","310"],
    ["122","120","322","320"],
    ["132","130","332","330"],
]

SCI_GRID = [
    ["003","001","203","201"],
    ["007","005","207","205"],
    ["013","011","213","211"],
    ["017","015","217","215"],
    ["023","021","223","221"],
    ["027","025","227","225"],
    ["033","031","233","231"],
    [None,"035",None,"235"],
    ["103","101","303","301"],
    ["107","105","307","305"],
    ["113","111","313","311"],
    ["117","115","317","315"],
    ["123","121","323","321"],
    ["127","125","327","325"],
    ["133","131","333","331"],
    [None,"135",None,"335"],
]

# All CER channels (Quartz + Plastic)
CER_ALL_GRID = [
    ["002", "000", "202", "200"],
    ["006", "004", "206", "204"],
    ["012", "010", "212", "210"],
    ["016", "014", "216", "214"],
    ["022", "020", "222", "220"],
    ["026", "024", "226", "224"],
    ["032", "030", "232", "230"],
    [None,  "034", None,  "234"],
    ["102", "100", "302", "300"],
    ["106", "104", "306", "304"],
    ["112", "110", "312", "310"],
    ["116", "114", "316", "314"],
    ["122", "120", "322", "320"],
    ["126", "124", "326", "324"],
    ["132", "130", "332", "330"],
    [None,  "134", None,  "334"],
]

FAMILIES = {
    "CER-Quartz": QUARTZ_GRID,
    "CER-Plastic": PLASTIC_GRID,
    "SCI": SCI_GRID,
    "CER-All": CER_ALL_GRID,
}

# Provided anchors
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

def parse_code(code):
    return int(code[0]), int(code[1]), int(code[2])

def branch_name(b, g, c):
    return f"tfinal_Board{b}_Group{g}_Channel{c}"

def prep_array(arr):
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
    mode = float(0.5 * (edges[np.argmax(h)] + edges[np.argmax(h)+1]))
    return mu, mode, h

# ================= GAUSS FIT (anchor position) =================
def _gauss(x, A, mu, sig):
    return A * np.exp(-0.5 * ((x - mu) / sig)**2)

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

# ================= CALIBRATION (fixed anchor per family) =================
def derive_family_calibration_fixed_anchor(root_file, grid, anchor_key, calib_stat="mode"):
    """
    calib_stat:
      - "mean": shift = anchor_mu - mean(channel)
      - "mode": shift = anchor_mu - mode(channel)

    anchor_mu is taken from a Gaussian fit to the ANCHOR peak (fallback to anchor mean).
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

                hstats = hist_stats(arr, bins)
                if hstats is None:
                    continue
                mu, mode, _ = hstats

                stats[(b, g, c)] = {"N": int(arr.size), "mu": float(mu), "mode": float(mode)}

    if anchor_key not in arrays or anchor_key not in stats:
        raise RuntimeError(f"Anchor {anchor_key} not usable/found in this family for {root_file}")

    # Anchor peak position (Gaussian fit)
    anchor_arr = arrays[anchor_key]
    fit_ok, mu_fit, sig_fit, _A = fit_gaussian_to_peak(anchor_arr, bins, window=0.5)
    anchor_mu = float(mu_fit) if (fit_ok and np.isfinite(mu_fit)) else float(stats[anchor_key]["mu"])

    # Compute shifts
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
    return shifts, (anchor_key, anchor_info), stats

# ================= PLOTTING =================
def mosaic_pre_post_to_pdf(pdf, root_file, grid, shifts, title):
    bins = np.linspace(*XLIM, NBINS + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    nrows = len(grid)
    ncols = max(len(r) for r in grid)

    with uproot.open(root_file) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.2 * nrows), sharex=True, sharey=True)
        axes = np.atleast_2d(axes)

        # compute global ymax
        global_ymax = 1
        cache = {}
        for r, row in enumerate(grid):
            for c, code in enumerate(row):
                if code is None:
                    cache[(r, c)] = None
                    continue
                b, g, ch = parse_code(code)
                k = branch_name(b, g, ch)
                if k not in keys:
                    cache[(r, c)] = ("missing", code)
                    continue

                raw = prep_array(tree[k].array(library="np"))
                if raw is None:
                    cache[(r, c)] = ("nostats", code)
                    continue

                pre_stats = hist_stats(raw, bins)
                if pre_stats is None:
                    cache[(r, c)] = ("nostats", code)
                    continue
                mu_pre, mode_pre, h_pre = pre_stats

                post = raw + shifts.get((b, g, ch), 0.0)
                post_stats = hist_stats(post, bins)
                if post_stats is None:
                    cache[(r, c)] = ("nostats", code)
                    continue
                mu_post, mode_post, h_post = post_stats

                global_ymax = max(global_ymax, int(h_pre.max()), int(h_post.max()))
                cache[(r, c)] = ("ok", code, mu_pre, mu_post, mode_pre, mode_post, h_pre, h_post)

        # draw
        for r, row in enumerate(grid):
            for c, code in enumerate(row):
                ax = axes[r, c]
                if code is None:
                    ax.axis("off")
                    continue

                ax.set_xlim(*XLIM)
                ax.set_yscale("log")
                ax.set_ylim(1, global_ymax * 1.15)
                ax.tick_params(labelsize=8, direction="in", top=True, right=True)

                entry = cache.get((r, c))
                if entry is None:
                    ax.axis("off")
                    continue

                status = entry[0]
                if status != "ok":
                    ax.text(0.5, 0.5, f"{entry[1]}\n({status})", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9)
                    continue

                _, code, mu_pre, mu_post, mode_pre, mode_post, h_pre, h_post = entry

                ln1, = ax.step(centers, h_pre, where="mid", lw=1.0, alpha=0.70)
                ln2, = ax.step(centers, h_post, where="mid", lw=1.4, alpha=0.95)
                ax.set_title(code, fontsize=9, pad=2)

                # Mean lines (green, dashed)
                ax.axvline(mu_pre,  color="green", linestyle="--", linewidth=1.2, alpha=0.65)
                ax.axvline(mu_post, color="green", linestyle="--", linewidth=1.6, alpha=0.90)

                # Mode lines (purple, dashed)
                ax.axvline(mode_pre,  color="purple", linestyle="--", linewidth=1.2, alpha=0.65)
                ax.axvline(mode_post, color="purple", linestyle="--", linewidth=1.6, alpha=0.90)

                txt = (f"μpre={mu_pre:.2f}\nμpost={mu_post:.2f}\n"
                       f"mpre={mode_pre:.2f}\nmpost={mode_post:.2f}")
                ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                        ha="left", va="top", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"))

        for ax in axes[-1, :]:
            if ax.axison and ax.get_visible():
                ax.set_xlabel(r"$|t_{\mathrm{final}}|$ [ns]")

        fig.text(0.01, 0.5, "Events", va="center", rotation=90)
        fig.suptitle(title, fontsize=14)
        fig.legend([ln1, ln2], ["PRE (raw)", "POST (shifted)"],
                   loc="center left", bbox_to_anchor=(0.86, 0.5),
                   fontsize=10, frameon=False)

        fig.subplots_adjust(left=0.05, right=0.84, top=0.94, bottom=0.05, hspace=0.12, wspace=0.05)
        pdf.savefig(fig)
        plt.close(fig)

def heatmap_to_pdf(pdf, root_file, grid, shifts, quantity, apply_shift, title):
    """
    Heatmap cell value = time of arrival (mean or mode) in ns.
    quantity: "mean" or "mode"
    apply_shift: False => PRE, True => POST
    """
    nrows = len(grid)
    ncols = max(len(r) for r in grid)
    mat = np.full((nrows, ncols), np.nan, dtype=float)

    bins = np.linspace(*XLIM, NBINS + 1)

    with uproot.open(root_file) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        for r, row in enumerate(grid):
            for c, code in enumerate(row):
                if code is None:
                    continue
                b, g, ch = parse_code(code)
                k = branch_name(b, g, ch)
                if k not in keys:
                    continue

                arr = prep_array(tree[k].array(library="np"))
                if arr is None:
                    continue

                if apply_shift:
                    arr = arr + shifts.get((b, g, ch), 0.0)

                st = hist_stats(arr, bins)
                if st is None:
                    continue
                mu, mode, _h = st
                mat[r, c] = mu if quantity == "mean" else mode

    vmin, vmax = XLIM

    fig, ax = plt.subplots(figsize=(12, 0.65 * nrows + 1.2))
    im = ax.imshow(mat, origin="upper", aspect="auto", vmin=vmin, vmax=vmax)

    for rr in range(nrows):
        for cc in range(ncols):
            if cc >= len(grid[rr]) or grid[rr][cc] is None:
                continue
            code = grid[rr][cc]
            val = mat[rr, cc]
            txt = f"{code}\n{val:.2f}" if np.isfinite(val) else f"{code}\n—"
            ax.text(cc, rr, txt, ha="center", va="center", fontsize=8)

    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    ax.set_xticklabels([""] * ncols)
    ax.set_yticklabels([""] * nrows)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{quantity}(|tfinal|) [ns]")
    ax.set_title(title)
    fig.subplots_adjust(left=0.03, right=0.90, top=0.90, bottom=0.05)

    pdf.savefig(fig)
    plt.close(fig)

# ================= MAIN =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1501_250928105227_converted_timingskim.root")
    ap.add_argument("--test", default="/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1511_250928180741_converted_timingskim.root")
    ap.add_argument("--outdir", default="/lustre/research/hep/akshriva/Dream-Timing/TRUE-HGtiming/calibration_studiesZ/MODE_CALIB_OUTPUT")
    ap.add_argument("--calib-stat", choices=["mean", "mode"], default="mean",
                    help="Use channel mean or histogram mode when computing shifts (anchor uses Gaussian-fit peak).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ref_label = _infer_run_label(args.reference)
    test_label = _infer_run_label(args.test)

    suffix = f"calib_{args.calib_stat}"
    ref_pdf_path = os.path.join(args.outdir, f"REF_file_{ref_label}_{suffix}.pdf")
    test_pdf_path = os.path.join(args.outdir, f"TEST_file_{test_label}_{suffix}.pdf")

    # --- Derive shifts from reference using fixed anchors per calibration family
    shifts_by_family = {}
    for fam in ["CER-Quartz", "CER-Plastic", "SCI"]:
        grid = FAMILIES[fam]
        anchor_key = ANCHORS[fam]
        shifts, anchor, stats = derive_family_calibration_fixed_anchor(
            args.reference, grid, anchor_key, calib_stat=args.calib_stat
        )
        shifts_by_family[fam] = shifts
        akey, ainfo = anchor
        print(f"[{fam}] anchor={akey}  mu={ainfo['mu']:.4f}  N={ainfo['N']}  fit_ok={ainfo['fit_ok']}  sig_fit={ainfo['sig_fit']:.3f}  calib_stat={ainfo['calib_stat']}")

    # CER-All shifts = Quartz + Plastic shifts
    shifts_cer_all = {}
    shifts_cer_all.update(shifts_by_family["CER-Quartz"])
    shifts_cer_all.update(shifts_by_family["CER-Plastic"])
    shifts_by_family["CER-All"] = shifts_cer_all

    def write_all_pages(pdf, root_file, tag):
        for fam in ["CER-Quartz", "CER-Plastic", "SCI", "CER-All"]:
            grid = FAMILIES[fam]
            shifts = shifts_by_family[fam]

            mosaic_pre_post_to_pdf(pdf, root_file, grid, shifts,
                                   title=f"{tag} — {fam} mosaic PRE vs POST")

            for qty in ["mean", "mode"]:
                heatmap_to_pdf(pdf, root_file, grid, shifts, quantity=qty, apply_shift=False,
                               title=f"{tag} — {fam} heatmap {qty} PRE")
                heatmap_to_pdf(pdf, root_file, grid, shifts, quantity=qty, apply_shift=True,
                               title=f"{tag} — {fam} heatmap {qty} POST")

    with PdfPages(ref_pdf_path) as pdf:
        write_all_pages(pdf, args.reference, tag=f"REFERENCE {ref_label} ({suffix})")

    with PdfPages(test_pdf_path) as pdf:
        write_all_pages(pdf, args.test, tag=f"TEST {test_label} (calib from {ref_label}; {suffix})")

    print("Saved:", ref_pdf_path)
    print("Saved:", test_pdf_path)

if __name__ == "__main__":
    main()
