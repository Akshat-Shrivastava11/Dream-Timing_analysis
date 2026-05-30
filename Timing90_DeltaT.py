#!/usr/bin/env python3
"""
Delta-T measurement between left and right channels of the same detector row.

For each left-right pair (same row, mirrored side) we compute:
    Δt = t_right - t_left
on the same events (event-by-event), fit a Gaussian, and report the sigma
as the combined timing resolution.

Usage:
    python delta_t_leftright_pairs.py \
        --ana-file /path/to/run1501_XXXXXXXXX.root \
        --outdir   ./DeltaT_LR_Pairs \
        --pid      electron
"""

import os
import re
import argparse
import numpy as np
import uproot
import awkward as ak
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mplhep as hep

plt.style.use(hep.style.CMS)

# ============================================================
# Detector grids
# ============================================================
QUARTZ_GRID = [
    [None,  "002", None,  None ],
    ["006", "004", "206", "204"],
    ["016", "014", "216", "214"],
    ["026", "024", "226", "224"],
    [None,  "030", None,  None ],
    [None,  "034", None,  None ],
    ["106", "104", "306", "304"],
    ["116", "114", "316", "314"],
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

# Family timing windows (same as main script)
FAMILY_WINDOWS = {
    "Quartz":  (-15.0, -12.0),
    "Plastic": (-14.5, -11.5),
}

FAMILY_COLORS = {
    "Quartz":  "#003366",
    "Plastic": "#e42536",
}

FAMILY_DISPLAY = {
    "Quartz":  "FSHA (Quartz)",
    "Plastic": "Toray PJR-FB750 (Plastic)",
}

# ============================================================
# Build left-right pairs from a grid
# The grid has 4 columns: [left-outer, left-inner, right-inner, right-outer]
# We pair col0↔col2 and col1↔col3 within each row.
# ============================================================
def build_lr_pairs(grid, family_name):
    """
    Returns list of (left_ch, right_ch) tuples for a 4-column detector grid.
    Columns 0 & 1 are the left side, columns 2 & 3 are the right side.
    Pairing: col0↔col2, col1↔col3
    """
    pairs = []
    for row in grid:
        # pad row to length 4 if needed
        row4 = list(row) + [None] * (4 - len(row))
        c0, c1, c2, c3 = row4[0], row4[1], row4[2], row4[3]
        if c0 is not None and c2 is not None:
            pairs.append((c0, c2))
        if c1 is not None and c3 is not None:
            pairs.append((c1, c3))
    return pairs

ALL_PAIRS = {
    "Quartz":  build_lr_pairs(QUARTZ_GRID,  "Quartz"),
    "Plastic": build_lr_pairs(PLASTIC_GRID, "Plastic"),
}

# ============================================================
# Configuration
# ============================================================
TREE_NAME  = "EventTree"
SUFFIX     = "_LP2_50"
AMP_THRESHOLD = 100.0   # kept but only used as a loose guard

PID_BRANCH_MAP = {
    "PSD":        "DRS_Board7_Group1_Channel1",
    "TTUMuonVeto":"DRS_Board7_Group2_Channel4",
    "Cer474":     "DRS_Board7_Group2_Channel5",
    "Cer519":     "DRS_Board7_Group2_Channel6",
    "Cer537":     "DRS_Board7_Group2_Channel7",
}

# ============================================================
# PID helpers  (copied verbatim from main script)
# ============================================================
def get_service_drs_cut(service_drs):
    cuts = {
        "PSD":        (100, 400, -3500.0, "Sum"),
        "TTUMuonVeto":(200, 400, -2e3,    "Sum"),
        "Cer474":     (800, 900, -2000.0,  "Sum"),
        "Cer519":     (450, 550, -1000.0,  "Sum"),
        "Cer537":     (400, 500, -500.0,   "Sum"),
    }
    return cuts.get(service_drs, (0, 1000, -5e4, "Sum"))

def get_particle_selection(particle_type):
    selections = {
        "muon":     {"TTUMuonVeto": True,  "PSD": False},
        "pion":     {"TTUMuonVeto": False, "PSD": False,
                     "Cer474": True, "Cer519": True, "Cer537": True},
        "electron": {"TTUMuonVeto": False, "PSD": True,
                     "Cer474": True, "Cer519": True, "Cer537": True},
        "proton":   {"TTUMuonVeto": False, "PSD": False,
                     "Cer474": False, "Cer519": False, "Cer537": False},
    }
    return selections.get(particle_type.lower(), {})

def compute_pid_mask(tree, particle_type):
    if particle_type is None:
        return np.ones(tree.num_entries, dtype=bool)
    requirements = get_particle_selection(particle_type)
    if not requirements:
        return np.ones(tree.num_entries, dtype=bool)

    available = set(tree.keys())
    mask = np.ones(tree.num_entries, dtype=bool)

    for det, must_fire in requirements.items():
        branch = PID_BRANCH_MAP.get(det)
        if not branch or branch not in available:
            print(f"  [WARN] PID branch missing: {det} ({branch}). Skipping.")
            continue
        ts_min, ts_max, val_cut, method = get_service_drs_cut(det)
        try:
            wf = tree[branch].array(library="ak")
            baseline = ak.mean(wf[:, :30], axis=1)
            wf_bl = wf - baseline
            win_sum = ak.sum(wf_bl[:, int(ts_min):int(ts_max)], axis=1)
            fired = ak.to_numpy(win_sum) < val_cut
            mask = mask & fired if must_fire else mask & (~fired)
        except Exception as e:
            print(f"  [WARN] PID cut failed for {det}: {e}")
    return mask

# ============================================================
# Timing extraction  (same formula as main script)
# ============================================================
def get_tfinal(tree, b, g, c):
    """
    t_final(b,g,c) = (t_{b,g,c} - t_{b,g,8}) - (t_{0,3,7} - t_{0,3,8})
    """
    br_sig     = f"DRS_Board{b}_Group{g}_Channel{c}{SUFFIX}"
    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8{SUFFIX}"
    br_trg     = f"DRS_Board0_Group3_Channel7{SUFFIX}"
    br_trg_ref = f"DRS_Board0_Group3_Channel8{SUFFIX}"

    keys = set(tree.keys())
    if any(br not in keys for br in [br_sig, br_sig_ref, br_trg, br_trg_ref]):
        return None

    sig     = tree[br_sig    ].array(library="np")
    sig_ref = tree[br_sig_ref].array(library="np")
    trg     = tree[br_trg    ].array(library="np")
    trg_ref = tree[br_trg_ref].array(library="np")
    return (sig - sig_ref) - (trg - trg_ref)

def parse_code(code_str):
    return int(code_str[0]), int(code_str[1]), int(code_str[2])

# ============================================================
# Minimal amplitude guard (loose – just removes dead events)
# ============================================================
def amp_ok(tree, code_str, threshold=AMP_THRESHOLD):
    b, g, c = parse_code(code_str)
    br = f"DRS_Board{b}_Group{g}_Channel{c}"
    if br not in set(tree.keys()):
        return np.ones(tree.num_entries, dtype=bool)
    wf = tree[br].array(library="ak")
    bl = ak.mean(wf[:, :30], axis=1)
    peak = ak.max(wf - bl, axis=1)
    return ak.to_numpy(peak) >= threshold

# ============================================================
# Gaussian (peak-normalised)
# ============================================================
def gaussian(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

# ============================================================
# Paper style helpers
# ============================================================
AXIS_FS  = 32
TICK_FS  = 26
CMS_FS   = 28
LEGEND_FS= 20

def apply_style():
    plt.rcParams.update({
        "figure.figsize":    (14, 10),
        "figure.dpi":        120,
        "savefig.dpi":       300,
        "font.size":         26,
        "axes.labelsize":    AXIS_FS,
        "xtick.labelsize":   TICK_FS,
        "ytick.labelsize":   TICK_FS,
        "legend.fontsize":   LEGEND_FS,
        "lines.linewidth":   3.0,
        "axes.linewidth":    1.8,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.top":         True,
        "ytick.right":       True,
        "xtick.major.size":  10,
        "ytick.major.size":  10,
        "xtick.minor.size":  6,
        "ytick.minor.size":  6,
    })

def decorate_axes(ax, xlabel, ylabel, particle_type):
    ax.set_xlabel(xlabel, fontsize=AXIS_FS, loc="right")
    ax.set_ylabel(ylabel, fontsize=AXIS_FS, loc="top")
    ax.tick_params(which="major", labelsize=TICK_FS, length=10,
                   width=1.8, direction="in", top=True, right=True)
    ax.tick_params(which="minor", length=6, width=1.4,
                   direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.grid(False)
    ptag = r"$e^{+}$" if (particle_type or "").lower() == "electron" else \
           (particle_type or "All").capitalize()
    hep.cms.label(ax=ax, exp="CaloX", data=False,
                  llabel=r"$\it{L\!-\!R\ \Delta t}$",
                  rlabel=f"40 GeV {ptag}",
                  fontsize=CMS_FS)

# ============================================================
# Core: compute delta-t for one pair and make the plot
# ============================================================
def process_pair(tree, pid_mask, ch_left, ch_right, family, run_label):
    """
    Returns dict with fit results, or None if not enough events.
    Keys: dt_arr, mu, sigma, fwhm, time_err, n,
          xmin, xmax, centers, hist_norm, x_smooth, y_gauss
    """
    b_l, g_l, c_l = parse_code(ch_left)
    b_r, g_r, c_r = parse_code(ch_right)

    t_left  = get_tfinal(tree, b_l, g_l, c_l)
    t_right = get_tfinal(tree, b_r, g_r, c_r)

    if t_left is None or t_right is None:
        print(f"    [SKIP] {ch_left}-{ch_right}: timing branches missing")
        return None

    # loose amplitude guard on both channels (just removes dead events)
    amp_l = amp_ok(tree, ch_left)
    amp_r = amp_ok(tree, ch_right)

    combined = pid_mask & amp_l & amp_r

    if len(t_left) != len(combined):
        print(f"    [SKIP] {ch_left}-{ch_right}: length mismatch")
        return None

    t_l = t_left [combined]
    t_r = t_right[combined]

    # Both channels must be finite
    fin = np.isfinite(t_l) & np.isfinite(t_r)
    t_l = t_l[fin]
    t_r = t_r[fin]

    # Both channels must be inside their family timing window
    tmin, tmax = FAMILY_WINDOWS[family]
    win = ((t_l >= tmin) & (t_l <= tmax) &
           (t_r >= tmin) & (t_r <= tmax))
    t_l = t_l[win]
    t_r = t_r[win]

    dt = t_r - t_l          # Δt = t_right − t_left
    n  = len(dt)

    if n < 25:
        print(f"    [SKIP] {ch_left}-{ch_right}: only {n} events after cuts")
        return None

    # Fit window: mean ± 4σ clipped to ±2 ns
    dt_std  = float(np.std(dt))
    dt_mean = float(np.mean(dt))
    halfwin = min(4.0 * dt_std, 2.0)
    xmin = dt_mean - halfwin
    xmax = dt_mean + halfwin

    dt_win = dt[(dt >= xmin) & (dt <= xmax)]
    if len(dt_win) < 25:
        # fall back to full range
        dt_win = dt
        xmin, xmax = float(dt.min()), float(dt.max())

    nbins   = 80
    bins    = np.linspace(xmin, xmax, nbins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    h, _  = np.histogram(dt_win, bins=bins)
    if h.max() == 0:
        return None
    h_norm = h / h.max()

    # Gaussian fit
    try:
        p0  = [dt_mean, max(dt_std, 0.01)]
        bds = ([xmin - 1.0, 1e-4], [xmax + 1.0, 5.0])
        popt, _ = curve_fit(gaussian, centers, h_norm, p0=p0, bounds=bds, maxfev=20000)
        mu    = float(popt[0])
        sigma = abs(float(popt[1]))
    except Exception:
        mu    = dt_mean
        sigma = dt_std

    fwhm     = 2.355 * sigma
    time_err = sigma / np.sqrt(n)

    x_smooth = np.linspace(xmin, xmax, 600)
    y_gauss  = gaussian(x_smooth, mu, sigma)

    print(f"    [FIT] {family} {ch_left}↔{ch_right}: "
          f"N={n:5d}, μ={mu:+.3f} ns, σ={sigma:.3f} ns, "
          f"FWHM={fwhm:.3f} ns")

    return dict(
        ch_left=ch_left, ch_right=ch_right,
        family=family, run_label=run_label,
        n=n, mu=mu, sigma=sigma, fwhm=fwhm, time_err=time_err,
        xmin=xmin, xmax=xmax,
        centers=centers, hist_norm=h_norm,
        x_smooth=x_smooth, y_gauss=y_gauss,
    )


def make_pair_plot(ax, res, particle_type):
    """Draw histogram + Gaussian fit on ax for one pair result dict."""
    color = FAMILY_COLORS[res["family"]]
    fam_label = FAMILY_DISPLAY[res["family"]]

    # Filled step histogram
    bw = res["centers"][1] - res["centers"][0]
    xl = res["centers"] - 0.5 * bw
    xr = res["centers"] + 0.5 * bw
    xs = np.empty(2 * len(res["centers"]))
    ys = np.empty(2 * len(res["centers"]))
    xs[0::2] = xl;  xs[1::2] = xr
    ys[0::2] = res["hist_norm"]; ys[1::2] = res["hist_norm"]
    ax.fill_between(xs, 0, ys, alpha=0.18, color=color, linewidth=0)
    ax.step(res["centers"], res["hist_norm"],
            where="mid", lw=1.6, alpha=0.65, color=color)

    # Gaussian curve
    ax.plot(res["x_smooth"], res["y_gauss"],
            lw=3.8, color=color, solid_capstyle="round",
            label=(rf"Gaussian fit: $\mu$ = {res['mu']:+.3f} ns, "
                   rf"$\sigma$ = {res['sigma']:.3f} ns"
                   f"\nFWHM = {res['fwhm']:.3f} ns, N = {res['n']}"))

    ax.set_xlim(res["xmin"], res["xmax"])
    ax.set_ylim(0, 1.45)

    decorate_axes(ax,
                  rf"$\Delta t$ = $t_{{\mathrm{{{res['ch_right']}}}}}$ − $t_{{\mathrm{{{res['ch_left']}}}}}$ [ns]",
                  "Normalized Events",
                  particle_type)

    # Pair / family label box
    ax.text(0.98, 0.965,
            f"{fam_label}\nPair: {res['ch_left']} (L) — {res['ch_right']} (R)",
            transform=ax.transAxes, ha="right", va="top", fontsize=22,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="none", alpha=0.85),
            zorder=10)

    leg = ax.legend(loc="upper left",
                    bbox_to_anchor=(0.02, 0.88),
                    frameon=True, fancybox=True, framealpha=0.88,
                    facecolor="white", edgecolor="none",
                    fontsize=20, handlelength=2.2)
    for t in leg.get_texts():
        t.set_fontweight("normal")


# ============================================================
# Summary page: all σ values per family
# ============================================================
def make_summary_page(pdf, all_results, particle_type):
    for family in ["Quartz", "Plastic"]:
        fam_res = [r for r in all_results if r["family"] == family]
        if not fam_res:
            continue

        fig, ax = plt.subplots(figsize=(16, 7))
        color = FAMILY_COLORS[family]
        labels = [f"{r['ch_left']}↔{r['ch_right']}" for r in fam_res]
        sigmas = np.array([r["sigma"] for r in fam_res]) * 1e3  # → ps
        errs   = np.array([r["time_err"] for r in fam_res]) * 1e3

        x = np.arange(len(fam_res))
        ax.errorbar(x, sigmas, yerr=errs, fmt="o", color=color,
                    capsize=5, markersize=9, elinewidth=2.2, zorder=3)
        ax.axhline(float(np.nanmean(sigmas)), color=color,
                   linestyle="--", linewidth=2.5,
                   label=rf"Mean $\sigma$ = {np.nanmean(sigmas):.0f} ps")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
        ax.set_ylabel(r"$\sigma_{\Delta t}$ [ps]", fontsize=AXIS_FS)
        ax.set_title(f"{FAMILY_DISPLAY[family]}: σ of Δt per left-right pair",
                     fontsize=28, loc="left")
        ax.tick_params(which="major", labelsize=TICK_FS,
                       length=10, width=1.8, direction="in", top=True, right=True)
        ax.tick_params(which="minor", length=6, width=1.4,
                       direction="in", top=True, right=True)
        ax.minorticks_on()
        ax.grid(False)
        leg = ax.legend(fontsize=LEGEND_FS, frameon=True, fancybox=True,
                        framealpha=0.88, facecolor="white", edgecolor="none")
        for t in leg.get_texts():
            t.set_fontweight("normal")

        ptag = r"$e^{+}$" if (particle_type or "").lower() == "electron" else \
               (particle_type or "All").capitalize()
        hep.cms.label(ax=ax, exp="CaloX", data=False,
                      llabel=r"$\it{L\!-\!R\ \Delta t}$",
                      rlabel=f"40 GeV {ptag}", fontsize=CMS_FS)

        fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.28)
        pdf.savefig(fig, dpi=220)
        plt.close(fig)


# ============================================================
# Main
# ============================================================
def run_label_from_path(path):
    m = re.search(r"(run\d+_\d{11,12})", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


def main():
    ap = argparse.ArgumentParser(
        description="Δt left-right channel pairs for run1501.")
    ap.add_argument("--ana-file", required=True,
                    help="Single ROOT file (run1501)")
    ap.add_argument("--outdir",   default="./DeltaT_LR_Pairs")
    ap.add_argument("--pid",      default="electron",
                    choices=["muon", "pion", "electron", "proton"])
    ap.add_argument("--families", nargs="+",
                    default=["Quartz", "Plastic"],
                    choices=["Quartz", "Plastic"],
                    help="Families to process")
    args = ap.parse_args()

    apply_style()
    os.makedirs(args.outdir, exist_ok=True)

    fpath = args.ana_file
    if not os.path.exists(fpath):
        raise SystemExit(f"[FATAL] File not found: {fpath}")

    run_label = run_label_from_path(fpath)
    print(f"\n[INIT] Delta-T left-right pairs")
    print(f"[INIT] File      : {os.path.basename(fpath)}")
    print(f"[INIT] Run label : {run_label}")
    print(f"[INIT] PID       : {args.pid}")
    print(f"[INIT] Outdir    : {args.outdir}")

    try:
        uf   = uproot.open(fpath)
        tree = uf[TREE_NAME]
    except Exception as e:
        raise SystemExit(f"[FATAL] Cannot open file/tree: {e}")

    print(f"[INIT] Tree entries: {tree.num_entries}")

    pid_mask = compute_pid_mask(tree, args.pid)
    print(f"[INIT] Events passing PID: {pid_mask.sum()} / {len(pid_mask)}")

    all_results = []

    pdf_path = os.path.join(args.outdir,
                            f"DeltaT_LR_Pairs_{args.pid}_{run_label}.pdf")
    print(f"\n[PLOT] Writing to {pdf_path}\n")

    with PdfPages(pdf_path) as pdf:
        for family in args.families:
            pairs  = ALL_PAIRS[family]
            color  = FAMILY_COLORS[family]
            n_pairs = len(pairs)
            print(f"\n[FAMILY] {family}  ({n_pairs} pairs)")

            for ch_left, ch_right in pairs:
                res = process_pair(tree, pid_mask,
                                   ch_left, ch_right, family, run_label)
                if res is None:
                    continue

                all_results.append(res)

                # individual page per pair
                fig, ax = plt.subplots(figsize=(14, 10))
                make_pair_plot(ax, res, args.pid)
                fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.14)
                pdf.savefig(fig, dpi=220)
                plt.close(fig)

        # Summary σ-per-pair plots
        if all_results:
            make_summary_page(pdf, all_results, args.pid)

    try:
        uf.close()
    except Exception:
        pass

    # ── Text summary ──────────────────────────────────────────────────────────
    txt_path = os.path.join(args.outdir,
                            f"DeltaT_LR_summary_{args.pid}_{run_label}.txt")
    hdr = (f"{'Family':<8} | {'Left':<6} | {'Right':<6} | "
           f"{'N':>6} | {'mu [ns]':>10} | {'sigma [ns]':>10} | "
           f"{'sigma [ps]':>10} | {'FWHM [ns]':>10} | {'time_err [ps]':>14}")
    sep = "=" * len(hdr)
    with open(txt_path, "w") as f:
        f.write(sep + "\n")
        f.write(hdr + "\n")
        f.write(sep + "\n")
        for r in all_results:
            f.write(
                f"{r['family']:<8} | {r['ch_left']:<6} | {r['ch_right']:<6} | "
                f"{r['n']:>6} | {r['mu']:>+10.4f} | {r['sigma']:>10.4f} | "
                f"{r['sigma']*1e3:>10.1f} | {r['fwhm']:>10.4f} | "
                f"{r['time_err']*1e3:>14.2f}\n"
            )

    print(f"\n[DONE] PDF    : {pdf_path}")
    print(f"[DONE] Table  : {txt_path}")
    print(f"[DONE] Total pairs fitted: {len(all_results)}")


if __name__ == "__main__":
    main()