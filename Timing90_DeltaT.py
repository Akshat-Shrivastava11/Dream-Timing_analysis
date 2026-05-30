#!/usr/bin/env python3
"""
Delta-T timing-resolution measurement for the six possible channel
combinations within each selected detector family row.

For each family, four channels are selected, giving:
    C(4, 2) = 6 combinations

For each combination we compute event-by-event:
    Δt = t_second - t_first
fit a Gaussian to the Δt distribution, and report sigma as the combined
pair timing resolution.

Default family rows:
    Quartz  : 106, 104, 306, 304
    Plastic : 102, 100, 302, 300

Usage:
python3 Timing90_DeltaT.py --ana-file /lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1501_250928105227_converted_timingskim.root          --outdir   ./DeltaT_LR_Pairs         --pid      electron
"""

import os
import re
import argparse
from itertools import combinations
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
# Six possible combinations per family
# For a 4-channel family row, C(4, 2) = 6 combinations.
# These rows contain the usual run1501 anchor channels 104 and 100.
# ============================================================
def build_grid_row_combinations(grid, require_complete_rows=True):
    """
    Build all C(4,2)=6 combinations for every complete 4-channel grid row.

    If require_complete_rows=True:
        only rows with all 4 channels are used.

    If require_complete_rows=False:
        partial rows are also used, giving C(N,2) for N available channels.
    """
    pairs = []

    for irow, row in enumerate(grid):
        row4 = list(row) + [None] * (4 - len(row))
        valid_channels = [ch for ch in row4 if ch is not None]

        if require_complete_rows and len(valid_channels) != 4:
            print(
                f"[GRID] Skipping incomplete row {irow}: {row4}",
                flush=True
            )
            continue

        row_pairs = list(combinations(valid_channels, 2))

        print(
            f"[GRID] Row {irow}: channels={valid_channels} -> {len(row_pairs)} pairs",
            flush=True
        )

        pairs.extend(row_pairs)

    return pairs


ALL_PAIRS = {
    "Quartz":  build_grid_row_combinations(QUARTZ_GRID,  require_complete_rows=True),
    "Plastic": build_grid_row_combinations(PLASTIC_GRID, require_complete_rows=True),
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





TIMING_CACHE = {}
AMP_CACHE = {}
KEYS_CACHE = None

def get_tree_keys(tree):
    global KEYS_CACHE
    if KEYS_CACHE is None:
        KEYS_CACHE = set(tree.keys())
    return KEYS_CACHE

# ============================================================
# Timing extraction  (same formula as main script)
# ============================================================
def get_tfinal(tree, b, g, c):
    code_key = f"{b}{g}{c}"
    if code_key in TIMING_CACHE:
        return TIMING_CACHE[code_key]

    br_sig     = f"DRS_Board{b}_Group{g}_Channel{c}{SUFFIX}"
    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8{SUFFIX}"
    br_trg     = f"DRS_Board0_Group3_Channel7{SUFFIX}"
    br_trg_ref = f"DRS_Board0_Group3_Channel8{SUFFIX}"

    keys = get_tree_keys(tree)
    if any(br not in keys for br in [br_sig, br_sig_ref, br_trg, br_trg_ref]):
        TIMING_CACHE[code_key] = None
        return None

    print(f"      [READ timing] {code_key}", flush=True)

    sig     = tree[br_sig].array(library="np")
    sig_ref = tree[br_sig_ref].array(library="np")

    if "TRIGGER_CORR" not in TIMING_CACHE:
        trg     = tree[br_trg].array(library="np")
        trg_ref = tree[br_trg_ref].array(library="np")
        TIMING_CACHE["TRIGGER_CORR"] = trg - trg_ref

    out = (sig - sig_ref) - TIMING_CACHE["TRIGGER_CORR"]
    TIMING_CACHE[code_key] = out
    return out

def parse_code(code_str):
    return int(code_str[0]), int(code_str[1]), int(code_str[2])

# ============================================================
# Minimal amplitude guard (loose – just removes dead events)
# ============================================================
def amp_ok(tree, code_str, threshold=AMP_THRESHOLD):
    if code_str in AMP_CACHE:
        return AMP_CACHE[code_str]

    b, g, c = parse_code(code_str)
    br = f"DRS_Board{b}_Group{g}_Channel{c}"

    keys = get_tree_keys(tree)
    if br not in keys:
        out = np.ones(tree.num_entries, dtype=bool)
        AMP_CACHE[code_str] = out
        return out

    print(f"      [READ waveform] {code_str}", flush=True)

    wf = tree[br].array(library="ak")
    bl = ak.mean(wf[:, :30], axis=1)
    peak = ak.max(wf - bl, axis=1)
    out = ak.to_numpy(peak) >= threshold

    AMP_CACHE[code_str] = out
    return out
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
                  llabel=r"$\it{Family\ combinations\ \Delta t}$",
                  rlabel=f"40 GeV {ptag}",
                  fontsize=CMS_FS)

# ============================================================
# Core: compute delta-t for one channel combination and make the plot
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

    dt = t_r - t_l          # Δt = t_second − t_first
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
        ch_left=ch_left,
        ch_right=ch_right,
        family=family,
        run_label=run_label,
        n=n,
        mu=mu,
        sigma=sigma,
        fwhm=fwhm,
        time_err=time_err,
        xmin=xmin,
        xmax=xmax,
        centers=centers,
        hist_norm=h_norm,
        x_smooth=x_smooth,
        y_gauss=y_gauss,
        dt_raw=dt.astype(np.float32),
        dt_windowed=dt_win.astype(np.float32),
    )
def make_pair_plot(ax, res, particle_type):
    """Draw histogram + Gaussian fit."""
    display_name = particle_type.capitalize() if particle_type else "All Particles"
    family_label = FAMILY_DISPLAY[res["family"]]

    # Histogram only
    bw = res["centers"][1] - res["centers"][0]
    xl = res["centers"] - 0.5 * bw
    xr = res["centers"] + 0.5 * bw
    xs = np.empty(2 * len(res["centers"]))
    ys = np.empty(2 * len(res["centers"]))
    xs[0::2] = xl
    xs[1::2] = xr
    ys[0::2] = res["hist_norm"]
    ys[1::2] = res["hist_norm"]

    hist_color = "#6A85C3"

    ax.fill_between(xs, 0, ys, alpha=0.35, color=hist_color, linewidth=0)

    ax.step(
        res["centers"],
        res["hist_norm"],
        where="mid",
        lw=2.2,
        color=hist_color,
        label="Data"
    )

    # Gaussian fit
    ax.plot(
        res["x_smooth"],
        res["y_gauss"],
        color="red",
        lw=2.8,
        label=(f"Gaussian Fit\n"
               f"μ = {res['mu']:+.3f} ns\n"
               f"σ = {res['sigma']*1000:.1f} ps")
    )

    ax.set_xlabel(
        rf"$t_{{\mathrm{{{res['ch_right']}}}}} - t_{{\mathrm{{{res['ch_left']}}}}}$ [ns]",
        fontsize=22
    )
    ax.set_ylabel("[A.U.]", fontsize=22)

    ax.set_xlim(res["xmin"], res["xmax"])
    ax.set_ylim(0, 1.2)

    # No "Preliminary"
    hep.cms.label(
        ax=ax,
        exp="CaloX",
        data=False,
        llabel="",
        rlabel="2025 Test Beam"
    )

    # Text block: no pair line, no "Family:" prefix
    ax.text(
        0.05, 0.90,
        f"Particle: {display_name}\n{family_label}",
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment="top"
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=20,
        handlelength=1.8,
        borderpad=0.5,
        labelspacing=0.35
    )
    ax.grid(True, linestyle="--", alpha=0.3)

    ax.tick_params(which="major", direction="in", top=True, right=True, length=8)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=4)
    ax.minorticks_on()

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
        ax.set_title(f"{FAMILY_DISPLAY[family]}: σ of Δt for all 6 combinations",
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
                      llabel=r"$\it{Family\ combinations\ \Delta t}$",
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
        description="Δt timing-resolution plots for all six family combinations.")
    ap.add_argument("--ana-file", required=True,
                    help="Single ROOT file (run1501)")
    ap.add_argument("--outdir",   default="/lustre/research/hep/akshriva/Dream-Timing/DeltaT_LR_Pairs/TIMING_PAIRS")
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
    #print(f"\n[INIT] Delta-T all six family combinations")
    print(f"\n[INIT] Delta-T all six combinations for every complete grid row")
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

    print(f"\n[PLOT] Writing individual PDFs to {args.outdir}\n")

    all_results = []

    for family in args.families:
        pairs = ALL_PAIRS[family]
        print(f"\n[FAMILY] {family}  ({len(pairs)} pairs)")

        for ch_left, ch_right in pairs:
            print(f"    [START] {family} {ch_left}-{ch_right}", flush=True)

            res = process_pair(tree, pid_mask, ch_left, ch_right, family, run_label)
            if res is None:
                continue

            all_results.append(res)

            fig, ax = plt.subplots(figsize=(10, 8))
            make_pair_plot(ax, res, args.pid)
            fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.13)

            # filename only uses channel names
            pdf_name = f"{ch_left}_{ch_right}.pdf"
            pdf_file = os.path.join(args.outdir, pdf_name)
            fig.savefig(pdf_file)
            plt.close(fig)

            print(f"    [SAVE] {pdf_file}")
            # Summary σ-per-pair plots
        #if all_results:
            #make_summary_page(pdf, all_results, args.pid)

    try:
        uf.close()
    except Exception:
        pass

    # ── Text summary ──────────────────────────────────────────────────────────
    txt_path = os.path.join(args.outdir,
                            f"DeltaT_Family6Combos_summary_{args.pid}_{run_label}.txt")
    hdr = (f"{'Family':<8} | {'First':<6} | {'Second':<6} | "
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

    print(f"\n[DONE] Individual PDFs saved in: {args.outdir}")
    print(f"[DONE] Table  : {txt_path}")
    print(f"[DONE] Total combinations fitted: {len(all_results)}")


if __name__ == "__main__":
    main()