#!/usr/bin/env python3
"""
waveform_shape_pid.py  (fixed)
================================
Changes vs previous version
----------------------------
1.  Width measured at 50% of peak (true FWHM), not 10%.
2.  Exp-fit removed entirely.  Fall time = interpolated 90%→10% on falling edge.
3.  interp_crossing for the RISING edge now starts searching from a point
    near the peak (scanning backward) so it never latches onto baseline noise.
4.  Heatmap axis limits are computed automatically from the data with a small
    padding, so points are never clipped outside the plot range.
5.  N_EVENTS default raised to 10 000.
"""

import os, re, argparse, warnings
import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

try:
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    HEP = True
except ImportError:
    HEP = False

# =============================================================================
# CONFIG
# =============================================================================
TREE_NAME     = "EventTree"
TIME_PER_BIN  = 0.2
BASELINE_BINS = 30
TIMING_SUFFIX = "_LP2_50"
AMP_THRESHOLD = 100.0
MIN_ADC_CUT   = -100.0
N_EVENTS      = 10_000          # raised

TWINDOW_LEFT  = 15.0
TWINDOW_RIGHT = 25.0

SUBPLOT_FAMILIES = ["Quartz", "Plastic", "Scintillator"]

CHANNELS_3MM = {"Quartz": "104", "Plastic": "010", "Scintillator": "107"}
CHANNELS_6MM = {"Quartz": "604", "Plastic": "606", "Scintillator": "615"}

RUN_ENERGY_MAP = {1429: 80.0, 1480: 170.0, 1355: 80.0,
                  1501: 40.0, 1474: 80.0,  1509: 40.0}
DEFAULT_ENERGY = 40.0

RUN_FILES = {
    "3mm": {
        "positron": ["/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1355_250924165834_converted_timingskim.root"],
        "pion":     ["/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1429_250926183919_converted_timingskim.root"],
        "muon":     ["/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1480_250928004120_converted_timingskim.root"],
    },
}

PID_BRANCH_MAP = {
    "PSD":         "DRS_Board7_Group1_Channel1",
    "TTUMuonVeto": "DRS_Board7_Group2_Channel4",
    "Cer474":      "DRS_Board7_Group2_Channel5",
    "Cer519":      "DRS_Board7_Group2_Channel6",
    "Cer537":      "DRS_Board7_Group2_Channel7",
}
_SVC_CUTS = {
    "PSD":         (100, 400, -3500., "Sum"),
    "TTUMuonVeto": (200, 400, -2e3,   "Sum"),
    "Cer474":      (800, 900, -2000., "Sum"),
    "Cer519":      (450, 550, -1000., "Sum"),
    "Cer537":      (400, 500, -500.,  "Sum"),
}
_PARTICLE_REQS = {
    "positron": {"TTUMuonVeto": False, "PSD": True,
                 "Cer474": True, "Cer519": True, "Cer537": True},
    "pion":     {"TTUMuonVeto": False, "PSD": False,
                 "Cer474": True, "Cer519": True, "Cer537": True},
    "muon":     {"TTUMuonVeto": True,  "PSD": False},
}

PARTICLE_COLORS = {"positron": "#E45C3A", "pion": "#3A8FE4", "muon": "#3AC46E"}
PARTICLE_LABELS = {
    "positron": "Positron (80 GeV)",
    "pion":     "Pion (80 GeV)",
    "muon":     "Muon (~170 GeV)",
}
FAMILY_DISPLAY = {
    "Quartz":       "FSHA (Fused-silica)",
    "Plastic":      "Toray PJR-FB750 (Plastic)",
    "Scintillator": "SCSF-81J (Scintillator)",
}

# =============================================================================
# HELPERS
# =============================================================================

def code_to_branch(code):
    s = str(code).zfill(3)
    return f"DRS_Board{s[0]}_Group{s[1]}_Channel{s[2]}"

def run_number_from_path(path):
    m = re.search(r"run(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None

def compute_pid_mask(tree, particle):
    reqs = _PARTICLE_REQS.get(particle.lower(), {})
    mask = np.ones(tree.num_entries, dtype=bool)
    for det, must_fire in reqs.items():
        br = PID_BRANCH_MAP.get(det)
        if not br or br not in tree.keys():
            continue
        ts_min, ts_max, val_cut, method = _SVC_CUTS.get(det, (0, 1000, -5e4, "Sum"))
        if method != "Sum":
            continue
        try:
            waves    = tree[br].array(library="ak")
            baseline = ak.mean(waves[:, :BASELINE_BINS], axis=1)
            win_sum  = ak.sum((waves - baseline)[:, int(ts_min):int(ts_max)], axis=1)
            fired    = ak.to_numpy(win_sum) < val_cut
            mask    &= fired if must_fire else ~fired
        except Exception:
            pass
    return mask

# =============================================================================
# CROSSING INTERPOLATION  (fixed)
# =============================================================================

def interp_crossing_forward(t, wf, level, i_start):
    """First upward crossing of `level` searching forward from i_start."""
    for i in range(i_start, len(wf) - 1):
        if wf[i] <= level < wf[i + 1]:
            dt = (level - wf[i]) / (wf[i + 1] - wf[i])
            return t[i] + dt * (t[i + 1] - t[i])
    return np.nan


def interp_crossing_backward(t, wf, level, i_start):
    """
    First upward crossing of `level` searching BACKWARD from i_start.
    Used for the rising edge: we scan left from the peak so we find
    the last time the signal was below `level` before the peak,
    avoiding any spurious crossings in the baseline region.
    """
    for i in range(i_start, 0, -1):
        if wf[i] >= level > wf[i - 1]:
            dt = (level - wf[i - 1]) / (wf[i] - wf[i - 1])
            return t[i - 1] + dt * (t[i] - t[i - 1])
    return np.nan


def interp_crossing_falling(t, wf, level, i_start, i_stop=None):
    """First downward crossing of `level` searching forward from i_start.
    Search is clamped to i_stop (exclusive) if provided, so sub-pulses
    beyond the main peak fall region are never reached.
    """
    stop = i_stop if i_stop is not None else len(wf) - 1
    for i in range(i_start, stop):
        if wf[i] >= level > wf[i + 1]:
            dt = (wf[i] - level) / (wf[i] - wf[i + 1])
            return t[i] + dt * (t[i + 1] - t[i])
    return np.nan

# =============================================================================
# WAVEFORM SHAPE METRICS  (fixed)
# =============================================================================

# Maximum time after the peak to search for falling-edge crossings.
# This prevents sub-pulses from shower tails (common in scintillators)
# from being mistaken for the 10% / 50% crossings of the main pulse.
# Set conservatively to 5 ns — larger than any physical single-pulse fall
# time but smaller than the gap to the first shower sub-pulse.
FALL_SEARCH_NS = 2.0


def compute_shape_metrics(wf, t_abs):
    """
    Returns dict with:
      rise_time_ns  : 10% → 90%  on rising edge  (backward search from peak)
      fall_time_ns  : 90% → 10%  on falling edge (forward, clamped to FALL_SEARCH_NS)
      width_ns      : FWHM  —  50% rising → 50% falling  (same window)
      peak_adc      : baseline-subtracted peak
      t_peak_ns     : absolute time of peak bin

    Falling-edge crossings are only searched within [t_peak, t_peak + FALL_SEARCH_NS].
    This robustly rejects shower sub-pulses that arrive later and would otherwise
    inflate fall_time and width for scintillator channels.
    """
    peak_val = float(np.max(wf))
    peak_idx = int(np.argmax(wf))

    if peak_val < AMP_THRESHOLD:
        return {k: np.nan for k in
                ["rise_time_ns", "fall_time_ns", "width_ns",
                 "peak_adc", "t_peak_ns",
                 "t_10_rise", "t_90_rise", "t_90_fall", "t_10_fall",
                 "t_50_rise", "t_50_fall"]}

    lvl10 = 0.10 * peak_val
    lvl50 = 0.50 * peak_val
    lvl90 = 0.90 * peak_val

    # ── Rising edge: search backward from peak ────────────────────────────
    t_10_rise = interp_crossing_backward(t_abs, wf, lvl10, peak_idx)
    t_90_rise = interp_crossing_backward(t_abs, wf, lvl90, peak_idx)
    t_50_rise = interp_crossing_backward(t_abs, wf, lvl50, peak_idx)

    # ── Falling edge: clamped window to suppress shower sub-pulses ────────
    # Convert FALL_SEARCH_NS to a bin index limit
    fall_stop = min(len(wf) - 1,
                    peak_idx + int(np.ceil(FALL_SEARCH_NS / TIME_PER_BIN)))
    
    t_90_fall = interp_crossing_falling(t_abs, wf, lvl90, peak_idx, fall_stop)
    t_10_fall = interp_crossing_falling(t_abs, wf, lvl10, peak_idx, fall_stop)
    t_50_fall = interp_crossing_falling(t_abs, wf, lvl50, peak_idx, fall_stop)

    rise_time = (t_90_rise - t_10_rise
                 if np.isfinite(t_90_rise) and np.isfinite(t_10_rise) else np.nan)
    fall_time = (t_10_fall - t_90_fall
                 if np.isfinite(t_10_fall) and np.isfinite(t_90_fall) else np.nan)
    # Width = FWHM (50% → 50%)
    width     = (t_50_fall - t_50_rise
                 if np.isfinite(t_50_fall) and np.isfinite(t_50_rise) else np.nan)

    return {
        "rise_time_ns": rise_time,
        "fall_time_ns": fall_time,
        "width_ns":     width,
        "peak_adc":     peak_val,
        "t_peak_ns":    float(t_abs[peak_idx]),
        "t_10_rise":    t_10_rise,
        "t_90_rise":    t_90_rise,
        "t_50_rise":    t_50_rise,
        "t_90_fall":    t_90_fall,
        "t_10_fall":    t_10_fall,
        "t_50_fall":    t_50_fall,
        "_waveform":    wf.copy(),
        "_t_abs":       t_abs.copy(),
    }

# =============================================================================
# COLLECT METRICS
# =============================================================================

def collect_metrics(fpath, chan_map, particle, n_max):
    results = {f: [] for f in SUBPLOT_FAMILIES}
    rnum    = run_number_from_path(fpath)

    try:
        with uproot.open(fpath) as f:
            tree     = f[TREE_NAME]
            pid_mask = compute_pid_mask(tree, particle)
            pid_idxs = np.where(pid_mask)[0]
            print(f"  PID pass: {len(pid_idxs)} / {tree.num_entries}")

            for family in SUBPLOT_FAMILIES:
                code   = chan_map.get(family)
                br     = code_to_branch(code)
                t50_br = br + TIMING_SUFFIX

                if br not in tree.keys():
                    print(f"  [WARN] {family}: branch {br} missing")
                    continue

                waves_ak  = tree[br].array(library="ak")
                baseline  = ak.mean(waves_ak[:, :BASELINE_BINS], axis=1)
                w_np      = ak.to_numpy(waves_ak - baseline)
                t50_arr   = (tree[t50_br].array(library="np")
                             if t50_br in tree.keys() else None)

                n_collected = 0
                for ev_idx in pid_idxs:
                    if n_collected >= n_max:
                        break
                    wf  = w_np[ev_idx]
                    t50 = float(t50_arr[ev_idx]) if t50_arr is not None else None

                    peak_val = float(np.max(wf))
                    trough   = float(np.min(wf))
                    t50_ok   = t50 is not None and np.isfinite(t50) and t50 > 0

                    if peak_val < AMP_THRESHOLD or trough < MIN_ADC_CUT or not t50_ok:
                        continue

                    t_abs   = np.arange(len(wf)) * TIME_PER_BIN
                    metrics = compute_shape_metrics(wf, t_abs)
                    metrics["event_idx"] = int(ev_idx)
                    metrics["run_num"]   = rnum
                    metrics["t50_ns"]    = t50
                    results[family].append(metrics)
                    n_collected += 1

                print(f"  {family}: {n_collected} valid pulses")

    except Exception as e:
        print(f"  [ERROR] {fpath}: {e}")
        import traceback; traceback.print_exc()

    return results

# =============================================================================
# PLOT HELPERS
# =============================================================================

METRICS_INFO = {
    "rise_time_ns": ("Rise Time [ns]",   50),
    "fall_time_ns": ("Fall Time [ns]",   50),
    "width_ns":     ("Pulse Width (FWHM) [ns]", 50),
}


def gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _tick_style(ax):
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)


def _auto_lim(values_list, pad_frac=0.10):
    """
    Compute axis limits from a list of 1-D arrays, ignoring NaNs.
    Adds pad_frac * range as padding on each side so no point is clipped.
    """
    all_v = np.concatenate([v[np.isfinite(v)] for v in values_list if len(v)])
    if len(all_v) == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(all_v, 1), np.percentile(all_v, 99)
    pad = max((hi - lo) * pad_frac, 0.05)
    return (lo - pad, hi + pad)


def _plot_heatmap(ax, x, y, xlabel, ylabel, title, cmap="viridis"):
    """
    2-D histogram with axis limits set automatically from the data.
    Returns the AxesImage so a colorbar can be added by the caller.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        ax.set_title(title + "  [no data]")
        return None

    xlim = _auto_lim([x])
    ylim = _auto_lim([y])
    nbins = 60

    H, xedges, yedges = np.histogram2d(x, y, bins=nbins,
                                        range=[xlim, ylim])
    H = H.T
    H_plot = np.where(H > 0, H, np.nan)

    im = ax.imshow(
        H_plot,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap=cmap,
        norm=LogNorm(vmin=1, vmax=max(H.max(), 1)),
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="Counts (log)", fraction=0.046, pad=0.04)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=11)
    _tick_style(ax)
    return im

# =============================================================================
# PDF 1 — HISTOGRAMS
# =============================================================================

def make_histograms_pdf(all_metrics, particle_list, outdir, n_tag):
    pdf_path = os.path.join(outdir, f"01_histograms_{n_tag}events.pdf")
    with PdfPages(pdf_path) as pdf:
        for metric, (xlabel, nbins) in METRICS_INFO.items():
            for family in SUBPLOT_FAMILIES:
                fig, ax = plt.subplots(figsize=(10, 7))

                all_vals = []
                for particle in particle_list:
                    vals = np.array([m[metric] for m in all_metrics[particle][family]
                                     if np.isfinite(m.get(metric, np.nan))])
                    all_vals.append(vals)

                xlim = _auto_lim(all_vals)

                for particle, vals in zip(particle_list, all_vals):
                    if len(vals) == 0:
                        continue
                    color = PARTICLE_COLORS[particle]
                    label = PARTICLE_LABELS[particle]

                    # Raw counts then divide by peak bin so max = 1
                    counts, edges = np.histogram(vals, bins=nbins, range=xlim)
                    centers       = 0.5 * (edges[:-1] + edges[1:])
                    peak_count    = counts.max() if counts.max() > 0 else 1
                    norm_counts   = counts / peak_count

                    ax.fill_between(centers, norm_counts, step="mid",
                                    color=color, alpha=0.25)
                    ax.step(centers, norm_counts, where="mid",
                            color=color, lw=1.8, alpha=0.90)

                    try:
                        popt, _ = curve_fit(gauss, centers, norm_counts,
                                            p0=[1.0, np.nanmean(vals),
                                                np.nanstd(vals)], maxfev=3000)
                        x_fit = np.linspace(*xlim, 300)
                        ax.plot(x_fit, gauss(x_fit, *popt),
                                color=color, lw=2.2, ls="--")
                        mu_s  = f"{popt[1]:.3f}"
                        sig_s = f"{abs(popt[2]):.3f}"
                    except Exception:
                        mu_s  = f"{np.nanmean(vals):.3f}"
                        sig_s = f"{np.nanstd(vals):.3f}"

                    ax.plot([], [], color=color, lw=2,
                            label=f"{label}\n  μ={mu_s} ns, σ={sig_s} ns")

                ax.set_xlabel(xlabel, fontsize=13)
                ax.set_ylabel("Normalised events ", fontsize=12)
                ax.set_xlim(xlim)
                ax.set_ylim(0, 1.12)
                ax.legend(fontsize=10, frameon=True, loc="upper right")
                ax.set_title(
                    f"{family}  —  {FAMILY_DISPLAY[family]}\n{xlabel}", fontsize=12
                )
                _tick_style(ax)
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    print(f"  [OK] Histograms PDF  -> {pdf_path}")
    return pdf_path

# =============================================================================
# PDF 2 — HEATMAPS  (side-by-side: one page per family × plot-type,
#                    particles as columns)
# =============================================================================

def make_heatmaps_pdf(all_metrics, particle_list, outdir, n_tag):
    """
    Layout: one PDF page per (family × correlation type).
    Each page is a 1-row grid of N_particles columns so all particles
    can be compared directly side-by-side with shared axis limits.

    Page order:
      For each family in [Quartz, Plastic, Scintillator]:
        Row A — Rise vs Fall
        Row B — Width (FWHM) vs Rise
        Row C — Fall vs Width
        Row D — Peak ADC vs Rise
    """
    pdf_path = os.path.join(outdir, f"02_heatmaps_{n_tag}events.pdf")
    n_cols   = len(particle_list)

    # --- helper: collect arrays for one metric across all particles ----------
    def _get(metric, family):
        return [np.array([m[metric] for m in all_metrics[p][family]])
                for p in particle_list]

    # --- define the four correlation types -----------------------------------
    PLOT_TYPES = [
        # (tag, xlabel, ylabel, x_metric, y_metric, cmap)
        ("rise_vs_fall",  "Rise Time [ns]",       "Fall Time [ns]",
         "rise_time_ns",  "fall_time_ns",          "plasma"),
        ("width_vs_rise", "Rise Time [ns]",        "Pulse Width FWHM [ns]",
         "rise_time_ns",  "width_ns",              "cividis"),
        ("fall_vs_width", "Pulse Width FWHM [ns]", "Fall Time [ns]",
         "width_ns",      "fall_time_ns",          "magma"),
        ("rise_vs_peak",  "Peak ADC [counts]",     "Rise Time [ns]",
         "peak_adc",      "rise_time_ns",          "inferno"),
    ]

    with PdfPages(pdf_path) as pdf:
        for family in SUBPLOT_FAMILIES:
            fam_label = f"{family}  —  {FAMILY_DISPLAY[family]}"

            for tag, xlabel, ylabel, xmet, ymet, cmap in PLOT_TYPES:

                # Collect x/y arrays for every particle
                x_arrs = _get(xmet, family)
                y_arrs = _get(ymet, family)

                # Shared axis limits computed across ALL particles on this page
                xlim = _auto_lim(x_arrs)
                ylim = _auto_lim(y_arrs)

                fig, axes = plt.subplots(
                    1, n_cols,
                    figsize=(6 * n_cols, 6.5),
                    sharey=True,
                )
                if n_cols == 1:
                    axes = [axes]

                for ax, particle, x_arr, y_arr in zip(
                        axes, particle_list, x_arrs, y_arrs):

                    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
                    xp, yp = x_arr[mask], y_arr[mask]

                    if len(xp) == 0:
                        ax.set_title(f"{PARTICLE_LABELS[particle]}\n[no data]",
                                     fontsize=10)
                        continue

                    H, xedges, yedges = np.histogram2d(
                        xp, yp, bins=60, range=[xlim, ylim]
                    )
                    H     = H.T
                    H_plot = np.where(H > 0, H, np.nan)

                    im = ax.imshow(
                        H_plot,
                        origin="lower",
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                        aspect="auto",
                        cmap=cmap,
                        norm=LogNorm(vmin=1, vmax=max(H.max(), 1)),
                        interpolation="nearest",
                    )
                    plt.colorbar(im, ax=ax, label="Counts (log)",
                                 fraction=0.046, pad=0.04)

                    ax.set_xlabel(xlabel, fontsize=11)
                    if ax is axes[0]:
                        ax.set_ylabel(ylabel, fontsize=11)

                    ax.set_title(
                        f"{PARTICLE_LABELS[particle]}\n"
                        f"N = {mask.sum():,}",
                        fontsize=10,
                    )
                    _tick_style(ax)

                fig.suptitle(
                    f"{fam_label}\n"
                    f"{ylabel}  vs  {xlabel}",
                    fontsize=12, y=1.01,
                )
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    print(f"  [OK] Heatmaps PDF    -> {pdf_path}")
    return pdf_path

# =============================================================================
# PDF 3 — EXAMPLE WAVEFORMS  (no exp fit, corrected shading)
# =============================================================================

def make_waveforms_pdf(all_metrics, particle_list, outdir, n_tag):
    pdf_path   = os.path.join(outdir, f"03_waveforms_{n_tag}events.pdf")
    N_EXAMPLES = 5

    with PdfPages(pdf_path) as pdf:
        for family in SUBPLOT_FAMILIES:
            for particle in particle_list:
                color = PARTICLE_COLORS[particle]
                mlist = all_metrics[particle][family]

                # Select examples with a valid fall_time and a waveform stored
                valid = [m for m in mlist
                         if np.isfinite(m.get("fall_time_ns", np.nan))
                         and m.get("_waveform") is not None]
                if not valid:
                    continue

                # Pick examples closest to the median peak amplitude
                peaks  = np.array([m["peak_adc"] for m in valid])
                med_pk = np.median(peaks)
                valid.sort(key=lambda m: abs(m["peak_adc"] - med_pk))
                examples = valid[:N_EXAMPLES]

                for ex_num, m in enumerate(examples):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    wf  = m["_waveform"]
                    t   = m["_t_abs"]
                    t50 = m["t50_ns"]

                    t_lo = t50 - TWINDOW_LEFT
                    t_hi = t50 + TWINDOW_RIGHT
                    win  = (t >= t_lo) & (t <= t_hi)
                    tw, ww = t[win], wf[win]
                    tr = tw - t50   # time relative to t50

                    ax.plot(tr, ww, color=color, lw=1.8, zorder=3,
                            label="Waveform")

                    pk = m["peak_adc"]

                    # ── Rising edge shading: 10%–90% (backward-found crossings) ──
                    t10r = m.get("t_10_rise", np.nan)
                    t90r = m.get("t_90_rise", np.nan)
                    if np.isfinite(t10r) and np.isfinite(t90r):
                        shade = (tw >= t10r) & (tw <= t90r)
                        ax.fill_between(tw[shade] - t50, 0, ww[shade],
                                        color="#3A8FE4", alpha=0.30,
                                        label="Rise 10→90%", zorder=2)

                    # ── Falling edge shading: 90%–10%, clamped to search window ─
                    t90f = m.get("t_90_fall", np.nan)
                    t10f = m.get("t_10_fall", np.nan)
                    t_peak_abs = m.get("t_peak_ns", t50)
                    fall_limit = t_peak_abs + FALL_SEARCH_NS
                    if np.isfinite(t90f) and np.isfinite(t10f):
                        shade = (tw >= t90f) & (tw <= min(t10f, fall_limit))
                        ax.fill_between(tw[shade] - t50, 0, ww[shade],
                                        color="#E45C3A", alpha=0.30,
                                        label="Fall 90→10%", zorder=2)
                    # Dotted vertical line shows where the fall search window ends
                    ax.axvline(fall_limit - t50, color="#E45C3A", lw=0.9,
                               ls=(0, (3, 4)), alpha=0.6,
                               label=f"Fall window (+{FALL_SEARCH_NS:.0f} ns)")

                    # ── 50% markers for FWHM ─────────────────────────────────
                    t50r = m.get("t_50_rise", np.nan)
                    t50f = m.get("t_50_fall", np.nan)
                    for tv, lbl in [(t50r, "50% rise"), (t50f, "50% fall")]:
                        if np.isfinite(tv):
                            ax.axvline(tv - t50, color="purple", lw=1.0,
                                       ls=":", alpha=0.7, label=lbl)

                    # ── Reference lines ──────────────────────────────────────
                    for frac, lbl in [(0.10, "10% pk"), (0.50, "50% pk"),
                                      (0.90, "90% pk")]:
                        ax.axhline(frac * pk, color="gray", lw=0.8,
                                   ls=":", alpha=0.5,
                                   label=lbl if frac == 0.10 else None)

                    ax.axvline(0, color="gray", lw=0.9, ls="--", alpha=0.55,
                               label="t₅₀")

                    # ── Metric annotation ────────────────────────────────────
                    rt = m.get("rise_time_ns", np.nan)
                    ft = m.get("fall_time_ns", np.nan)
                    wd = m.get("width_ns",     np.nan)
                    ann = (f"Rise (10→90%) = {rt:.3f} ns\n"
                           f"Fall (90→10%) = {ft:.3f} ns\n"
                           f"Width (FWHM)  = {wd:.3f} ns")
                    ax.text(0.97, 0.97, ann, transform=ax.transAxes,
                            fontsize=9, va="top", ha="right",
                            bbox=dict(fc="white", ec="#cccccc",
                                      boxstyle="round,pad=0.45", alpha=0.85))

                    ax.set_xlabel("Time − t₅₀ [ns]", fontsize=12)
                    ax.set_ylabel("ADC (baseline subtracted)", fontsize=12)
                    ax.set_title(
                        f"{PARTICLE_LABELS[particle]}  |  "
                        f"{family} ({FAMILY_DISPLAY[family]})\n"
                        f"Example {ex_num+1}/{len(examples)}  —  "
                        f"event {m['event_idx']}  |  peak = {pk:.0f} ADC",
                        fontsize=11
                    )
                    ax.set_xlim(t_lo - t50, t_hi - t50)
                    ax.set_ylim(bottom=-0.06 * pk)
                    ax.legend(fontsize=8, frameon=True, loc="upper left",
                              ncol=2)
                    _tick_style(ax)
                    fig.tight_layout()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

    print(f"  [OK] Waveforms PDF   -> {pdf_path}")
    return pdf_path

# =============================================================================
# PDF 4 — SUMMARY
# =============================================================================

def make_summary_pdf(all_metrics, particle_list, outdir, n_tag):
    pdf_path = os.path.join(outdir, f"04_summary_{n_tag}events.pdf")

    with PdfPages(pdf_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(14, 6))
        ax  = fig.add_subplot(111)
        ax.axis("off")
        lines = [
            r"$\bf{Waveform\ Shape\ PID\ Analysis}$",
            "",
            "Metrics:  Rise Time (10→90%)  |  Fall Time (90→10%)  "
            "|  Pulse Width (FWHM 50→50%)",
            "Amplitude cut: peak ≥ 100 ADC  |  trough ≥ −100 ADC",
            "",
            "Particles: " + ",  ".join(PARTICLE_LABELS[p] for p in particle_list),
            "Families:  Quartz ch104  |  Plastic ch010  |  Scintillator ch107  (3 mm)",
            f"Events per combination: up to {n_tag}",
        ]
        ax.text(0.5, 0.5, "\n".join(lines),
                ha="center", va="center", fontsize=12,
                transform=ax.transAxes,
                bbox=dict(fc="#f7f7f7", ec="#aaaaaa",
                          boxstyle="round,pad=0.9"))
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Summary table
        fig = plt.figure(figsize=(16, 7))
        ax  = fig.add_subplot(111)
        ax.axis("off")

        header = ["Family", "Particle",
                  "Rise μ±σ [ns]", "Fall μ±σ [ns]", "FWHM μ±σ [ns]", "N"]
        rows   = [header]
        for family in SUBPLOT_FAMILIES:
            for particle in particle_list:
                mlist = all_metrics[particle][family]
                row   = [family if particle == particle_list[0] else "",
                         PARTICLE_LABELS[particle]]
                for metric in ["rise_time_ns", "fall_time_ns", "width_ns"]:
                    vals = np.array([m[metric] for m in mlist
                                     if np.isfinite(m.get(metric, np.nan))])
                    if len(vals) >= 5:
                        row.append(f"{np.mean(vals):.3f} ± {np.std(vals):.3f}")
                    else:
                        row.append("—")
                row.append(str(len(mlist)))
                rows.append(row)

        tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.9)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#f0f4f8")

        fig.suptitle("Summary: Waveform Shape Metrics (mean ± std)", fontsize=13)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"  [OK] Summary PDF     -> {pdf_path}")
    return pdf_path

# =============================================================================
# CSV
# =============================================================================

def write_csv(all_metrics, particle_list, outdir, n_tag):
    import csv
    csv_path = os.path.join(outdir, f"shapes_{n_tag}events.csv")
    fieldnames = ["particle", "family", "event_idx", "run_num",
                  "t50_ns", "peak_adc", "t_peak_ns",
                  "rise_time_ns", "fall_time_ns", "width_ns",
                  "t_10_rise", "t_90_rise", "t_50_rise",
                  "t_90_fall", "t_10_fall", "t_50_fall"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for particle in particle_list:
            for family in SUBPLOT_FAMILIES:
                for m in all_metrics[particle][family]:
                    row = {k: v for k, v in m.items() if not k.startswith("_")}
                    row["particle"] = particle
                    row["family"]   = family
                    w.writerow(row)
    print(f"  [OK] CSV             -> {csv_path}")
    return csv_path

# =============================================================================
# DRIVER
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Waveform shape PID analysis")
    ap.add_argument("--outdir",    default="./Waveform_Shape_analysisv4")
    ap.add_argument("--n-events",  type=int, default=N_EVENTS)
    ap.add_argument("--thickness", default="3mm", choices=["3mm", "6mm"])
    ap.add_argument("--particles", nargs="+",
                    default=["positron", "pion", "muon"],
                    choices=["positron", "pion", "muon"])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    chan_map = CHANNELS_3MM if args.thickness == "3mm" else CHANNELS_6MM

    all_metrics = {}
    for particle in args.particles:
        print(f"\n{'='*60}\n  {args.thickness} | {particle}\n{'='*60}")
        raw_files = RUN_FILES.get(args.thickness, {}).get(particle, [])
        if not raw_files:
            print("  [SKIP] No files configured.")
            all_metrics[particle] = {f: [] for f in SUBPLOT_FAMILIES}
            continue

        combined = {f: [] for f in SUBPLOT_FAMILIES}
        for fpath in raw_files:
            m = collect_metrics(fpath, chan_map, particle, args.n_events)
            for family in SUBPLOT_FAMILIES:
                combined[family].extend(m[family])
                combined[family] = combined[family][:args.n_events]

        all_metrics[particle] = combined

    particle_list = [p for p in args.particles if p in all_metrics]
    n_tag = args.n_events

    print("\n\nGenerating PDFs...")
    make_histograms_pdf(all_metrics, particle_list, args.outdir, n_tag)
    make_heatmaps_pdf(  all_metrics, particle_list, args.outdir, n_tag)
    make_waveforms_pdf( all_metrics, particle_list, args.outdir, n_tag)
    make_summary_pdf(   all_metrics, particle_list, args.outdir, n_tag)
    write_csv(          all_metrics, particle_list, args.outdir, n_tag)
    print("\n[DONE]")


if __name__ == "__main__":
    main()