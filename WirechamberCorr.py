#!/usr/bin/env python3
import os
import re
import glob
import argparse
import numpy as np
import uproot
import awkward as ak

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import mplhep as hep

# ============================================================
# STYLE
# ============================================================
plt.style.use(hep.style.CMS)

# ============================================================
# CONFIGURATION
# ============================================================
TREE_NAME = "EventTree"
AMP_THRESHOLD = 100.0
MIN_ADC_CUT = -100.0

FAMILIES = {
    "Plastic": {"channels": ["100", "102", "112", "110"], "tmin": -14.5, "tmax": -11.5, "legend": "Cherenkov-Plastic", "color": "red"},
    "Quartz":  {"channels": ["104", "106", "304", "114"], "tmin": -15.0, "tmax": -11.5, "legend": "Cherenkov-Quartz",  "color": "blue"},
    "SCI":     {"channels": ["105", "107", "111", "117"], "tmin": -13.5, "tmax":  -9.5, "legend": "Scintillating",     "color": "green"},
}

TARGET_CHANNELS = ["104","105","100","102",
                   #"107","304", "114","112", "110"
                   ]
#TARGET_CHANNELS = ["104"]
WC_CHANNELS = {
    "L1": "DRS_Board7_Group0_Channel0",
    "R1": "DRS_Board7_Group0_Channel1",
    "U1": "DRS_Board7_Group0_Channel2",
    "D1": "DRS_Board7_Group0_Channel3",
}

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

# ============================================================
# HELPERS
# ============================================================
def _run_label(path: str) -> str:
    m = re.search(r"(run\d+_\d{11,12})", os.path.basename(path))
    if m:
        return m.group(1)
    m = re.search(r"(run\d+)", os.path.basename(path))
    if m:
        return m.group(1)
    return os.path.splitext(os.path.basename(path))[0]

def _sort_files(files):
    def key(p):
        m = re.search(r"run(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0
    return sorted(files, key=key)

def _parse_code(code_str):
    return int(code_str[0]), int(code_str[1]), int(code_str[2])

def _branch_tfinal(code_str):
    b, g, c = _parse_code(code_str)
    return f"tfinal_Board{b}_Group{g}_Channel{c}"

def get_family_info(ch_code):
    for fam, info in FAMILIES.items():
        if ch_code in info["channels"]:
            return fam, info
    return "Unknown", {"tmin": -20, "tmax": 20, "legend": "Unknown", "color": "black"}

def display_particle_name(pid):
    if pid is None:
        return "No PID"
    return "Positron" if pid == "electron" else pid.capitalize()

def safe_corr(a, b):
    try:
        if len(a) < 2 or len(b) < 2:
            return np.nan
        if np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return np.corrcoef(a, b)[0, 1]
    except Exception:
        return np.nan

# ============================================================
# PID & ADC MASKS
# ============================================================
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
    if not requirements:
        return None

    n_entries = tree.num_entries
    final_mask = np.ones(n_entries, dtype=bool)
    available_keys = set(tree.keys())

    for det, must_fire in requirements.items():
        branch_name = PID_BRANCH_MAP.get(det)
        if not branch_name or branch_name not in available_keys:
            continue

        ts_min, ts_max, val_cut, method = get_service_drs_cut(det)

        try:
            waveforms = tree[branch_name].array(library="ak")
            if method == "Sum":
                baseline = ak.mean(waveforms[:, :30], axis=1)
                waveforms_blsub = waveforms - baseline
                window_sum = ak.sum(waveforms_blsub[:, int(ts_min):int(ts_max)], axis=1)
                is_fired = ak.to_numpy(window_sum) < val_cut
            else:
                continue

            final_mask = final_mask & is_fired if must_fire else final_mask & (~is_fired)
        except Exception:
            continue

    return final_mask

def compute_adc_mask(tree, code_str):
    b, g, c = int(code_str[0]), int(code_str[1]), int(code_str[2])
    drs_br = f"DRS_Board{b}_Group{g}_Channel{c}"
    if drs_br not in tree:
        return np.ones(tree.num_entries, dtype=bool)

    waves = tree[drs_br].array(library="ak")
    baseline = ak.mean(waves[:, :30], axis=1)
    waves_blsub = waves - baseline

    peak = ak.max(waves_blsub, axis=1)
    min_adc = ak.min(waves_blsub, axis=1)

    mask = (peak >= AMP_THRESHOLD) & (min_adc >= MIN_ADC_CUT)
    return ak.to_numpy(mask)

# ============================================================
# TFINAL STRATEGIES
# ============================================================
def get_tfinal_3mm_baseline(tree, b, g, c, suffix):
    br_sig     = f"DRS_Board{b}_Group{g}_Channel{c}{suffix}"
    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8{suffix}"
    br_trg     = f"DRS_Board0_Group3_Channel7{suffix}"
    br_trg_ref = f"DRS_Board0_Group3_Channel8{suffix}"

    keys = tree.keys()
    if any(br not in keys for br in [br_sig, br_sig_ref, br_trg, br_trg_ref]):
        return None

    arr_sig     = tree[br_sig].array(library="np")
    arr_sig_ref = tree[br_sig_ref].array(library="np")
    arr_trg     = tree[br_trg].array(library="np")
    arr_trg_ref = tree[br_trg_ref].array(library="np")

    return (arr_sig - arr_sig_ref) - (arr_trg - arr_trg_ref)

def get_tfinal_3mm_mcp6(tree, b, g, c, suffix):
    br_sig     = f"DRS_Board{b}_Group{g}_Channel{c}{suffix}"
    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8{suffix}"
    br_trg     = f"DRS_Board0_Group3_Channel6{suffix}"
    br_trg_ref = f"DRS_Board0_Group3_Channel8{suffix}"

    keys = tree.keys()
    if any(br not in keys for br in [br_sig, br_sig_ref, br_trg, br_trg_ref]):
        return None

    arr_sig     = tree[br_sig].array(library="np")
    arr_sig_ref = tree[br_sig_ref].array(library="np")
    arr_trg     = tree[br_trg].array(library="np")
    arr_trg_ref = tree[br_trg_ref].array(library="np")

    return (arr_sig - arr_sig_ref) - (arr_trg - arr_trg_ref)

def get_tfinal_3mm_avg(tree, b, g, c, suffix):
    br_sig     = f"DRS_Board{b}_Group{g}_Channel{c}{suffix}"
    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8{suffix}"
    br_mcp1    = f"DRS_Board0_Group3_Channel6{suffix}"
    br_mcp2    = f"DRS_Board0_Group3_Channel7{suffix}"
    br_trg_ref = f"DRS_Board0_Group3_Channel8{suffix}"

    keys = tree.keys()
    if any(br not in keys for br in [br_sig, br_sig_ref, br_mcp1, br_mcp2, br_trg_ref]):
        return None

    arr_sig     = tree[br_sig].array(library="np")
    arr_sig_ref = tree[br_sig_ref].array(library="np")
    arr_mcp1    = tree[br_mcp1].array(library="np")
    arr_mcp2    = tree[br_mcp2].array(library="np")
    arr_trg_ref = tree[br_trg_ref].array(library="np")

    mcp_avg = (arr_mcp1 + arr_mcp2) / 2.0
    return (arr_sig - arr_sig_ref) - (mcp_avg - arr_trg_ref)

def get_tfinal(tree, ch_code, mode="branch", suffix="_LP2_50"):
    b, g, c = _parse_code(ch_code)

    if mode == "branch":
        br = _branch_tfinal(ch_code)
        if br not in tree:
            return None
        return ak.to_numpy(tree[br].array(library="ak"))

    if mode == "baseline":
        return get_tfinal_3mm_baseline(tree, b, g, c, suffix)

    if mode == "mcp6":
        return get_tfinal_3mm_mcp6(tree, b, g, c, suffix)

    if mode == "avg":
        return get_tfinal_3mm_avg(tree, b, g, c, suffix)

    raise ValueError(f"Unknown tfinal mode: {mode}")

# ============================================================
# WIRE CHAMBER
# ============================================================
def get_hit_times_vectorized(events):
    if events.ndim != 2:
        return np.zeros(len(events))

    baselines = np.mean(events[:, :20], axis=1, keepdims=True)
    corrected = events - baselines
    hit_indices = np.argmin(corrected, axis=1)
    return hit_indices

def get_wc_positions(tree):
    try:
        keys = set(tree.keys())
        required = [WC_CHANNELS["L1"], WC_CHANNELS["R1"], WC_CHANNELS["U1"], WC_CHANNELS["D1"]]
        if not all(k in keys for k in required):
            return None, None

        L1 = ak.to_numpy(tree[WC_CHANNELS["L1"]].array(library="ak"))
        R1 = ak.to_numpy(tree[WC_CHANNELS["R1"]].array(library="ak"))
        U1 = ak.to_numpy(tree[WC_CHANNELS["U1"]].array(library="ak"))
        D1 = ak.to_numpy(tree[WC_CHANNELS["D1"]].array(library="ak"))

        L1_t = get_hit_times_vectorized(L1)
        R1_t = get_hit_times_vectorized(R1)
        U1_t = get_hit_times_vectorized(U1)
        D1_t = get_hit_times_vectorized(D1)

        x_positions = L1_t - R1_t
        y_positions = U1_t - D1_t
        return x_positions, y_positions

    except Exception as e:
        print(f"    [WC] Error calculating positions: {e}")
        return None, None


# ============================================================
# GAUSSIAN FIT HELPERS
# ============================================================
def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def get_hist_mode(arr, bins=80, fit_range=None):
    if len(arr) < 3:
        return np.nan, None, None

    h, edges = np.histogram(arr, bins=bins, range=fit_range)
    if np.sum(h) == 0:
        return np.nan, h, edges

    centers = 0.5 * (edges[:-1] + edges[1:])
    mode_idx = int(np.argmax(h))
    mode_val = centers[mode_idx]
    return mode_val, h, edges

def fit_gaussian_about_mode(arr, bins=80, hist_range=None, fit_window_ns=0.6):
    """
    Fit Gaussian around the histogram mode.
    Returns dict with:
      mode, mu_fit, sigma_fit, amp_fit, success
    """
    if len(arr) < 10:
        return {
            "mode": np.nan,
            "mu_fit": np.nan,
            "sigma_fit": np.nan,
            "amp_fit": np.nan,
            "success": False,
        }

    mode_val, h, edges = get_hist_mode(arr, bins=bins, fit_range=hist_range)
    if not np.isfinite(mode_val):
        return {
            "mode": np.nan,
            "mu_fit": np.nan,
            "sigma_fit": np.nan,
            "amp_fit": np.nan,
            "success": False,
        }

    centers = 0.5 * (edges[:-1] + edges[1:])

    fit_mask = (centers >= mode_val - fit_window_ns) & (centers <= mode_val + fit_window_ns)
    xfit = centers[fit_mask]
    yfit = h[fit_mask]

    if len(xfit) < 5 or np.max(yfit) <= 0:
        return {
            "mode": mode_val,
            "mu_fit": mode_val,
            "sigma_fit": np.std(arr),
            "amp_fit": np.max(h) if len(h) else np.nan,
            "success": False,
        }

    # initial guesses
    amp0 = float(np.max(yfit))
    mu0 = float(mode_val)
    sig0 = max(np.std(arr), 0.05)

    try:
        from scipy.optimize import curve_fit

        popt, _ = curve_fit(
            gaussian,
            xfit,
            yfit,
            p0=[amp0, mu0, sig0],
            bounds=([
                0.0,
                mode_val - fit_window_ns,
                1e-3
            ], [
                np.inf,
                mode_val + fit_window_ns,
                5.0
            ]),
            maxfev=20000
        )

        amp_fit, mu_fit, sigma_fit = popt
        sigma_fit = abs(float(sigma_fit))

        return {
            "mode": float(mode_val),
            "mu_fit": float(mu_fit),
            "sigma_fit": sigma_fit,
            "amp_fit": float(amp_fit),
            "success": True,
        }

    except Exception:
        return {
            "mode": float(mode_val),
            "mu_fit": float(mode_val),
            "sigma_fit": float(np.std(arr)),
            "amp_fit": float(np.max(yfit)),
            "success": False,
        }
# ============================================================
# PLOTTING
# ============================================================

def draw_corner_page(pdf, t_plot, x_plot, y_plot, ch_code, run_id, particle_type,
                     tfinal_mode, family_info, wc_range, bins_1d, bins_2d,
                     t_range, abs_tfinal):
    color = family_info["color"]
    family_legend = family_info["legend"]
    display_name = display_particle_name(particle_type)

    t_label = "|t_final| [ns]" if abs_tfinal else "t_final [ns]"
    labels = [t_label, "Wire Chamber X (L - R)", "Wire Chamber Y (U - D)"]
    data = [t_plot, x_plot, y_plot]
    ranges = [t_range, (-wc_range, wc_range), (-wc_range, wc_range)]

    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    plt.subplots_adjust(wspace=0.08, hspace=0.08)

    # correlations
    corr_tx = safe_corr(t_plot, x_plot)
    corr_ty = safe_corr(t_plot, y_plot)
    corr_xy = safe_corr(x_plot, y_plot)

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]

            # upper triangle
            if i < j:
                ax.axis("off")
                if (i, j) == (0, 1):
                    text = rf"$\rho$(t,X) = {corr_tx:.4f}" if np.isfinite(corr_tx) else r"$\rho$(t,X) = nan"
                elif (i, j) == (0, 2):
                    text = rf"$\rho$(t,Y) = {corr_ty:.4f}" if np.isfinite(corr_ty) else r"$\rho$(t,Y) = nan"
                elif (i, j) == (1, 2):
                    text = rf"$\rho$(X,Y) = {corr_xy:.4f}" if np.isfinite(corr_xy) else r"$\rho$(X,Y) = nan"
                else:
                    text = ""

                ax.text(
                    0.5, 0.5, text,
                    ha="center", va="center",
                    fontsize=14, fontweight="bold"
                )
                continue

            # diagonal
            if i == j:
                vals = data[i]
                rr = ranges[i]

                ax.hist(
                    vals,
                    bins=bins_1d,
                    range=rr,
                    histtype="stepfilled",
                    color=color if i == 0 else "gray",
                    alpha=0.45,
                    edgecolor=color if i == 0 else "black",
                    linewidth=1.5
                )
                ax.set_xlim(rr)

                mu = np.mean(vals)
                sig = np.std(vals)

                stats_text = f"$\\mu$ = {mu:.3f}\n$\\sigma$ = {sig:.3f}\nN = {len(vals)}"
                ax.text(
                    0.97, 0.95, stats_text,
                    transform=ax.transAxes,
                    ha="right", va="top",
                    fontsize=11,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
                )

                if i < 2:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(labels[i])

                if j == 0:
                    ax.set_ylabel("Events")

                ax.minorticks_on()
                continue

            # lower triangle
            x = data[j]
            y = data[i]
            xr = ranges[j]
            yr = ranges[i]

            h = ax.hist2d(
                x, y,
                bins=bins_2d,
                range=[xr, yr],
                norm=LogNorm(),
                cmap="viridis"
            )

            ax.set_xlim(xr)
            ax.set_ylim(yr)

            if i == 2:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(labels[i])
            else:
                ax.set_yticklabels([])

            ax.minorticks_on()

    # CaloX label
    hep.cms.label(
        ax=axes[0, 0],
        exp="CaloX",
        data=True,
        rlabel=f"40 GeV {display_name} | Ch {ch_code} | {run_id}"
    )

    # overall title/info
    fig.suptitle(
        f"{family_legend} | Corner Plot | tfinal mode: {tfinal_mode}",
        fontsize=16,
        y=0.995
    )

    # one colorbar only for lower-left-most used 2D panel
    cax = axes[2, 0]
    mappable = cax.collections[0] if cax.collections else None
    if mappable is not None:
        cb = fig.colorbar(mappable, ax=axes, fraction=0.02, pad=0.01)
        cb.set_label("Counts (log)")

    pdf.savefig(fig)
    plt.close(fig)

def draw_tx_corner_page(pdf, t_plot, x_plot, ch_code, run_id, particle_type,
                        tfinal_mode, family_info, wc_range, bins_1d, bins_2d,
                        t_range):
    color = family_info["color"]
    family_legend = family_info["legend"]
    display_name = display_particle_name(particle_type)

    t_label = "t_final [ns]"
    x_label = "Wire Chamber X (L - R)"

    corr_tx = safe_corr(t_plot, x_plot)

    fitres = fit_gaussian_about_mode(
        t_plot,
        bins=bins_1d,
        hist_range=t_range,
        fit_window_ns=0.6
    )
    mode_t = fitres["mode"]
    mu_t = fitres["mu_fit"]
    sig_t = fitres["sigma_fit"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    plt.subplots_adjust(left=0.10, right=0.93, bottom=0.10, top=0.90,
                        wspace=0.16, hspace=0.16)

    # =========================================================
    # (0,0) tfinal histogram + fit
    # =========================================================
    ax = axes[0, 0]
    counts, edges, _ = ax.hist(
        t_plot,
        bins=bins_1d,
        range=t_range,
        histtype="stepfilled",
        color=color,
        alpha=0.40,
        edgecolor=color,
        linewidth=1.4
    )
    ax.set_xlim(t_range)
    ax.set_ylabel("Events")
    ax.set_xticklabels([])
    ax.minorticks_on()

    if np.isfinite(mu_t) and np.isfinite(sig_t) and sig_t > 0:
        xline = np.linspace(t_range[0], t_range[1], 600)
        binw = edges[1] - edges[0]
        amp_hist = len(t_plot) * binw
        yline = gaussian(xline, amp_hist, mu_t, sig_t)
        ax.plot(xline, yline, color="black", linewidth=2.0, label="Gaussian fit")

    ax.text(
        0.97, 0.95,
        f"mode = {mode_t:.3f}\n"
        f"$\\mu_{{fit}}$ = {mu_t:.3f}\n"
        f"$\\sigma_{{fit}}$ = {sig_t:.3f}\n"
        f"N = {len(t_plot)}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10.5,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
    )

    hep.cms.label(
        ax=ax,
        exp="CaloX",
        data=True,
        rlabel=f"40 GeV {display_name} | Ch {ch_code} | {run_id}"
    )

    # =========================================================
    # (0,1) correlation text
    # =========================================================
    ax = axes[0, 1]
    ax.axis("off")
    txt = rf"$\rho$(t,X) = {corr_tx:.4f}" if np.isfinite(corr_tx) else r"$\rho$(t,X) = nan"
    ax.text(0.5, 0.58, txt, ha="center", va="center", fontsize=16, fontweight="bold")

    # =========================================================
    # (1,0) tfinal vs X 2D histogram
    # =========================================================
    ax = axes[1, 0]
    h = ax.hist2d(
        t_plot,
        x_plot,
        bins=bins_2d,
        range=[t_range, (-wc_range, wc_range)],
        norm=LogNorm(),
        cmap="viridis"
    )
    ax.set_xlabel(t_label)
    ax.set_ylabel(x_label)
    ax.minorticks_on()

    # =========================================================
    # (1,1) X histogram
    # =========================================================
    ax = axes[1, 1]
    ax.hist(
        x_plot,
        bins=bins_1d,
        range=(-wc_range, wc_range),
        histtype="stepfilled",
        color="gray",
        alpha=0.45,
        edgecolor="black",
        linewidth=1.2
    )
    ax.set_xlim(-wc_range, wc_range)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Events")
    ax.minorticks_on()

    mu_x = np.mean(x_plot)
    sig_x = np.std(x_plot)
    ax.text(
        0.97, 0.95,
        f"$\\mu$ = {mu_x:.3f}\n$\\sigma$ = {sig_x:.3f}\nN = {len(x_plot)}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10.5,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
    )

    cb = fig.colorbar(h[3], ax=axes, fraction=0.030, pad=0.02)
    cb.set_label("Counts (log)")

    fig.suptitle(
        f"{family_legend} | t_final vs X | tfinal mode: {tfinal_mode}",
        fontsize=15,
        y=0.965
    )

    pdf.savefig(fig)
    plt.close(fig)

def draw_xcut_study(pdf, t_plot, x_plot, ch_code, run_id, particle_type,
                    tfinal_mode, family_info, t_range, abs_tfinal,
                    xcuts=None, bins_1d=80, min_events_per_cut=30):
    color = family_info["color"]

    family_legend = family_info["legend"]
    display_name = display_particle_name(particle_type)

    t_label = "t_final [ns]"
    if xcuts is None:
        xcuts = [200, 150, 120, 100, 80,75,70, 60, 40, 20]

    stats = []

    # =========================================================
    # PAGE 1: Gaussian fits ONLY (No Histograms)
    # =========================================================
    fig, ax = plt.subplots(figsize=(11, 8))
    plt.subplots_adjust(left=0.11, right=0.97, bottom=0.12, top=0.88)

    # Use a colormap to distinguish different X-cuts since hists are gone
    colors = plt.cm.viridis(np.linspace(0, 1, len(xcuts)))

    for idx, xcut in enumerate(xcuts):
        sel = np.abs(x_plot) < xcut
        t_sel = t_plot[sel]
        
        if len(t_sel) < min_events_per_cut:
            stats.append({"xcut": xcut, "N": len(t_sel), "mean": np.nan, "sigma": np.nan, "mode": np.nan, "success": False})
            continue

        fitres = fit_gaussian_about_mode(t_sel, bins=bins_1d, hist_range=t_range, fit_window_ns=0.6)
        
        stats.append({
            "xcut": xcut, "N": len(t_sel), "mean": fitres["mu_fit"],
            "sigma": fitres["sigma_fit"], "mode": fitres["mode"], "success": fitres["success"]
        })

        mu = fitres["mu_fit"]
        sig = fitres["sigma_fit"]

        if np.isfinite(mu) and np.isfinite(sig) and sig > 0:
            xline = np.linspace(t_range[0], t_range[1], 600)
            # Normalizing the Gaussian to area 1 for comparison across cuts
            yline = (1.0 / (sig * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xline - mu) / sig)**2)
            
            # Legend now corresponds strictly to the Gaussian fit parameters
            ax.plot(xline, yline, linewidth=2.5, color=colors[idx],
                    label=fr"$|X|<{xcut}$: $\mu={mu:.3f}$, $\sigma={sig:.3f}$ ns (N={len(t_sel)})")

    hep.cms.label(ax=ax, exp="CaloX", data=True, rlabel=f"40 GeV {display_name} | Ch {ch_code} | {run_id}")

    ax.set_xlabel(t_label)
    ax.set_ylabel("Probability Density")
    #ax.set_title(f"{family_legend} | Gaussian Timing Resolution vs X-Cut | {tfinal_mode}", fontsize=14, pad=10)
    ax.set_xlim(t_range)
    ax.minorticks_on()
    # Move legend to the side or keep inside; font size adjusted for clarity
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    pdf.savefig(fig)
    plt.close(fig)

    # =========================================================
    # PAGE 2: sigma_fit / mean_fit / N vs X cut
    # =========================================================
    fig, axes = plt.subplots(3, 1, figsize=(11, 11.5), sharex=True)
    plt.subplots_adjust(left=0.12, right=0.97, bottom=0.08, top=0.93, hspace=0.12)

    xvals = np.array([d["xcut"] for d in stats], dtype=float)
    nvals = np.array([d["N"] for d in stats], dtype=float)
    muvals = np.array([d["mean"] for d in stats], dtype=float)
    sigvals = np.array([d["sigma"] for d in stats], dtype=float)
    modevals = np.array([d["mode"] for d in stats], dtype=float)

    # sigma_fit
    ax = axes[0]
    ax.plot(xvals, sigvals, marker="o", linewidth=2, color=color)
    ax.set_ylabel(r"$\sigma_{\mathrm{fit}}(t_{final})$ [ns]")
    #ax.set_title(f"{family_legend} | X-cut scan from Gaussian fits | mode: {tfinal_mode}", fontsize=14, pad=8)
    ax.minorticks_on()
    ax.grid(alpha=0.25)

    finite = np.isfinite(sigvals)
    if np.any(finite):
        finite_x = xvals[finite]
        finite_sig = sigvals[finite]
        best_local = np.argmin(finite_sig)
        best_x = finite_x[best_local]
        best_sig = finite_sig[best_local]

        ax.axvline(best_x, linestyle="--", linewidth=1.4, color="black")
        ax.text(
            0.03, 0.94,
            fr"Best: $|X|<{best_x:.0f}$" "\n"
            fr"$\sigma_{{fit}}$ = {best_sig:.3f} ns",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
        )

    hep.cms.label(
        ax=ax,
        exp="CaloX",
        data=True,
        rlabel=f"40 GeV {display_name} | Ch {ch_code} | {run_id}"
    )

    # mean_fit + mode
    ax = axes[1]
    ax.plot(xvals, muvals, marker="o", linewidth=2, color=color, label=r"$\mu_{\mathrm{fit}}$")
    ax.plot(xvals, modevals, marker="s", linewidth=1.5, linestyle="--", color="black", label="mode")
    ax.set_ylabel(r"$\mu_{\mathrm{fit}}(t_{final})$ [ns]")
    ax.minorticks_on()
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    # N
    ax = axes[2]
    ax.plot(xvals, nvals, marker="o", linewidth=2, color=color)
    ax.set_ylabel("Events")
    ax.set_xlabel(r"X cut: $|X| < x_{\mathrm{cut}}$")
    ax.minorticks_on()
    ax.grid(alpha=0.25)

    pdf.savefig(fig)
    plt.close(fig)

# ============================================================
# MAIN LOGIC
# ============================================================
def process_channel(ch_code, files, args):
    fam_name, fam_info = get_family_info(ch_code)
    pid_tag = args.pid if args.pid else "NoPID"

    os.makedirs(args.outdir, exist_ok=True)
    out_pdf = os.path.join(args.outdir, f"Corner_CaloX_Ch{ch_code}_{pid_tag}_{args.tfinal_mode}.pdf")

    print(f"\n[CHANNEL] --------------------------------------------------")
    print(f"[CHANNEL] Channel {ch_code} ({fam_name})")
    print(f"[CHANNEL] Output: {out_pdf}")

    with PdfPages(out_pdf) as pdf:
        for fpath in files:
            run_id = _run_label(fpath)
            print(f"  -> Processing {run_id}...")

            try:
                with uproot.open(fpath) as uf:
                    if TREE_NAME not in uf:
                        print(f"     [SKIP] Tree {TREE_NAME} missing.")
                        continue

                    tree = uf[TREE_NAME]

                    # tfinal
                    t_data = get_tfinal(tree, ch_code, mode=args.tfinal_mode, suffix=args.suffix)
                    if t_data is None:
                        print(f"     [SKIP] tfinal unavailable for mode={args.tfinal_mode}")
                        continue

                    # WC
                    wc_x, wc_y = get_wc_positions(tree)
                    if wc_x is None or wc_y is None:
                        print(f"     [SKIP] WC data missing.")
                        continue

                    n_events = min(len(t_data), len(wc_x), len(wc_y))

                    total_mask = np.ones(n_events, dtype=bool)

                    if args.pid:
                        pid_mask = compute_pid_mask(tree, args.pid)
                        if pid_mask is not None:
                            total_mask &= pid_mask[:n_events]

                    if args.apply_adc_cut:
                        adc_mask = compute_adc_mask(tree, ch_code)
                        total_mask &= adc_mask[:n_events]

                    t_final = np.asarray(t_data[:n_events])[total_mask]
                    x_final = np.asarray(wc_x[:n_events])[total_mask]
                    y_final = np.asarray(wc_y[:n_events])[total_mask]

                    finite_mask = np.isfinite(t_final) & np.isfinite(x_final) & np.isfinite(y_final)
                    t_final = t_final[finite_mask]
                    x_final = x_final[finite_mask]
                    y_final = y_final[finite_mask]

                    # signed t_final study: no abs applied here

                    # timing range
                    if args.tmin is not None and args.tmax is not None:
                        tmin, tmax = args.tmin, args.tmax
                    else:
                        tmin, tmax = fam_info["tmin"], fam_info["tmax"]

                    range_mask = (
                        (t_final >= tmin) & (t_final <= tmax) &
                        (x_final >= -args.wc_range) & (x_final <= args.wc_range) &
                        (y_final >= -args.wc_range) & (y_final <= args.wc_range)
                    )

                    t_plot = t_final[range_mask]
                    x_plot = x_final[range_mask]
                    y_plot = y_final[range_mask]

                    if len(t_plot) < args.min_events:
                        print(f"     [SKIP] Not enough events after cuts: {len(t_plot)}")
                        continue

                    # New X-only diagnostic page
                    draw_tx_corner_page(
                        pdf=pdf,
                        t_plot=t_plot,
                        x_plot=x_plot,
                        ch_code=ch_code,
                        run_id=run_id,
                        particle_type=args.pid,
                        tfinal_mode=args.tfinal_mode,
                        family_info=fam_info,
                        wc_range=args.wc_range,
                        bins_1d=args.bins_1d,
                        bins_2d=args.bins_2d,
                        t_range=(tmin, tmax),
                    )

                    # New X-cut scan study
                    draw_xcut_study(
                        pdf=pdf,
                        t_plot=t_plot,
                        x_plot=x_plot,
                        ch_code=ch_code,
                        run_id=run_id,
                        particle_type=args.pid,
                        tfinal_mode=args.tfinal_mode,
                        family_info=fam_info,
                        t_range=(tmin, tmax),
                        abs_tfinal=args.abs_tfinal,
                        xcuts=args.xcuts,
                        bins_1d=args.bins_1d,
                        min_events_per_cut=args.min_events_per_cut,
                    )

                    # Old full corner plot kept here if you still want it
                    if args.make_full_corner:
                        draw_corner_page(
                            pdf=pdf,
                            t_plot=t_plot,
                            x_plot=x_plot,
                            y_plot=y_plot,
                            ch_code=ch_code,
                            run_id=run_id,
                            particle_type=args.pid,
                            tfinal_mode=args.tfinal_mode,
                            family_info=fam_info,
                            wc_range=args.wc_range,
                            bins_1d=args.bins_1d,
                            bins_2d=args.bins_2d,
                            t_range=(tmin, tmax),
                            abs_tfinal=args.abs_tfinal,
                        )

            except Exception as e:
                print(f"     [ERR] Failed on {run_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"[CHANNEL] Done: {out_pdf}")

# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ana-files", nargs="*", help="Input ROOT files")
    ap.add_argument("--ana-glob", default=None, help="Glob for input ROOT files")
    ap.add_argument("--outdir", default="./WirechamberCorr", help="Output directory")

    ap.add_argument("--pid", default="electron", choices=["muon", "pion", "electron", "proton"],
                    help="Apply PID selection")

    ap.add_argument("--channels", nargs="+", default=TARGET_CHANNELS,
                    help="Channels to process, e.g. 100 104 105 110")

    ap.add_argument("--tfinal-mode", default="baseline",
                    choices=["branch", "baseline", "mcp6", "avg"],
                    help="How to build tfinal")

    ap.add_argument("--suffix", default="_LP2_50",
                    help="Suffix for raw timing branches used in baseline/mcp6/avg modes")

    ap.add_argument("--abs-tfinal", action="store_true",
                    help="Take abs(tfinal) before plotting")
    
    ap.add_argument("--apply-adc-cut", action="store_true",
                    help="Apply ADC mask using waveform amplitude cut")

    ap.add_argument("--tmin", type=float, default=None, help="Timing min for plotting")
    ap.add_argument("--tmax", type=float, default=None, help="Timing max for plotting")
    ap.add_argument("--wc-range", type=float, default=250.0, help="Wire chamber range +/-")

    ap.add_argument("--bins-1d", type=int, default=80, help="Bins for diagonal hists")
    ap.add_argument("--bins-2d", type=int, default=80, help="Bins for 2D hists")
    ap.add_argument("--min-events", type=int, default=20, help="Minimum events after cuts")

    # Added only
    ap.add_argument("--xcuts", nargs="+", type=float,
                    default=[200, 150, 120, 100, 80,75,70, 60, 40, 20],
                    help="Scan cuts of the form |X| < cut")
    ap.add_argument("--min-events-per-cut", type=int, default=30,
                    help="Minimum events required for each X-cut point")
    ap.add_argument("--make-full-corner", action="store_true",
                    help="Also keep the old full t/X/Y corner page")

    args = ap.parse_args()

    files = []
    if args.ana_files:
        files.extend(args.ana_files)
    if args.ana_glob:
        files.extend(glob.glob(args.ana_glob))

    files = _sort_files(list(set(files)))

    if not files:
        raise SystemExit("No input files found.")

    print(f"Found {len(files)} files.")
    print(f"Output dir: {args.outdir}")
    print(f"Channels: {args.channels}")
    print(f"tfinal mode: {args.tfinal_mode}")
    print(f"PID: {args.pid}")
    print(f"X cuts: {args.xcuts}")

    for ch in args.channels:
        process_channel(ch, files, args)

    print("\nAll done.")

if __name__ == "__main__":
    main()