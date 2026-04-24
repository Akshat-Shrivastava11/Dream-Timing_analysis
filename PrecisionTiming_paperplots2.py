#!/usr/bin/env python3
"""
Paper-level CaloX timing z-scan plotting script.

Outputs are written to TiminingZscan_summary_forpaper by default.

What this script makes:
  1. A CSV table with Gaussian fit results for every family/channel/location.
  2. A multipage PDF with one paper-style histogram + Gaussian overlay per channel.
     The legend uses z-location, not run number.
  3. Individual PNG/PDF versions of each channel overlay.
  4. A special run1501 overlay for channels 107, 100, and 104.
  5. A clean z-scan schematic diagram.

Example:
python make_timing_zscan_forpaper.py \
  --ana-files /path/to/run1501_*.root /path/to/run1507_*.root /path/to/run1511_*.root /path/to/run1513_*.root \
  --outdir TiminingZscan_summary_forpaper \
  --pid electron
"""

import os
import re
import csv
import glob
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import uproot
import awkward as ak
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyArrowPatch
import mplhep as hep

plt.style.use(hep.style.CMS)

# ============================================================
# Grids and channel lists
# ============================================================
QUARTZ_GRID = [
    [None,  "002", None,  None],
    ["006", "004", "206", "204"],
    ["016", "014", "216", "214"],
    ["026", "024", "226", "224"],
    [None,  "030", None,  None],
    [None,  "034", None,  None],
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

SCI_ALL_GRID = [
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

SCI_SELECTED_CHANNELS = [
    "103", "101", "303", "301",
    "107", "105", "307", "305",
    "113", "111", "313", "311",
]


def extract_channels(grid):
    return [ch for row in grid for ch in row if ch is not None]


FAMILY_DISPLAY_NAMES = {
    "Plastic": "Toray PJR-FB750 (Plastic)",
    "Quartz":  "FSHA (Fused-silica)",
    "SCI":     "SCSF-81J (Scintillator)",
}

FAMILY_COLORS = {
    "Plastic": "#d62728",
    "Quartz":  "#1f77b4",
    "SCI":     "#2ca02c",
}

# These are your hard timing windows.
FAMILY_WINDOWS = {
    "Plastic": (-14.5, -11.5),
    "Quartz":  (-15.0, -11.5),
    "SCI":     (-13.5,  -9.5),
}


def build_families(sci_mode: str = "all"):
    sci_channels = extract_channels(SCI_ALL_GRID) if sci_mode == "all" else SCI_SELECTED_CHANNELS
    return {
        "Plastic": {
            "channels": extract_channels(PLASTIC_GRID),
            "tmin": FAMILY_WINDOWS["Plastic"][0],
            "tmax": FAMILY_WINDOWS["Plastic"][1],
            "color": FAMILY_COLORS["Plastic"],
        },
        "Quartz": {
            "channels": extract_channels(QUARTZ_GRID),
            "tmin": FAMILY_WINDOWS["Quartz"][0],
            "tmax": FAMILY_WINDOWS["Quartz"][1],
            "color": FAMILY_COLORS["Quartz"],
        },
        "SCI": {
            "channels": sci_channels,
            "tmin": FAMILY_WINDOWS["SCI"][0],
            "tmax": FAMILY_WINDOWS["SCI"][1],
            "color": FAMILY_COLORS["SCI"],
        },
    }

# ============================================================
# Configuration
# ============================================================
TREE_NAME = "EventTree"
AMP_THRESHOLD = 100.0
MIN_ADC_CUT = -100.0
WC_X_CUT = 100.0

WC_CHANNELS = {
    "L1": "DRS_Board7_Group0_Channel0",
    "R1": "DRS_Board7_Group0_Channel1",
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
# Data containers
# ============================================================
@dataclass
class FitRecord:
    file_path: str
    run_label: str
    z_mm: float
    family: str
    channel: str
    n: int
    mu: float
    sigma: float
    fwhm: float
    time_err: float
    xlim: Tuple[float, float]
    centers: np.ndarray
    hist_norm: np.ndarray
    x_smooth: np.ndarray
    y_gauss: np.ndarray

    @property
    def z_cm(self):
        return self.z_mm / 10.0

    @property
    def location_label(self):
        if self.z_mm == -999.0 or not np.isfinite(self.z_mm):
            return "z unknown"
        return f"z = {self.z_cm:+.1f} cm"


# ============================================================
# Style helpers
# ============================================================
def apply_paper_style():
    plt.rcParams.update({
        "figure.figsize": (14, 10),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 18,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
        "lines.linewidth": 3.0,
        "axes.linewidth": 1.3,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })


def particle_display_name(particle_type: Optional[str]) -> str:
    if particle_type is None:
        return "All particles"
    if particle_type.lower() == "electron":
        return "Positron"
    return particle_type.capitalize()


def suffix_display_name(suffix: str) -> str:
    if "LP2_50" in suffix:
        return r"$LP2_{50}$"
    if "LP2" in suffix:
        return r"$LP2$"
    return suffix.strip("_")


def setup_paper_axes(ax, xlabel, ylabel, particle_type, suffix, llabel="Timing z-scan"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", length=8, width=1.2, direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor", length=4, width=1.0, direction="in", top=True, right=True)
    rlabel = f"40 GeV {particle_display_name(particle_type)} | {suffix_display_name(suffix)}"
    hep.cms.label(ax=ax, exp="CaloX", data=False, llabel=llabel, rlabel=rlabel, fontsize=20)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")

# ============================================================
# Selection and timing helpers
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
    if particle_type is None:
        return None

    requirements = get_particle_selection(particle_type)
    if not requirements:
        return None

    n_entries = tree.num_entries
    final_mask = np.ones(n_entries, dtype=bool)
    available_keys = set(tree.keys())

    for det, must_fire in requirements.items():
        branch_name = PID_BRANCH_MAP.get(det)
        if not branch_name or branch_name not in available_keys:
            print(f"    [WARN] PID branch missing for {det}: {branch_name}. Skipping this PID requirement.")
            continue

        ts_min, ts_max, val_cut, method = get_service_drs_cut(det)
        try:
            waveforms = tree[branch_name].array(library="ak")
            if method != "Sum":
                continue

            baseline = ak.mean(waveforms[:, :30], axis=1)
            waveforms_blsub = waveforms - baseline
            window_sum = ak.sum(waveforms_blsub[:, int(ts_min):int(ts_max)], axis=1)
            is_fired = ak.to_numpy(window_sum) < val_cut
            final_mask = final_mask & is_fired if must_fire else final_mask & (~is_fired)
        except Exception as e:
            print(f"    [WARN] PID cut failed for {det}: {e}")
            continue

    return final_mask


def compute_adc_mask(tree, code_str):
    b, g, c = int(code_str[0]), int(code_str[1]), int(code_str[2])
    drs_br = f"DRS_Board{b}_Group{g}_Channel{c}"
    if drs_br not in tree.keys():
        return np.ones(tree.num_entries, dtype=bool)

    waves = tree[drs_br].array(library="ak")
    baseline = ak.mean(waves[:, :30], axis=1)
    waves_blsub = waves - baseline
    peak = ak.max(waves_blsub, axis=1)
    min_adc = ak.min(waves_blsub, axis=1)
    mask = (peak >= AMP_THRESHOLD) & (min_adc >= MIN_ADC_CUT)
    return ak.to_numpy(mask)


def get_hit_times_vectorized(events):
    if events.ndim != 2:
        return np.zeros(len(events))
    baselines = np.mean(events[:, :20], axis=1, keepdims=True)
    corrected = events - baselines
    return np.argmin(corrected, axis=1)


def compute_wc_mask(tree, limit=WC_X_CUT):
    br_l1 = WC_CHANNELS["L1"]
    br_r1 = WC_CHANNELS["R1"]

    if br_l1 not in tree.keys() or br_r1 not in tree.keys():
        print("    [WARN] Wirechamber waveform branches missing. Skipping WC cut.")
        return np.ones(tree.num_entries, dtype=bool)

    L1 = ak.to_numpy(tree[br_l1].array(library="ak"))
    R1 = ak.to_numpy(tree[br_r1].array(library="ak"))
    L1_t = get_hit_times_vectorized(L1)
    R1_t = get_hit_times_vectorized(R1)
    x_positions = L1_t - R1_t
    return np.abs(x_positions) < limit


def get_z_position(run_label):
    # Relative z map used in the timing z-scan.
    if "run1513" in run_label:
        if "192918" in run_label:
            return 163.5
        if "194230" in run_label:
            return -182.3

    match = re.search(r"run(\d+)", run_label)
    run_num = int(match.group(1)) if match else None
    z_map = {
        1501:  50.0,
        1507:   0.0,
        1511: -50.0,
    }
    return z_map.get(run_num, -999.0)


def parse_channel_code(code_str):
    return int(code_str[0]), int(code_str[1]), int(code_str[2])


def run_label_from_path(path: str) -> str:
    m = re.search(r"(run\d+_\d{11,12})", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


def get_tfinal_3mm(tree, b, g, c, suffix):
    """
    t_final(b,g,c) = [t(b,g,c) - t(b,g,8)] - [t(0,3,7) - t(0,3,8)]
    """
    br_sig     = f"DRS_Board{b}_Group{g}_Channel{c}{suffix}"
    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8{suffix}"
    br_trg     = f"DRS_Board0_Group3_Channel7{suffix}"
    br_trg_ref = f"DRS_Board0_Group3_Channel8{suffix}"

    keys = set(tree.keys())
    if any(br not in keys for br in [br_sig, br_sig_ref, br_trg, br_trg_ref]):
        return None

    arr_sig     = tree[br_sig].array(library="np")
    arr_sig_ref = tree[br_sig_ref].array(library="np")
    arr_trg     = tree[br_trg].array(library="np")
    arr_trg_ref = tree[br_trg_ref].array(library="np")
    return (arr_sig - arr_sig_ref) - (arr_trg - arr_trg_ref)

# ============================================================
# Fit helpers
# ============================================================
def gaussian_peak_1(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def mode_from_hist(arr, bins):
    h, _ = np.histogram(arr, bins=bins)
    if h.sum() == 0:
        return np.nan, 0, h
    centers = 0.5 * (bins[1:] + bins[:-1])
    idx = int(np.argmax(h))
    return float(centers[idx]), int(h[idx]), h


def fit_timing_distribution(arr_time, xlim, nbins):
    bins = np.linspace(xlim[0], xlim[1], nbins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    mode, _, h = mode_from_hist(arr_time, bins)
    if h.max() <= 0:
        return None

    h_norm = h / h.max()
    arr_std = float(np.std(arr_time))
    if not np.isfinite(arr_std) or arr_std <= 0:
        arr_std = 0.2

    try:
        p0 = [mode, arr_std]
        bounds = ([xlim[0] - 2.0, 0.001], [xlim[1] + 2.0, 10.0])
        popt, _ = curve_fit(gaussian_peak_1, centers, h_norm, p0=p0, bounds=bounds, maxfev=20000)
        mu = float(popt[0])
        sigma = abs(float(popt[1]))
    except Exception:
        mu = float(mode)
        sigma = arr_std

    x_smooth = np.linspace(xlim[0], xlim[1], 600)
    y_gauss = gaussian_peak_1(x_smooth, mu, sigma)
    fwhm = 2.355 * sigma
    time_err = sigma / np.sqrt(len(arr_time))

    return {
        "centers": centers,
        "hist_norm": h_norm,
        "mu": mu,
        "sigma": sigma,
        "fwhm": fwhm,
        "time_err": time_err,
        "x_smooth": x_smooth,
        "y_gauss": y_gauss,
    }


def sorted_files(paths):
    def key(p):
        b = os.path.basename(p)
        mrun = re.search(r"run(\d+)", b)
        r = int(mrun.group(1)) if mrun else 10**9
        mts = re.search(r"_(\d{11,12})(?:_|\.|$)", b)
        ts = int(mts.group(1)) if mts else 10**18
        return r, ts, b
    return sorted(paths, key=key)

# ============================================================
# Collect all Gaussian fits
# ============================================================
def collect_fit_records(files, tree_name, particle_type, suffix, families, nbins, min_events, use_wc_cut=True):
    records: List[FitRecord] = []

    for fidx, fpath in enumerate(files, start=1):
        run_label = run_label_from_path(fpath)
        z_mm = get_z_position(run_label)
        print(f"\n[FILE {fidx}/{len(files)}] {os.path.basename(fpath)}  ->  {run_label}, z={z_mm:.1f} mm")

        try:
            uf = uproot.open(fpath)
            tree = uf[tree_name]
        except Exception as e:
            print(f"  [ERROR] Could not open file/tree: {e}")
            continue

        try:
            pid_mask = compute_pid_mask(tree, particle_type) if particle_type else None
            wc_mask = compute_wc_mask(tree) if use_wc_cut else np.ones(tree.num_entries, dtype=bool)
        except Exception as e:
            print(f"  [WARN] Problem building global masks: {e}")
            pid_mask = None
            wc_mask = np.ones(tree.num_entries, dtype=bool)

        for family, cfg in families.items():
            xlim = (cfg["tmin"], cfg["tmax"])
            for code_str in cfg["channels"]:
                b, g, c = parse_channel_code(code_str)

                try:
                    arr_raw = get_tfinal_3mm(tree, b, g, c, suffix)
                    if arr_raw is None:
                        continue

                    adc_mask = compute_adc_mask(tree, code_str)
                    combined_mask = adc_mask & wc_mask
                    if pid_mask is not None:
                        combined_mask = combined_mask & pid_mask

                    if len(arr_raw) != len(combined_mask):
                        print(f"  [SKIP] {family} ch {code_str}: timing/mask length mismatch")
                        continue

                    arr_time = arr_raw[combined_mask]
                    arr_time = arr_time[np.isfinite(arr_time)]
                    arr_time = arr_time[(arr_time >= xlim[0]) & (arr_time <= xlim[1])]
                    n_final = int(len(arr_time))
                    if n_final < min_events:
                        continue

                    fit = fit_timing_distribution(arr_time, xlim=xlim, nbins=nbins)
                    if fit is None:
                        continue

                    # Keep the same protection you had for pathological scintillator fits.
                    if family == "SCI" and fit["sigma"] < 0.050:
                        print(f"  [SKIP] {family} ch {code_str}: sigma too low ({fit['sigma']:.4f} ns)")
                        continue

                    records.append(FitRecord(
                        file_path=fpath,
                        run_label=run_label,
                        z_mm=z_mm,
                        family=family,
                        channel=code_str,
                        n=n_final,
                        mu=fit["mu"],
                        sigma=fit["sigma"],
                        fwhm=fit["fwhm"],
                        time_err=fit["time_err"],
                        xlim=xlim,
                        centers=fit["centers"],
                        hist_norm=fit["hist_norm"],
                        x_smooth=fit["x_smooth"],
                        y_gauss=fit["y_gauss"],
                    ))

                    print(
                        f"  [FIT] {family:7s} ch {code_str}: "
                        f"{FitRecord(fpath, run_label, z_mm, family, code_str, n_final, fit['mu'], fit['sigma'], fit['fwhm'], fit['time_err'], xlim, fit['centers'], fit['hist_norm'], fit['x_smooth'], fit['y_gauss']).location_label:>12s}, "
                        f"N={n_final:5d}, mu={fit['mu']:.3f}, sigma={fit['sigma']:.3f}"
                    )

                except Exception as e:
                    print(f"  [WARN] Failed {family} ch {code_str}: {e}")
                    continue

        try:
            uf.close()
        except Exception:
            pass

    return records


def write_fit_table(records: List[FitRecord], out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_label", "z_mm", "z_cm", "location_label", "family", "family_display", "channel",
            "mean_ns", "sigma_ns", "time_err_ns", "fwhm_ns", "n_events", "file_path",
        ])
        for r in sorted(records, key=lambda x: (x.family, int(x.channel), x.z_mm, x.run_label)):
            writer.writerow([
                r.run_label, f"{r.z_mm:.3f}", f"{r.z_cm:.3f}", r.location_label,
                r.family, FAMILY_DISPLAY_NAMES.get(r.family, r.family), r.channel,
                f"{r.mu:.6f}", f"{r.sigma:.6f}", f"{r.time_err:.6f}", f"{r.fwhm:.6f}", r.n, r.file_path,
            ])
    print(f"\n[TABLE] Saved: {out_csv}")

# ============================================================
# Plotting
# ============================================================
def records_by_family_channel(records):
    grouped: Dict[Tuple[str, str], List[FitRecord]] = {}
    for r in records:
        grouped.setdefault((r.family, r.channel), []).append(r)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda x: (x.z_mm, x.run_label))
    return grouped


def color_map_for_locations(records: List[FitRecord]):
    # Consistent location colors across all channel plots.
    unique_z = sorted({r.z_mm for r in records if r.z_mm != -999.0})
    fallback_z = sorted({r.z_mm for r in records if r.z_mm == -999.0})
    z_values = unique_z + fallback_z
    cmap = plt.get_cmap("tab10")
    return {z: cmap(i % 10) for i, z in enumerate(z_values)}


def plot_one_channel_overlay(ax, recs: List[FitRecord], particle_type, suffix, location_colors):
    first = recs[0]
    ax.set_xlim(*first.xlim)
    ax.set_ylim(0.0, 1.28)

    # Remove LP2_50 / LP50 from header
    setup_paper_axes(
        ax,
        "Time of Arrival [ns]",
        "Normalized Events",
        particle_type,
        suffix=None,
    )

    for r in recs:
        color = location_colors.get(r.z_mm, "black")
        label = rf"{r.location_label}: $\mu={r.mu:.2f}$ ns, $\sigma={r.sigma:.2f}$ ns, N={r.n}"
        ax.step(r.centers, r.hist_norm, where="mid", lw=2.0, alpha=0.35, color=color)
        ax.plot(r.x_smooth, r.y_gauss, lw=3.6, color=color, label=label)

    title = f"{FAMILY_DISPLAY_NAMES.get(first.family, first.family)} | Channel {first.channel}"
    ax.text(
        0.03, 0.90, title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=24,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="none",
            alpha=0.75,
        ),
    )

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=15,
        handlelength=2.8,
    )

    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.xaxis.label.set_size(24)
    ax.yaxis.label.set_size(24)

def make_all_channel_location_overlays(records: List[FitRecord], outdir: str, particle_type, suffix, save_individual=True):
    if not records:
        print("[PLOT] No records available for channel overlays.")
        return

    location_colors = color_map_for_locations(records)
    grouped = records_by_family_channel(records)

    pdf_path = os.path.join(outdir, "paper_all_channel_location_overlays.pdf")
    print(f"\n[PLOT] Writing all channel/location overlays to {pdf_path}")

    indiv_dir = os.path.join(outdir, "individual_channel_overlays")
    if save_individual:
        os.makedirs(indiv_dir, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        for (family, channel), recs in sorted(grouped.items(), key=lambda x: (x[0][0], int(x[0][1]))):
            fig, ax = plt.subplots(figsize=(14, 10))
            plot_one_channel_overlay(ax, recs, particle_type, suffix, location_colors)
            fig.tight_layout()
            pdf.savefig(fig)

            if save_individual:
                base = f"paper_overlay_{safe_name(family)}_ch{channel}"
                fig.savefig(os.path.join(indiv_dir, base + ".png"), bbox_inches="tight")
                fig.savefig(os.path.join(indiv_dir, base + ".pdf"), bbox_inches="tight")

            plt.close(fig)

    print(f"[PLOT] Saved: {pdf_path}")
    if save_individual:
        print(f"[PLOT] Individual PNG/PDF files saved in: {indiv_dir}")


def find_run_records(records: List[FitRecord], run_substring: str, requested):
    selected = []
    for fam, ch in requested:
        matches = [r for r in records if r.family == fam and r.channel == ch and run_substring in r.run_label]
        if not matches:
            print(f"  [WARN] Could not find {run_substring} record for {fam} channel {ch}")
            continue
        # If there are multiple run1501 files, take the one with the largest N.
        selected.append(sorted(matches, key=lambda r: r.n, reverse=True)[0])
    return selected


def make_run1501_anchor_overlay(records: List[FitRecord], outdir: str, particle_type, suffix, run_substring="run1501"):
    # Legend/order: Quartz, Plastic, SCI
    requested = [
        ("Quartz", "104"),
        ("Plastic", "100"),
        ("SCI", "107"),
    ]

    selected = find_run_records(records, run_substring, requested)
    if not selected:
        print(f"[PLOT] No {run_substring} anchor records found. Skipping Z-scan overlay.")
        return

    # Force selected order to match requested order
    order_map = {pair: i for i, pair in enumerate(requested)}
    selected = sorted(selected, key=lambda r: order_map.get((r.family, r.channel), 999))

    xmin = min(r.xlim[0] for r in selected)
    xmax = max(r.xlim[1] for r in selected)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, 1.28)

    # No LP50 in title/header
    setup_paper_axes(
        ax,
        "Time of Arrival [ns]",
        "Normalized Events",
        particle_type,
        suffix=None,
        llabel="Z-scan",
    )

    for r in selected:
        color = FAMILY_COLORS.get(r.family, "black")
        fam_label = FAMILY_DISPLAY_NAMES.get(r.family, r.family)
        label = (
            rf"{fam_label}, Ch {r.channel}: "
            rf"$\mu={r.mu:.2f}$ ns, $\sigma={r.sigma:.2f}$ ns, N={r.n}"
        )
        ax.step(r.centers, r.hist_norm, where="mid", lw=2.2, alpha=0.35, color=color)
        ax.plot(r.x_smooth, r.y_gauss, lw=3.8, color=color, label=label)

    locs = sorted({r.location_label for r in selected})
    loc_text = locs[0] if len(locs) == 1 else ", ".join(locs)

    ax.text(
        0.03, 0.90,
        f"Reference location: {loc_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=24,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="none",
            alpha=0.75,
        ),
    )

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=14,
        handlelength=2.8,
    )

    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.tick_params(axis="both", which="minor", labelsize=16)
    ax.xaxis.label.set_size(24)
    ax.yaxis.label.set_size(24)

    fig.tight_layout()

    out_png = os.path.join(outdir, f"paper_{run_substring}_channels_104_100_107_overlay.png")
    out_pdf = os.path.join(outdir, f"paper_{run_substring}_channels_104_100_107_overlay.pdf")
    fig.savefig(out_png, bbox_inches="tight", dpi=250)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[PLOT] Saved: {out_png}")
    print(f"[PLOT] Saved: {out_pdf}")


def make_clean_zscan_diagram(outdir: str):
    os.makedirs(outdir, exist_ok=True)

    out_png = os.path.join(outdir, "paper_zscan_diagram.png")
    out_pdf = os.path.join(outdir, "paper_zscan_diagram.pdf")

    positions_cm = [-18.2, -5.0, 0.0, 5.0, 16.4]
    pos_labels = [f"Pos. {i}" for i in range(1, 6)]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    # Light orange module
    module_x0, module_y0 = 0.18, 0.35
    module_w, module_h = 0.70, 0.20

    rect = plt.Rectangle(
        (module_x0, module_y0),
        module_w,
        module_h,
        transform=ax.transAxes,
        facecolor="#f9a01b",
        edgecolor="black",
        linewidth=2.2,
        alpha=0.88,
    )
    ax.add_patch(rect)

    # Position x coordinates
    xs = np.linspace(module_x0 + 0.08, module_x0 + module_w - 0.08, 5)

    # Beam label moved left so it does not overlap
    ax.text(
        module_x0 - 0.10,
        0.84,
        "40 GeV\npositron beam",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=18,
        color="#174a7c",
    )

    # Arrows and labels
    for x, label, z in zip(xs, pos_labels, positions_cm):
        ax.text(
            x,
            0.91,
            label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            color="#174a7c",
        )

        ax.annotate(
            "",
            xy=(x, module_y0 + module_h + 0.015),
            xytext=(x, 0.88),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="-|>",
                lw=3.0,
                color="#174a7c",
                mutation_scale=28,
            ),
        )

        ax.plot(
            [x, x],
            [module_y0 - 0.09, module_y0 + module_h],
            transform=ax.transAxes,
            linestyle="--",
            color="black",
            linewidth=2.2,
        )

        ax.text(
            x,
            0.20,
            f"{z:+.1f} cm" if z != 0 else "0.0 cm",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=15,
            color="black",
        )

    # HG-DREAM label
    ax.text(
        module_x0 + module_w / 2,
        module_y0 + module_h / 2,
        "HG-DREAM",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="black",
    )

    # Z axis
    ax.annotate(
        "",
        xy=(0.94, 0.23),
        xytext=(0.16, 0.23),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", lw=3.0, color="black", mutation_scale=28),
    )

    ax.text(
        0.96,
        0.23,
        "z",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="black",
    )

    ax.text(
        0.53,
        0.12,
        "Relative beam position",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=22,
        color="black",
    )

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[PLOT] Saved: {out_png}")
    print(f"[PLOT] Saved: {out_pdf}")
# ============================================================
# Main
# ============================================================
def resolve_files(args):
    if args.ana_files:
        files = list(args.ana_files)
    else:
        files = sorted(glob.glob(args.ana_glob))
    return sorted_files(files)


def main():
    parser = argparse.ArgumentParser(description="Make paper-level CaloX timing z-scan plots.")
    parser.add_argument("--ana-files", nargs="+", default=None, help="Explicit list of input ROOT files.")
    parser.add_argument("--ana-glob", default=None, help="Glob for input ROOT files.")
    parser.add_argument("--tree", default=TREE_NAME, help="Tree name in ROOT files.")
    parser.add_argument("--outdir", default="TiminingZscan_summary_forpaper2", help="Output directory.")
    parser.add_argument("--pid", default="electron", choices=["muon", "pion", "electron", "proton"], help="PID selection.")
    parser.add_argument("--suffix", default="_LP2_50", help="Timing suffix, e.g. _LP2_50.")
    parser.add_argument("--nbins", type=int, default=100, help="Histogram bins for paper overlays.")
    parser.add_argument("--min-events", type=int, default=25, help="Minimum entries required for a fit.")
    parser.add_argument("--sci-channels", choices=["all", "selected"], default="all", help="Use all SCI channels or the selected subset from your older script.")
    parser.add_argument("--no-wc-cut", action="store_true", help="Disable wirechamber cut.")
    parser.add_argument("--no-individual", action="store_true", help="Do not save individual PNG/PDF files for every channel.")
    parser.add_argument("--run-overlay", default="run1501", help="Run substring for the 107/100/104 anchor overlay.")

    args = parser.parse_args()

    if args.ana_files is None and args.ana_glob is None:
        raise SystemExit("[FATAL] Provide either --ana-files or --ana-glob")

    apply_paper_style()
    os.makedirs(args.outdir, exist_ok=True)

    files = resolve_files(args)
    if not files:
        raise SystemExit("[FATAL] No files matched your input.")

    print("\n[INIT] Paper-level CaloX timing z-scan plotting")
    print(f"[INIT] Number of input files: {len(files)}")
    print(f"[INIT] Output directory: {args.outdir}")
    print(f"[INIT] PID: {args.pid}")
    print(f"[INIT] Timing suffix: {args.suffix}")
    print(f"[INIT] SCI channel mode: {args.sci_channels}")
    print(f"[INIT] Wirechamber cut: {'OFF' if args.no_wc_cut else 'ON'}")

    families = build_families(args.sci_channels)

    records = collect_fit_records(
        files=files,
        tree_name=args.tree,
        particle_type=args.pid,
        suffix=args.suffix,
        families=families,
        nbins=args.nbins,
        min_events=args.min_events,
        use_wc_cut=(not args.no_wc_cut),
    )

    if not records:
        raise SystemExit("[FATAL] No successful fits were produced. Check input files, branch names, PID cuts, and windows.")

    table_path = os.path.join(args.outdir, "timing_gaussian_fit_summary.csv")
    write_fit_table(records, table_path)

    make_all_channel_location_overlays(
        records=records,
        outdir=args.outdir,
        particle_type=args.pid,
        suffix=args.suffix,
        save_individual=(not args.no_individual),
    )

    make_run1501_anchor_overlay(
        records=records,
        outdir=args.outdir,
        particle_type=args.pid,
        suffix=args.suffix,
        run_substring=args.run_overlay,
    )

    make_clean_zscan_diagram(args.outdir)

    print("\n[DONE] Paper-level timing z-scan outputs saved to:")
    print(f"       {args.outdir}")


if __name__ == "__main__":
    main()
