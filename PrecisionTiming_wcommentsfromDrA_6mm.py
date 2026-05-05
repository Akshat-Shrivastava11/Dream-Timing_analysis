#!/usr/bin/env python3
"""
Make 6 mm HG-DREAM / CaloX z-vs-time plots using all available 6 mm channels.

Key behavior:
  * Uses the 6 mm channel/run map with y1000, y1065, y936, y1028 channel configs.
  * Collapses all y positions into the same z-offset coordinate.
    Example: run1501/run1502/run1504/run1506 are all z = +5.0 cm.
  * Fits a Gaussian timing peak for every run/family/channel.
  * Saves a CSV table of all fits.
  * Makes:
      1. One combined family-level z vs mean TOA plot.
      2. One per-family z vs mean TOA plot using all channels.
      3. Optional per-channel z-vs-TOA pages when enough z points exist.

Example:
python3 make_6mm_z_vs_t_all_channels.py \
  --base-dir /lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples \
  --outdir TiminingZscan_summary_6mm_allY_collapsed \
  --pid electron \
  --suffix _LP2_50

Or use explicit files:
python3 make_6mm_z_vs_t_all_channels.py \
  --ana-files /path/to/run1501_*.root /path/to/run1502_*.root ... \
  --outdir TiminingZscan_summary_6mm_allY_collapsed \
  --pid electron
"""

import os
import re
import csv
import glob
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import uproot
import awkward as ak
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator
import mplhep as hep

plt.style.use(hep.style.CMS)

# ============================================================
# Defaults
# ============================================================
TREE_NAME = "EventTree"
BASE_DIR = "/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples"
NBINS = 160
MIN_EVENTS = 500
AMP_THRESHOLD = 100.0
MIN_ADC_CUT = -100.0
WC_X_CUT = 100.0

# ============================================================
# 6 mm channel configs by y position
# ============================================================
Y_CONFIGS = {
    "y1000": {
        "SCI":         {"channels": ["620", "621"],                 "tmin":  8.0, "tmax": 11.0},
        "Plastic":     {"channels": ["612", "611", "610", "613"], "tmin": 10.5, "tmax": 12.5},
        "Quartz":      {"channels": ["631", "630", "627", "637"], "tmin": 10.0, "tmax": 13.5},
    },
    "y1065": {
        "Quartz":      {"channels": ["523", "522", "521", "520"], "tmin": 10.0, "tmax": 13.5},
    },
    "y936": {
        "SCI":         {"channels": ["604", "605"],                 "tmin":  8.0, "tmax": 11.0},
        "Plastic":     {"channels": ["607", "606"],                 "tmin": 11.0, "tmax": 12.5},
        "Quartz":      {"channels": ["617", "616", "615", "614"], "tmin": 11.0, "tmax": 12.6},
    },
    "y1028": {
        "SCI":         {"channels": ["421", "420"],                 "tmin":  7.0, "tmax": 10.5},
        "Plastic":     {"channels": ["425", "423", "422", "424"], "tmin": 10.5, "tmax": 12.5},
        "Quartz":      {"channels": ["413", "412", "411", "410"], "tmin": 11.0, "tmax": 12.5},
    },
}

# Explicit mapping from run label to y group.
RUN_TO_YGROUP = {
    # y = 1000 mm
    "run1502_250928113749": "y1000",
    "run1508_250928161049": "y1000",
    "run1512_250928183645": "y1000",

    # y = 1065 mm
    "run1501_250928105227": "y1065",
    "run1507_250928160030": "y1065",
    "run1511_250928180741": "y1065",
    "run1513_250928192918": "y1065",
    "run1513_250928194230": "y1065",

    # y = 936 mm
    "run1504_250928133854": "y936",
    "run1509_250928164817": "y936",
    "run1512_250928185722": "y936",

    # y = 1028 mm
    "run1506_250928143030": "y1028",
    "run1506_250928145724": "y1028",
    "run1510_250928172949": "y1028",
}

# Collapse all y scans into the same relative z coordinate.
# These are the shifted coordinates used in the 3 mm paper-style script:
# run1501/1502/1504/1506 -> +50 mm, run1507/1508/1509/1510 -> 0 mm,
# run1511/1512 -> -50 mm, run1513 special positions -> +163.5 and -182.3 mm.
def get_z_position_collapsed(run_label: str) -> float:
    if "run1513" in run_label:
        if "192918" in run_label:
            return 163.5
        if "194230" in run_label:
            return -182.3

    m = re.search(r"run(\d+)", run_label)
    run_num = int(m.group(1)) if m else None

    z_map_mm = {
        # +5 cm plane
        1501:  50.0,
        1502:  50.0,
        1504:  50.0,
        1506:  50.0,

        # 0 cm plane
        1507:   0.0,
        1508:   0.0,
        1509:   0.0,
        1510:   0.0,

        # -5 cm plane
        1511: -50.0,
        1512: -50.0,
    }
    return z_map_mm.get(run_num, -999.0)

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

FAMILY_ORDER = ["SCI", "Plastic", "Quartz"]

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

WC_CHANNELS = {
    "L1": "DRS_Board7_Group0_Channel0",
    "R1": "DRS_Board7_Group0_Channel1",
}

# ============================================================
# Data model
# ============================================================
@dataclass
class FitRecord:
    file_path: str
    run_label: str
    y_group: str
    z_mm: float
    family: str
    channel: str
    n: int
    mu: float
    sigma: float
    time_err: float
    fwhm: float
    tmin: float
    tmax: float
    centers: np.ndarray
    hist_norm: np.ndarray
    x_smooth: np.ndarray
    y_gauss: np.ndarray

    @property
    def z_cm(self) -> float:
        return self.z_mm / 10.0

# ============================================================
# Style helpers
# ============================================================
def apply_paper_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 22,
        "axes.labelsize": 28,
        "axes.titlesize": 24,
        "xtick.labelsize": 23,
        "ytick.labelsize": 23,
        "legend.fontsize": 18,
        "axes.linewidth": 1.6,
        "xtick.major.size": 10,
        "ytick.major.size": 10,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })


def particle_display_name(particle_type: Optional[str]) -> str:
    if particle_type is None:
        return "All particles"
    if particle_type.lower() == "electron":
        return r"$e^{+}$"
    return particle_type.capitalize()


def setup_axes(ax, xlabel: str, ylabel: str, particle_type: Optional[str], llabel="Z-Scan"):
    ax.set_xlabel(xlabel, loc="right")
    ax.set_ylabel(ylabel, loc="top")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=10)
    ax.tick_params(which="minor", length=5)
    ax.grid(False)

    hep.cms.label(
        ax=ax,
        exp="CaloX",
        data=False,
        llabel=llabel,
        rlabel=f"40 GeV {particle_display_name(particle_type)}",
        fontsize=24,
    )

# ============================================================
# Selection helpers
# ============================================================
def get_service_drs_cut(service_drs: str) -> tuple:
    cuts = {
        "HoleVeto": (100, 350, -2e3, "Sum"),
        "PSD": (100, 400, -3500.0, "Sum"),
        "TTUMuonVeto": (200, 400, -2e3, "Sum"),
        "Cer474": (800, 900, -2000.0, "Sum"),
        "Cer519": (450, 550, -1000.0, "Sum"),
        "Cer537": (400, 500, -500.0, "Sum"),
    }
    return cuts.get(service_drs, (0, 1000, -5e4, "Sum"))


def get_particle_selection(particle_type: str) -> dict:
    selections = {
        "muon": {"TTUMuonVeto": True, "PSD": False},
        "pion": {"TTUMuonVeto": False, "PSD": False, "Cer474": True, "Cer519": True, "Cer537": True},
        "electron": {"TTUMuonVeto": False, "PSD": True, "Cer474": True, "Cer519": True, "Cer537": True},
        "proton": {"TTUMuonVeto": False, "PSD": False, "Cer474": False, "Cer519": False, "Cer537": False},
    }
    return selections.get(particle_type.lower(), {})


def compute_pid_mask(tree, particle_type: Optional[str]):
    if particle_type is None:
        return None
    requirements = get_particle_selection(particle_type)
    if not requirements:
        return None

    final_mask = np.ones(tree.num_entries, dtype=bool)
    keys = set(tree.keys())

    for det, must_fire in requirements.items():
        branch_name = PID_BRANCH_MAP.get(det)
        if not branch_name or branch_name not in keys:
            print(f"    [WARN] PID branch missing for {det}: {branch_name}. Skipping that PID term.")
            continue

        ts_min, ts_max, val_cut, method = get_service_drs_cut(det)
        if method != "Sum":
            continue

        try:
            waves = tree[branch_name].array(library="ak")
            baseline = ak.mean(waves[:, :30], axis=1)
            waves_blsub = waves - baseline
            window_sum = ak.sum(waves_blsub[:, int(ts_min):int(ts_max)], axis=1)
            is_fired = ak.to_numpy(window_sum) < val_cut
            final_mask = final_mask & is_fired if must_fire else final_mask & (~is_fired)
        except Exception as e:
            print(f"    [WARN] PID cut failed for {det}: {e}")

    return final_mask


def compute_adc_mask(tree, code_str: str):
    b, g, c = int(code_str[0]), int(code_str[1]), int(code_str[2])
    drs_br = f"DRS_Board{b}_Group{g}_Channel{c}"
    if drs_br not in tree.keys():
        return np.ones(tree.num_entries, dtype=bool)

    waves = tree[drs_br].array(library="ak")
    baseline = ak.mean(waves[:, :30], axis=1)
    waves_blsub = waves - baseline
    peak = ak.max(waves_blsub, axis=1)
    min_adc = ak.min(waves_blsub, axis=1)
    return ak.to_numpy((peak >= AMP_THRESHOLD) & (min_adc >= MIN_ADC_CUT))


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
        print("    [WARN] Wirechamber branches missing. Skipping WC cut.")
        return np.ones(tree.num_entries, dtype=bool)

    L1 = ak.to_numpy(tree[br_l1].array(library="ak"))
    R1 = ak.to_numpy(tree[br_r1].array(library="ak"))
    x_positions = get_hit_times_vectorized(L1) - get_hit_times_vectorized(R1)
    return np.abs(x_positions) < limit

# ============================================================
# Timing and fitting
# ============================================================
def parse_code(code_str: str) -> Tuple[int, int, int]:
    return int(code_str[0]), int(code_str[1]), int(code_str[2])


def compute_tfinal_6mm(tree, b: int, g: int, c: int, suffix="_LP2_50"):
    """
    Same timing definition as your current scripts:
      t_final = [t(b,g,c) - t(b,g,8)] - [t(0,3,7) - t(0,3,8)]

    NOTE: no abs() here. If your old 6 mm plot looked positive only because of abs(),
    use --abs-time to reproduce that behavior.
    """
    br_sig     = f"DRS_Board{b}_Group{g}_Channel{c}{suffix}"
    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8{suffix}"
    br_trg     = f"DRS_Board0_Group3_Channel7{suffix}"
    br_trg_ref = f"DRS_Board0_Group3_Channel8{suffix}"

    keys = set(tree.keys())
    missing = [br for br in [br_sig, br_sig_ref, br_trg, br_trg_ref] if br not in keys]
    if missing:
        return None

    arr_sig     = tree[br_sig].array(library="np")
    arr_sig_ref = tree[br_sig_ref].array(library="np")
    arr_trg     = tree[br_trg].array(library="np")
    arr_trg_ref = tree[br_trg_ref].array(library="np")

    if not (arr_sig.shape == arr_sig_ref.shape == arr_trg.shape == arr_trg_ref.shape):
        return None

    return (arr_sig - arr_sig_ref) - (arr_trg - arr_trg_ref)


def gaussian_peak(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def mode_from_hist(arr, bins):
    h, _ = np.histogram(arr, bins=bins)
    if h.sum() == 0 or h.max() == 0:
        return np.nan, h
    centers = 0.5 * (bins[1:] + bins[:-1])
    return float(centers[int(np.argmax(h))]), h


def fit_time_peak(arr_time, tmin, tmax, nbins):
    bins = np.linspace(tmin, tmax, nbins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    mode, h = mode_from_hist(arr_time, bins)
    if not np.isfinite(mode) or h.max() == 0:
        return None

    h_norm = h / h.max()
    arr_std = float(np.std(arr_time))
    if not np.isfinite(arr_std) or arr_std <= 0:
        arr_std = 0.25

    try:
        popt, _ = curve_fit(
            gaussian_peak,
            centers,
            h_norm,
            p0=[mode, arr_std],
            bounds=([tmin - 2.0, 0.001], [tmax + 2.0, 10.0]),
            maxfev=20000,
        )
        mu = float(popt[0])
        sigma = abs(float(popt[1]))
    except Exception:
        mu = float(mode)
        sigma = arr_std

    x_smooth = np.linspace(tmin, tmax, 600)
    y_gauss = gaussian_peak(x_smooth, mu, sigma)

    return {
        "mu": mu,
        "sigma": sigma,
        "time_err": sigma / np.sqrt(len(arr_time)),
        "fwhm": 2.355 * sigma,
        "centers": centers,
        "hist_norm": h_norm,
        "x_smooth": x_smooth,
        "y_gauss": y_gauss,
    }

# ============================================================
# File/run helpers
# ============================================================
def run_label_from_path(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"(run\d+_\d{11,12})", base)
    return m.group(1) if m else os.path.splitext(base)[0]


def y_group_for_run(run_label: str) -> Optional[str]:
    for key, group in RUN_TO_YGROUP.items():
        if key in run_label:
            return group
    return None


def sorted_files(files: List[str]) -> List[str]:
    def key(p):
        base = os.path.basename(p)
        mrun = re.search(r"run(\d+)", base)
        r = int(mrun.group(1)) if mrun else 10**9
        mts = re.search(r"_(\d{11,12})(?:_|\.|$)", base)
        ts = int(mts.group(1)) if mts else 10**18
        return r, ts, base
    return sorted(files, key=key)


def resolve_files(args) -> List[str]:
    if args.ana_files:
        return sorted_files(list(args.ana_files))
    if args.ana_glob:
        return sorted_files(glob.glob(args.ana_glob))

    files = []
    for run_label in RUN_TO_YGROUP:
        fpath = os.path.join(args.base_dir, f"{run_label}_converted_timingskim.root")
        if os.path.exists(fpath):
            files.append(fpath)
        else:
            print(f"[WARN] Mapped file not found: {fpath}")
    return sorted_files(files)

# ============================================================
# Record collection
# ============================================================
def collect_records(files, args) -> List[FitRecord]:
    records: List[FitRecord] = []

    for i, fpath in enumerate(files, start=1):
        run_label = run_label_from_path(fpath)
        y_group = y_group_for_run(run_label)
        if y_group is None:
            print(f"[SKIP] {os.path.basename(fpath)}: run not in RUN_TO_YGROUP")
            continue

        z_mm = get_z_position_collapsed(run_label)
        y_cfg = Y_CONFIGS[y_group]

        print(f"\n[FILE {i}/{len(files)}] {os.path.basename(fpath)}")
        print(f"  run={run_label}, y_group={y_group}, collapsed z={z_mm:+.1f} mm")

        try:
            uf = uproot.open(fpath)
            tree = uf[args.tree]
        except Exception as e:
            print(f"  [ERROR] Could not open file/tree: {e}")
            continue

        try:
            pid_mask = None if args.no_pid_cut else (compute_pid_mask(tree, args.pid) if args.pid else None)
            adc_cache = {}
            wc_mask = compute_wc_mask(tree) if args.use_wc_cut else np.ones(tree.num_entries, dtype=bool)
        except Exception as e:
            print(f"  [WARN] Failed global masks: {e}")
            pid_mask = None
            adc_cache = {}
            wc_mask = np.ones(tree.num_entries, dtype=bool)

        for family in FAMILY_ORDER:
            if family not in y_cfg:
                continue
            fam_cfg = y_cfg[family]
            tmin, tmax = fam_cfg["tmin"], fam_cfg["tmax"]

            for ch_str in fam_cfg["channels"]:
                b, g, c = parse_code(ch_str)
                try:
                    arr_raw = compute_tfinal_6mm(tree, b, g, c, suffix=args.suffix)
                    if arr_raw is None:
                        print(f"    [MISS] {family:7s} ch {ch_str}: missing timing branches")
                        continue

                    if not args.signed_time:
                        arr_raw = np.abs(arr_raw)

                    if args.use_adc_cut:
                        if ch_str not in adc_cache:
                            adc_cache[ch_str] = compute_adc_mask(tree, ch_str)
                        adc_mask = adc_cache[ch_str]
                    else:
                        adc_mask = np.ones(tree.num_entries, dtype=bool)

                    combined_mask = adc_mask & wc_mask
                    if pid_mask is not None:
                        combined_mask = combined_mask & pid_mask

                    if len(arr_raw) != len(combined_mask):
                        print(f"    [SKIP] {family:7s} ch {ch_str}: array/mask length mismatch")
                        continue

                    arr = arr_raw[combined_mask]
                    arr = arr[np.isfinite(arr)]
                    arr = arr[(arr >= tmin) & (arr <= tmax)]
                    n = int(len(arr))

                    if n < args.min_events:
                        print(f"    [SKIP] {family:7s} ch {ch_str}: N={n} < {args.min_events}")
                        continue

                    fit = fit_time_peak(arr, tmin, tmax, args.nbins)
                    if fit is None:
                        continue

                    mu = fit["mu"]
                    sigma = fit["sigma"]
                    time_err = fit["time_err"]
                    fwhm = fit["fwhm"]

                    # Same idea as the paper script's SCI protection, but configurable
                    # and applied to all families. This removes pathological Gaussian
                    # fits that are too narrow (or optionally too broad).
                    if sigma < args.min_sigma_cut:
                        print(
                            f"    [SKIP] {family:7s} ch {ch_str}: "
                            f"sigma too low ({sigma:.4f} < {args.min_sigma_cut:.4f} ns)"
                        )
                        continue
                    if args.max_sigma_cut is not None and sigma > args.max_sigma_cut:
                        print(
                            f"    [SKIP] {family:7s} ch {ch_str}: "
                            f"sigma too high ({sigma:.4f} > {args.max_sigma_cut:.4f} ns)"
                        )
                        continue

                    records.append(FitRecord(
                        file_path=fpath,
                        run_label=run_label,
                        y_group=y_group,
                        z_mm=z_mm,
                        family=family,
                        channel=ch_str,
                        n=n,
                        mu=mu,
                        sigma=sigma,
                        time_err=time_err,
                        fwhm=fwhm,
                        tmin=tmin,
                        tmax=tmax,
                        centers=fit["centers"],
                        hist_norm=fit["hist_norm"],
                        x_smooth=fit["x_smooth"],
                        y_gauss=fit["y_gauss"],
                    ))

                    print(
                        f"    [FIT] {family:7s} ch {ch_str}: "
                        f"z={z_mm/10.0:+.1f} cm, N={n:5d}, mu={mu:.3f}, sigma={sigma:.3f}"
                    )

                except Exception as e:
                    print(f"    [WARN] {family:7s} ch {ch_str} failed: {e}")

        try:
            uf.close()
        except Exception:
            pass

    return records

# ============================================================
# Tables
# ============================================================
def write_csv(records: List[FitRecord], out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "run_label", "y_group", "z_mm", "z_cm", "family", "family_display", "channel",
            "mean_ns", "sigma_ns", "time_err_ns", "fwhm_ns", "n_events", "tmin", "tmax", "file_path"
        ])
        for r in sorted(records, key=lambda x: (x.family, x.channel, x.z_mm, x.run_label)):
            w.writerow([
                r.run_label, r.y_group, f"{r.z_mm:.3f}", f"{r.z_cm:.3f}",
                r.family, FAMILY_DISPLAY_NAMES.get(r.family, r.family), r.channel,
                f"{r.mu:.6f}", f"{r.sigma:.6f}", f"{r.time_err:.6f}", f"{r.fwhm:.6f}",
                r.n, f"{r.tmin:.3f}", f"{r.tmax:.3f}", r.file_path,
            ])
    print(f"\n[TABLE] Saved: {out_csv}")

# ============================================================
# Fit/plot helpers
# ============================================================
def weighted_line_fit(z_cm, mu, yerr):
    z_cm = np.asarray(z_cm, dtype=float)
    mu = np.asarray(mu, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    yerr = np.where((np.isfinite(yerr)) & (yerr > 0), yerr, np.nanmedian(yerr[yerr > 0]) if np.any(yerr > 0) else 1.0)
    weights = 1.0 / yerr

    if len(z_cm) < 2:
        return None

    try:
        params, cov = np.polyfit(z_cm, mu, 1, w=weights, cov=True)
        slope, intercept = params
        slope_err = float(np.sqrt(cov[0, 0]))
        intercept_err = float(np.sqrt(cov[1, 1]))
    except Exception:
        slope, intercept = np.polyfit(z_cm, mu, 1, w=weights)
        slope_err = np.nan
        intercept_err = np.nan

    v_cm_ns = 1.0 / abs(slope) if slope != 0 else np.nan
    v_err_cm_ns = abs(slope_err / slope**2) if slope != 0 and np.isfinite(slope_err) else np.nan
    return slope, intercept, slope_err, intercept_err, v_cm_ns, v_err_cm_ns


def yerr_for_records(records: List[FitRecord], weight_mode: str) -> np.ndarray:
    if weight_mode == "time_err":
        return np.array([r.time_err for r in records], dtype=float)

    # Default matches the second/paper-style script: use fitted Gaussian sigma
    # as the y-error/weight in the z-vs-TOA linear fit.
    return np.array([r.sigma for r in records], dtype=float)


def fit_family(records: List[FitRecord], family: str, weight_mode: str = "sigma"):
    fam_recs = [
        r for r in records
        if r.family == family
        and r.z_mm != -999.0
        and np.isfinite(r.mu)
        and np.isfinite(r.sigma)
        and r.sigma > 0
    ]
    if len(fam_recs) < 2:
        return None, fam_recs
    z = np.array([r.z_cm for r in fam_recs])
    mu = np.array([r.mu for r in fam_recs])
    err = yerr_for_records(fam_recs, weight_mode)
    return weighted_line_fit(z, mu, err), fam_recs

# ============================================================
# Plots
# ============================================================
def make_combined_family_plot(records: List[FitRecord], outdir: str, particle_type: Optional[str], weight_mode: str = "sigma"):
    pdf_path = os.path.join(outdir, f"Z_vs_TOA_6mm_allY_collapsed_{particle_type or 'all'}.pdf")
    png_path = pdf_path.replace(".pdf", ".png")
    txt_path = os.path.join(outdir, f"Z_vs_TOA_6mm_allY_collapsed_{particle_type or 'all'}_fits.txt")

    fig, ax = plt.subplots(figsize=(18, 13.5))
    setup_axes(ax, "Z Position [cm]", "Mean Time of Arrival [ns]", particle_type, llabel="6 mm Z-Scan")

    all_mu = [r.mu for r in records if np.isfinite(r.mu)]
    if all_mu:
        ymin, ymax = min(all_mu), max(all_mu)
        pad = max(0.35, 0.15 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(-20.5, 18.5)

    lines = []
    labels = []
    fit_rows = []

    text_y = 0.94
    for family in FAMILY_ORDER:
        result, fam_recs = fit_family(records, family, weight_mode=weight_mode)
        if not fam_recs:
            continue

        color = FAMILY_COLORS[family]
        z = np.array([r.z_cm for r in fam_recs])
        mu = np.array([r.mu for r in fam_recs])
        err = yerr_for_records(fam_recs, weight_mode)

        ax.errorbar(
            z, mu, yerr=err,
            fmt="o", color=color, capsize=5, markersize=8,
            elinewidth=2.0, alpha=0.45,
        )

        if result is not None:
            slope, intercept, slope_err, intercept_err, v_cm_ns, v_err_cm_ns = result
            z_fit = np.linspace(np.min(z) - 2.0, np.max(z) + 2.0, 300)
            line, = ax.plot(z_fit, slope * z_fit + intercept, color=color, lw=3.5)
            lines.append(line)

            if np.isfinite(v_err_cm_ns):
                label = f"{FAMILY_DISPLAY_NAMES[family]}: {v_cm_ns:.3g} ± {v_err_cm_ns:.1g} cm/ns"
                text = f"{FAMILY_DISPLAY_NAMES[family]}  {v_cm_ns:.3g} ± {v_err_cm_ns:.1g} cm/ns"
            else:
                label = f"{FAMILY_DISPLAY_NAMES[family]}: {v_cm_ns:.3g} cm/ns"
                text = f"{FAMILY_DISPLAY_NAMES[family]}  {v_cm_ns:.3g} cm/ns"
            labels.append(label)

            fit_rows.append((family, len(fam_recs), slope, slope_err, intercept, intercept_err, v_cm_ns, v_err_cm_ns))

            ax.text(
                0.97, text_y, text,
                transform=ax.transAxes,
                ha="right", va="top",
                color=color,
                fontsize=24,
            )
            text_y -= 0.065

    if lines:
        ax.legend(lines, labels, loc="lower right", fontsize=18, frameon=True)

    fig.subplots_adjust(left=0.10, right=0.985, top=0.88, bottom=0.14)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    with open(txt_path, "w") as f:
        f.write(f"6 mm all-y collapsed z-vs-TOA fit results; z-fit weight mode = {weight_mode}\n")
        f.write("family,n_points,slope_ns_per_cm,slope_err,intercept_ns,intercept_err,velocity_cm_per_ns,velocity_err_cm_per_ns\n")
        for row in fit_rows:
            f.write(",".join(str(x) for x in row) + "\n")

    print(f"[PLOT] Saved: {pdf_path}")
    print(f"[PLOT] Saved: {png_path}")
    print(f"[FIT]  Saved: {txt_path}")


def make_family_pages(records: List[FitRecord], outdir: str, particle_type: Optional[str], min_points_per_channel=3, weight_mode: str = "sigma"):
    pdf_path = os.path.join(outdir, f"Z_vs_TOA_6mm_per_family_channels_{particle_type or 'all'}.pdf")

    with PdfPages(pdf_path) as pdf:
        for family in FAMILY_ORDER:
            fam_records = [r for r in records if r.family == family and r.z_mm != -999.0]
            if not fam_records:
                continue

            # Page 1: all points + family average fit
            fig, ax = plt.subplots(figsize=(15, 10))
            setup_axes(ax, "Z Position [cm]", "Mean Time of Arrival [ns]", particle_type, llabel=f"6 mm {family}")
            color = FAMILY_COLORS[family]

            z = np.array([r.z_cm for r in fam_records])
            mu = np.array([r.mu for r in fam_records])
            err = np.array([r.time_err for r in fam_records])
            ax.errorbar(z, mu, yerr=err, fmt="o", color=color, alpha=0.35, capsize=3, markersize=6)

            result = weighted_line_fit(z, mu, err) if len(z) >= 2 else None
            if result is not None:
                slope, intercept, slope_err, intercept_err, v, v_err = result
                zfit = np.linspace(min(z) - 2, max(z) + 2, 250)
                ax.plot(zfit, slope * zfit + intercept, color=color, lw=3.0,
                        label=f"all {family}: v = {v:.3g} ± {v_err:.1g} cm/ns")
                ax.legend(loc="best", frameon=True)

            ax.set_title(f"{FAMILY_DISPLAY_NAMES[family]}: all channels and all y positions collapsed")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Page 2: per-channel lines, if enough points exist.
            by_channel: Dict[str, List[FitRecord]] = {}
            for r in fam_records:
                by_channel.setdefault(r.channel, []).append(r)

            channel_fits = []
            for ch, recs in by_channel.items():
                if len(recs) < min_points_per_channel:
                    continue
                zc = np.array([r.z_cm for r in recs])
                muc = np.array([r.mu for r in recs])
                ec = yerr_for_records(recs, weight_mode)
                fit = weighted_line_fit(zc, muc, ec)
                if fit is not None:
                    channel_fits.append((ch, recs, fit))

            if channel_fits:
                fig, ax = plt.subplots(figsize=(15, 10))
                setup_axes(ax, "Z Position [cm]", "Mean Time of Arrival [ns]", particle_type, llabel=f"6 mm {family}")
                cmap = plt.get_cmap("tab20", len(channel_fits))

                handles, labels = [], []
                allz, allmu = [], []
                for i, (ch, recs, fit) in enumerate(sorted(channel_fits, key=lambda x: int(x[0]))):
                    c = cmap(i)
                    zc = np.array([r.z_cm for r in recs])
                    muc = np.array([r.mu for r in recs])
                    ec = yerr_for_records(recs, weight_mode)
                    slope, intercept, slope_err, intercept_err, v, v_err = fit
                    zfit = np.linspace(min(zc) - 1.0, max(zc) + 1.0, 120)
                    ax.errorbar(zc, muc, yerr=ec, fmt="o", color=c, alpha=0.6, capsize=2, markersize=5)
                    line, = ax.plot(zfit, slope * zfit + intercept, color=c, lw=2.0)
                    handles.append(line)
                    labels.append(f"Ch {ch}: v={v:.3g}±{v_err:.1g} cm/ns")
                    allz.extend(zc.tolist())
                    allmu.extend(muc.tolist())

                ax.legend(handles, labels, loc="best", fontsize=10, frameon=True, ncol=1)
                ax.set_title(f"{FAMILY_DISPLAY_NAMES[family]}: per-channel z-vs-TOA fits")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    print(f"[PLOT] Saved: {pdf_path}")


# ============================================================
# Histogram / Gaussian overlay inspection plots
# ============================================================
def make_histogram_overlay_pages(records: List[FitRecord], outdir: str, particle_type: Optional[str], suffix: str):
    """Make paper-style timing histograms with Gaussian overlays.

    One page per family/channel. Each page overlays all available z/y records
    for that channel, so pathological fits are easy to spot.
    """
    if not records:
        return

    pdf_path = os.path.join(outdir, f"timing_hist_gaussian_overlays_6mm_{particle_type or 'all'}.pdf")
    indiv_dir = os.path.join(outdir, "individual_timing_hist_overlays")
    os.makedirs(indiv_dir, exist_ok=True)

    grouped: Dict[Tuple[str, str], List[FitRecord]] = {}
    for r in records:
        grouped.setdefault((r.family, r.channel), []).append(r)

    z_values = sorted({r.z_mm for r in records if r.z_mm != -999.0})
    cmap = plt.get_cmap("tab10")
    z_colors = {z: cmap(i % 10) for i, z in enumerate(z_values)}

    with PdfPages(pdf_path) as pdf:
        for (family, channel), recs in sorted(
            grouped.items(),
            key=lambda x: (FAMILY_ORDER.index(x[0][0]), int(x[0][1]))
        ):
            recs = sorted(recs, key=lambda r: (r.z_mm, r.y_group, r.run_label))
            first = recs[0]

            fig, ax = plt.subplots(figsize=(18, 12))
            setup_axes(ax, "Time of Arrival [ns]", "Normalized Events", particle_type, llabel="6 mm timing fits")
            ax.set_xlim(first.tmin, first.tmax)
            ax.set_ylim(0.0, 1.65)

            handles, labels = [], []
            for r in recs:
                color = z_colors.get(r.z_mm, "black")
                ax.step(r.centers, r.hist_norm, where="mid", color=color, alpha=0.35, lw=1.8)
                line, = ax.plot(r.x_smooth, r.y_gauss, color=color, lw=3.0)
                handles.append(line)
                labels.append(
                    rf"{r.z_cm:+.1f} cm, {r.y_group}, Ch {r.channel}: "
                    rf"$\mu$={r.mu:.2f} ns, $\sigma$={r.sigma:.2f} ns, N={r.n}"
                )

            ax.text(
                0.98, 0.96,
                f"{FAMILY_DISPLAY_NAMES.get(family, family)} | Channel {channel}",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=24,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
            )
            ax.legend(
                handles, labels,
                loc="upper right",
                bbox_to_anchor=(0.985, 0.86),
                fontsize=13,
                frameon=True,
                title="6 mm z/y positions",
            )
            fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.14)
            pdf.savefig(fig, dpi=220)

            base = f"hist_overlay_6mm_{family}_ch{channel}".replace("/", "_")
            fig.savefig(os.path.join(indiv_dir, base + ".png"), bbox_inches="tight")
            fig.savefig(os.path.join(indiv_dir, base + ".pdf"), bbox_inches="tight")
            plt.close(fig)

    print(f"[PLOT] Saved: {pdf_path}")
    print(f"[PLOT] Individual histogram overlays saved in: {indiv_dir}")

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="6 mm z-vs-TOA plotting with all y groups collapsed into common z offsets.")
    parser.add_argument("--base-dir", default=BASE_DIR, help="Directory containing *_converted_timingskim.root files.")
    parser.add_argument("--ana-files", nargs="+", default=None, help="Explicit input ROOT files.")
    parser.add_argument("--ana-glob", default=None, help="Glob for input ROOT files.")
    parser.add_argument("--tree", default=TREE_NAME, help="ROOT tree name.")
    parser.add_argument("--outdir", default="TiminingZscan_summary_6mm_allY_collapsed", help="Output directory.")
    parser.add_argument("--pid", default="electron", choices=["muon", "pion", "electron", "proton"], help="PID selection.")
    parser.add_argument("--suffix", default="_LP2_50", help="Timing branch suffix.")
    parser.add_argument("--nbins", type=int, default=NBINS, help="Histogram bins for Gaussian timing fits.")
    parser.add_argument("--min-events", type=int, default=MIN_EVENTS, help="Minimum events required per channel/run fit.")
    parser.add_argument("--min-sigma-cut", type=float, default=0.089, help="Drop Gaussian fits with sigma below this value [ns]. Applies to all families. Set 0 to disable.")
    parser.add_argument("--max-sigma-cut", type=float, default=None, help="Optional: drop Gaussian fits with sigma above this value [ns].")
    parser.add_argument("--fit-weight", choices=["sigma", "time_err"], default="sigma", help="Y-error used in z-vs-TOA line fit. 'sigma' matches the paper script; 'time_err' uses sigma/sqrt(N).")
    # Defaults for this fixed 6 mm version:
    #   * WC cut OFF
    #   * ADC cut OFF
    #   * PID cut ON unless --no-pid-cut is used
    #   * abs(t_final) ON, matching the older 6 mm script with positive timing windows
    parser.add_argument("--use-wc-cut", action="store_true", help="Enable wirechamber cut. Default is OFF.")
    parser.add_argument("--use-adc-cut", action="store_true", help="Enable per-channel ADC/amplitude cut. Default is OFF.")
    parser.add_argument("--no-pid-cut", action="store_true", help="Disable PID cuts too, useful for debugging zero-event issues.")
    parser.add_argument("--signed-time", action="store_true", help="Use signed t_final instead of abs(t_final). Default uses abs(t_final).")
    parser.add_argument("--min-points-per-channel", type=int, default=3, help="Minimum z points for per-channel fit pages.")

    args = parser.parse_args()
    apply_paper_style()
    os.makedirs(args.outdir, exist_ok=True)

    files = resolve_files(args)
    if not files:
        raise SystemExit("[FATAL] No input files found. Check --base-dir, --ana-glob, or --ana-files.")

    print("\n[INIT] 6 mm all-y collapsed z-vs-TOA plotting")
    print(f"[INIT] files: {len(files)}")
    print(f"[INIT] outdir: {args.outdir}")
    print(f"[INIT] pid: {args.pid}")
    print(f"[INIT] suffix: {args.suffix}")
    print(f"[INIT] WC cut: {'ON' if args.use_wc_cut else 'OFF'}")
    print(f"[INIT] ADC cut: {'ON' if args.use_adc_cut else 'OFF'}")
    print(f"[INIT] PID cut: {'OFF' if args.no_pid_cut else 'ON'}")
    print(f"[INIT] abs(t_final): {'OFF' if args.signed_time else 'ON'}")
    sigma_cut_msg = f"{args.min_sigma_cut} <= sigma"
    if args.max_sigma_cut is not None:
        sigma_cut_msg += f" <= {args.max_sigma_cut}"
    sigma_cut_msg += " ns"
    print(f"[INIT] sigma cut: {sigma_cut_msg}")
    print(f"[INIT] z-fit weighting: {args.fit_weight}")

    records = collect_records(files, args)
    if not records:
        raise SystemExit("[FATAL] No successful fits. Check timing windows, branch names, and cuts.")

    csv_path = os.path.join(args.outdir, f"timing_gaussian_fit_summary_6mm_allY_collapsed_{args.pid}.csv")
    write_csv(records, csv_path)

    make_combined_family_plot(records, args.outdir, args.pid, weight_mode=args.fit_weight)
    make_family_pages(
        records,
        args.outdir,
        args.pid,
        min_points_per_channel=args.min_points_per_channel,
        weight_mode=args.fit_weight,
    )
    make_histogram_overlay_pages(records, args.outdir, args.pid, args.suffix)

    print("\n[DONE] Outputs written to:")
    print(f"       {args.outdir}")


if __name__ == "__main__":
    main()
