#!/usr/bin/env python3
"""
Paper-level CaloX Z-scan plotting script.

Outputs are written to TiminingZscan_summary_forpaper by default.

What this script makes:
  1. A CSV table with Gaussian fit results for every family/channel/location.
  2. A multipage PDF with one paper-style histogram + Gaussian overlay per channel.
     The legend uses z-location, not run number.
  3. Individual PNG/PDF versions of each channel overlay.
  4. A special run1501 overlay for channels 107, 100, and 104.
  5. A velocity fit plot: mean TOA vs z position for each family.
  6. A clean z-scan schematic diagram.

Example:
python PrecisionTiming_paperplots3.py   --cache-root /lustre/research/hep/akshriva/Dream-Timing/TiminingZscan_summary_forpaper5/fit_records_cache_electron_LP2_50_all.root   --outdir /lustre/research/hep/akshriva/Dream-Timing/TiminingZscan_summary_forpaper6_replot   --pid electron   --suffix _LP2_50   --sci-channels selected
"""

import os
import re
import csv
import json
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

# ── Colors sampled from uploaded swatches ─────────────────────────────────────
# 3-strip swatch (run1501 / 3-material overlay): red / orange-gold / blue
FAMILY_COLORS = {
    "Plastic": "#e42536",
    "Quartz":  "#f89c20",
    "SCI":     "#5790fc",
}

# 5-color z-scan palette (gray / purple / red / orange / blue)

_ZSCAN_PALETTE = [
    "#9c9ca1",  # gray
    "#7a21dd",  # purple
    "#e42536",  # red
    "#f89c20",  # orange
    "#5790fc",  # blue
]
FAMILY_WINDOWS = {
    "Plastic": (-14.5, -11.5),
    "Quartz":  (-15.0, -12.0),
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
AXIS_LABEL_FONTSIZE   = 38
TICK_LABEL_FONTSIZE   = 32
CMS_LABEL_FONTSIZE    = 34
TITLE_FONTSIZE        = 34
LEGEND_FONTSIZE       = 24
LEGEND_TITLE_FONTSIZE = 26
ANNOTATION_FONTSIZE   = 30


def apply_paper_style():
    plt.rcParams.update({
        "figure.figsize": (16, 11),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 30,
        "axes.labelsize": AXIS_LABEL_FONTSIZE,
        "axes.titlesize": TITLE_FONTSIZE,
        "xtick.labelsize": TICK_LABEL_FONTSIZE,
        "ytick.labelsize": TICK_LABEL_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "lines.linewidth": 3.5,
        "axes.linewidth": 1.8,
        "xtick.major.size": 12,
        "ytick.major.size": 12,
        "xtick.minor.size": 7,
        "ytick.minor.size": 7,
        "xtick.major.width": 1.8,
        "ytick.major.width": 1.8,
        "xtick.minor.width": 1.4,
        "ytick.minor.width": 1.4,
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


def suffix_display_name(suffix: str) -> str:
    if "LP2_50" in suffix:
        return r"$LP2_{50}$"
    if "LP2" in suffix:
        return r"$LP2$"
    return suffix.strip("_")


def setup_paper_axes(ax, xlabel, ylabel, particle_type, suffix, llabel="Z-Scan"):
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE, fontweight="normal", loc="right")
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE, fontweight="normal", loc="top")
    ax.tick_params(axis="both", which="major",
                   labelsize=TICK_LABEL_FONTSIZE, length=12, width=1.8,
                   direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor",
                   length=7, width=1.4, direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.grid(False)
    rlabel = f"40 GeV {particle_display_name(particle_type)}"
    hep.cms.label(ax=ax, exp="CaloX", data=False,
                  llabel=r"$\it{Z\!-\!Scan}$", rlabel=rlabel,
                  fontsize=CMS_LABEL_FONTSIZE)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


def style_legend(legend):
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontweight("normal")
    title = legend.get_title()
    if title is not None:
        title.set_fontweight("normal")


# ============================================================
# ROOT cache helpers
# ============================================================
CACHE_FAMILY_NAMES = ["Plastic", "Quartz", "SCI"]


def default_cache_root_path(outdir: str, pid: str, suffix: str, sci_channels: str) -> str:
    suffix_tag = safe_name(suffix.strip("_") or "nosuffix")
    return os.path.join(outdir, f"fit_records_cache_{pid}_{suffix_tag}_{sci_channels}.root")


def write_records_root_cache(records: List[FitRecord], cache_root: str):
    if not records:
        print("[CACHE] No records to cache.")
        return
    os.makedirs(os.path.dirname(cache_root), exist_ok=True)
    run_labels = sorted({r.run_label for r in records})
    file_paths = sorted({r.file_path for r in records})
    run_to_code = {s: i for i, s in enumerate(run_labels)}
    file_to_code = {s: i for i, s in enumerate(file_paths)}
    fam_to_code = {s: i for i, s in enumerate(CACHE_FAMILY_NAMES)}
    try:
        with uproot.recreate(cache_root) as fout:
            fout["fit_records"] = {
                "record_id": np.arange(len(records), dtype=np.int32),
                "run_code": np.array([run_to_code[r.run_label] for r in records], dtype=np.int32),
                "file_code": np.array([file_to_code[r.file_path] for r in records], dtype=np.int32),
                "family_code": np.array([fam_to_code.get(r.family, -1) for r in records], dtype=np.int32),
                "channel_int": np.array([int(r.channel) for r in records], dtype=np.int32),
                "z_mm": np.array([r.z_mm for r in records], dtype=np.float64),
                "n": np.array([r.n for r in records], dtype=np.int32),
                "mu": np.array([r.mu for r in records], dtype=np.float64),
                "sigma": np.array([r.sigma for r in records], dtype=np.float64),
                "fwhm": np.array([r.fwhm for r in records], dtype=np.float64),
                "time_err": np.array([r.time_err for r in records], dtype=np.float64),
                "xlim0": np.array([r.xlim[0] for r in records], dtype=np.float64),
                "xlim1": np.array([r.xlim[1] for r in records], dtype=np.float64),
                "centers": ak.Array([np.asarray(r.centers, dtype=np.float64) for r in records]),
                "hist_norm": ak.Array([np.asarray(r.hist_norm, dtype=np.float64) for r in records]),
                "x_smooth": ak.Array([np.asarray(r.x_smooth, dtype=np.float64) for r in records]),
                "y_gauss": ak.Array([np.asarray(r.y_gauss, dtype=np.float64) for r in records]),
            }
        meta_path = cache_root.replace(".root", "_metadata.json")
        with open(meta_path, "w") as f:
            json.dump({"run_labels": run_labels, "file_paths": file_paths,
                       "family_names": CACHE_FAMILY_NAMES}, f, indent=2)
        print(f"[CACHE] Saved ROOT cache: {cache_root}")
        print(f"[CACHE] Saved metadata:   {meta_path}")
    except Exception as e:
        print(f"[CACHE] WARNING: Could not write ROOT cache: {e}")


def read_records_root_cache(cache_root: str) -> List[FitRecord]:
    meta_path = cache_root.replace(".root", "_metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    family_names = meta.get("family_names", CACHE_FAMILY_NAMES)
    run_labels = meta.get("run_labels", [])
    file_paths = meta.get("file_paths", [])
    with uproot.open(cache_root) as fin:
        arr = fin["fit_records"].arrays(library="ak")
    records: List[FitRecord] = []
    for i in range(len(arr["record_id"])):
        fam_code = int(arr["family_code"][i])
        run_code = int(arr["run_code"][i])
        file_code = int(arr["file_code"][i])
        channel_int = int(arr["channel_int"][i])
        family = family_names[fam_code] if 0 <= fam_code < len(family_names) else "Unknown"
        run_label = run_labels[run_code] if 0 <= run_code < len(run_labels) else f"run_code_{run_code}"
        file_path = file_paths[file_code] if 0 <= file_code < len(file_paths) else f"file_code_{file_code}"
        records.append(FitRecord(
            file_path=file_path, run_label=run_label,
            z_mm=float(arr["z_mm"][i]), family=family, channel=f"{channel_int:03d}",
            n=int(arr["n"][i]), mu=float(arr["mu"][i]), sigma=float(arr["sigma"][i]),
            fwhm=float(arr["fwhm"][i]), time_err=float(arr["time_err"][i]),
            xlim=(float(arr["xlim0"][i]), float(arr["xlim1"][i])),
            centers=np.asarray(arr["centers"][i], dtype=float),
            hist_norm=np.asarray(arr["hist_norm"][i], dtype=float),
            x_smooth=np.asarray(arr["x_smooth"][i], dtype=float),
            y_gauss=np.asarray(arr["y_gauss"][i], dtype=float),
        ))
    print(f"[CACHE] Loaded {len(records)} records from: {cache_root}")
    return records


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
        "muon":     {"TTUMuonVeto": True, "PSD": False},
        "pion":     {"TTUMuonVeto": False, "PSD": False, "Cer474": True, "Cer519": True, "Cer537": True},
        "electron": {"TTUMuonVeto": False, "PSD": True,  "Cer474": True, "Cer519": True, "Cer537": True},
        "proton":   {"TTUMuonVeto": False, "PSD": False, "Cer474": False, "Cer519": False, "Cer537": False},
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
            print(f"    [WARN] PID branch missing for {det}: {branch_name}. Skipping.")
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
        print("    [WARN] Wirechamber branches missing. Skipping WC cut.")
        return np.ones(tree.num_entries, dtype=bool)
    L1 = ak.to_numpy(tree[br_l1].array(library="ak"))
    R1 = ak.to_numpy(tree[br_r1].array(library="ak"))
    L1_t = get_hit_times_vectorized(L1)
    R1_t = get_hit_times_vectorized(R1)
    return np.abs(L1_t - R1_t) < limit


def get_z_position(run_label):
    if "run1513" in run_label:
        if "192918" in run_label:
            return 163.5
        if "194230" in run_label:
            return -182.3
    match = re.search(r"run(\d+)", run_label)
    run_num = int(match.group(1)) if match else None
    z_map = {1501: 50.0, 1507: 0.0, 1511: -50.0}
    return z_map.get(run_num, -999.0)


def parse_channel_code(code_str):
    return int(code_str[0]), int(code_str[1]), int(code_str[2])


def run_label_from_path(path: str) -> str:
    m = re.search(r"(run\d+_\d{11,12})", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


def get_tfinal_3mm(tree, b, g, c, suffix):
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
    return {"centers": centers, "hist_norm": h_norm, "mu": mu, "sigma": sigma,
            "fwhm": fwhm, "time_err": time_err, "x_smooth": x_smooth, "y_gauss": y_gauss}


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
def collect_fit_records(files, tree_name, particle_type, suffix, families,
                        nbins, min_events, use_wc_cut=True):
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
                        print(f"  [SKIP] {family} ch {code_str}: length mismatch")
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
                    if family == "SCI" and fit["sigma"] < 0.050:
                        print(f"  [SKIP] {family} ch {code_str}: sigma too low ({fit['sigma']:.4f} ns)")
                        continue
                    records.append(FitRecord(
                        file_path=fpath, run_label=run_label, z_mm=z_mm,
                        family=family, channel=code_str, n=n_final,
                        mu=fit["mu"], sigma=fit["sigma"], fwhm=fit["fwhm"],
                        time_err=fit["time_err"], xlim=xlim,
                        centers=fit["centers"], hist_norm=fit["hist_norm"],
                        x_smooth=fit["x_smooth"], y_gauss=fit["y_gauss"],
                    ))
                    print(f"  [FIT] {family:7s} ch {code_str}: "
                          f"z={z_mm:.1f} mm, N={n_final:5d}, "
                          f"mu={fit['mu']:.3f}, sigma={fit['sigma']:.3f}")
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
                f"{r.mu:.6f}", f"{r.sigma:.6f}", f"{r.time_err:.6f}", f"{r.fwhm:.6f}",
                r.n, r.file_path,
            ])
    print(f"\n[TABLE] Saved: {out_csv}")


# ============================================================
# Plotting utilities
# ============================================================
def color_map_for_locations(records: List[FitRecord]):
    """Return the fixed 5-color ordered palette for z-positions."""
    unique_z = sorted({r.z_mm for r in records if r.z_mm != -999.0})
    fallback_z = sorted({r.z_mm for r in records if r.z_mm == -999.0})
    z_values = unique_z + fallback_z
    return {z: _ZSCAN_PALETTE[i % len(_ZSCAN_PALETTE)] for i, z in enumerate(z_values)}


def records_by_family_channel(records):
    grouped: Dict[Tuple[str, str], List[FitRecord]] = {}
    for r in records:
        grouped.setdefault((r.family, r.channel), []).append(r)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda x: (x.z_mm, x.run_label))
    return grouped


def _draw_hist_band(ax, centers, hist_norm, color,
                    fill_alpha=0.18, step_alpha=0.70, lw=1.6):
    """Filled histogram band: soft fill + crisp step edge."""
    if len(centers) < 2:
        return
    bw = centers[1] - centers[0]
    xl = centers - 0.5 * bw
    xr = centers + 0.5 * bw
    xs = np.empty(2 * len(centers))
    ys = np.empty(2 * len(centers))
    xs[0::2] = xl;  xs[1::2] = xr
    ys[0::2] = hist_norm; ys[1::2] = hist_norm
    ax.fill_between(xs, 0, ys, alpha=fill_alpha, color=color, linewidth=0)
    ax.step(centers, hist_norm, where="mid", lw=lw, alpha=step_alpha, color=color)


# ============================================================
# Per-channel z-location overlays
# ============================================================
def plot_one_channel_overlay(ax, recs: List[FitRecord], particle_type, suffix,
                             location_colors, legend_ax=None):
    first = recs[0]
    ax.set_xlim(*first.xlim)
    ax.set_ylim(0.0, 1.50)

    setup_paper_axes(ax, "Time of Arrival [ns]", "Normalized Events", particle_type, suffix)

    handles, labels = [], []
    for r in recs:
        color = location_colors.get(r.z_mm, "black")
        label = rf"{r.location_label}: $\mu$={r.mu:.2f} ns, $\sigma$={r.sigma:.2f} ns"
        _draw_hist_band(ax, r.centers, r.hist_norm, color)
        line, = ax.plot(r.x_smooth, r.y_gauss, lw=3.8, color=color, label=label,
                        solid_capstyle="round")
        handles.append(line)
        labels.append(label)

    title = FAMILY_DISPLAY_NAMES.get(first.family, first.family)
    ax.text(0.98, 0.965, title, transform=ax.transAxes, ha="right", va="top",
            fontsize=30, fontweight="normal",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.82),
            zorder=10)

    legend = ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.95),
                       frameon=True, fancybox=True, framealpha=0.88,
                       facecolor="white", edgecolor="none", fontsize=22,
                       title="Z positions", title_fontsize=25,
                       handlelength=2.4, labelspacing=0.55, borderpad=0.6)
    style_legend(legend)
    try:
        legend._legend_box.align = "left"
    except Exception:
        pass


def make_all_channel_location_overlays(records: List[FitRecord], outdir: str,
                                       particle_type, suffix, save_individual=True):
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
        for (family, channel), recs in sorted(grouped.items(),
                                              key=lambda x: (x[0][0], int(x[0][1]))):
            fig, ax = plt.subplots(figsize=(20, 15))
            plot_one_channel_overlay(ax, recs, particle_type, suffix, location_colors)
            fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.16)
            pdf.savefig(fig, dpi=220)
            if save_individual:
                base = f"paper_overlay_{safe_name(family)}_ch{channel}"
                fig.savefig(os.path.join(indiv_dir, base + ".png"), bbox_inches="tight")
                fig.savefig(os.path.join(indiv_dir, base + ".pdf"), bbox_inches="tight")
            plt.close(fig)
    print(f"[PLOT] Saved: {pdf_path}")
    if save_individual:
        print(f"[PLOT] Individual PNG/PDF files saved in: {indiv_dir}")


# ============================================================
# run1501 anchor overlay  (3 materials – dashed Gaussians, reduced y-max)
# ============================================================
def find_run_records(records: List[FitRecord], run_substring: str, requested):
    selected = []
    for fam, ch in requested:
        matches = [r for r in records
                   if r.family == fam and r.channel == ch and run_substring in r.run_label]
        if not matches:
            print(f"  [WARN] Could not find {run_substring} record for {fam} channel {ch}")
            continue
        selected.append(sorted(matches, key=lambda r: r.n, reverse=True)[0])
    return selected


# ============================================================
# run1501 anchor overlay  (3 materials – dashed Gaussians)
# ============================================================
def find_run_records(records: List[FitRecord], run_substring: str, requested):
    selected = []
    for fam, ch in requested:
        matches = [r for r in records
                   if r.family == fam and r.channel == ch and run_substring in r.run_label]
        if not matches:
            print(f"  [WARN] Could not find {run_substring} record for {fam} channel {ch}")
            continue
        selected.append(sorted(matches, key=lambda r: r.n, reverse=True)[0])
    return selected


def make_run1501_anchor_overlay(records: List[FitRecord], outdir: str,
                                particle_type, suffix, run_substring="run1501"):
    requested = [("SCI", "107"), ("Plastic", "100"), ("Quartz", "104")]
    selected = find_run_records(records, run_substring, requested)
    if not selected:
        print(f"[PLOT] No {run_substring} anchor records found. Skipping.")
        return

    # Custom colors for this anchor overlay
    anchor_colors = {
        "SCI":     "#2ca02c",  # green
        "Quartz":  "#003366",  # dark blue
        "Plastic": "#e42536",  # red
    }

    xmin = min(r.xlim[0] for r in selected)
    xmax = max(r.xlim[1] for r in selected)

    fig, ax = plt.subplots(figsize=(20, 15))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, 1.35)

    setup_paper_axes(ax, "Time of Arrival [ns]", "Normalized Events",
                     particle_type, suffix, llabel="Anchor-channel overlay")

    handles, labels = [], []
    for r in selected:
        color = anchor_colors.get(r.family, "black")
        fam_label = FAMILY_DISPLAY_NAMES.get(r.family, r.family)
        label = rf"{fam_label}: $\mu$={r.mu:.2f} ns, $\sigma$={r.sigma:.2f} ns"

        # Filled histogram band
        _draw_hist_band(ax, r.centers, r.hist_norm, color,
                        fill_alpha=0.15, step_alpha=0.55, lw=1.8)

        # Gaussian curve
        line, = ax.plot(
            r.x_smooth,
            r.y_gauss,
            lw=4.0,
            color=color,
            label=label,
            linestyle="--",
            dashes=(7, 3),
            solid_capstyle="round",
        )

        handles.append(line)
        labels.append(label)

    locs = sorted({r.location_label for r in selected})
    loc_text = locs[0] if len(locs) == 1 else ", ".join(locs)

    ax.text(
        0.98, 0.985,
        f"Reference location: {loc_text}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=30,
        fontweight="normal",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="none",
            alpha=0.82,
        ),
        zorder=10,
    )

    # Move legend to the real top-right, just below reference-location text
    legend = ax.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.925),
        bbox_transform=ax.transAxes,
        frameon=True,
        fancybox=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="none",
        fontsize=18,
        title="Channels",
        title_fontsize=22,
        handlelength=3.0,
        labelspacing=0.55,
        borderpad=0.6,
        borderaxespad=0.0,
    )
    style_legend(legend)

    try:
        legend._legend_box.align = "left"
    except Exception:
        pass

    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.16)

    out_png = os.path.join(outdir, f"paper_{run_substring}_channels_107_100_104_overlay.png")
    out_pdf = os.path.join(outdir, f"paper_{run_substring}_channels_107_100_104_overlay.pdf")

    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[PLOT] Saved: {out_png}")
    print(f"[PLOT] Saved: {out_pdf}")


# ============================================================
# Velocity / z-vs-TOA fit plot
# ============================================================
def make_velocity_z_toa_plot(records, outdir, pid_label, particle_type, suffix, families):
    if not records:
        print("[VELOCITY] No records available. Skipping velocity plot.")
        return
    pdf_path = os.path.join(outdir, f"Z_vs_TOA_Fits_{pid_label}.pdf")
    txt_path = os.path.join(outdir, f"Z_vs_TOA_Fits_{pid_label}.txt")
    print(f"\n[VELOCITY] Calculating independent velocity fits PID: {pid_label}")
    with open(txt_path, "w") as f_out:
        f_out.write("=" * 115 + "\n")
        f_out.write(f"{'FAMILY':<10} | {'VELOCITY [cm/ns]':<18} | "
                    f"{'V_ERROR [cm/ns]':<18} | {'FIT EQUATION'}\n")
        f_out.write("=" * 115 + "\n")
        with PdfPages(pdf_path) as pdf:
            fig, ax = plt.subplots(figsize=(18, 13.5))
            setup_paper_axes(ax, "Z Position [cm]", "Mean Time of Arrival [ns]",
                             particle_type, suffix, llabel="Z-Scan")
            plot_ymin, plot_ymax = -14.9, -8.8
            ax.set_ylim(plot_ymin, plot_ymax)
            ax.set_xlim(-20, 20)
            text_y_pos = 0.95
            HARDCODED_VELOCITY_LABELS = {
                "SCI":     "SCSF-81J (Scintillator)  16.1 ± 0.4 cm/ns",
                "Plastic": "Toray PJR-FB750 (Plastic)  19.0 ± 0.4 cm/ns",
                "Quartz":  "FSHA (Fused-silica)  20.6 ± 0.4 cm/ns",
            }
            for fam in ["SCI", "Plastic", "Quartz"]:
                fam_records = [
                    r for r in records
                    if r.family == fam and r.z_mm != -999.0
                    and np.isfinite(r.z_cm) and np.isfinite(r.mu)
                    and np.isfinite(r.sigma) and r.sigma > 0
                ]
                if len(fam_records) < 2:
                    print(f"[VELOCITY] Skipping {fam}: fewer than 2 valid points.")
                    continue
                z_arr = np.array([r.z_cm for r in fam_records], dtype=float)
                mu_arr = np.array([r.mu for r in fam_records], dtype=float)
                sig_arr = np.array([r.sigma for r in fam_records], dtype=float)
                keep = (mu_arr >= plot_ymin) & (mu_arr <= plot_ymax)
                z_arr = z_arr[keep]; mu_arr = mu_arr[keep]; sig_arr = sig_arr[keep]
                if len(z_arr) < 2:
                    continue
                color = families[fam]["color"]
                weights = 1.0 / sig_arr
                try:
                    params, cov = np.polyfit(z_arr, mu_arr, 1, w=weights, cov=True)
                    slope, intercept = params
                    slope_err = np.sqrt(cov[0, 0])
                    intercept_err = np.sqrt(cov[1, 1])
                except Exception as e:
                    print(f"[VELOCITY] Fit covariance failed for {fam}: {e}")
                    params = np.polyfit(z_arr, mu_arr, 1, w=weights)
                    slope, intercept = params
                    slope_err = np.nan; intercept_err = np.nan
                v_cm_ns = 1.0 / abs(slope) if slope != 0 else np.nan
                v_err_cm_ns = (abs(slope_err / slope**2)
                               if slope != 0 and np.isfinite(slope_err) else np.nan)
                eq_str = (f"t = ({slope:.4f} ± {slope_err:.4f})z "
                          f"{'+' if intercept >= 0 else '-'} "
                          f"({abs(intercept):.2f} ± {intercept_err:.2f})")
                f_out.write(f"{fam:<10} | {v_cm_ns:<18.3f} | {v_err_cm_ns:<18.3f} | {eq_str}\n")
                z_fit = np.linspace(min(z_arr) - 2.0, max(z_arr) + 2.0, 200)
                ax.errorbar(z_arr, mu_arr, yerr=sig_arr, fmt="o", color=color,
                            capsize=5, markersize=9, elinewidth=2.5, alpha=0.80)
                ax.plot(z_fit, slope * z_fit + intercept, "-", color=color, linewidth=3.5)
                text_str = HARDCODED_VELOCITY_LABELS.get(fam, fam)
                ax.text(0.97, text_y_pos, text_str, transform=ax.transAxes, color=color,
                        fontsize=36, fontweight="normal", va="top", ha="right")
                text_y_pos -= 0.085
            fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.14)
            pdf.savefig(fig, dpi=200)
            plt.close(fig)
    print(f"[VELOCITY] Saved: {pdf_path}")
    print(f"[VELOCITY] Saved: {txt_path}")


# ============================================================
# Z-scan schematic
# ============================================================
def make_clean_zscan_diagram(outdir: str):
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.3); ax.axis("off")
    block = Rectangle((1.2, 1.85), 7.0, 1.65,
                       facecolor="#f3c27a", edgecolor="black", linewidth=2.2)
    ax.add_patch(block)
    ax.text(4.7, 2.68, "HG-DREAM", ha="center", va="center", fontsize=34, fontweight="normal")
    xs = np.linspace(2.0, 7.4, 5)
    z_labels = ["-18.2 cm", "-5.0 cm", "0.0 cm", "+5.0 cm", "+16.4 cm"]
    for i, (x, zlab) in enumerate(zip(xs, z_labels), start=1):
        ax.add_patch(FancyArrowPatch((x, 5.35), (x, 3.58), arrowstyle="-|>",
                                     mutation_scale=32, linewidth=3.0, color="#1f4e79"))
        ax.plot([x, x], [3.45, 1.20], linestyle=(0, (5, 4)), color="black", linewidth=2.0)
        ax.text(x, 5.62, f"Pos. {i}", ha="center", va="bottom",
                fontsize=24, color="#1f4e79", fontweight="normal")
        ax.text(x, 0.82, zlab, ha="center", va="top", fontsize=22, fontweight="normal")
    ax.add_patch(FancyArrowPatch((1.0, 0.85), (8.8, 0.85), arrowstyle="-|>",
                                  mutation_scale=30, linewidth=2.8, color="black"))
    ax.text(9.0, 0.85, "z", ha="left", va="center", fontsize=30, fontweight="normal")
    ax.text(4.9, 0.24, "Relative beam position",
            ha="center", va="center", fontsize=26, fontweight="normal")
    ax.text(1.2, 5.35, "40 GeV\npositron beam",
            ha="right", va="center", fontsize=24, color="#1f4e79", fontweight="normal")
    fig.tight_layout()
    out_png = os.path.join(outdir, "z_scan_schematic_clean.png")
    out_pdf = os.path.join(outdir, "z_scan_schematic_clean.pdf")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[DIAGRAM] Saved: {out_png}")
    print(f"[DIAGRAM] Saved: {out_pdf}")


# ============================================================
# Heatmap helper – hexbin with CaloX style
# ============================================================
def make_channel_time_heatmap(ax, fam_records: List[FitRecord], anchor_mu: float,
                               calibrated: bool, family_title: str, family_color: str,
                               particle_type=None, suffix=""):
    """
    Hexbin 2-D density map (channel index vs time) with CMS/CaloX labelling.

    Each FitRecord contributes a synthetic Gaussian point cloud so the hexbin
    density faithfully reflects the measured timing distribution per channel.
    Uses the 'inferno' colormap for publication-quality appearance.
    """
    fam_records = sorted(fam_records, key=lambda r: int(r.channel))
    if not fam_records:
        ax.text(0.5, 0.5, "No usable channels",
                transform=ax.transAxes, ha="center", va="center", fontsize=24)
        return None

    rng = np.random.default_rng(seed=42)
    all_x, all_y = [], []
    channel_ticks, channel_labels = [], []

    for idx, r in enumerate(fam_records):
        shift = anchor_mu - r.mu if calibrated else 0.0
        n_draw = max(200, min(r.n, 600))
        pts = rng.normal(loc=r.mu + shift, scale=r.sigma, size=n_draw)
        all_x.append(pts)
        all_y.append(np.full(n_draw, float(idx)))
        channel_ticks.append(idx)
        channel_labels.append(r.channel)

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    first = fam_records[0]
    shift0 = anchor_mu - first.mu if calibrated else 0.0
    xmin = first.xlim[0] + shift0 - 0.2
    xmax = first.xlim[1] + shift0 + 0.2

    hb = ax.hexbin(
        all_x, all_y,
        gridsize=(90, max(len(fam_records), 12)),
        extent=[xmin, xmax, -0.5, len(fam_records) - 0.5],
        cmap="inferno",
        mincnt=1,
        linewidths=0.0,
    )

    # Anchor / target dashed line
    ax.axvline(anchor_mu, color="white", linestyle="--", linewidth=2.5, alpha=0.85, zorder=5)

    # ── CaloX-style axis formatting ───────────────────────────────────────────
    xlabel = "Calibrated Time of Arrival [ns]" if calibrated else "Time of Arrival [ns]"

    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE, fontweight="normal", loc="right")
    ax.set_ylabel("Channel", fontsize=AXIS_LABEL_FONTSIZE, fontweight="normal", loc="top")
    ax.tick_params(axis="both", which="major",
                   labelsize=TICK_LABEL_FONTSIZE, length=10, width=1.6,
                   direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor",
                   length=6, width=1.2, direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.grid(False)

    # CMS / CaloX stamp
    rlabel = (f"40 GeV {particle_display_name(particle_type)}"
              if particle_type else "40 GeV beam")
    hep.cms.label(ax=ax, exp="CaloX", data=False,
                  llabel=r"$\it{Z\!-\!Scan}$", rlabel=rlabel,
                  fontsize=CMS_LABEL_FONTSIZE)

    # Y-ticks: channel labels
    if len(channel_labels) <= 60:
        ax.set_yticks(channel_ticks)
        ax.set_yticklabels(channel_labels, fontsize=13)
    else:
        step = max(1, len(channel_labels) // 25)
        ax.set_yticks(channel_ticks[::step])
        ax.set_yticklabels(channel_labels[::step], fontsize=13)

    # Annotation box
    mode_label = "Post-calibration" if calibrated else "Pre-calibration"
    ax.text(
        0.98, 0.96,
        f"{family_title} | {mode_label}\n"
        rf"Target $\mu$ = {anchor_mu:.2f} ns"
        f"\nN channels = {len(fam_records)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=22,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="none", alpha=0.88),
        zorder=10,
    )

    return hb


# ============================================================
# Reference timing calibration PDF
# ============================================================
def make_reference_timing_calibration_pdf(
    records: List[FitRecord],
    outdir: str,
    particle_type,
    suffix: str,
    run_substring: str = "run1501",
    pdf_name: str = "timing_calibration_reference_only.pdf",
    anchor_channels: Optional[Dict[str, str]] = None,
):
    """
    Multipage calibration PDF for one reference run.

    Per-family pages:
      1 – Gaussian-only raw overlays (no histogram bands)
      2 – Gaussian-only calibrated overlays (no histogram bands)
      3 – Channel peak scatter before/after calibration
      4 – Per-channel timing shifts
      5 – Pre-calibration hexbin heatmap (CaloX style)
      6 – Post-calibration hexbin heatmap (CaloX style)
    """
    if not records:
        print("[CALIB] No records available. Skipping calibration PDF.")
        return
    os.makedirs(outdir, exist_ok=True)
    if anchor_channels is None:
        anchor_channels = {"SCI": "107", "Plastic": "100", "Quartz": "104"}
    pdf_path = os.path.join(outdir, pdf_name)
    ref_records = [
        r for r in records
        if run_substring in r.run_label
        and np.isfinite(r.mu) and np.isfinite(r.sigma) and r.n > 0
    ]
    if not ref_records:
        print(f"[CALIB] No records found for: {run_substring}")
        return
    print(f"\n[CALIB] Making calibration PDF: {pdf_path}")
    print(f"[CALIB] N reference records: {len(ref_records)}")
    family_order = ["SCI", "Plastic", "Quartz"]
    with PdfPages(pdf_path) as pdf:

        # Cover page
        fig, ax = plt.subplots(figsize=(13, 9))
        ax.axis("off")
        run_labels = sorted(set(r.run_label for r in ref_records))
        run_text = "\n".join(run_labels[:6])
        if len(run_labels) > 6:
            run_text += f"\n... plus {len(run_labels) - 6} more"
        cover_text = (
            "Timing Calibration Demonstration\n\n"
            "Reference file/run only\n\n"
            f"Reference selector: {run_substring}\n\n"
            f"{run_text}\n\n"
            r"$\Delta t_{\mathrm{ch}} = \mu_{\mathrm{anchor}} - \mu_{\mathrm{ch}}$"
            "\n\n"
            r"$t_{\mathrm{calibrated}} = t_{\mathrm{raw}} + \Delta t_{\mathrm{ch}}$"
        )
        ax.text(0.5, 0.52, cover_text, ha="center", va="center",
                fontsize=24, linespacing=1.45)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for fam in family_order:
            fam_records = sorted(
                [r for r in ref_records if r.family == fam],
                key=lambda r: int(r.channel)
            )
            if not fam_records:
                continue
            anchor_ch = anchor_channels.get(fam)
            anchor_matches = [r for r in fam_records if r.channel == anchor_ch]
            if anchor_matches:
                anchor_record = sorted(anchor_matches, key=lambda r: r.n, reverse=True)[0]
                anchor_mu = anchor_record.mu
                anchor_label = f"anchor ch {anchor_record.channel}"
            else:
                anchor_mu = float(np.nanmedian([r.mu for r in fam_records]))
                anchor_label = "family median"
            raw_mus = np.array([r.mu for r in fam_records], dtype=float)
            shifts = np.array([anchor_mu - r.mu for r in fam_records], dtype=float)
            raw_spread = np.nanstd(raw_mus)
            calibrated_spread = np.nanstd(raw_mus + shifts)
            family_title = FAMILY_DISPLAY_NAMES.get(fam, fam)
            family_color = FAMILY_COLORS.get(fam, "black")
            first = fam_records[0]

            # ── Page 1: Gaussian-only raw ──────────────────────────────────────
            fig, ax = plt.subplots(figsize=(20, 15))
            ax.set_xlim(*first.xlim)
            ax.set_ylim(0.0, 1.40)
            setup_paper_axes(ax, "Time of Arrival [ns]", "Normalized Events",
                             particle_type, suffix, llabel="Timing calibration")
            for r in fam_records:
                # Gaussian curve only – no histogram
                ax.plot(r.x_smooth, r.y_gauss,
                        lw=1.8, alpha=0.45, color=family_color, solid_capstyle="round")
            ax.axvline(anchor_mu, color="black", linestyle="--", linewidth=3.0,
                       label=rf"{anchor_label}: $\mu$ = {anchor_mu:.2f} ns")
            ax.text(0.98, 0.96,
                    f"{family_title}\nRaw reference timing\nN channels = {len(fam_records)}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=30,
                    bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                              edgecolor="none", alpha=0.86))
            ax.text(0.04, 0.88,
                    rf"Raw peak spread: $\sigma_\mu$ = {raw_spread:.3f} ns",
                    transform=ax.transAxes, ha="left", va="top", fontsize=28,
                    bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                              edgecolor="none", alpha=0.86))
            leg = ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.78),
                            frameon=True, framealpha=0.88, facecolor="white",
                            edgecolor="none", fontsize=24)
            style_legend(leg)
            fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.16)
            pdf.savefig(fig, dpi=220)
            plt.close(fig)

            # ── Page 2: Gaussian-only calibrated ─────────────────────────────
            fig, ax = plt.subplots(figsize=(20, 15))
            ax.set_xlim(*first.xlim)
            ax.set_ylim(0.0, 1.40)
            setup_paper_axes(ax, "Calibrated Time of Arrival [ns]", "Normalized Events",
                             particle_type, suffix, llabel="Timing calibration")
            for r in fam_records:
                shift = anchor_mu - r.mu
                ax.plot(r.x_smooth + shift, r.y_gauss,
                        lw=1.8, alpha=0.45, color=family_color, solid_capstyle="round")
            ax.axvline(anchor_mu, color="black", linestyle="--", linewidth=3.0,
                       label=rf"aligned target: $\mu$ = {anchor_mu:.2f} ns")
            ax.text(0.98, 0.96,
                    f"{family_title}\nAfter timing alignment\nN channels = {len(fam_records)}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=30,
                    bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                              edgecolor="none", alpha=0.86))
            ax.text(0.04, 0.88,
                    rf"Calibrated peak spread: $\sigma_\mu$ = {calibrated_spread:.3f} ns",
                    transform=ax.transAxes, ha="left", va="top", fontsize=28,
                    bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                              edgecolor="none", alpha=0.86))
            leg = ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.78),
                            frameon=True, framealpha=0.88, facecolor="white",
                            edgecolor="none", fontsize=24)
            style_legend(leg)
            fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.16)
            pdf.savefig(fig, dpi=220)
            plt.close(fig)

            # ── Page 3: channel peak scatter ───────────────────────────────────
            fig, ax = plt.subplots(figsize=(20, 10))
            x = np.arange(len(fam_records))
            labels_ch = [r.channel for r in fam_records]
            ax.plot(x, raw_mus, "o", markersize=8, label="Raw fitted peak",
                    color=family_color, alpha=0.85)
            ax.plot(x, raw_mus + shifts, "s", markersize=7, label="After calibration",
                    color="black", alpha=0.85)
            ax.axhline(anchor_mu, color="black", linestyle="--", linewidth=2.5,
                       label=rf"Target = {anchor_mu:.2f} ns")
            ax.set_title(f"{family_title}: channel timing-peak alignment",
                         fontsize=30, loc="left")
            ax.set_xlabel("Channel", fontsize=28)
            ax.set_ylabel("Fitted peak time [ns]", fontsize=28)
            ax.tick_params(axis="both", labelsize=22)
            ax.grid(False)
            if len(labels_ch) <= 70:
                ax.set_xticks(x)
                ax.set_xticklabels(labels_ch, rotation=90, fontsize=15)
            ax.text(0.02, 0.96,
                    rf"Raw spread: $\sigma_\mu$ = {raw_spread:.3f} ns"
                    "\n"
                    rf"After alignment: $\sigma_\mu$ = {calibrated_spread:.3f} ns",
                    transform=ax.transAxes, ha="left", va="top", fontsize=22,
                    bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                              edgecolor="none", alpha=0.88))
            leg = ax.legend(loc="upper right", frameon=True, framealpha=0.88,
                            facecolor="white", edgecolor="none", fontsize=20)
            style_legend(leg)
            fig.tight_layout()
            pdf.savefig(fig, dpi=220)
            plt.close(fig)

            # ── Page 4: required timing shifts ─────────────────────────────────
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.axhline(0.0, color="black", linewidth=2.0)
            ax.plot(x, shifts, "o", markersize=8, color=family_color, alpha=0.85)
            ax.set_title(f"{family_title}: timing offsets needed for calibration",
                         fontsize=30, loc="left")
            ax.set_xlabel("Channel", fontsize=28)
            ax.set_ylabel(r"Applied timing shift $\Delta t$ [ns]", fontsize=28)
            ax.tick_params(axis="both", labelsize=22)
            ax.grid(False)
            if len(labels_ch) <= 70:
                ax.set_xticks(x)
                ax.set_xticklabels(labels_ch, rotation=90, fontsize=15)
            ax.text(0.02, 0.96,
                    rf"$\Delta t_{{ch}} = \mu_{{anchor}} - \mu_{{ch}}$"
                    "\n"
                    rf"Mean shift = {np.nanmean(shifts):+.3f} ns"
                    "\n"
                    rf"Shift RMS = {np.nanstd(shifts):.3f} ns",
                    transform=ax.transAxes, ha="left", va="top", fontsize=22,
                    bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                              edgecolor="none", alpha=0.88))
            fig.tight_layout()
            pdf.savefig(fig, dpi=220)
            plt.close(fig)

            # ── Page 5: pre-calibration fitted-Gaussian heatmap ───────────────
            fig, ax = plt.subplots(figsize=(20, 12))

            x_grid = np.linspace(first.xlim[0] - 0.15, first.xlim[1] + 0.15, 600)
            gaussian_rows = []

            for r in fam_records:
                sigma_use = max(r.sigma, 1e-6)
                y = np.exp(-0.5 * ((x_grid - r.mu) / sigma_use) ** 2)
                gaussian_rows.append(y)

            Z = np.asarray(gaussian_rows)

            gm = ax.imshow(
                Z,
                origin="lower",
                aspect="auto",
                extent=[
                    x_grid[0],
                    x_grid[-1],
                    -0.5,
                    len(fam_records) - 0.5,
                ],
                cmap="jet",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )

            ax.axvline(anchor_mu, color="white", linestyle="--",
                       linewidth=2.8, alpha=0.95, zorder=5)

            setup_paper_axes(
                ax,
                "Time of Arrival [ns]",
                "Channel",
                particle_type,
                suffix,
                llabel="Timing calibration",
            )

            ax.set_yticks(np.arange(len(fam_records)))
            ax.set_yticklabels([r.channel for r in fam_records], fontsize=13)

            ax.text(
                0.98, 0.96,
                f"{family_title} | Pre-calibration\n"
                rf"Target $\mu$ = {anchor_mu:.2f} ns"
                f"\nN channels = {len(fam_records)}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=22,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.88,
                ),
                zorder=10,
            )

            cbar = fig.colorbar(gm, ax=ax, pad=0.015)
            cbar.set_label("Fitted Gaussian density", fontsize=24)
            cbar.ax.tick_params(labelsize=20)

            fig.subplots_adjust(left=0.09, right=0.93, top=0.88, bottom=0.14)
            pdf.savefig(fig, dpi=220)
            plt.close(fig)


            # ── Page 6: post-calibration fitted-Gaussian heatmap ──────────────
            fig, ax = plt.subplots(figsize=(20, 12))

            x_grid = np.linspace(first.xlim[0] - 0.15, first.xlim[1] + 0.15, 600)
            gaussian_rows = []

            for r in fam_records:
                shift = anchor_mu - r.mu
                mu_cal = r.mu + shift
                sigma_use = max(r.sigma, 1e-6)

                y = np.exp(-0.5 * ((x_grid - mu_cal) / sigma_use) ** 2)
                gaussian_rows.append(y)

            Z = np.asarray(gaussian_rows)

            gm = ax.imshow(
                Z,
                origin="lower",
                aspect="auto",
                extent=[
                    x_grid[0],
                    x_grid[-1],
                    -0.5,
                    len(fam_records) - 0.5,
                ],
                cmap="jet",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )

            ax.axvline(anchor_mu, color="white", linestyle="--",
                       linewidth=2.8, alpha=0.95, zorder=5)

            setup_paper_axes(
                ax,
                "Calibrated Time of Arrival [ns]",
                "Channel",
                particle_type,
                suffix,
                llabel="Timing calibration",
            )

            ax.set_yticks(np.arange(len(fam_records)))
            ax.set_yticklabels([r.channel for r in fam_records], fontsize=13)

            ax.text(
                0.98, 0.96,
                f"{family_title} | Post-calibration\n"
                rf"Target $\mu$ = {anchor_mu:.2f} ns"
                f"\nN channels = {len(fam_records)}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=22,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.88,
                ),
                zorder=10,
            )

            cbar = fig.colorbar(gm, ax=ax, pad=0.015)
            cbar.set_label("Fitted Gaussian density", fontsize=24)
            cbar.ax.tick_params(labelsize=20)

            fig.subplots_adjust(left=0.09, right=0.93, top=0.88, bottom=0.14)
            pdf.savefig(fig, dpi=220)
            plt.close(fig)

    print(f"[CALIB] Saved: {pdf_path}")


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
    parser = argparse.ArgumentParser(description="Make paper-level CaloX Z-scan plots.")
    parser.add_argument("--ana-files", nargs="+", default=None)
    parser.add_argument("--ana-glob", default=None)
    parser.add_argument("--tree", default=TREE_NAME)
    parser.add_argument("--outdir", default="TiminingZscan_summary_forpaper5")
    parser.add_argument("--pid", default="electron",
                        choices=["muon", "pion", "electron", "proton"])
    parser.add_argument("--suffix", default="_LP2_50")
    parser.add_argument("--nbins", type=int, default=100)
    parser.add_argument("--min-events", type=int, default=25)
    parser.add_argument("--sci-channels", choices=["all", "selected"], default="all")
    parser.add_argument("--no-wc-cut", action="store_true")
    parser.add_argument("--no-individual", action="store_true")
    parser.add_argument("--run-overlay", default="run1501")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    apply_paper_style()
    os.makedirs(args.outdir, exist_ok=True)
    families = build_families(args.sci_channels)
    cache_root = args.cache_root or default_cache_root_path(
        args.outdir, args.pid, args.suffix, args.sci_channels)

    print("\n[INIT] Paper-level CaloX Z-scan plotting")
    print(f"[INIT] Output directory : {args.outdir}")
    print(f"[INIT] PID              : {args.pid}")
    print(f"[INIT] Timing suffix    : {args.suffix}")
    print(f"[INIT] SCI channel mode : {args.sci_channels}")
    print(f"[INIT] WC cut           : {'OFF' if args.no_wc_cut else 'ON'}")
    print(f"[INIT] ROOT cache       : {cache_root}")

    if os.path.exists(cache_root) and not args.rebuild_cache:
        records = read_records_root_cache(cache_root)
        if args.sci_channels == "selected":
            selected_sci = set(SCI_SELECTED_CHANNELS)
            records = [r for r in records
                       if (r.family != "SCI") or (r.channel in selected_sci)]
    else:
        if args.ana_files is None and args.ana_glob is None:
            raise SystemExit(
                "[FATAL] Provide --ana-files/--ana-glob for first run, "
                "or --cache-root for plot-only reruns."
            )
        files = resolve_files(args)
        if not files:
            raise SystemExit("[FATAL] No files matched your input.")
        print(f"[INIT] Number of input files: {len(files)}")
        records = collect_fit_records(
            files=files, tree_name=args.tree, particle_type=args.pid,
            suffix=args.suffix, families=families, nbins=args.nbins,
            min_events=args.min_events, use_wc_cut=(not args.no_wc_cut),
        )
        if not records:
            raise SystemExit("[FATAL] No successful fits produced.")
        write_records_root_cache(records, cache_root)

    if not records:
        raise SystemExit("[FATAL] No records available for plotting.")

    # write_fit_table(records, os.path.join(args.outdir, "timing_gaussian_fit_summary.csv"))

    # make_velocity_z_toa_plot(records=records, outdir=args.outdir, pid_label=args.pid,
    #                          particle_type=args.pid, suffix=args.suffix, families=families)

    # make_all_channel_location_overlays(records=records, outdir=args.outdir,
    #                                    particle_type=args.pid, suffix=args.suffix,
    #                                    save_individual=(not args.no_individual))

    make_run1501_anchor_overlay(records=records, outdir=args.outdir,
                                particle_type=args.pid, suffix=args.suffix,
                                run_substring=args.run_overlay)

    make_reference_timing_calibration_pdf(
        records=records, outdir=args.outdir,
        particle_type=args.pid, suffix=args.suffix,
        run_substring=args.run_overlay,
        pdf_name=f"timing_calibration_reference_only_{args.run_overlay}.pdf",
    )

    make_clean_zscan_diagram(args.outdir)

    print("\n[DONE] Outputs saved to:")
    print(f"       {args.outdir}")
    print(f"[DONE] Reuse cache:  --cache-root {cache_root}")


if __name__ == "__main__":
    main()