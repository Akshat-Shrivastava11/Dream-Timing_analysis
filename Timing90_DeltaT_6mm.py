#!/usr/bin/env python3
"""
6 mm HG-DREAM / CaloX Delta-T timing-resolution script.

This is the 6 mm version of the 3 mm Delta-T pair script:
  * uses the 6 mm channel maps / y groups,
  * uses RUN_TO_YGROUP to choose the correct channels for each run,
  * computes t_final dynamically from timing branches,
  * makes one PDF per channel pair,
  * writes one CSV/TXT summary table.

Default timing definition:
  t_final = [t(B,G,C) - t(B,G,8)] - [t(0,3,7) - t(0,3,8)]

By default the script uses abs(t_final), matching the older 6 mm timing-window
scripts with positive windows. Use --signed-tfinal if you want signed times.

Example, automatic run dict input:
python3 Timing90_DeltaT_6mm.py \
  --base-dir /lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples \
  --outdir /lustre/research/hep/akshriva/Dream-Timing/DeltaT_6mm \
  --pid electron \
  --suffix _LP2_50

Example, explicit files:
python3 Timing90_DeltaT_6mm.py \
  --ana-files /lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1504_250928133854_converted_timingskim.root \
  --outdir ./DeltaT_6mm_test \
  --pid electron
"""

import os
import re
import csv
import glob
import argparse
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import uproot
import awkward as ak
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.CMS)

# ============================================================
# Defaults
# ============================================================
TREE_NAME = "EventTree"
BASE_DIR = "/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples"
DEFAULT_SUFFIX = "_LP2_50"
DEFAULT_NBINS = 80
MIN_EVENTS = 25
AMP_THRESHOLD = 100.0
MIN_ADC_CUT = -100.0

# ============================================================
# 6 mm full detector grids
# These are kept here so the script is tied to the 6 mm physical grid.
# The run-specific pair choices below are taken from these grids through Y_CONFIGS.
# ============================================================
QUARTZ_GRID = [
    [None,  None,  "617", "616", "615", "614", None,  None],
    [None,  None,  "625", "624", "623", "622", None,  None],
    [None,  "637", "631", "630", "627", "626", "636", None],
    ["515", "514", "635", "634", "633", "632", "501", "500"],
    [None,  None,  "002", None,  None,  None,  None,  None],
    ["517", "516", "006", "004", "206", "204", "503", "502"],
    [None,  None,  "016", "014", "216", "214", None,  None],
    ["521", "520", "026", "024", "226", "224", "505", "504"],
    [None,  None,  "030", None,  None,  None,  None,  None],
    [None,  None,  "530", "034", "534", "234", None,  None],
    ["523", "522", "106", "104", "306", "304", "507", "506"],
    [None,  None,  "116", "114", "316", "314", None,  None],
    ["525", "524", "126", "124", "326", "324", "511", "510"],
    [None,  None,  "532", "134", "536", "334", None,  None],
    ["527", "526", "403", "402", "401", "400", "513", "512"],
    [None,  "437", "407", "406", "405", "404", "436", None],
    [None,  None,  "413", "412", "411", "410", None,  None],
    [None,  None,  "417", "416", "415", "414", None,  None],
]

PLASTIC_GRID = [
    [None,  None,  "603", "602", "601", "600", None,  None],
    [None,  None,  None,  "607", "606", None,  None,  None],
    [None,  None,  "613", "612", "611", "610", None,  None],
    [None,  None,  None,  None,  None,  None,  None,  None],
    [None,  None,  None,  None,  None,  None,  None,  None],
    [None,  None,  None,  None,  None,  None,  None,  None],
    [None,  None,  None,  "000", "202", "200", None,  None],
    [None,  None,  "012", "010", "212", "210", None,  None],
    [None,  None,  "022", "020", "222", "220", None,  None],
    [None,  None,  "032", None,  "232", "230", None,  None],
    [None,  None,  None,  None,  None,  None,  None,  None],
    [None,  None,  "102", "100", "302", "300", None,  None],
    [None,  None,  "112", "110", "312", "310", None,  None],
    [None,  None,  "122", "120", "322", "320", None,  None],
    [None,  None,  "132", "130", "332", "330", None,  None],
    [None,  None,  None,  None,  None,  None,  None,  None],
    [None,  None,  None,  None,  None,  None,  None,  None],
    [None,  None,  None,  None,  None,  None,  None,  None],
    [None,  None,  "425", "424", "423", "422", None,  None],
    [None,  None,  None,  "427", "426", None,  None,  None],
    [None,  None,  "433", "432", "431", "430", None,  None],
]

SCI_GRID = [
    [None, None, "605", "604", None, None],
    [None, None, None, None, None, None],
    [None, None, "621", "620", None, None],
    [None, None, None, None, None, None],
    [None, None, None, None, None, None],
    [None, "003", "001", "203", "201", None],
    [None, "007", "005", "207", "205", None],
    [None, "013", "011", "213", "211", None],
    [None, "017", "015", "217", "215", None],
    [None, "023", "021", "223", "221", None],
    [None, "027", "025", "227", "225", None],
    [None, "033", "031", "233", "231", None],
    [None, "531", "035", "535", "235", None],
    [None, "103", "101", "303", "301", None],
    [None, "107", "105", "307", "305", None],
    [None, "113", "111", "313", "311", None],
    [None, "117", "115", "317", "315", None],
    [None, "123", "121", "323", "321", None],
    [None, "127", "125", "327", "325", None],
    [None, "133", "131", "333", "331", None],
    [None, "533", "135", "537", "335", None],
    [None, None, None, None, None, None],
    [None, None, None, None, None, None],
    [None, None, "421", "420", None, None],
    [None, None, None, None, None, None],
    [None, None, "425", "434", None, None],
]

GRID_BY_FAMILY = {
    "SCI": SCI_GRID,
    "Plastic": PLASTIC_GRID,
    "Quartz": QUARTZ_GRID,
}

# ============================================================
# 6 mm channel configs by y position.
# These are the actual channels used for each run group.
# ============================================================
Y_CONFIGS = {
    "y1000": {
        "SCI":     {"channels": ["620", "621"],                 "tmin":  8.0, "tmax": 11.0},
        "Plastic": {"channels": ["612", "611", "610", "613"], "tmin": 10.5, "tmax": 12.5},
        "Quartz":  {"channels": ["631", "630", "627", "637"], "tmin": 10.0, "tmax": 13.5},
    },
    "y1065": {
        "Quartz":  {"channels": ["523", "522", "521", "520"], "tmin": 10.0, "tmax": 13.5},
    },
    "y936": {
        "SCI":     {"channels": ["604", "605"],                 "tmin":  8.0, "tmax": 11.0},
        "Plastic": {"channels": ["607", "606"],                 "tmin": 11.0, "tmax": 12.5},
        "Quartz":  {"channels": ["617", "616", "615", "614"], "tmin": 11.0, "tmax": 12.6},
    },
    "y1028": {
        "SCI":     {"channels": ["421", "420"],                 "tmin":  7.0, "tmax": 10.5},
        "Plastic": {"channels": ["425", "423", "422", "424"], "tmin": 10.5, "tmax": 12.5},
        "Quartz":  {"channels": ["413", "412", "411", "410"], "tmin": 11.0, "tmax": 12.5},
    },
}

# Explicit run -> y group dictionary.
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

FAMILY_DISPLAY = {
    "SCI": "SCSF-81J (Scintillator)",
    "Plastic": "Toray PJR-FB750 (Plastic)",
    "Quartz": "FSHA (Quartz)",
}

FAMILY_COLORS = {
    "SCI": "#2ca02c",
    "Plastic": "#e42536",
    "Quartz": "#003366",
}

FAMILY_ORDER = ["SCI", "Plastic", "Quartz"]

# ============================================================
# PID helpers
# ============================================================
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


def get_service_drs_cut(service_drs: str) -> Tuple[int, int, float, str]:
    cuts = {
        "HoleVeto": (100, 350, -2e3, "Sum"),
        "PSD": (100, 400, -3500.0, "Sum"),
        "TTUMuonVeto": (200, 400, -2e3, "Sum"),
        "Cer474": (800, 900, -2000.0, "Sum"),
        "Cer519": (450, 550, -1000.0, "Sum"),
        "Cer537": (400, 500, -500.0, "Sum"),
    }
    return cuts.get(service_drs, (0, 1000, -5e4, "Sum"))


def get_particle_selection(particle_type: str) -> Dict[str, bool]:
    selections = {
        "muon": {"TTUMuonVeto": True, "PSD": False},
        "pion": {"TTUMuonVeto": False, "PSD": False, "Cer474": True, "Cer519": True, "Cer537": True},
        "electron": {"TTUMuonVeto": False, "PSD": True, "Cer474": True, "Cer519": True, "Cer537": True},
        "proton": {"TTUMuonVeto": False, "PSD": False, "Cer474": False, "Cer519": False, "Cer537": False},
    }
    return selections.get((particle_type or "").lower(), {})


def compute_pid_mask(tree, particle_type: Optional[str]) -> np.ndarray:
    if particle_type is None:
        return np.ones(tree.num_entries, dtype=bool)

    requirements = get_particle_selection(particle_type)
    if not requirements:
        print(f"  [PID] No PID requirements for '{particle_type}'. Using all events.")
        return np.ones(tree.num_entries, dtype=bool)

    keys = set(tree.keys())
    mask = np.ones(tree.num_entries, dtype=bool)

    print(f"  [PID] Applying {particle_type} selection")
    for det, must_fire in requirements.items():
        branch = PID_BRANCH_MAP.get(det)
        if not branch or branch not in keys:
            print(f"    [WARN] PID branch missing: {det} ({branch}). Skipping.")
            continue

        ts_min, ts_max, val_cut, method = get_service_drs_cut(det)
        if method != "Sum":
            print(f"    [WARN] Unsupported PID method for {det}: {method}. Skipping.")
            continue

        try:
            wf = tree[branch].array(library="ak")
            baseline = ak.mean(wf[:, :30], axis=1)
            wf_bl = wf - baseline
            win_sum = ak.sum(wf_bl[:, int(ts_min):int(ts_max)], axis=1)
            fired = ak.to_numpy(win_sum) < val_cut
            mask = mask & fired if must_fire else mask & (~fired)
            status = "fire" if must_fire else "veto"
            print(f"    [PID] {det:<12} {status:<4} -> pass now {int(mask.sum())}/{len(mask)}")
        except Exception as e:
            print(f"    [WARN] PID cut failed for {det}: {e}")

    return mask

# ============================================================
# Timing/cache helpers
# ============================================================
class FileCache:
    def __init__(self, tree, suffix: str):
        self.tree = tree
        self.suffix = suffix
        self.keys = set(tree.keys())
        self.timing: Dict[str, Optional[np.ndarray]] = {}
        self.adc: Dict[str, np.ndarray] = {}
        self.trigger_corr: Optional[np.ndarray] = None


def parse_code(code_str: str) -> Tuple[int, int, int]:
    code = re.sub(r"[^0-9]", "", str(code_str))[:3]
    if len(code) != 3:
        raise ValueError(f"Bad channel code: {code_str}")
    return int(code[0]), int(code[1]), int(code[2])


def compute_tfinal_6mm(cache: FileCache, code_str: str, abs_tfinal: bool = True) -> Optional[np.ndarray]:
    """
    6 mm timing definition:
      t_final = [t(B,G,C) - t(B,G,8)] - [t(0,3,7) - t(0,3,8)]

    If abs_tfinal=True, returns |t_final| to match your 6 mm positive timing windows.
    """
    key = f"{code_str}_{cache.suffix}_{'abs' if abs_tfinal else 'signed'}"
    if key in cache.timing:
        return cache.timing[key]

    b, g, c = parse_code(code_str)
    br_sig = f"DRS_Board{b}_Group{g}_Channel{c}{cache.suffix}"
    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8{cache.suffix}"
    br_trg = f"DRS_Board0_Group3_Channel7{cache.suffix}"
    br_trg_ref = f"DRS_Board0_Group3_Channel8{cache.suffix}"

    missing = [br for br in [br_sig, br_sig_ref, br_trg, br_trg_ref] if br not in cache.keys]
    if missing:
        print(f"      [MISS timing] {code_str}: missing {missing}")
        cache.timing[key] = None
        return None

    print(f"      [READ timing] {code_str}", flush=True)
    sig = cache.tree[br_sig].array(library="np")
    sig_ref = cache.tree[br_sig_ref].array(library="np")

    if cache.trigger_corr is None:
        trg = cache.tree[br_trg].array(library="np")
        trg_ref = cache.tree[br_trg_ref].array(library="np")
        if trg.shape != trg_ref.shape:
            cache.timing[key] = None
            return None
        cache.trigger_corr = trg - trg_ref

    if not (sig.shape == sig_ref.shape == cache.trigger_corr.shape):
        print(f"      [SKIP timing] {code_str}: shape mismatch")
        cache.timing[key] = None
        return None

    out = (sig - sig_ref) - cache.trigger_corr
    if abs_tfinal:
        out = np.abs(out)

    cache.timing[key] = out
    return out


def compute_adc_mask(cache: FileCache, code_str: str) -> np.ndarray:
    if code_str in cache.adc:
        return cache.adc[code_str]

    b, g, c = parse_code(code_str)
    br = f"DRS_Board{b}_Group{g}_Channel{c}"
    if br not in cache.keys:
        out = np.ones(cache.tree.num_entries, dtype=bool)
        cache.adc[code_str] = out
        return out

    print(f"      [READ waveform] {code_str}", flush=True)
    try:
        wf = cache.tree[br].array(library="ak")
        baseline = ak.mean(wf[:, :30], axis=1)
        wf_bl = wf - baseline
        peak = ak.max(wf_bl, axis=1)
        min_adc = ak.min(wf_bl, axis=1)
        out = ak.to_numpy((peak >= AMP_THRESHOLD) & (min_adc >= MIN_ADC_CUT))
    except Exception as e:
        print(f"      [WARN] ADC mask failed for {code_str}: {e}. Using all events.")
        out = np.ones(cache.tree.num_entries, dtype=bool)

    cache.adc[code_str] = out
    return out

# ============================================================
# Fitting and plotting
# ============================================================
def gaussian_peak(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def fit_delta_t(dt: np.ndarray, nbins: int, dt_halfwidth: float):
    dt = np.asarray(dt, dtype=float)
    dt = dt[np.isfinite(dt)]
    if len(dt) < MIN_EVENTS:
        return None

    raw_mean = float(np.mean(dt))
    raw_std = float(np.std(dt))
    if not np.isfinite(raw_std) or raw_std <= 0:
        return None

    # Use a compact, robust fit window centered on the observed peak.
    halfwin = min(max(4.0 * raw_std, 0.25), dt_halfwidth)
    xmin = raw_mean - halfwin
    xmax = raw_mean + halfwin
    dt_fit = dt[(dt >= xmin) & (dt <= xmax)]

    if len(dt_fit) < MIN_EVENTS:
        # Fallback to the central 98% range.
        lo, hi = np.percentile(dt, [1.0, 99.0])
        xmin, xmax = float(lo), float(hi)
        dt_fit = dt[(dt >= xmin) & (dt <= xmax)]

    if len(dt_fit) < MIN_EVENTS or xmin >= xmax:
        return None

    bins = np.linspace(xmin, xmax, nbins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    h, _ = np.histogram(dt_fit, bins=bins)
    if h.max() == 0:
        return None
    h_norm = h / h.max()

    try:
        p0 = [float(np.mean(dt_fit)), max(float(np.std(dt_fit)), 0.01)]
        bounds = ([xmin - 1.0, 1e-4], [xmax + 1.0, 10.0])
        popt, _ = curve_fit(gaussian_peak, centers, h_norm, p0=p0, bounds=bounds, maxfev=20000)
        mu = float(popt[0])
        sigma = abs(float(popt[1]))
    except Exception as e:
        print(f"      [WARN] Gaussian fit failed: {e}. Using raw window stats.")
        mu = float(np.mean(dt_fit))
        sigma = float(np.std(dt_fit))

    x_smooth = np.linspace(xmin, xmax, 600)
    y_gauss = gaussian_peak(x_smooth, mu, sigma)

    return {
        "dt_raw": dt.astype(np.float32),
        "dt_fit": dt_fit.astype(np.float32),
        "n": int(len(dt)),
        "n_fit": int(len(dt_fit)),
        "mu": mu,
        "sigma": sigma,
        "fwhm": 2.355 * sigma,
        "time_err": sigma / np.sqrt(len(dt)),
        "xmin": float(xmin),
        "xmax": float(xmax),
        "centers": centers,
        "hist_norm": h_norm,
        "x_smooth": x_smooth,
        "y_gauss": y_gauss,
    }


def particle_label(particle_type: Optional[str]) -> str:
    if particle_type is None:
        return "All"
    if particle_type.lower() == "electron":
        return r"$e^{+}$"
    return particle_type.capitalize()


def make_pair_plot(res: Dict, out_pdf: str, particle_type: Optional[str], suffix: str):
    fig, ax = plt.subplots(figsize=(10.5, 8.5))

    color = FAMILY_COLORS.get(res["family"], "#6A85C3")
    centers = res["centers"]
    hist_norm = res["hist_norm"]

    ax.fill_between(centers, hist_norm, step="mid", alpha=0.30, color=color, linewidth=0)
    ax.step(centers, hist_norm, where="mid", lw=2.2, color=color, label="Data")
    ax.plot(
        res["x_smooth"],
        res["y_gauss"],
        color="red",
        lw=2.8,
        label=(
            "Gaussian Fit\n"
            rf"$\mu$ = {res['mu']:+.3f} ns\n"
            rf"$\sigma$ = {res['sigma'] * 1000.0:.1f} ps"
        ),
    )

    ax.set_xlabel(
        rf"$t_{{\mathrm{{{res['ch_right']}}}}} - t_{{\mathrm{{{res['ch_left']}}}}}$ [ns]",
        fontsize=24,
        loc="right",
    )
    ax.set_ylabel("A.U.", fontsize=28, loc="top")
    ax.set_xlim(res["xmin"], res["xmax"])
    ax.set_ylim(0, 1.22)

    hep.cms.label(
        ax=ax,
        exp="CaloX",
        data=False,
        llabel="",
        rlabel=f"40 GeV {particle_label(particle_type)}",
        fontsize=22,
    )

    ax.text(
        0.05,
        0.94,
        f"{FAMILY_DISPLAY.get(res['family'], res['family'])}\n",
        #f"{res['run_label']} | {res['y_group']}\n"
        #f"{suffix}, N = {res['n']}",
        transform=ax.transAxes,
        fontsize=18,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=20,
        handlelength=1.8,
        borderpad=0.5,
        labelspacing=0.35,
    )
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.minorticks_on()
    ax.tick_params(which="major", direction="in", top=True, right=True, length=8, labelsize=20)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=4)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.91, bottom=0.13)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


def make_run_summary_plots(all_results: List[Dict], outdir: str, particle_type: Optional[str]):
    if not all_results:
        return

    grouped: Dict[Tuple[str, str, str], List[Dict]] = {}
    for r in all_results:
        grouped.setdefault((r["run_label"], r["y_group"], r["family"]), []).append(r)

    for (run_label, y_group, family), records in sorted(grouped.items()):
        records = sorted(records, key=lambda x: (x["ch_left"], x["ch_right"]))
        labels = [f"{r['ch_left']}↔{r['ch_right']}" for r in records]
        sigmas_ps = np.array([r["sigma"] for r in records]) * 1000.0
        errs_ps = np.array([r["time_err"] for r in records]) * 1000.0
        x = np.arange(len(records))

        fig, ax = plt.subplots(figsize=(max(12, 0.55 * len(records)), 7.5))
        color = FAMILY_COLORS.get(family, "black")
        ax.errorbar(x, sigmas_ps, yerr=errs_ps, fmt="o", color=color, capsize=4, markersize=7)
        ax.axhline(np.nanmean(sigmas_ps), color=color, ls="--", lw=2.0,
                   label=rf"Mean $\sigma_{{\Delta t}}$ = {np.nanmean(sigmas_ps):.0f} ps")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
        ax.set_ylabel(r"$\sigma_{\Delta t}$ [ps]", loc="top")
        ax.set_xlabel("Channel pair", loc="right")
        ax.set_title(f"{FAMILY_DISPLAY.get(family, family)} | {run_label} | {y_group}", loc="left", fontsize=18)
        ax.legend(frameon=True, fontsize=14)
        ax.grid(True, axis="y", ls="--", alpha=0.25)
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        hep.cms.label(ax=ax, exp="CaloX", data=False, llabel="", rlabel=f"40 GeV {particle_label(particle_type)}", fontsize=18)

        run_dir = os.path.join(outdir, run_label, family)
        os.makedirs(run_dir, exist_ok=True)
        fig.savefig(os.path.join(run_dir, f"summary_sigma_{family}.pdf"), bbox_inches="tight", dpi=300)
        plt.close(fig)

# ============================================================
# File / run helpers
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
        run_num = int(mrun.group(1)) if mrun else 10**9
        mts = re.search(r"_(\d{11,12})(?:_|\.|$)", base)
        ts = int(mts.group(1)) if mts else 10**18
        return run_num, ts, base
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


def validate_yconfigs_against_grids():
    by_family = {}
    for family, grid in GRID_BY_FAMILY.items():
        by_family[family] = {ch for row in grid for ch in row if ch is not None}

    missing = []
    for y_group, fams in Y_CONFIGS.items():
        for family, cfg in fams.items():
            for ch in cfg["channels"]:
                if ch not in by_family.get(family, set()):
                    missing.append((y_group, family, ch))

    if missing:
        print("[WARN] Some Y_CONFIG channels are not present in the full 6 mm grid:")
        for y_group, family, ch in missing:
            print(f"       {y_group:6s} {family:8s} ch {ch}")


def pairs_for_run(run_label: str, families: List[str]) -> List[Tuple[str, str, str, str, float, float]]:
    y_group = y_group_for_run(run_label)
    if y_group is None:
        return []

    out = []
    cfg = Y_CONFIGS[y_group]
    for family in FAMILY_ORDER:
        if family not in families:
            continue
        if family not in cfg:
            continue
        channels = cfg[family]["channels"]
        if len(channels) < 2:
            continue
        tmin = cfg[family]["tmin"]
        tmax = cfg[family]["tmax"]
        for ch_left, ch_right in combinations(channels, 2):
            out.append((y_group, family, ch_left, ch_right, tmin, tmax))
    return out

# ============================================================
# Core processing
# ============================================================
def process_pair(
    cache: FileCache,
    pid_mask: np.ndarray,
    run_label: str,
    y_group: str,
    family: str,
    ch_left: str,
    ch_right: str,
    tmin: float,
    tmax: float,
    args,
):
    t_left = compute_tfinal_6mm(cache, ch_left, abs_tfinal=(not args.signed_tfinal))
    t_right = compute_tfinal_6mm(cache, ch_right, abs_tfinal=(not args.signed_tfinal))

    if t_left is None or t_right is None:
        print(f"    [SKIP] {family} {ch_left}-{ch_right}: missing timing")
        return None

    combined = pid_mask.copy()
    if args.use_adc_cut:
        combined = combined & compute_adc_mask(cache, ch_left) & compute_adc_mask(cache, ch_right)

    if len(t_left) != len(combined) or len(t_right) != len(combined):
        print(f"    [SKIP] {family} {ch_left}-{ch_right}: array/mask length mismatch")
        return None

    tl = t_left[combined]
    tr = t_right[combined]
    good = np.isfinite(tl) & np.isfinite(tr)
    tl = tl[good]
    tr = tr[good]

    if not args.no_time_window:
        win = (tl >= tmin) & (tl <= tmax) & (tr >= tmin) & (tr <= tmax)
        tl = tl[win]
        tr = tr[win]

    if len(tl) < args.min_events:
        print(f"    [SKIP] {family} {ch_left}-{ch_right}: N={len(tl)} < {args.min_events}")
        return None

    dt = tr - tl
    fit = fit_delta_t(dt, nbins=args.nbins, dt_halfwidth=args.dt_halfwidth)
    if fit is None:
        print(f"    [SKIP] {family} {ch_left}-{ch_right}: fit failed")
        return None

    res = dict(
        run_label=run_label,
        y_group=y_group,
        family=family,
        ch_left=ch_left,
        ch_right=ch_right,
        tmin=tmin,
        tmax=tmax,
        suffix=args.suffix,
        signed_tfinal=args.signed_tfinal,
        **fit,
    )

    print(
        f"    [FIT] {family:7s} {ch_left}↔{ch_right}: "
        f"N={res['n']:5d}, mu={res['mu']:+.4f} ns, "
        f"sigma={res['sigma']:.4f} ns ({res['sigma']*1000.0:.1f} ps)"
    )
    return res


def write_summary_tables(results: List[Dict], outdir: str, pid: Optional[str], suffix: str):
    safe_suffix = suffix.strip("_")
    csv_path = os.path.join(outdir, f"DeltaT_6mm_summary_{pid or 'all'}_{safe_suffix}.csv")
    txt_path = csv_path.replace(".csv", ".txt")

    fields = [
        "run_label", "y_group", "family", "ch_left", "ch_right",
        "n", "n_fit", "mu_ns", "sigma_ns", "sigma_ps", "fwhm_ns", "time_err_ps",
        "tmin", "tmax", "suffix", "signed_tfinal",
    ]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in results:
            w.writerow([
                r["run_label"], r["y_group"], r["family"], r["ch_left"], r["ch_right"],
                r["n"], r["n_fit"], f"{r['mu']:.6f}", f"{r['sigma']:.6f}",
                f"{r['sigma']*1000.0:.3f}", f"{r['fwhm']:.6f}", f"{r['time_err']*1000.0:.3f}",
                f"{r['tmin']:.3f}", f"{r['tmax']:.3f}", r["suffix"], int(r["signed_tfinal"]),
            ])

    with open(txt_path, "w") as f:
        hdr = (
            f"{'Run':<24} | {'Y':<5} | {'Family':<8} | {'First':<6} | {'Second':<6} | "
            f"{'N':>7} | {'mu [ns]':>10} | {'sigma [ps]':>12} | {'FWHM [ns]':>10}"
        )
        sep = "=" * len(hdr)
        f.write(sep + "\n")
        f.write(hdr + "\n")
        f.write(sep + "\n")
        for r in results:
            f.write(
                f"{r['run_label']:<24} | {r['y_group']:<5} | {r['family']:<8} | "
                f"{r['ch_left']:<6} | {r['ch_right']:<6} | {r['n']:>7d} | "
                f"{r['mu']:>+10.4f} | {r['sigma']*1000.0:>12.2f} | {r['fwhm']:>10.4f}\n"
            )

    print(f"\n[TABLE] Saved: {csv_path}")
    print(f"[TABLE] Saved: {txt_path}")

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="6 mm Delta-T pair timing-resolution script.")
    parser.add_argument("--base-dir", default=BASE_DIR, help="Directory containing *_converted_timingskim.root files.")
    parser.add_argument("--ana-files", nargs="+", default=None, help="Explicit input ROOT files.")
    parser.add_argument("--ana-glob", default=None, help="Glob for input ROOT files.")
    parser.add_argument("--tree", default=TREE_NAME, help="ROOT tree name.")
    parser.add_argument("--outdir", default="DeltaT_6mm_pairs", help="Output directory.")
    parser.add_argument("--pid", default="electron", choices=["muon", "pion", "electron", "proton"], help="PID selection.")
    parser.add_argument("--no-pid-cut", action="store_true", help="Disable PID cut.")
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX, help="Timing suffix, e.g. _LP2_50 or _t_peak.")
    parser.add_argument("--families", nargs="+", default=FAMILY_ORDER, choices=FAMILY_ORDER, help="Families to process.")
    parser.add_argument("--nbins", type=int, default=DEFAULT_NBINS, help="Delta-t histogram bins.")
    parser.add_argument("--min-events", type=int, default=MIN_EVENTS, help="Minimum events required per pair.")
    parser.add_argument("--dt-halfwidth", type=float, default=2.0, help="Max half-width of the delta-t fit window [ns].")
    parser.add_argument("--use-adc-cut", action="store_true", help="Enable per-channel ADC cuts. Default is OFF.")
    parser.add_argument("--no-time-window", action="store_true", help="Do not require both channels to be inside the family t_final window.")
    parser.add_argument("--signed-tfinal", action="store_true", help="Use signed t_final. Default uses abs(t_final).")
    parser.add_argument("--skip-summary-plots", action="store_true", help="Skip summary sigma plots.")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    validate_yconfigs_against_grids()

    files = resolve_files(args)
    if not files:
        raise SystemExit("[FATAL] No input files found. Check --base-dir, --ana-files, or --ana-glob.")

    print("\n[INIT] 6 mm Delta-T pair timing")
    print(f"[INIT] files        : {len(files)}")
    print(f"[INIT] outdir       : {args.outdir}")
    print(f"[INIT] pid          : {'OFF' if args.no_pid_cut else args.pid}")
    print(f"[INIT] suffix       : {args.suffix}")
    print(f"[INIT] families     : {args.families}")
    print(f"[INIT] abs(t_final) : {'OFF' if args.signed_tfinal else 'ON'}")
    print(f"[INIT] ADC cut      : {'ON' if args.use_adc_cut else 'OFF'}")
    print(f"[INIT] time window  : {'OFF' if args.no_time_window else 'ON'}")

    all_results: List[Dict] = []

    for idx, fpath in enumerate(files, start=1):
        run_label = run_label_from_path(fpath)
        pairs = pairs_for_run(run_label, args.families)
        if not pairs:
            print(f"\n[SKIP FILE {idx}/{len(files)}] {os.path.basename(fpath)}: run not in RUN_TO_YGROUP or no requested families")
            continue

        print(f"\n[FILE {idx}/{len(files)}] {os.path.basename(fpath)}")
        print(f"  run_label : {run_label}")
        print(f"  y_group   : {pairs[0][0]}")
        print(f"  pairs     : {len(pairs)}")

        try:
            uf = uproot.open(fpath)
            tree = uf[args.tree]
        except Exception as e:
            print(f"  [ERROR] Could not open file/tree: {e}")
            continue

        try:
            cache = FileCache(tree, args.suffix)
            pid_mask = np.ones(tree.num_entries, dtype=bool) if args.no_pid_cut else compute_pid_mask(tree, args.pid)
            print(f"  [MASK] events passing PID/global cuts: {int(pid_mask.sum())}/{len(pid_mask)}")

            for y_group, family, ch_left, ch_right, tmin, tmax in pairs:
                print(f"  [START] {family} {ch_left}-{ch_right}", flush=True)
                res = process_pair(cache, pid_mask, run_label, y_group, family, ch_left, ch_right, tmin, tmax, args)
                if res is None:
                    continue

                all_results.append(res)

                # Keep file names channel-only, but avoid overwrites by using run/family subdirectories.
                pair_dir = os.path.join(args.outdir, run_label, family)
                os.makedirs(pair_dir, exist_ok=True)
                out_pdf = os.path.join(pair_dir, f"{ch_left}_{ch_right}.pdf")
                make_pair_plot(res, out_pdf, args.pid if not args.no_pid_cut else None, args.suffix)
                print(f"    [SAVE] {out_pdf}")

        finally:
            try:
                uf.close()
            except Exception:
                pass

    if all_results:
        write_summary_tables(all_results, args.outdir, args.pid if not args.no_pid_cut else None, args.suffix)
        if not args.skip_summary_plots:
            make_run_summary_plots(all_results, args.outdir, args.pid if not args.no_pid_cut else None)
    else:
        raise SystemExit("[FATAL] No successful pair fits. Check branch names, timing windows, and cuts.")

    print("\n[DONE]")
    print(f"[DONE] Output directory: {args.outdir}")
    print(f"[DONE] Total fitted pairs: {len(all_results)}")


if __name__ == "__main__":
    main()
