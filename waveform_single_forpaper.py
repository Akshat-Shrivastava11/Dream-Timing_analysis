#!/usr/bin/env python3
"""
waveform_per_page_t50.py
=========================
For each (thickness × particle × family) combination produces:

  1. PDF  – one page per waveform, x-axis centred at t50 = 0
  2. INDEX CSV  – one row per waveform with all metadata
  3. WAVEFORM CSV  – full ADC + time data for every selected waveform
                     so any plot can be reproduced without the ROOT file

Output structure
----------------
<outdir>/
  <thickness>/
    <particle>/
      Waveforms_<family>_<thickness>_<particle>_<N>events.pdf
      Waveforms_<family>_<thickness>_<particle>_<N>events_index.csv
      Waveforms_<family>_<thickness>_<particle>_<N>events_waveforms.csv

INDEX CSV columns
-----------------
  page, run_label, run_number, event_index, thickness, particle, family,
  channel_code, branch_name, beam_energy_GeV, t50_ns,
  waveform_peak_adc, waveform_trough_adc

WAVEFORM CSV columns
--------------------
  page, run_label, event_index, bin_index,
  time_ns (absolute), time_rel_ns (relative to t50, i.e. centred at 0),
  adc (baseline-subtracted)

Usage
-----
python waveform_per_page_t50.py \\
    --outdir ./WaveformPages \\
    --n-waveforms 100 \\
    --pid electron       # optional: restrict particle type
    --thickness 3mm      # optional: restrict thickness
    --family Quartz      # optional: restrict sensor family
"""

import os, re, csv, argparse
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ── optional CMS style ────────────────────────────────────────────────────────
try:
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    HEP_AVAILABLE = True
except ImportError:
    HEP_AVAILABLE = False

# =============================================================================
# GLOBAL CONFIG
# =============================================================================
TREE_NAME     = "EventTree"
TIME_PER_BIN  = 0.2          # ns / DRS4 bin
BASELINE_BINS = 30            # leading bins used for baseline estimate
TIMING_SUFFIX = "_LP2_50"     # branch suffix storing t50 [ns]
AMP_THRESHOLD = 100.0         # min baseline-subtracted peak  [ADC]
MIN_ADC_CUT   = -100.0        # max allowed trough (rejects saturated / noisy)
N_WAVEFORMS   = 100           # default waveforms per combination

# x-axis display window around t50 = 0  (used in PDF only; CSV stores full waveform)
TWINDOW_LEFT  = 15.0          # ns before t50
TWINDOW_RIGHT = 25.0          # ns after  t50

# =============================================================================
# BEAM ENERGY MAP  (run number → beam energy in GeV)
# Add / edit entries here whenever you add new runs.
# Runs NOT listed fall back to DEFAULT_ENERGY.
# =============================================================================
DEFAULT_ENERGY = 40.0   # GeV

RUN_ENERGY_MAP = {
    # ── 3 mm runs ──────────────────────────────────────────────────
    1429: 40.0,   # 3mm pion
    1480: 40.0,   # 3mm muon
    1355: 40.0,   # 3mm electron
    1501: 40.0,   # 3mm electron 90 deg
    # ── 6 mm runs ──────────────────────────────────────────────────
    1474: 40.0,   # 6mm pion
    1509: 40.0,   # 6mm electron
    # ── add more below as needed ───────────────────────────────────
    # 1600: 20.0,
    # 1601: 60.0,
    # 1602: 100.0,
}

# =============================================================================
# CHANNEL CODES  (3-digit string: Board | Group | Channel)
# =============================================================================
CHANNELS_3MM = {"Quartz": "104", "Plastic": "010", "Scintillator": "107"}
CHANNELS_6MM = {"Quartz": "604", "Plastic": "606", "Scintillator": "615"}
MCP1_CODE    = "037"   # Board 0, Group 3, Channel 7  (reference MCP)

FAMILY_COLORS = {
    "Quartz":       "tab:orange",
    "Plastic":      "tab:blue",
    "Scintillator": "tab:red",
    "MCP1":         "tab:green",
}

FAMILY_DISPLAY = {
    "Quartz":       "FSHA (Fused-silica)",
    "Plastic":      "Toray PJR-FB750 (Plastic)",
    "Scintillator": "SCSF-81J (Scintillator)",
    "MCP1":         "MCP1 Reference",
}
# =============================================================================
# RUN FILE MAP  – edit paths to match your NTuple location
# =============================================================================

# =============================================================================
# RUN FILE MAP  – edit paths to match your NTuple location
# =============================================================================
RUN_FILES = {
    "3mm": {
        "pion":          ["/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1429_250926183919_converted_timingskim.root"],
        "muon":          ["/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1480_250928004120_converted_timingskim.root"],
        "electron":      ["/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1355_250924165834_converted_timingskim.root"],
        "electron_90deg":["/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1501_250928105227_converted_timingskim.root"],
    },
    "6mm": {
        "pion":     ["/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1474_250927193729_converted_timingskim.root"],
        "muon":     [],
        "electron": ["/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/run1509_250928164817_converted_timingskim.root"],
    },
}
 
# =============================================================================
# PID  (mirrors PrecisionTiming_wcommentsfromDrA.py exactly)
# =============================================================================
PID_BRANCH_MAP = {
    "PSD":         "DRS_Board7_Group1_Channel1",
    "HoleVeto":    "DRS_Board7_Group1_Channel6",
    "NC":          "DRS_Board7_Group1_Channel7",
    "T3":          "DRS_Board7_Group2_Channel0",
    "T4":          "DRS_Board7_Group2_Channel1",
    "KT1":         "DRS_Board7_Group2_Channel2",
    "KT2":         "DRS_Board7_Group2_Channel3",
    "TTUMuonVeto": "DRS_Board7_Group2_Channel4",
    "Cer474":      "DRS_Board7_Group2_Channel5",
    "Cer519":      "DRS_Board7_Group2_Channel6",
    "Cer537":      "DRS_Board7_Group2_Channel7",
}
 
_SVC_CUTS = {
    "HoleVeto":    (100, 350, -2e3,   "Sum"),
    "PSD":         (100, 400, -3500., "Sum"),
    "TTUMuonVeto": (200, 400, -2e3,   "Sum"),
    "Cer474":      (800, 900, -2000., "Sum"),
    "Cer519":      (450, 550, -1000., "Sum"),
    "Cer537":      (400, 500, -500.,  "Sum"),
}
 
_PARTICLE_REQS = {
    "muon":          {"TTUMuonVeto": True,  "PSD": False},
    "pion":          {"TTUMuonVeto": False, "PSD": False,
                      "Cer474": True, "Cer519": True, "Cer537": True},
    "electron":      {"TTUMuonVeto": False, "PSD": True,
                      "Cer474": True, "Cer519": True, "Cer537": True},
    "electron_90deg":{"TTUMuonVeto": False, "PSD": True,
                      "Cer474": True, "Cer519": True, "Cer537": True},
}
 
 
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
            continue
    return mask
 
# =============================================================================
# HELPERS
# =============================================================================
 
def code_to_branch(code):
    s = str(code).zfill(3)
    return f"DRS_Board{s[0]}_Group{s[1]}_Channel{s[2]}"
 
def run_number_from_path(path):
    m = re.search(r"run(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None
 
def run_label_from_path(path):
    m = re.search(r"(run\d+)", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]
 
def get_beam_energy(run_number):
    return RUN_ENERGY_MAP.get(run_number, DEFAULT_ENERGY) if run_number is not None else DEFAULT_ENERGY
 
def display_particle(p):
    if p.lower() == "electron":       return "Positron"
    if p.lower() == "electron_90deg": return "Positron (90°)"
    return p.capitalize()
 
def modal_t50(t50_values):
    arr = np.asarray(t50_values)
    if len(arr) == 0:
        return np.nan
    hist, edges = np.histogram(arr, bins=100)
    peak_idx = int(np.argmax(hist))
    return 0.5 * (edges[peak_idx] + edges[peak_idx + 1])
 
# =============================================================================
# COLLECT VALID EVENTS
# =============================================================================
 
def collect_valid_events(fpath, branch_code, particle):
    """
    Returns list of dicts:
      { run_label, run_number, event_index, t50_ns, waveform }
    waveform is the FULL baseline-subtracted array (all bins, not windowed).
    """
    br     = code_to_branch(branch_code)
    t50_br = br + TIMING_SUFFIX
    rnum   = run_number_from_path(fpath)
    rl     = run_label_from_path(fpath)
    records = []
 
    try:
        with uproot.open(fpath) as f:
            tree = f[TREE_NAME]
 
            if br not in tree.keys():
                print(f"  [SKIP] Branch {br} not found in {os.path.basename(fpath)}")
                return records
            if t50_br not in tree.keys():
                print(f"  [SKIP] Timing branch {t50_br} not found in {os.path.basename(fpath)}")
                return records
 
            pid_mask  = compute_pid_mask(tree, particle)
            waves_ak  = tree[br].array(library="ak")
            baseline  = ak.mean(waves_ak[:, :BASELINE_BINS], axis=1)
            w_sub_ak  = waves_ak - baseline
            peak_ak   = ak.max(w_sub_ak, axis=1)
            trough_ak = ak.min(w_sub_ak, axis=1)
            adc_mask  = ak.to_numpy(
                (peak_ak  >= AMP_THRESHOLD) &
                (trough_ak >= MIN_ADC_CUT)
            )
            t50_arr = tree[t50_br].array(library="np")
            t50_ok  = np.isfinite(t50_arr) & (t50_arr > 0)
 
            combined = pid_mask & adc_mask & t50_ok
            good_idx = np.where(combined)[0]
 
            print(f"  [INFO] {os.path.basename(fpath)} | {br}"
                  f" | PID {pid_mask.sum()}  ADC {adc_mask.sum()}"
                  f"  t50-ok {t50_ok.sum()}  combined {combined.sum()}")
 
            w_np = ak.to_numpy(w_sub_ak)   # (N_entries, N_bins) – full waveform
            for idx in good_idx:
                records.append({
                    "run_label":   rl,
                    "run_number":  rnum,
                    "event_index": int(idx),
                    "t50_ns":      float(t50_arr[idx]),
                    "waveform":    w_np[idx].copy(),
                })
 
    except Exception as e:
        print(f"  [ERROR] {fpath}: {e}")
 
    return records
 
# =============================================================================
# DRAW ONE PAGE
# =============================================================================
 
def draw_single_waveform(ax, record, family, branch_code, particle, thickness,
                         page_num, total):
    wf     = record["waveform"]
    t50    = record["t50_ns"]
    rl     = record["run_label"]
    ev     = record["event_index"]
    rnum   = record["run_number"]
    energy = get_beam_energy(rnum)
 
    t_rel = np.arange(len(wf)) * TIME_PER_BIN - t50
    win   = (t_rel >= -TWINDOW_LEFT) & (t_rel <= TWINDOW_RIGHT)
 
    color = FAMILY_COLORS.get(family, "tab:gray")
    ax.plot(t_rel[win], wf[win], color=color, lw=1.8)
    ax.axvline(0.0, color="black", ls="--", lw=1.5, alpha=0.75, label=r"$t_{50} = 0$")
 
    ax.set_xlim(-TWINDOW_LEFT, TWINDOW_RIGHT)
    ax.set_xlabel(r"$t - t_{50}$  [ns]", fontsize=14)
    ax.set_ylabel("ADC counts (baseline subtracted)", fontsize=13)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", labelsize=12, length=7,
                   direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor", length=4,
                   direction="in", top=True, right=True)
    ax.legend(fontsize=11, frameon=False, loc="upper right")
 
    disp_p  = display_particle(particle)
    r_label = (f"{energy:.0f} GeV {disp_p} | {thickness} | "
               f"{FAMILY_DISPLAY.get(family, family)}")
    if HEP_AVAILABLE:
        try:
            hep.cms.label(ax=ax, exp="CaloX", data=False, rlabel=r_label, llabel="")
        except Exception:
            ax.set_title(f"CaloX — {r_label}", fontsize=11)
    else:
        ax.set_title(f"CaloX — {r_label}", fontsize=11)
 
    branch_name = code_to_branch(branch_code)
    info = (
        f"Run        : {rl}\n"
        f"Event      : {ev}\n"
        f"Channel    : {branch_code}  ({branch_name})\n"
        f"Family     : {family}  —  {FAMILY_DISPLAY.get(family, '')}\n"
        f"Beam energy: {energy:.0f} GeV\n"
        f"t\u2085\u2080        : {t50:.3f} ns\n"
        f"Waveform   : {page_num} / {total}"
    )
    ax.text(
        0.02, 0.97, info,
        transform=ax.transAxes, fontsize=9,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="white", edgecolor="gray", alpha=0.90),
    )
 
# =============================================================================
# CSV SCHEMA  – one row per DRS4 bin
# =============================================================================
CSV_HEADER = ["event", "run_num", "particle_type", "channel_num", "time_ns", "adc"]
 
# =============================================================================
# PROCESS ONE COMBINATION
# =============================================================================
 
def process_combination(thickness, particle, files, family, branch_code, outdir, n_waves):
    print(f"\n{'='*65}")
    print(f"  {thickness} | {particle} | {family}  (code {branch_code})")
    print(f"{'='*65}")
 
    os.makedirs(outdir, exist_ok=True)
 
    # ── gather all valid events ───────────────────────────────────────────────
    all_records = []
    for fpath in files:
        all_records.extend(collect_valid_events(fpath, branch_code, particle))
 
    if not all_records:
        print(f"  [WARN] No valid events — skipping.")
        return
 
    # ── select n_waves events closest to modal t50 ────────────────────────────
    all_t50s   = np.array([r["t50_ns"] for r in all_records])
    mode       = modal_t50(all_t50s)
    chosen_idx = np.argsort(np.abs(all_t50s - mode))[:n_waves]
    chosen     = [all_records[i] for i in chosen_idx]
 
    print(f"  [INFO] Total valid: {len(all_records)}  |  modal t50 = {mode:.3f} ns")
    print(f"  [INFO] Writing {len(chosen)} waveforms")
 
    safe_p   = particle.replace("_", "")
    stem     = f"Waveforms_{family}_{thickness}_{safe_p}_{len(chosen)}events"
    pdf_path = os.path.join(outdir, stem + ".pdf")
    csv_path = os.path.join(outdir, stem + ".csv")
 
    with PdfPages(pdf_path) as pdf, open(csv_path, "w", newline="") as csv_fh:
 
        writer = csv.DictWriter(csv_fh, fieldnames=CSV_HEADER)
        writer.writeheader()
 
        for page, record in enumerate(chosen, start=1):
            wf   = record["waveform"]
            t50  = record["t50_ns"]
            rnum = record["run_number"]
            ev   = record["event_index"]
 
            # time axis relative to t50  (t50 = 0)
            t_rel = np.arange(len(wf)) * TIME_PER_BIN - t50
 
            # apply the same window used in the plot so CSV and PDF match exactly
            win      = (t_rel >= -TWINDOW_LEFT) & (t_rel <= TWINDOW_RIGHT)
            t_window = t_rel[win]
            w_window = wf[win]
 
            # ── PDF page ──────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(12, 6))
            draw_single_waveform(
                ax=ax, record=record, family=family,
                branch_code=branch_code, particle=particle,
                thickness=thickness, page_num=page, total=len(chosen),
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
 
            # ── CSV: one row per bin IN THE PLOT WINDOW only ──────────────────
            # time_ns ranges from -TWINDOW_LEFT to +TWINDOW_RIGHT  (t50 = 0)
            # this matches the x-axis of the paired PDF page exactly
            rnum_str = str(rnum) if rnum is not None else ""
            for t_val, adc_val in zip(t_window, w_window):
                writer.writerow({
                    "event":         ev,
                    "run_num":       rnum_str,
                    "particle_type": particle,
                    "channel_num":   branch_code,
                    "time_ns":       f"{t_val:.4f}",
                    "adc":           f"{adc_val:.4f}",
                })
 
    if chosen:
        _t = np.arange(len(chosen[0]["waveform"])) * TIME_PER_BIN - chosen[0]["t50_ns"]
        n_bins_window = int(np.sum((_t >= -TWINDOW_LEFT) & (_t <= TWINDOW_RIGHT)))
    else:
        n_bins_window = 0
    print(f"  [OK] PDF → {pdf_path}")
    print(f"  [OK] CSV → {csv_path}")
    print(f"       ({len(chosen)} waveforms x {n_bins_window} windowed bins = "
          f"{len(chosen) * n_bins_window:,} rows  "
          f"[window: -{TWINDOW_LEFT} to +{TWINDOW_RIGHT} ns])")
 
# =============================================================================
# DRIVER
# =============================================================================
 
def main():
    ap = argparse.ArgumentParser(
        description=(
            "One waveform per PDF page, centred at t50, PID-selected. "
            "Each PDF is paired with a CSV: "
            "event | run_num | particle_type | channel_num | time_ns | adc"
        )
    )
    ap.add_argument("--outdir",      default="./paperwaveforms4_wcsv",
                    help="Root output directory  (default: ./paperwaveforms3_wcsv)")
    ap.add_argument("--n-waveforms", type=int, default=N_WAVEFORMS,
                    help=f"Waveforms per combination  (default: {N_WAVEFORMS})")
    ap.add_argument("--pid",       default=None,
                    choices=["muon", "pion", "electron", "electron_90deg"],
                    help="Restrict to one particle type  (default: all)")
    ap.add_argument("--thickness", default=None, choices=["3mm", "6mm"],
                    help="Restrict to one thickness       (default: both)")
    ap.add_argument("--family",    default=None,
                    choices=["Quartz", "Plastic", "Scintillator", "MCP1"],
                    help="Restrict to one sensor family   (default: all)")
    args = ap.parse_args()
 
    os.makedirs(args.outdir, exist_ok=True)
 
    for thickness, particles in RUN_FILES.items():
        if args.thickness and thickness != args.thickness:
            continue
 
        chan_map = CHANNELS_3MM if thickness == "3mm" else CHANNELS_6MM
        families = {**chan_map, "MCP1": MCP1_CODE}
 
        for particle, raw_files in particles.items():
            if args.pid and particle != args.pid:
                continue
            if not raw_files:
                print(f"[SKIP] No files configured for {thickness} | {particle}")
                continue
 
            for family, branch_code in families.items():
                if args.family and family != args.family:
                    continue
 
                subdir = os.path.join(args.outdir, thickness, particle)
                process_combination(
                    thickness   = thickness,
                    particle    = particle,
                    files       = raw_files,
                    family      = family,
                    branch_code = branch_code,
                    outdir      = subdir,
                    n_waves     = args.n_waveforms,
                )
 
    print("\n[DONE] All PDFs and CSVs written.")
 
 
if __name__ == "__main__":
    main()
 