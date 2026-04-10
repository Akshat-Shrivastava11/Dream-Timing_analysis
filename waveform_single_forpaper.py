#!/usr/bin/env python3
"""
waveform_per_page_t50.py
=========================
For each (thickness × particle × family) combination:
  - Apply PID + ADC amplitude cuts
  - Select up to N_WAVEFORMS events (those whose t50 is closest to the modal t50)
  - Write one PDF where EACH PAGE shows ONE baseline-subtracted waveform
    with the x-axis shifted so that t50 = 0  (window: -TWINDOW_LEFT … +TWINDOW_RIGHT ns)

Output layout:
  <outdir>/
    <thickness>/
      <particle>/
        Waveforms_<family>_<thickness>_<particle>_<N>events.pdf

Usage
-----
python waveform_per_page_t50.py \\
    --outdir ./WaveformPages \\
    --n-waveforms 100 \\
    --pid electron          # optional override; otherwise all particles in RUN_FILES are run
"""

import os, re, argparse
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
BASELINE_BINS = 30            # leading bins used for baseline
TIMING_SUFFIX = "_LP2_50"     # branch suffix storing t50 [ns]
AMP_THRESHOLD = 100.0         # min baseline-subtracted peak  [ADC]
MIN_ADC_CUT   = -100.0        # max allowed trough (rejects saturated / noisy)
N_WAVEFORMS   = 100           # waveforms per combination

# x-axis window around t50 = 0
TWINDOW_LEFT  = 15.0          # ns shown before t50
TWINDOW_RIGHT = 25.0          # ns shown after  t50

# =============================================================================
# CHANNEL CODES  (Board | Group | Channel, zero-padded to 3 digits)
# =============================================================================
CHANNELS_3MM = {"Quartz": "104", "Plastic": "010", "Scintillator": "107"}
CHANNELS_6MM = {"Quartz": "604", "Plastic": "606", "Scintillator": "615"}
MCP1_CODE    = "037"   # Board 0, Group 3, Channel 7

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
    "MCP1":         "MCP1 (Reference)",
}

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
    """'104'  →  'DRS_Board1_Group0_Channel4'"""
    s = str(code).zfill(3)
    return f"DRS_Board{s[0]}_Group{s[1]}_Channel{s[2]}"


def run_label(path):
    m = re.search(r"(run\d+)", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


def display_particle(p):
    if p.lower() == "electron":      return "Positron"
    if p.lower() == "electron_90deg": return "Positron (90°)"
    return p.capitalize()


def modal_t50(t50_values):
    """Return the bin-centre of the tallest bin in a 100-bin histogram."""
    if len(t50_values) == 0:
        return np.nan
    hist, edges = np.histogram(t50_values, bins=100)
    peak_idx    = np.argmax(hist)
    return 0.5 * (edges[peak_idx] + edges[peak_idx + 1])

# =============================================================================
# COLLECT VALID EVENTS for one (file, family_branch)
# =============================================================================

def collect_valid_events(fpath, branch_code, particle, pid_mask_cache=None):
    """
    Returns list of (event_index, t50_ns, waveform_array) tuples that pass
    PID + ADC cuts and have a valid finite t50 > 0.

    pid_mask_cache: pass a pre-computed mask to avoid re-reading PID branches.
    """
    br      = code_to_branch(branch_code)
    t50_br  = br + TIMING_SUFFIX
    records = []

    try:
        with uproot.open(fpath) as f:
            tree = f[TREE_NAME]

            if br not in tree.keys() or t50_br not in tree.keys():
                print(f"  [SKIP] Branch {br} or {t50_br} not found in {os.path.basename(fpath)}")
                return records

            # ── PID mask ──────────────────────────────────────────────────────
            if pid_mask_cache is not None:
                pid_mask = pid_mask_cache
            else:
                pid_mask = compute_pid_mask(tree, particle)

            # ── read waveforms & t50 ─────────────────────────────────────────
            waves_ak = tree[br].array(library="ak")
            baseline = ak.mean(waves_ak[:, :BASELINE_BINS], axis=1)
            w_sub_ak = waves_ak - baseline          # baseline-subtracted (ak)

            peak_ak  = ak.max(w_sub_ak, axis=1)
            trough_ak= ak.min(w_sub_ak, axis=1)

            adc_mask = ak.to_numpy(
                (peak_ak  >=  AMP_THRESHOLD) &
                (trough_ak >= MIN_ADC_CUT)
            )

            t50_arr  = tree[t50_br].array(library="np")
            t50_valid= np.isfinite(t50_arr) & (t50_arr > 0)

            combined = pid_mask & adc_mask & t50_valid
            good_idx = np.where(combined)[0]

            print(f"  [INFO] {os.path.basename(fpath)} | branch {br} "
                  f"| PID passed {pid_mask.sum()}, ADC {adc_mask.sum()}, "
                  f"t50 valid {t50_valid.sum()}, combined {combined.sum()}")

            # read waveforms for good events only (uproot entry slicing)
            w_np = ak.to_numpy(w_sub_ak)   # shape (N_entries, N_bins)

            for idx in good_idx:
                records.append((idx, float(t50_arr[idx]), w_np[idx].copy()))

    except Exception as e:
        print(f"  [ERROR] {fpath}: {e}")

    return records

# =============================================================================
# DRAW ONE PAGE: single waveform centred at t50 = 0
# =============================================================================

def draw_single_waveform(ax, waveform, t50, event_idx, run_name,
                         family, particle, thickness, page_num, total):
    """
    Plot one baseline-subtracted waveform on *ax*.
    x-axis is time relative to t50 (t50 = 0 by construction).
    """
    n_bins  = len(waveform)
    t_raw   = np.arange(n_bins) * TIME_PER_BIN          # absolute time [ns]
    t_rel   = t_raw - t50                                # shift so t50 = 0

    # ── restrict to display window ────────────────────────────────────────────
    win_mask = (t_rel >= -TWINDOW_LEFT) & (t_rel <= TWINDOW_RIGHT)
    t_plot   = t_rel[win_mask]
    w_plot   = waveform[win_mask]

    color = FAMILY_COLORS.get(family, "tab:gray")

    ax.plot(t_plot, w_plot, color=color, lw=1.8)
    ax.axvline(0.0, color="black", ls="--", lw=1.5, alpha=0.75,
               label=r"$t_{50} = 0$")

    ax.set_xlim(-TWINDOW_LEFT, TWINDOW_RIGHT)
    ax.set_xlabel(r"$t - t_{50}$  [ns]", fontsize=14)
    ax.set_ylabel("ADC counts (baseline subtracted)", fontsize=13)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", labelsize=12, length=7,
                   direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor", length=4,
                   direction="in", top=True, right=True)
    ax.legend(fontsize=11, frameon=False, loc="upper right")

    # ── CMS label / fallback title ────────────────────────────────────────────
    disp_p   = display_particle(particle)
    r_label  = f"40 GeV {disp_p} | {thickness} | {FAMILY_DISPLAY.get(family, family)}"
    if HEP_AVAILABLE:
        try:
            hep.cms.label(ax=ax, exp="CaloX", data=False,
                          rlabel=r_label, llabel="")
        except Exception:
            ax.set_title(f"CaloX — {r_label}", fontsize=11)
    else:
        ax.set_title(f"CaloX — {r_label}", fontsize=11)

    # ── info box (top-left) ───────────────────────────────────────────────────
    info = (f"Run: {run_name}   Event: {event_idx}\n"
            f"$t_{{50}}$ = {t50:.3f} ns   |   "
            f"Waveform {page_num}/{total}")
    ax.text(0.02, 0.97, info,
            transform=ax.transAxes, fontsize=10,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="gray", alpha=0.85))

# =============================================================================
# PROCESS ONE COMBINATION  (thickness × particle × family)
# =============================================================================

def process_combination(thickness, particle, files, family, branch_code, outdir, n_waves):
    print(f"\n{'='*60}")
    print(f"  Processing: {thickness} | {particle} | {family}  (code {branch_code})")
    print(f"{'='*60}")

    os.makedirs(outdir, exist_ok=True)

    # ── gather events from all files ──────────────────────────────────────────
    all_records = []   # list of (fpath, ev_idx, t50, waveform)
    for fpath in files:
        recs = collect_valid_events(fpath, branch_code, particle)
        for (ev_idx, t50, wf) in recs:
            all_records.append((fpath, ev_idx, t50, wf))

    if not all_records:
        print(f"  [WARN] No valid events found. Skipping.")
        return

    # ── pick n_waves events closest to the modal t50 ─────────────────────────
    all_t50s = np.array([r[2] for r in all_records])
    mode     = modal_t50(all_t50s)
    print(f"  [INFO] Total valid events: {len(all_records)}  |  modal t50 = {mode:.3f} ns")

    distances   = np.abs(all_t50s - mode)
    sorted_idx  = np.argsort(distances)
    chosen_idx  = sorted_idx[:n_waves]
    chosen      = [all_records[i] for i in chosen_idx]

    print(f"  [INFO] Selected {len(chosen)} waveforms for PDF.")

    # ── write PDF ─────────────────────────────────────────────────────────────
    safe_particle = particle.replace("_", "")
    pdf_path = os.path.join(
        outdir,
        f"Waveforms_{family}_{thickness}_{safe_particle}_{len(chosen)}events.pdf"
    )

    with PdfPages(pdf_path) as pdf:
        for page, (fpath, ev_idx, t50, wf) in enumerate(chosen, start=1):
            rl  = run_label(fpath)
            fig, ax = plt.subplots(figsize=(12, 6))

            draw_single_waveform(
                ax=ax,
                waveform=wf,
                t50=t50,
                event_idx=ev_idx,
                run_name=rl,
                family=family,
                particle=particle,
                thickness=thickness,
                page_num=page,
                total=len(chosen),
            )

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"  [OK] Wrote {len(chosen)}-page PDF: {pdf_path}")

# =============================================================================
# DRIVER
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="One waveform per page, centred at t50, with PID cuts."
    )
    ap.add_argument("--outdir",      default="./paperwaveforms",
                    help="Root output directory")
    ap.add_argument("--n-waveforms", type=int, default=N_WAVEFORMS,
                    help="Number of waveforms (pages) per combination")
    ap.add_argument("--pid",         default=None,
                    choices=["muon", "pion", "electron", "electron_90deg"],
                    help="Run only this particle type (default: all)")
    ap.add_argument("--thickness",   default=None,
                    choices=["3mm", "6mm"],
                    help="Run only this thickness (default: both)")
    ap.add_argument("--family",      default=None,
                    choices=["Quartz", "Plastic", "Scintillator", "MCP1"],
                    help="Run only this sensor family (default: all)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    for thickness, particles in RUN_FILES.items():
        if args.thickness and thickness != args.thickness:
            continue

        chan_map = CHANNELS_3MM if thickness == "3mm" else CHANNELS_6MM

        # Build list of families to process for this thickness
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
                    thickness    = thickness,
                    particle     = particle,
                    files        = raw_files,
                    family       = family,
                    branch_code  = branch_code,
                    outdir       = subdir,
                    n_waves      = args.n_waveforms,
                )

    print("\n[DONE] All waveform PDFs written.")


if __name__ == "__main__":
    main()