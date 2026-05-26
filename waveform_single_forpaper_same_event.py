#!/usr/bin/env python3
"""
waveform_per_page_t50.py
=========================
For each (thickness × particle) combination produces:

  1. PDF  – one page per event, with 3 subplots side-by-side:
              Quartz | Plastic | Scintillator
             x-axis is absolute time [ns]. The display window is centred
             around the t50 of the first family with a valid pulse.
             A vertical dashed line marks each family's own t50 value.
  2. INDEX CSV  – one row per event (page)
  3. WAVEFORM CSV  – windowed ADC + time data for every selected event

Output structure
----------------
<outdir>/
  <thickness>/
    <particle>/
      Waveforms_<thickness>_<particle>_<N>events.pdf
      Waveforms_<thickness>_<particle>_<N>events_index.csv
      Waveforms_<thickness>_<particle>_<N>events_waveforms.csv

INDEX CSV columns
-----------------
  page, run_label, run_number, event_index, thickness, particle,
  beam_energy_GeV,
  t50_Quartz_ns, t50_Plastic_ns, t50_Scintillator_ns,
  peak_Quartz_adc, peak_Plastic_adc, peak_Scintillator_adc

WAVEFORM CSV columns
--------------------
  page, run_label, event_index, family, bin_index, time_ns, adc

Usage
-----
python waveform_per_page_t50.py \\
    --outdir ./WaveformPages \\
    --n-waveforms 100 \\
    --pid electron       # optional: restrict particle type
    --thickness 3mm      # optional: restrict thickness
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
# PAPER STYLE  (mirrors PrecisionTiming_paperplots3.py)
# =============================================================================
AXIS_LABEL_FONTSIZE   = 30
TICK_LABEL_FONTSIZE   = 24
CMS_LABEL_FONTSIZE    = 26
TITLE_FONTSIZE        = 26
LEGEND_FONTSIZE       = 20
ANNOTATION_FONTSIZE   = 22


def apply_paper_style():
    plt.rcParams.update({
        "figure.dpi":           120,
        "savefig.dpi":          300,
        "font.size":            24,
        "axes.labelsize":       AXIS_LABEL_FONTSIZE,
        "axes.titlesize":       TITLE_FONTSIZE,
        "xtick.labelsize":      TICK_LABEL_FONTSIZE,
        "ytick.labelsize":      TICK_LABEL_FONTSIZE,
        "legend.fontsize":      LEGEND_FONTSIZE,
        "font.weight":          "normal",
        "axes.labelweight":     "normal",
        "axes.titleweight":     "normal",
        "lines.linewidth":      2.8,
        "axes.linewidth":       1.6,
        "xtick.major.size":     10,
        "ytick.major.size":     10,
        "xtick.minor.size":     5,
        "ytick.minor.size":     5,
        "xtick.major.width":    1.6,
        "ytick.major.width":    1.6,
        "xtick.minor.width":    1.2,
        "ytick.minor.width":    1.2,
        "xtick.direction":      "in",
        "ytick.direction":      "in",
        "xtick.top":            True,
        "ytick.right":          True,
    })


# =============================================================================
# GLOBAL CONFIG
# =============================================================================
TREE_NAME     = "EventTree"
TIME_PER_BIN  = 0.2          # ns / DRS4 bin
BASELINE_BINS = 30            # leading bins used for baseline estimate
TIMING_SUFFIX = "_LP2_50"     # branch suffix storing t50 [ns]
AMP_THRESHOLD = 100.0         # min baseline-subtracted peak  [ADC]
MIN_ADC_CUT   = -100.0        # max allowed trough (rejects saturated / noisy)
N_WAVEFORMS   = 1000          # default waveforms per combination

# Display window around t50 — x-axis is absolute time [ns]
TWINDOW_LEFT  = 15.0          # ns before t50
TWINDOW_RIGHT = 25.0          # ns after  t50

# The 3 families shown as subplots on every page, left → right
SUBPLOT_FAMILIES = ["Quartz", "Plastic", "Scintillator"]

# =============================================================================
# BEAM ENERGY MAP
# =============================================================================
DEFAULT_ENERGY = 40.0

RUN_ENERGY_MAP = {
    1429: 80.0,
    1480: 170.0,
    1355: 80.0,
    1501: 40.0,
    1474: 80.0,
    1509: 40.0,
}

# =============================================================================
# CHANNEL CODES  (3-digit string: Board | Group | Channel)
# =============================================================================
CHANNELS_3MM = {"Quartz": "104", "Plastic": "010", "Scintillator": "107"}
CHANNELS_6MM = {"Quartz": "604", "Plastic": "606", "Scintillator": "615"}

# Match the color palette from PrecisionTiming_paperplots3.py
FAMILY_COLORS = {
    "Quartz":       "#f89c20",   # orange-gold
    "Plastic":      "#e42536",   # red
    "Scintillator": "#5790fc",   # blue
}

FAMILY_DISPLAY = {
    "Quartz":       "FSHA (Fused-silica)",
    "Plastic":      "Toray PJR-FB750 (Plastic)",
    "Scintillator": "SCSF-81J (Scintillator)",
}

# =============================================================================
# RUN FILE MAP
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
# PID
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
    if p.lower() == "electron":        return r"$e^{+}$"
    if p.lower() == "electron_90deg":  return r"$e^{+}$ (90°)"
    if p.lower() == "muon":            return r"$\mu$"
    if p.lower() == "pion":            return r"$\pi$"
    return p.capitalize()

def modal_t50(t50_values):
    arr = np.asarray(t50_values)
    if len(arr) == 0:
        return np.nan
    hist, edges = np.histogram(arr, bins=100)
    peak_idx = int(np.argmax(hist))
    return 0.5 * (edges[peak_idx] + edges[peak_idx + 1])

# =============================================================================
# COLLECT VALID EVENTS  (all 3 families simultaneously, per event)
# =============================================================================

def collect_valid_events_multi(fpath, chan_map, particle):
    """
    Returns a list of dicts, one per event that passes PID AND has at least
    one family with a valid pulse. Each dict:
      {
        run_label, run_number, event_index,
        ref_t50_ns,   # t50 of first SUBPLOT_FAMILY with a valid pulse
        families: {
          "Quartz":       { waveform, t50_ns, has_pulse },
          "Plastic":      { waveform, t50_ns, has_pulse },
          "Scintillator": { waveform, t50_ns, has_pulse },
        }
      }
    """
    rnum    = run_number_from_path(fpath)
    rl      = run_label_from_path(fpath)
    records = []

    try:
        with uproot.open(fpath) as f:
            tree      = f[TREE_NAME]
            n_entries = tree.num_entries
            pid_mask  = compute_pid_mask(tree, particle)

            # Load waveforms + t50 for all 3 families upfront
            family_data = {}
            for family in SUBPLOT_FAMILIES:
                code   = chan_map.get(family)
                br     = code_to_branch(code)
                t50_br = br + TIMING_SUFFIX

                missing = []
                if br not in tree.keys():
                    missing.append(br)
                if t50_br not in tree.keys():
                    missing.append(t50_br)
                if missing:
                    print(f"  [WARN] {family}: missing branches {missing} — blanked out")
                    family_data[family] = None
                    continue

                waves_ak  = tree[br].array(library="ak")
                baseline  = ak.mean(waves_ak[:, :BASELINE_BINS], axis=1)
                w_sub_ak  = waves_ak - baseline
                peak_np   = ak.to_numpy(ak.max(w_sub_ak,  axis=1))
                trough_np = ak.to_numpy(ak.min(w_sub_ak,  axis=1))
                t50_arr   = tree[t50_br].array(library="np")
                w_np      = ak.to_numpy(w_sub_ak)

                family_data[family] = {
                    "w_np":    w_np,
                    "t50_arr": t50_arr,
                    "peak":    peak_np,
                    "trough":  trough_np,
                    "code":    code,
                }

                good = ((peak_np >= AMP_THRESHOLD) & (trough_np >= MIN_ADC_CUT) &
                        np.isfinite(t50_arr) & (t50_arr > 0))
                print(f"  [INFO] {os.path.basename(fpath)} | {family} | "
                      f"PID pass {pid_mask.sum()} | "
                      f"pulse pass {good.sum()}")

            # Event loop
            for ev_idx in range(n_entries):
                if not pid_mask[ev_idx]:
                    continue

                fam_results = {}
                any_pulse   = False
                ref_t50     = None

                for family in SUBPLOT_FAMILIES:
                    fd = family_data.get(family)
                    if fd is None:
                        fam_results[family] = {
                            "waveform":  None,
                            "t50_ns":    np.nan,
                            "has_pulse": False,
                        }
                        continue

                    peak   = float(fd["peak"][ev_idx])
                    trough = float(fd["trough"][ev_idx])
                    t50    = float(fd["t50_arr"][ev_idx])
                    t50_ok = np.isfinite(t50) and t50 > 0
                    pulse  = (peak >= AMP_THRESHOLD) and (trough >= MIN_ADC_CUT) and t50_ok

                    fam_results[family] = {
                        "waveform":  fd["w_np"][ev_idx].copy(),
                        "t50_ns":    t50 if t50_ok else np.nan,
                        "has_pulse": pulse,
                    }

                    if pulse:
                        any_pulse = True
                        if ref_t50 is None:
                            ref_t50 = t50   # first valid pulse sets the window

                if not any_pulse:
                    continue

                records.append({
                    "run_label":   rl,
                    "run_number":  rnum,
                    "event_index": int(ev_idx),
                    "ref_t50_ns":  ref_t50,
                    "families":    fam_results,
                })

    except Exception as e:
        print(f"  [ERROR] {fpath}: {e}")

    return records

# =============================================================================
# DRAW ONE PAGE  (3 subplots for one event, paper-quality style)
# =============================================================================

def setup_paper_axes(ax, xlabel, ylabel, is_leftmost=False):
    """Apply paper-quality axis formatting matching PrecisionTiming_paperplots3."""
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE, fontweight="normal", loc="right")
    if is_leftmost:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE, fontweight="normal", loc="top")
    ax.tick_params(axis="both", which="major",
                   labelsize=TICK_LABEL_FONTSIZE, length=10, width=1.6,
                   direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor",
                   length=5, width=1.2, direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.grid(False)


def draw_event_page(fig, axes, record, thickness, particle, page_num, total, chan_map):
    rnum    = record["run_number"]
    ev      = record["event_index"]
    energy  = get_beam_energy(rnum)
    ref_t50 = record["ref_t50_ns"]

    disp_p  = display_particle(particle)

    for ax_idx, (ax, family) in enumerate(zip(axes, SUBPLOT_FAMILIES)):
        fd    = record["families"][family]
        color = FAMILY_COLORS.get(family, "tab:gray")
        code  = chan_map.get(family, "???")
        is_leftmost = (ax_idx == 0)

        if fd["waveform"] is None:
            ax.text(0.5, 0.5, f"{family}\n(no branch data)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=ANNOTATION_FONTSIZE)
        else:
            wf  = fd["waveform"]
            t50 = fd["t50_ns"]

            # Absolute time axis
            t_abs = np.arange(len(wf)) * TIME_PER_BIN

            # Window centred on ref_t50
            win = (t_abs >= ref_t50 - TWINDOW_LEFT) & (t_abs <= ref_t50 + TWINDOW_RIGHT)

            ax.plot(t_abs[win], wf[win], color=color, lw=2.0,
                    solid_capstyle="round")

            # Dashed t50 line + top-right text label (matches screenshot style)
            if np.isfinite(t50):
                ax.axvline(t50, color="black", ls="--", lw=1.6, alpha=0.80)
                ax.text(
                    0.98, 0.965,
                    rf"$\mathrm{{---}}\ t_{{50}}$",
                    transform=ax.transAxes,
                    ha="right", va="top",
                    fontsize=LEGEND_FONTSIZE,
                    color="black",
                )

            ax.set_xlim(ref_t50 - TWINDOW_LEFT, ref_t50 + TWINDOW_RIGHT)

        # Paper-quality axis formatting
        setup_paper_axes(
            ax,
            xlabel="Time [ns]",
            ylabel="ADC (baseline sub.)",
            is_leftmost=is_leftmost,
        )

        # Family name — top-right below t50 label, plain rectangle outline
        family_title = FAMILY_DISPLAY.get(family, family)
        ax.text(
            0.98, 0.865,
            family_title,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=ANNOTATION_FONTSIZE,
            fontweight="normal",
            bbox=dict(boxstyle="square,pad=0.25", facecolor="white",
                      edgecolor="black", linewidth=0.8, alpha=1.0),
            zorder=10,
        )

    # ── CaloX on the top-left (anchored to axes[0]); run info as figure-level
    #    right-aligned text — keeps both labels from ever colliding. ────────────
    r_label = f"{energy:.0f} GeV {disp_p} | {thickness} | Event {ev}"

    if HEP_AVAILABLE:
        try:
            # Place only the experiment stamp on the leftmost axis.
            # rlabel="" so nothing appears on the right side of axes[0].
            hep.cms.label(
                ax=axes[0],
                exp="CaloX",
                data=True,
                label="",
                rlabel="",
                fontsize=CMS_LABEL_FONTSIZE,
            )
        except Exception:
            axes[0].text(
                0.0, 1.02, "CaloX",
                transform=axes[0].transAxes,
                ha="left", va="bottom",
                fontsize=CMS_LABEL_FONTSIZE,
                fontweight="bold",
            )

    # Run info: pinned to the top-right corner of the figure in figure coords.
    # axes[2] right edge ≈ right=0.98 from subplots_adjust, top ≈ 0.88 → 1.0
    # Use axes[2].transAxes → figure transform for exact right-edge alignment.
    axes[2].text(
        1.0, 1.02,
        r_label,
        transform=axes[2].transAxes,
        ha="right", va="bottom",
        fontsize=CMS_LABEL_FONTSIZE,
        fontweight="normal",
    )

# =============================================================================
# CSV SCHEMAS
# =============================================================================
INDEX_HEADER = [
    "page", "run_label", "run_number", "event_index", "thickness", "particle",
    "beam_energy_GeV",
    "t50_Quartz_ns", "t50_Plastic_ns", "t50_Scintillator_ns",
    "peak_Quartz_adc", "peak_Plastic_adc", "peak_Scintillator_adc",
]

WAVE_HEADER = ["event", "run_num", "particle_type", "channel_num", "time_ns", "adc"]

# =============================================================================
# PROCESS ONE (thickness, particle) COMBINATION
# =============================================================================

def process_combination(thickness, particle, files, chan_map, outdir, n_waves):
    print(f"\n{'='*65}")
    print(f"  {thickness} | {particle}  (Quartz | Plastic | Scintillator)")
    print(f"{'='*65}")

    os.makedirs(outdir, exist_ok=True)

    all_records = []
    for fpath in files:
        all_records.extend(collect_valid_events_multi(fpath, chan_map, particle))

    if not all_records:
        print(f"  [WARN] No valid events — skipping.")
        return

    # Select n_waves events whose ref_t50 is closest to the modal ref_t50
    all_t50s   = np.array([r["ref_t50_ns"] for r in all_records])
    mode       = modal_t50(all_t50s)
    chosen_idx = np.argsort(np.abs(all_t50s - mode))[:n_waves]
    chosen     = [all_records[i] for i in chosen_idx]

    print(f"  [INFO] Total valid events : {len(all_records)}")
    print(f"         Modal ref t50      : {mode:.3f} ns")
    print(f"         Writing            : {len(chosen)} pages")

    safe_p    = particle.replace("_", "")
    stem      = f"Waveforms_{thickness}_{safe_p}_{len(chosen)}events"
    pdf_path  = os.path.join(outdir, stem + ".pdf")
    idx_path  = os.path.join(outdir, stem + "_index.csv")
    wave_path = os.path.join(outdir, stem + "_waveforms.csv")

    with (PdfPages(pdf_path) as pdf,
          open(idx_path,  "w", newline="") as idx_fh,
          open(wave_path, "w", newline="") as wave_fh):

        idx_writer  = csv.DictWriter(idx_fh,  fieldnames=INDEX_HEADER)
        wave_writer = csv.DictWriter(wave_fh, fieldnames=WAVE_HEADER)
        idx_writer.writeheader()
        wave_writer.writeheader()

        for page, record in enumerate(chosen, start=1):
            rnum    = record["run_number"]
            rl      = record["run_label"]
            ev      = record["event_index"]
            energy  = get_beam_energy(rnum)
            ref_t50 = record["ref_t50_ns"]

            # ── PDF page ──────────────────────────────────────────────────
            # Landscape: wide figure, tighter vertical margins
            fig, axes = plt.subplots(1, 3, sharey=False, figsize=(30, 7))
            fig.subplots_adjust(left=0.05, right=0.99, top=0.88,
                                bottom=0.14, wspace=0.25)
            draw_event_page(fig, axes, record, thickness, particle,
                            page, len(chosen), chan_map)
            pdf.savefig(fig, bbox_inches="tight", dpi=200)
            plt.close(fig)

            # ── Index CSV row ──────────────────────────────────────────────
            def fmt_t50(fd):
                return f"{fd['t50_ns']:.4f}" if np.isfinite(fd["t50_ns"]) else ""

            def fmt_peak(fd):
                if fd["waveform"] is None:
                    return ""
                return f"{float(np.max(fd['waveform'])):.2f}"

            fd_q = record["families"]["Quartz"]
            fd_p = record["families"]["Plastic"]
            fd_s = record["families"]["Scintillator"]

            idx_writer.writerow({
                "page":                  page,
                "run_label":             rl,
                "run_number":            rnum if rnum is not None else "",
                "event_index":           ev,
                "thickness":             thickness,
                "particle":              particle,
                "beam_energy_GeV":       f"{energy:.1f}",
                "t50_Quartz_ns":         fmt_t50(fd_q),
                "t50_Plastic_ns":        fmt_t50(fd_p),
                "t50_Scintillator_ns":   fmt_t50(fd_s),
                "peak_Quartz_adc":       fmt_peak(fd_q),
                "peak_Plastic_adc":      fmt_peak(fd_p),
                "peak_Scintillator_adc": fmt_peak(fd_s),
            })

            # ── Waveform CSV rows ──────────────────────────────────────────
            rnum_str = str(rnum) if rnum is not None else ""
            for family in SUBPLOT_FAMILIES:
                fd = record["families"][family]
                if fd["waveform"] is None:
                    continue
                wf          = fd["waveform"]
                t_abs       = np.arange(len(wf)) * TIME_PER_BIN
                win         = ((t_abs >= ref_t50 - TWINDOW_LEFT) &
                               (t_abs <= ref_t50 + TWINDOW_RIGHT))
                channel_num = chan_map.get(family, "???")
                for t_val, adc_val in zip(t_abs[win], wf[win]):
                    wave_writer.writerow({
                        "event":         ev,
                        "run_num":       rnum_str,
                        "particle_type": particle,
                        "channel_num":   channel_num,
                        "time_ns":       f"{t_val:.4f}",
                        "adc":           f"{adc_val:.4f}",
                    })

    print(f"  [OK] PDF   → {pdf_path}")
    print(f"  [OK] Index → {idx_path}")
    print(f"  [OK] Waves → {wave_path}")

# =============================================================================
# DRIVER
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=(
            "One PDF page per event, 3 subplots (Quartz | Plastic | Scintillator). "
            "x-axis is absolute time [ns] with a vertical dashed line at each "
            "family's t50. Requires at least one family to have a valid pulse."
        )
    )
    ap.add_argument("--outdir",      default="./paper_waveforms_triple_wCSVS",
                    help="Root output directory  (default: ./paper_waveforms_triplet_1000evts)")
    ap.add_argument("--n-waveforms", type=int, default=N_WAVEFORMS,
                    help=f"Events per combination  (default: {N_WAVEFORMS})")
    ap.add_argument("--pid",       default=None,
                    choices=["muon", "pion", "electron", "electron_90deg"],
                    help="Restrict to one particle type  (default: all)")
    ap.add_argument("--thickness", default=None, choices=["3mm", "6mm"],
                    help="Restrict to one thickness       (default: both)")
    args = ap.parse_args()

    apply_paper_style()
    os.makedirs(args.outdir, exist_ok=True)

    for thickness, particles in RUN_FILES.items():
        if args.thickness and thickness != args.thickness:
            continue

        chan_map = CHANNELS_3MM if thickness == "3mm" else CHANNELS_6MM

        for particle, raw_files in particles.items():
            if args.pid and particle != args.pid:
                continue
            if not raw_files:
                print(f"[SKIP] No files configured for {thickness} | {particle}")
                continue

            subdir = os.path.join(args.outdir, thickness, particle)
            process_combination(
                thickness = thickness,
                particle  = particle,
                files     = raw_files,
                chan_map  = chan_map,
                outdir    = subdir,
                n_waves   = args.n_waveforms,
            )

    print("\n[DONE] All PDFs and CSVs written.")


if __name__ == "__main__":
    main()