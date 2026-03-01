#!/usr/bin/env python3
import os
import re
import glob
import argparse
import numpy as np
import uproot
import awkward as ak
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator

# ================= DEFAULTS =================
TREE_NAME = "EventTree"
NBINS = 200

# ================= ADC CUT CONFIG =================
AMP_THRESHOLD = 100.0  # Waveform must peak above this (baseline subtracted)
MIN_ADC_CUT = -100.0

# ================= DETECTOR FAMILY CONFIG =================
FAMILIES = {
    "Plastic": {"channels": ["100","102","112", "110"], "tmin": 11.5, "tmax": 14.5, "legend": "Cherenkov-Plastic", "color": "red"},
    "Quartz":  {"channels": ["104","106", "304","114"], "tmin": 11.5, "tmax": 15.0, "legend": "Cherenkov-Quartz", "color": "blue"},
    "SCI":     {"channels": ["105", "107","111","117"], "tmin":  9.5, "tmax": 13.5, "legend": "Scintilating",      "color": "green"}
}

# ================= Z POSITION MAPPING =================
def get_z_position(run_label):
    if "run1513" in run_label:
        if "192918" in run_label: return -54.5
        if "194230" in run_label: return -400.3
    match = re.search(r"run(\d+)", run_label)
    run_num = int(match.group(1)) if match else None
    z_map = {1501: -168.0, 1507: -218.0, 1511: -268.0}
    return z_map.get(run_num, -999.0)

# ================= PID CONFIGURATION =================
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
        "muon": {
            "TTUMuonVeto": True,
            "PSD": False,
        },
        "pion": {
            "TTUMuonVeto": False,
            "PSD": False,
            "Cer474": True, "Cer519": True, "Cer537": True
        },
        "electron": {
            "TTUMuonVeto": False,
            "PSD": True,
            "Cer474": True, "Cer519": True, "Cer537": True
        },
        "proton": {
            "TTUMuonVeto": False,
            "PSD": False,
            "Cer474": False, "Cer519": False, "Cer537": False
        },
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
                window = waveforms_blsub[:, int(ts_min):int(ts_max)]
                window_sum = ak.sum(window, axis=1)
                window_sum_np = ak.to_numpy(window_sum)
                is_fired = window_sum_np < val_cut
            else:
                continue

            if must_fire:
                final_mask = final_mask & is_fired
            else:
                final_mask = final_mask & (~is_fired)
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

# ================= HELPERS =================
def _parse_code(code_str):
    return int(code_str[0]), int(code_str[1]), int(code_str[2])

def _branch(b, g, c):
    return f"tfinal_Board{b}_Group{g}_Channel{c}"

def _run_label(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"(run\d+_\d{11,12})", base)
    if m:
        return m.group(1)
    return os.path.splitext(base)[0]

def _resolve_files(args):
    if args.ana_files:
        files = list(args.ana_files)
    else:
        files = sorted(glob.glob(args.ana_glob))

    if args.run_min is not None and args.run_max is not None:
        keep = []
        for f in files:
            m = re.search(r"run(\d+)", os.path.basename(f))
            if not m:
                continue
            r = int(m.group(1))
            if args.run_min <= r <= args.run_max:
                keep.append(f)
        files = keep

    def _sort_key(p):
        b = os.path.basename(p)
        mrun = re.search(r"run(\d+)", b)
        r = int(mrun.group(1)) if mrun else 10**9
        mts = re.search(r"_(\d{11,12})(?:_|\.|$)", b)
        ts = int(mts.group(1)) if mts else 10**18
        return (r, ts, b)

    return sorted(files, key=_sort_key)

def _mode_from_hist(arr, bins):
    h, _ = np.histogram(arr, bins=bins)
    if h.sum() == 0:
        return (np.nan, 0, h)
    idx = int(np.argmax(h))
    centers = 0.5 * (bins[1:] + bins[:-1])
    return (float(centers[idx]), int(h[idx]), h)

def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-(x - mean)**2 / (2 * sigma**2))

def style_paper_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, loc='right', fontsize=12, fontweight='medium')
    ax.set_ylabel(ylabel, loc='top', fontsize=12, fontweight='medium')
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    
    ax.tick_params(which='both', direction='in', top=True, right=True, length=6)
    ax.tick_params(which='minor', length=3)

# ================= PLOTTING FUNCTION =================
def create_z_toa_plot(plot_data, outdir, pid_label):
    pdf_path = os.path.join(outdir, f"Z_vs_TOA_Fits_{pid_label}.pdf")
    
    print("\n--- Fit Results & Speeds ---")
    with PdfPages(pdf_path) as pdf:
        # PAGE 1: With Points & Error Bars
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        
        # PAGE 2: Lines Only
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        
        # Style both axes identically
        for ax in [ax1, ax2]:
            style_paper_axes(ax, "Z Position [mm]", "Time of Arrival Mean [ns]")
            ax.set_title(f"TOA vs. Z Position ({pid_label})", pad=15)
        
        for family_name, data in plot_data.items():
            if not data["z"]:
                continue
                
            z_arr = np.array(data["z"])
            mu_arr = np.array(data["mu"])
            sig_arr = np.array(data["sig"])
            
            legend_name = FAMILIES[family_name]["legend"]
            color = FAMILIES[family_name]["color"]
            
            # Linear Fit
            weights = np.where(sig_arr > 0, 1.0 / sig_arr, 1.0) 
            slope, intercept = np.polyfit(z_arr, mu_arr, 1, w=weights)
            
            # ---> SPEED CALCULATION IS HERE <---
            # slope units: ns/mm. Inverse slope: mm/ns. 1 mm/ns = 1e6 m/s
            if slope != 0:
                speed_m_s = abs(1.0 / slope) * 1e6
                exponent = int(np.floor(np.log10(speed_m_s)))
                mantissa = speed_m_s / (10**exponent)
                
                # Math text formatting for a beautiful LaTeX-style legend
                speed_str = r"$v = {:.2f} \times 10^{{{}}}$ m/s".format(mantissa, exponent)
                print(f"{family_name:<10} : {speed_m_s:.2e} m/s")
            else:
                speed_str = r"$v = $ undefined"
                print(f"{family_name:<10} : undefined (slope=0)")
                
            legend_label_with_speed = f"{legend_name} ({speed_str})"
            
            # Generate points for the line
            z_fit = np.linspace(min(z_arr) - 20, max(z_arr) + 20, 100)
            t_fit = slope * z_fit + intercept
            
            # Add to Figure 1 (With Points)
            ax1.errorbar(z_arr, mu_arr, yerr=sig_arr, fmt='o', color=color, 
                         elinewidth=1.5, capsize=3, markersize=4, alpha=0.8)
            ax1.plot(z_fit, t_fit, '-', color=color, linewidth=2, label=legend_label_with_speed)
            
            # Add to Figure 2 (Lines Only)
            ax2.plot(z_fit, t_fit, '-', color=color, linewidth=2, label=legend_label_with_speed)

        # Finalize and save
        for fig, ax in [(fig1, ax1), (fig2, ax2)]:
            ax.legend(loc="best", frameon=True, fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nPaper-worthy plot saved to: {pdf_path}")

# ================= MAIN DATA EXTRACTION =================
def generate_stats_table(files, outpath, tree_name, particle_type=None):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    
    header_fmt = "{:<10} | {:<10} | {:<10} | {:<8} | {:<12} | {:<12} | {:<12} | {:<10}"
    row_fmt    = "{:<10} | {:<10.1f} | {:<10} | {:<8} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<10}"
    
    # Dictionary to hold data for plotting: {Family: {"z": [], "mu": [], "sig": []}}
    plot_data = {fam: {"z": [], "mu": [], "sig": []} for fam in FAMILIES.keys()}
    
    with open(outpath, "w") as f_out:
        separator = "=" * 102
        f_out.write(separator + "\n")
        f_out.write(header_fmt.format("Run", "Position_Z", "Family", "Channel", "Time_Mean", "Time_Sigma", "FWHM", "N_Events") + "\n")
        f_out.write(separator + "\n")
        
        for fpath in files:
            rl = _run_label(fpath)
            print(f"\n--- Processing Run: {rl} ---")
            
            try:
                uf = uproot.open(fpath)
                tree = uf[tree_name]
                keys = set(tree.keys())
                z_pos = get_z_position(rl)
                
                pid_mask = None
                if particle_type:
                    pid_mask = compute_pid_mask(tree, particle_type)

            except Exception as e:
                print(f"[WARN] Failed to open {fpath}: {e}")
                continue

            for family_name, fam_cfg in FAMILIES.items():
                tmin = fam_cfg["tmin"]
                tmax = fam_cfg["tmax"]
                bins = np.linspace(tmin, tmax, NBINS + 1)
                centers = 0.5 * (bins[1:] + bins[:-1])

                for code_str in fam_cfg["channels"]:
                    b, g, ch = _parse_code(code_str)
                    k = _branch(b, g, ch)
                    
                    if k not in keys: 
                        continue
                    
                    try:
                        arr_raw = tree[k].array(library="np")
                        n_initial = len(arr_raw)
                        
                        if pid_mask is not None:
                            arr_pid = arr_raw[pid_mask]
                            n_pid = len(arr_pid)
                        else:
                            n_pid = n_initial
                        
                        adc_mask = compute_adc_mask(tree, code_str)
                        combined_mask = pid_mask & adc_mask if pid_mask is not None else adc_mask
                        
                        if arr_raw.shape[0] == combined_mask.shape[0]:
                            arr_adc = arr_raw[combined_mask]
                            n_adc = len(arr_adc)
                        else:
                            print(f" [ERROR] Shape mismatch in {rl} ch {code_str}")
                            continue
                            
                        arr_time = np.abs(arr_adc)
                        arr_time = arr_time[(arr_time >= tmin) & (arr_time <= tmax)]
                        n_final = len(arr_time)
                        
                        print(f"  [{family_name}] Ch {code_str} Cutflow: Initial={n_initial} -> PID={n_pid} -> ADC={n_adc} -> TimeCut={n_final}")
                        
                    except Exception as e:
                        print(f" [ERROR] Failed processing in {rl} ch {code_str}: {e}")
                        continue

                    if n_final < 50: 
                        print(f"    -> Skipping fit, too few events ({n_final} < 50)")
                        continue
                        
                    mode, max_counts, h = _mode_from_hist(arr_time, bins)
                    if h.sum() == 0: 
                        continue

                    if h.max() > 0:
                        h = h / h.max()

                    fit_window = 1.5 
                    mask = (centers >= mode - fit_window) & (centers <= mode + fit_window)
                    x_fit = centers[mask]
                    y_fit = h[mask]

                    fit_mu, fit_sig, fwhm = np.nan, np.nan, np.nan

                    if len(x_fit) > 4:
                        try:
                            p0 = [1.0, mode, 0.3]
                            popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0)
                            fit_mu = popt[1]
                            fit_sig = abs(popt[2])
                            fwhm = 2.355 * fit_sig
                        except:
                            fit_mu = mode
                            fit_sig = float(arr_time.std())
                            fwhm = 2.355 * fit_sig
                    else:
                        fit_mu = mode
                        fit_sig = float(arr_time.std())
                        fwhm = 2.355 * fit_sig

                    m_run = re.search(r"run(\d+)", rl)
                    run_display = m_run.group(1) if m_run else rl

                    f_out.write(row_fmt.format(
                        run_display, z_pos, family_name, code_str, fit_mu, fit_sig, fwhm, n_final
                    ) + "\n")
                    
                    # Store data for plotting if Z position is valid
                    if z_pos != -999.0:
                        plot_data[family_name]["z"].append(z_pos)
                        plot_data[family_name]["mu"].append(fit_mu)
                        plot_data[family_name]["sig"].append(fit_sig)
            
            uf.close()
            f_out.write("-" * 102 + "\n")
            
        f_out.write(separator + "\n")
    print(f"\nTable successfully saved to: {outpath}")
    
    # Generate the requested plots
    pid_label = f"PID_{particle_type}" if particle_type else "AllParticles"
    create_z_toa_plot(plot_data, os.path.dirname(outpath), pid_label)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ana-files", nargs="+", default=None, help="Explicit list of input ROOT files.")
    ap.add_argument("--ana-glob", default=None, help="Glob for input ROOT files.")
    ap.add_argument("--run-min", type=int, default=None, help="Keep only runs >= run-min")
    ap.add_argument("--run-max", type=int, default=None, help="Keep only runs <= run-max")
    ap.add_argument("--tree", default=TREE_NAME, help="Tree name")
    ap.add_argument("--outdir", default="./TRUE-HGtiming/calibration_studiesZ/tables", help="Output directory")
    ap.add_argument("--pid", default='electron', choices=["muon", "pion", "electron", "proton"], help="Apply PID selection")

    args = ap.parse_args()

    if args.ana_files is None and args.ana_glob is None:
        raise SystemExit("ERROR: provide either --ana-files or --ana-glob")

    files = _resolve_files(args)
    if len(files) == 0:
        raise SystemExit("ERROR: no files matched your selection")

    print(f"Found {len(files)} files.")
    
    pid_label = f"PID_{args.pid}" if args.pid else "AllParticles"
    output_txt_path = os.path.join(args.outdir, f"Timing_Statistics_{pid_label}.txt")

    generate_stats_table(files, output_txt_path, args.tree, particle_type=args.pid)
    print("All done.")

if __name__ == "__main__":
    main()