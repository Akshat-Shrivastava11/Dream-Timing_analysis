#!/usr/bin/env python3
import os
import re
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
BASE_DIR = "/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples"

# ================= ADC CUT CONFIG =================
AMP_THRESHOLD = 100.0  # Waveform must peak above this (baseline subtracted)
MIN_ADC_CUT = -100.0

# ================= DETECTOR FAMILY METADATA (For Plotting) =================
FAMILY_META = {
    "Plastic-CER": {"legend": "Cherenkov-Plastic", "color": "red"},
    "Quartz-CER":  {"legend": "Cherenkov-Quartz",  "color": "blue"},
    "SCI":         {"legend": "Scintillating",     "color": "green"}
}

# ================= DYNAMIC RUN CONFIGURATIONS =================
Y_CONFIGS = {
    "y1000": {
        "SCI":         {"channels": ["620","621"], "tmin": 8.0,  "tmax": 11.0},
        "Plastic-CER": {"channels": ["612","611","610","613"], "tmin": 10.5, "tmax": 12.5},
        "Quartz-CER":  {"channels": ["631","630","627","637"], "tmin": 10.0, "tmax": 13.5}
    },
    "y1065": {
        #"SCI":         {"channels": ["105"], "tmin": 8.5,  "tmax": 15.0},
        #"Plastic-CER": {"channels": ["100"], "tmin": 10.0, "tmax": 15.0},
        "Quartz-CER":  {"channels": ["523","522","521","520"], "tmin": 10.0, "tmax": 13.5}
    },
    "y936": {
        "SCI":         {"channels": ["604","605"], "tmin": 8.0,  "tmax": 11.0},
        "Plastic-CER": {"channels": ["607","606"], "tmin": 11.0, "tmax": 12.5},
        "Quartz-CER":  {"channels": ["617","616","615","614"], "tmin": 11.0, "tmax": 12.6}
    },
    "y1028": {
        "SCI":         {"channels": ["421","420"], "tmin": 7.0, "tmax": 10.5},
        "Plastic-CER": {"channels": ["425","423","422","424"], "tmin": 10.5, "tmax": 12.5},
        "Quartz-CER":  {"channels": ["413","412","411","410"], "tmin": 11.0, "tmax": 12.5}
    }
}

RUN_MAP = {
    # y: 1000mm
    "run1502_250928113749": "y1000",
    "run1508_250928161049": "y1000",
    "run1512_250928183645": "y1000",
    
    # y: 1065mm
    "run1501_250928105227": "y1065",
    "run1511_250928180741": "y1065",
    "run1507_250928160030": "y1065",
    "run1513_250928192918": "y1065",
    "run1513_250928194230": "y1065",
    
    # y: 936mm
    "run1504_250928133854": "y936",
    "run1509_250928164817": "y936",
    "run1512_250928185722": "y936",
    
    # y: 1028mm
    "run1506_250928143030": "y1028",
    "run1506_250928145724": "y1028",
    "run1510_250928172949": "y1028"
}

def get_run_group_and_config(run_label):
    for key, group in RUN_MAP.items():
        if key in run_label:
            return group, Y_CONFIGS[group]
    return None, {}

# ================= Z POSITION MAPPING =================
# ================= Z POSITION MAPPING =================
def get_z_position(run_label):
    if "run1513" in run_label:
        if "192918" in run_label: return -54.5
        if "194230" in run_label: return -400.3
    match = re.search(r"run(\d+)", run_label)
    run_num = int(match.group(1)) if match else None
    
    z_map = {
        # y1065 runs
        1501: -168.0, 1507: -218.0, 1511: -268.0,
        
        # y936 runs
        1504: -168.0, 1509: -218.0, 1512: -268.0,
        
        # y1000 runs
        1502: -168.0,    # <-- ADD ACTUAL Z POSITIONS HERE
        1508: -218.0,  # <-- ADD ACTUAL Z POSITIONS HERE
        # 1512 is already covered above

        # y1028 runs
        1506: -168.0,  # <-- ADD ACTUAL Z POSITIONS HERE
        1510: -218.0   # <-- ADD ACTUAL Z POSITIONS HERE
    }
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

def compute_tfinal_6mm(tree, b, g, c, suffix=""):
    br_sig     = f"DRS_Board{b}_Group{g}_Channel{c}{suffix}"
    br_sig_ref = f"DRS_Board{b}_Group{g}_Channel8{suffix}"
    br_trg     = f"DRS_Board0_Group3_Channel7{suffix}"
    br_trg_ref = f"DRS_Board0_Group3_Channel8{suffix}"
    
    keys = tree.keys()
    for br in [br_sig, br_sig_ref, br_trg, br_trg_ref]:
        if br not in keys:
            return None
            
    arr_sig     = tree[br_sig].array(library="np")
    arr_sig_ref = tree[br_sig_ref].array(library="np")
    arr_trg     = tree[br_trg].array(library="np")
    arr_trg_ref = tree[br_trg_ref].array(library="np")
    
    if not (arr_sig.shape == arr_sig_ref.shape == arr_trg.shape == arr_trg_ref.shape):
        return None
        
    t_final = (arr_sig - arr_sig_ref) - (arr_trg - arr_trg_ref)
    
    return np.abs(t_final)

def _run_label(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"(run\d+_\d{11,12})", base)
    if m:
        return m.group(1)
    return os.path.splitext(base)[0]

def _resolve_files(args):
    files = []
    # Build exact paths strictly from the dictionary
    for run_label in RUN_MAP.keys():
        fpath = os.path.join(BASE_DIR, f"{run_label}_converted_timingskim.root")
        if os.path.exists(fpath):
            files.append(fpath)
        else:
            print(f"[WARN] File explicitly mapped but not found on disk: {fpath}")

    # Optional min/max run filtering if user provides them via arguments
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

def style_paper_axes(ax, xlabel, ylabel, particle_type, group_name):
    ax.set_xlabel(xlabel, loc='right', fontsize=14)
    ax.set_ylabel(ylabel, loc='top', fontsize=14)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=12)
    ax.tick_params(which='major', length=8)
    ax.tick_params(which='minor', length=4)
    
    display_name = "Positron" if particle_type.lower() == "electron" else particle_type.capitalize()
    
    y_str = group_name.replace("y", "y=") + "mm"
    header_text = r"$\mathbf{CaloX}$" + f"  40 GeV {display_name} ({y_str})"
    
    ax.text(0.0, 1.02, header_text, 
            transform=ax.transAxes, fontsize=14, 
            va='bottom', ha='left', color='black')

def create_z_toa_plot(plot_data, outdir, pid_label, particle_type, group_name):
    pdf_path = os.path.join(outdir, f"Z_vs_TOA_Fits_{pid_label}_{group_name}.pdf")
    
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8, 7))
        
        style_paper_axes(ax, "Z Position [mm]", "Time of Arrival Mean [ns]", particle_type, group_name)
        
        plotted_any = False
        
        for family_name, data in plot_data.items():
            if not data["z"]: continue
                
            plotted_any = True
            z_arr = np.array(data["z"])
            mu_arr = np.array(data["mu"])
            sig_arr = np.array(data["sig"])
            
            color = FAMILY_META[family_name]["color"]
            legend_label = FAMILY_META[family_name]["legend"]
            
            weights = np.where(sig_arr > 0, 1.0 / sig_arr, 1.0) 
            
            if len(z_arr) > 1:
                slope, intercept = np.polyfit(z_arr, mu_arr, 1, w=weights)
                
                speed_m_s = abs(1.0 / slope) * 1e6 if slope != 0 else 0
                exponent = int(np.floor(np.log10(speed_m_s))) if speed_m_s > 0 else 0
                mantissa = speed_m_s / (10**exponent) if speed_m_s > 0 else 0
                
                speed_legend = r"{} ($v \approx {:.2f} \times 10^{{{}}}$ m/s)".format(
                    legend_label, mantissa, exponent
                )
                
                z_fit = np.linspace(min(z_arr) - 20, max(z_arr) + 20, 100)
                t_fit = slope * z_fit + intercept
                
                ax.errorbar(z_arr, mu_arr, yerr=sig_arr, fmt='o', color=color, capsize=3, markersize=4)
                ax.plot(z_fit, t_fit, '-', color=color, linewidth=2, label=speed_legend)
            else:
                ax.errorbar(z_arr, mu_arr, yerr=sig_arr, fmt='o', color=color, capsize=3, markersize=4, label=legend_label)

        if plotted_any:
            ax.legend(loc="upper left", frameon=False, fontsize=10)
            fig.subplots_adjust(top=0.92) 
            pdf.savefig(fig)
        
        plt.close(fig)

# ================= MAIN DATA EXTRACTION =================
def generate_stats_table(files, outpath, tree_name, particle_type=None):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    
    header_fmt = "{:<10} | {:<10} | {:<10} | {:<8} | {:<12} | {:<12} | {:<12} | {:<10}"
    row_fmt    = "{:<10} | {:<10.1f} | {:<10} | {:<8} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<10}"
    
    y_groups = list(Y_CONFIGS.keys())
    plot_data_all = {
        grp: {fam: {"z": [], "mu": [], "sig": []} for fam in FAMILY_META.keys()} 
        for grp in y_groups
    }
    
    with open(outpath, "w") as f_out:
        separator = "=" * 102
        f_out.write(separator + "\n")
        f_out.write(header_fmt.format("Run", "Position_Z", "Family", "Channel", "Time_Mean", "Time_Sigma", "FWHM", "N_Events") + "\n")
        f_out.write(separator + "\n")
        
        for fpath in files:
            rl = _run_label(fpath)
            
            group_name, run_configs = get_run_group_and_config(rl)
            if not run_configs:
                continue
                
            print(f"\n--- Processing Run: {rl} ({group_name}) ---")
            
            try:
                uf = uproot.open(fpath)
                tree = uf[tree_name]
                z_pos = get_z_position(rl)
                
                pid_mask = None
                if particle_type:
                    pid_mask = compute_pid_mask(tree, particle_type)

            except Exception as e:
                print(f"[WARN] Failed to open {fpath}: {e}")
                continue

            for family_name, fam_cfg in run_configs.items():
                tmin = fam_cfg["tmin"]
                tmax = fam_cfg["tmax"]
                bins = np.linspace(tmin, tmax, NBINS + 1)
                centers = 0.5 * (bins[1:] + bins[:-1])

                for code_str in fam_cfg["channels"]:
                    b, g, ch = _parse_code(code_str)
                    
                    arr_raw = compute_tfinal_6mm(tree, b, g, ch, suffix="_LP2_50") 
                    
                    if arr_raw is None:
                        print(f" [WARN] Required timing branches not found for ch {code_str} (suffix _LP2_50).")
                        continue
                    
                    try:
                        n_initial = len(arr_raw)
                        
                        if pid_mask is not None:
                            arr_pid = arr_raw[pid_mask]
                            n_pid = len(arr_pid)
                        else:
                            n_pid = n_initial
                        
                        adc_mask = compute_adc_mask(tree, code_str)
                        
                        if pid_mask is not None:
                            combined_mask = pid_mask 
                        else:
                            combined_mask = pid_mask
                        
                        if arr_raw.shape[0] == combined_mask.shape[0]:
                            arr_adc = arr_raw[combined_mask]
                            n_adc = len(arr_adc)
                        else:
                            print(f" [ERROR] Shape mismatch in {rl} ch {code_str}")
                            continue
                            
                        arr_time = arr_adc 
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
                    
                    if z_pos != -999.0:
                        plot_data_all[group_name][family_name]["z"].append(z_pos)
                        plot_data_all[group_name][family_name]["mu"].append(fit_mu)
                        plot_data_all[group_name][family_name]["sig"].append(fit_sig)
            
            uf.close()
            f_out.write("-" * 102 + "\n")
            
        f_out.write(separator + "\n")
    print(f"\nTable successfully saved to: {outpath}")
    
    pid_label = f"PID_{particle_type}" if particle_type else "AllParticles"
    
    for grp, pdata in plot_data_all.items():
        has_data = any(len(d["z"]) > 0 for d in pdata.values())
        if has_data:
            create_z_toa_plot(pdata, os.path.dirname(outpath), pid_label, particle_type, grp)
            print(f"Generated plot for {grp}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-min", type=int, default=None, help="Keep only runs >= run-min")
    ap.add_argument("--run-max", type=int, default=None, help="Keep only runs <= run-max")
    ap.add_argument("--tree", default=TREE_NAME, help="Tree name")
    ap.add_argument("--outdir", default="./TRUE-HGtiming/calibration_studiesZ/tables6mm", help="Output directory")
    ap.add_argument("--pid", default='electron', choices=["muon", "pion", "electron", "proton"], help="Apply PID selection")

    args = ap.parse_args()

    files = _resolve_files(args)
    if len(files) == 0:
        raise SystemExit(f"ERROR: No files found! Check that BASE_DIR path is correct.")

    print(f"Found {len(files)} explicitly mapped files.")
    
    pid_label = f"PID_{args.pid}" if args.pid else "AllParticles"
    output_txt_path = os.path.join(args.outdir, f"Timing_Statistics_{pid_label}.txt")

    generate_stats_table(files, output_txt_path, args.tree, particle_type=args.pid)
    print("All done.")

if __name__ == "__main__":
    main()