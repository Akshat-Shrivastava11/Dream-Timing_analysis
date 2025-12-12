import sys
import re
import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt
import gc
import os
from scipy.stats import norm
from scipy.optimize import curve_fit

# === Parse input ===
#filename = sys.argv[1]
#match = re.search(r"run(\d+)", filename)
#run_number = match.group(1) if match else "unknown"

# === Load ROOT file using uproot ===
run_number = "1468"
filename =  "/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1468_250927145556_TimingDAQ.root"
file = uproot.open(filename)
tree = file["EventTree"]

print(f"Loaded file: {filename} with {tree.num_entries} events")

# === Define channel map ===
channel_map = {
    "DRS_Board0_Group3_Channel0": "ch0",
    "DRS_Board0_Group3_Channel1": "ch1",
    "DRS_Board0_Group3_Channel2": "ch2",
    "DRS_Board0_Group3_Channel3": "ch3",
    "DRS_Board0_Group3_Channel4": "ch4",
    "DRS_Board0_Group3_Channel5": "ch5",
    "DRS_Board0_Group3_Channel6": "ch6",
    "DRS_Board0_Group3_Channel7": "ch7",
    "DRS_Board0_Group3_Channel8": "ch8",
    "DRS_Board7_Group0_Channel0": "br1",
    "DRS_Board7_Group0_Channel1": "br2",
    "DRS_Board7_Group0_Channel2": "br3",
    "DRS_Board7_Group0_Channel3": "br4",
}

# === Load branches ===
branches_to_load = list(channel_map.keys())
data = tree.arrays(branches_to_load, library="ak")

# Rename for convenience
channels = {alias: data[root_name] for root_name, alias in channel_map.items()}

# === Output container ===
delta_mcp_rising, mcp1_time, mcp2_time = [], [], []
ch0_time, ch1_time, ch2_time, ch3_time, ch4_time, ch5_time = [], [], [], [], [], []
# === Helper functions (same as original) ===

def get_min(vec):
    try:
        arr = np.asarray(vec, dtype=np.float32)
        if arr.size == 0:
            return None, None
        min_idx = np.argmin(arr)
        return min_idx, arr[min_idx]
    except Exception as e:
        print("get_min() failed:", e)
        return None, None

def find_baseline(event_info):
    try:
        baseline = np.asarray(event_info[:int(0.1 * len(event_info))])
        avg = np.mean(baseline)
        rms = np.std(baseline)
    except:
        avg = 0
        rms = 99999
    return avg, rms

def fit_region(event_info, baseline):
    try:
        event_info = np.asarray(event_info)
        if event_info.size == 0:
            raise ValueError("empty waveform")
        min_idx = np.argmin(event_info)
        min_val = event_info[min_idx]
        amplitude = baseline - min_val
        th_low = baseline - 0.1 * amplitude
        th_high = baseline - 0.9 * amplitude
        idxs = np.arange(len(event_info))
        mask = (event_info >= th_high) & (event_info <= th_low) & (idxs < min_idx) & (np.abs(idxs - min_idx) < 10)
        return event_info[mask], idxs[mask], amplitude
    except Exception as e:
        print("fit_region() failed:", e)
        return np.array([]), np.array([]), 0

def fit_rising_edge(region, indices):
    if len(region) < 2:
        return None, None
    return np.polyfit(indices, region, 1)

def extract_true_time(slope, intercept, baseline, amplitude):
    try:
        target = baseline - 0.5 * amplitude
        ts = (target - intercept) / slope
        return ts, 200 * ts  # 200 ps per sample?
    except Exception:
        return None, None

def plot_waveform(event_idx, signal, slope, intercept, time):
    idxs = np.arange(len(signal))
    plt.plot(idxs, signal, label="Signal")
    if slope is not None and intercept is not None:
        plt.plot(idxs, slope * idxs + intercept, '--', label="Fit")
        plt.title(f"Event {event_idx}, Time: {time:.2f} ps")
    #ymin = min(np.min(signal)-50)
    #ymax = max(np.max(signal)+50)
    plt.ylim(2000, 3000)
    plt.legend()
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"/lustre/research/hep/akshriva/Dream-Timing/MCP/Delta_t50_Results//waveform_run{run_number}_event{event_idx}.png")
    plt.close()

def plot_debug(event_idx, signal):
    idxs = np.arange(len(signal))
    plt.plot(idxs, signal, label="pulse")
    #ymin = min(np.min(signal)-50)
    #ymax = max(np.max(signal)+50)
    plt.ylim(2000, 3000)
    plt.legend()
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"/lustre/research/hep/akshriva/Dream-Timing/MCP/Delta_t50_Results//debug_pulse_run{run_number}_event{event_idx}.png")
    plt.close()

# === Main event loop ===
nEvents = tree.num_entries

for i in range(nEvents):
    # ---------- MCP1 (ch6) ----------
    try:
        waveform = channels["ch6"][i]
        baseline, rms = find_baseline(waveform)
        region, indices, amplitude = fit_region(waveform, baseline)
        if i < 10:
            plot_debug(i, waveform)
        if amplitude >= 5 * rms:
            slope, intercept = fit_rising_edge(region, indices)
            ts, t_ps = extract_true_time(slope, intercept, baseline, amplitude)
            mcp1_time.append(t_ps)
            #if i < 10:
            #    plot_waveform(i, waveform, slope, intercept, t_ps)
        else:
            #print(f"[Event MCP1 {i}] Skipped: amplitude below threshold ({amplitude:.2f} < {5*rms:.2f})")
            mcp1_time.append(None)

    except Exception as e:
        print(f"[Event MCP1 {i}] Failed: {e}")
        mcp1_time.append(None)

    if i % 1000 == 0:
        print(f"Processed MCP1 {i}/{nEvents}")
        gc.collect()

    # ---------- MCP2 (ch7) ----------
    try:
        waveform = channels["ch7"][i]
        baseline, rms = find_baseline(waveform)
        region, indices, amplitude = fit_region(waveform, baseline)

        if amplitude >= 5 * rms:
            slope, intercept = fit_rising_edge(region, indices)
            ts, t_ps = extract_true_time(slope, intercept, baseline, amplitude)
            mcp2_time.append(t_ps)
            # if i < 10:
            #     plot_waveform(i, waveform, slope, intercept, t_ps)
        else:
            #print(f"[Event MCP2 {i}] Skipped: amplitude below threshold ({amplitude:.2f} < {5*rms:.2f})")
            mcp2_time.append(None)

    except Exception as e:
        print(f"[Event MCP2 {i}] Failed: {e}")
        mcp2_time.append(None)

    if i % 1000 == 0:
        print(f"Processed MCP2 {i}/{nEvents}")
        gc.collect()

    # ---------- Cerenkov27 (ch0) ----------
    try:
        waveform = channels["ch0"][i]
        baseline, rms = find_baseline(waveform)
        region, indices, amplitude = fit_region(waveform, baseline)
        if amplitude >= 5 * rms:
            slope, intercept = fit_rising_edge(region, indices)
            ts, t_ps = extract_true_time(slope, intercept, baseline, amplitude)
            ch0_time.append(t_ps)
            # if i < 30:
            #     plot_waveform(i, waveform, slope, intercept, t_ps)
        else:
            #print(f"[Event Ceren27 {i}] Skipped: amplitude below threshold ({amplitude:.2f} < {5*rms:.2f})")
            ch0_time.append(None)

    except Exception as e:
        print(f"[Event Ceren27 {i}] Failed: {e}")
        ch0_time.append(None)

    if i % 1000 == 0:
        print(f"Processed Ceren27 {i}/{nEvents}")
        gc.collect()

    # ---------- Cerenkov29 (ch2) ----------
    try:
        waveform = channels["ch2"][i]
        baseline, rms = find_baseline(waveform)
        region, indices, amplitude = fit_region(waveform, baseline)

        if amplitude >= 5 * rms:
            slope, intercept = fit_rising_edge(region, indices)
            ts, t_ps = extract_true_time(slope, intercept, baseline, amplitude)
            ch2_time.append(t_ps)
            # if i < 10:
            #     plot_waveform(i, waveform, slope, intercept, t_ps)
        else:
            #print(f"[Event Ceren29 {i}] Skipped: amplitude below threshold ({amplitude:.2f} < {5*rms:.2f})")
            ch2_time.append(None)

    except Exception as e:
        print(f"[Event Ceren29 {i}] Failed: {e}")
        ch2_time.append(None)

    if i % 1000 == 0:
        print(f"Processed Ceren29 {i}/{nEvents}")
        gc.collect()
    
        # ---------- Cerenkov31 (ch4) ----------
    try:
        waveform = channels["ch4"][i]
        baseline, rms = find_baseline(waveform)
        region, indices, amplitude = fit_region(waveform, baseline)

        if amplitude >= 5 * rms:
            slope, intercept = fit_rising_edge(region, indices)
            ts, t_ps = extract_true_time(slope, intercept, baseline, amplitude)
            ch4_time.append(t_ps)
            # if i < 10:
            #     plot_waveform(i, waveform, slope, intercept, t_ps)
        else:
            #print(f"[Event Ceren31 {i}] Skipped: amplitude below threshold ({amplitude:.2f} < {5*rms:.2f})")
            ch4_time.append(None)

    except Exception as e:
        print(f"[Event Ceren31 {i}] Failed: {e}")
        ch4_time.append(None)

    if i % 1000 == 0:
        print(f"Processed Ceren31 {i}/{nEvents}")
        gc.collect()
    
        # ---------- Scin28 (ch1) ----------
    try:
        waveform = channels["ch1"][i]
        baseline, rms = find_baseline(waveform)
        region, indices, amplitude = fit_region(waveform, baseline)

        if amplitude >= 5 * rms:
            slope, intercept = fit_rising_edge(region, indices)
            ts, t_ps = extract_true_time(slope, intercept, baseline, amplitude)
            ch1_time.append(t_ps)
            # if i < 10:
            #     plot_waveform(i, waveform, slope, intercept, t_ps)
        else:
            #print(f"[Event SCin28 {i}] Skipped: amplitude below threshold ({amplitude:.2f} < {5*rms:.2f})")
            ch1_time.append(None)

    except Exception as e:
        print(f"[Event SCin28 {i}] Failed: {e}")
        ch1_time.append(None)

    if i % 1000 == 0:
        print(f"Processed Scin28 {i}/{nEvents}")
        gc.collect()
    
        # ---------- Scin30 (ch3) ----------
    try:
        waveform = channels["ch3"][i]
        baseline, rms = find_baseline(waveform)
        region, indices, amplitude = fit_region(waveform, baseline)

        if amplitude >= 5 * rms:
            slope, intercept = fit_rising_edge(region, indices)
            ts, t_ps = extract_true_time(slope, intercept, baseline, amplitude)
            ch3_time.append(t_ps)
            # if i < 10:
            #     plot_waveform(i, waveform, slope, intercept, t_ps)
        else:
            #print(f"[Event SCin30 {i}] Skipped: amplitude below threshold ({amplitude:.2f} < {5*rms:.2f})")
            ch3_time.append(None)

    except Exception as e:
        print(f"[Event SCin30 {i}] Failed: {e}")
        ch3_time.append(None)

    if i % 1000 == 0:
        print(f"Processed Scin30 {i}/{nEvents}")
        gc.collect()
    
        # ---------- Scin32 (ch5) ----------
    try:
        waveform = channels["ch5"][i]
        baseline, rms = find_baseline(waveform)
        region, indices, amplitude = fit_region(waveform, baseline)

        if amplitude >= 5 * rms:
            slope, intercept = fit_rising_edge(region, indices)
            ts, t_ps = extract_true_time(slope, intercept, baseline, amplitude)
            ch5_time.append(t_ps)
            # if i < 10:
            #     plot_waveform(i, waveform, slope, intercept, t_ps)
        else:
            #print(f"[Event SCin32 {i}] Skipped: amplitude below threshold ({amplitude:.2f} < {5*rms:.2f})")
            ch5_time.append(None)

    except Exception as e:
        print(f"[Event SCin32 {i}] Failed: {e}")
        ch5_time.append(None)

    if i % 1000 == 0:
        print(f"Processed Scin32 {i}/{nEvents}")
        gc.collect()
# Done
print("Finished processing.")

# mask = ~np.isnan(mcp1_time) & ~np.isnan(mcp2_time)

# # Perform subtraction only on the non-NaN elements
# result = np.full_like(mcp1_time, np.nan) # Initialize result array with NaNs
# result[mask] = mcp1_time[mask] - mcp2_time[mask]

# === Plotting ===
def plot_and_save(data, title, xlabel, filename):
    data = np.array([np.nan if x is None else x for x in data], dtype=np.float64)

    # Remove NaNs, Infs, and unphysical extremes
    data_clean = data[~np.isnan(data) & ~np.isinf(data) & (data > -1e6) & (data < 1e6)]

    # Compute stats directly from NumPy array
    if len(data_clean) == 0:
        print(f"[WARNING] No valid data for {filename}")
        return

    mean = np.mean(data_clean)
    std = np.std(data_clean)
    entries = len(data_clean)
    plt.figure(figsize=(8, 6))
    plt.hist([x for x in data_clean if x is not None], bins=50, range=[-50000,200000], histtype='stepfilled', color='red', alpha=0.7,
             label=f"N={entries}\nμ={mean:.2f}\nσ={std:.2f}")
    plt.title(f"{title} (Run {run_number})")
    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"/lustre/research/hep/akshriva/Dream-Timing/MCP/Delta_t50_Results/{filename}_run{run_number}.png")
    plt.close()


def plot_diff_and_save(data1, data2, title, xlabel, filename,plot_lim):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    data1 = np.array([np.nan if x is None else x for x in data1], dtype=np.float64)
    data2 = np.array([np.nan if x is None else x for x in data2], dtype=np.float64)

    # --- Mask to filter out NaNs and Infs ---
    valid_mask = (
        ~np.isnan(data1) & ~np.isinf(data1) &
        ~np.isnan(data2) & ~np.isinf(data2)
    )

    if not np.any(valid_mask):
        print(f"[WARNING] No valid data for {filename}")
        return

    # --- Compute difference ---
    data_diff = data1[valid_mask] - data2[valid_mask]

    # --- Apply cut for fitting/stats ---
    fit_mask = (data_diff > -200000) & (data_diff < 200000)
    data_cut = data_diff[fit_mask]

    if len(data_cut) == 0:
        print(f"[WARNING] No data in fitting range for {filename}")
        return

    # --- Define Gaussian function ---
    def gauss(x, A, mu, sigma):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    # --- Histogram parameters ---
    bins = 50
    counts, bin_edges = np.histogram(data_cut, bins=bins, range=plot_lim)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # --- Initial fit guess ---
    p0 = [np.max(counts), np.mean(data_cut), np.std(data_cut)]

    try:
        popt, _ = curve_fit(gauss, bin_centers, counts, p0=p0)
        A_fit, mu_fit, sigma_fit = popt
    except RuntimeError:
        print(f"[WARNING] Gaussian fit failed for {filename}")
        mu_fit, sigma_fit = np.nan, np.nan

    # --- Basic stats on filtered data ---
    mean = np.mean(data_cut)
    std = np.std(data_cut)
    entries = len(data_cut)

    # --- Plotting ---
    plt.figure(figsize=(8, 6))

    # Histogram
    plt.hist(data_cut, bins=bins, range=plot_lim, histtype='stepfilled',
             color='red', alpha=0.7, label="Data")

    # Gaussian fit curve
    if not np.isnan(mu_fit):
        x_fit = np.linspace(plot_lim[0], plot_lim[1], 1000)
        plt.plot(x_fit, gauss(x_fit, *popt), 'k-', linewidth=2, label="Gaussian fit")

    # Stats box (top-left)
    plt.text(0.05, 0.95,
             f"Entries = {entries}\n"
             f"μ = {mean:.2f}\n"
             f"σ = {std:.2f}\n"
             f"Fit μ = {mu_fit:.2f}\n"
             f"Fit σ = {sigma_fit:.2f}",
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='left',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    # Labels and formatting
    plt.title(f"{title} (Run {run_number})")
    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"/lustre/research/hep/akshriva/Dream-Timing/MCP/Delta_t50_Results/{filename}_run{run_number}.pdf")
    plt.close()


###########################################################################################
#################           START OF PLOTTING TIME DIFFERENCES ############################
###########################################################################################

plot_diff_and_save(mcp1_time,mcp2_time, "MCP1 - MCP2 Time Difference", "Δt [ps]", "mcp_delta_between_gaus", [-250, 1500])
plot_and_save(mcp1_time, "MCP1 Time", "Time [ps]", "mcp1_time")
plot_and_save(mcp2_time, "MCP2 Time", "Time [ps]", "mcp2_time")

channel_times = [
    ("ch0", ch0_time),
    ("ch1", ch1_time),
    ("ch2", ch2_time),
    ("ch3", ch3_time),
    ("ch4", ch4_time),
    ("ch5", ch5_time),
]

for name, arr in channel_times:
    plot_diff_and_save(arr, mcp1_time, f"{name} time difference - MCP1", "Δt [ps]", f"mcp1_{name}_delta", [0, 20000])
    plot_diff_and_save(arr, mcp2_time, f"{name} time difference - MCP2", "Δt [ps]", f"mcp2_{name}_delta", [0, 20000])
    plot_and_save(arr, f"{name} Time", "Time [ps]", f"{name}_time")