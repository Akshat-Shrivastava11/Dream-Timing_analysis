import uproot
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import os
from scipy.signal import medfilt
import numpy as np
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import os

ALL_CHANNELS = [
    "DRS_Board0_Group0_Channel0", "DRS_Board0_Group0_Channel1", "DRS_Board0_Group0_Channel2",
    "DRS_Board0_Group0_Channel3", "DRS_Board0_Group0_Channel4", "DRS_Board0_Group0_Channel5",
    "DRS_Board0_Group0_Channel6", "DRS_Board0_Group0_Channel7", "DRS_Board0_Group0_Channel8",

    "DRS_Board0_Group1_Channel0", "DRS_Board0_Group1_Channel1", "DRS_Board0_Group1_Channel2",
    "DRS_Board0_Group1_Channel3", "DRS_Board0_Group1_Channel4", "DRS_Board0_Group1_Channel5",
    "DRS_Board0_Group1_Channel6", "DRS_Board0_Group1_Channel7", "DRS_Board0_Group1_Channel8",

    "DRS_Board0_Group2_Channel0", "DRS_Board0_Group2_Channel1", "DRS_Board0_Group2_Channel2",
    "DRS_Board0_Group2_Channel3", "DRS_Board0_Group2_Channel4", "DRS_Board0_Group2_Channel5",
    "DRS_Board0_Group2_Channel6", "DRS_Board0_Group2_Channel7", "DRS_Board0_Group2_Channel8",

    "DRS_Board0_Group3_Channel0", "DRS_Board0_Group3_Channel1", "DRS_Board0_Group3_Channel2",
    "DRS_Board0_Group3_Channel3", "DRS_Board0_Group3_Channel4", "DRS_Board0_Group3_Channel5",
    "DRS_Board0_Group3_Channel6", "DRS_Board0_Group3_Channel7", "DRS_Board0_Group3_Channel8",

    "DRS_Board1_Group0_Channel0", "DRS_Board1_Group0_Channel1", "DRS_Board1_Group0_Channel2",
    "DRS_Board1_Group0_Channel3", "DRS_Board1_Group0_Channel4", "DRS_Board1_Group0_Channel5",
    "DRS_Board1_Group0_Channel6", "DRS_Board1_Group0_Channel7", "DRS_Board1_Group0_Channel8",

    "DRS_Board1_Group1_Channel0", "DRS_Board1_Group1_Channel1", "DRS_Board1_Group1_Channel2",
    "DRS_Board1_Group1_Channel3", "DRS_Board1_Group1_Channel4", "DRS_Board1_Group1_Channel5",
    "DRS_Board1_Group1_Channel6", "DRS_Board1_Group1_Channel7", "DRS_Board1_Group1_Channel8",

    "DRS_Board1_Group2_Channel0", "DRS_Board1_Group2_Channel1", "DRS_Board1_Group2_Channel2",
    "DRS_Board1_Group2_Channel3", "DRS_Board1_Group2_Channel4", "DRS_Board1_Group2_Channel5",
    "DRS_Board1_Group2_Channel6", "DRS_Board1_Group2_Channel7", "DRS_Board1_Group2_Channel8",

    "DRS_Board1_Group3_Channel0", "DRS_Board1_Group3_Channel1", "DRS_Board1_Group3_Channel2",
    "DRS_Board1_Group3_Channel3", "DRS_Board1_Group3_Channel4", "DRS_Board1_Group3_Channel5",
    "DRS_Board1_Group3_Channel6", "DRS_Board1_Group3_Channel7", "DRS_Board1_Group3_Channel8",

    "DRS_Board2_Group0_Channel0", "DRS_Board2_Group0_Channel1", "DRS_Board2_Group0_Channel2",
    "DRS_Board2_Group0_Channel3", "DRS_Board2_Group0_Channel4", "DRS_Board2_Group0_Channel5",
    "DRS_Board2_Group0_Channel6", "DRS_Board2_Group0_Channel7", "DRS_Board2_Group0_Channel8",

    "DRS_Board2_Group1_Channel0", "DRS_Board2_Group1_Channel1", "DRS_Board2_Group1_Channel2",
    "DRS_Board2_Group1_Channel3", "DRS_Board2_Group1_Channel4", "DRS_Board2_Group1_Channel5",
    "DRS_Board2_Group1_Channel6", "DRS_Board2_Group1_Channel7", "DRS_Board2_Group1_Channel8",

    "DRS_Board2_Group2_Channel0", "DRS_Board2_Group2_Channel1", "DRS_Board2_Group2_Channel2",
    "DRS_Board2_Group2_Channel3", "DRS_Board2_Group2_Channel4", "DRS_Board2_Group2_Channel5",
    "DRS_Board2_Group2_Channel6", "DRS_Board2_Group2_Channel7", "DRS_Board2_Group2_Channel8",

    "DRS_Board2_Group3_Channel0", "DRS_Board2_Group3_Channel1", "DRS_Board2_Group3_Channel2",
    "DRS_Board2_Group3_Channel3", "DRS_Board2_Group3_Channel4", "DRS_Board2_Group3_Channel5",
    "DRS_Board2_Group3_Channel6", "DRS_Board2_Group3_Channel7", "DRS_Board2_Group3_Channel8",

    "DRS_Board3_Group0_Channel0", "DRS_Board3_Group0_Channel1", "DRS_Board3_Group0_Channel2",
    "DRS_Board3_Group0_Channel3", "DRS_Board3_Group0_Channel4", "DRS_Board3_Group0_Channel5",
    "DRS_Board3_Group0_Channel6", "DRS_Board3_Group0_Channel7", "DRS_Board3_Group0_Channel8",

    "DRS_Board3_Group1_Channel0", "DRS_Board3_Group1_Channel1", "DRS_Board3_Group1_Channel2",
    "DRS_Board3_Group1_Channel3", "DRS_Board3_Group1_Channel4", "DRS_Board3_Group1_Channel5",
    "DRS_Board3_Group1_Channel6", "DRS_Board3_Group1_Channel7", "DRS_Board3_Group1_Channel8",

    "DRS_Board3_Group2_Channel0", "DRS_Board3_Group2_Channel1", "DRS_Board3_Group2_Channel2",
    "DRS_Board3_Group2_Channel3", "DRS_Board3_Group2_Channel4", "DRS_Board3_Group2_Channel5",
    "DRS_Board3_Group2_Channel6", "DRS_Board3_Group2_Channel7", "DRS_Board3_Group2_Channel8",

    "DRS_Board3_Group3_Channel0", "DRS_Board3_Group3_Channel1", "DRS_Board3_Group3_Channel2",
    "DRS_Board3_Group3_Channel3", "DRS_Board3_Group3_Channel4", "DRS_Board3_Group3_Channel5",
    "DRS_Board3_Group3_Channel6", "DRS_Board3_Group3_Channel7", "DRS_Board3_Group3_Channel8",

    "DRS_Board4_Group0_Channel0", "DRS_Board4_Group0_Channel1", "DRS_Board4_Group0_Channel2",
    "DRS_Board4_Group0_Channel3", "DRS_Board4_Group0_Channel4", "DRS_Board4_Group0_Channel5",
    "DRS_Board4_Group0_Channel6", "DRS_Board4_Group0_Channel7", "DRS_Board4_Group0_Channel8",

    "DRS_Board4_Group1_Channel0", "DRS_Board4_Group1_Channel1", "DRS_Board4_Group1_Channel2",
    "DRS_Board4_Group1_Channel3", "DRS_Board4_Group1_Channel4", "DRS_Board4_Group1_Channel5",
    "DRS_Board4_Group1_Channel6", "DRS_Board4_Group1_Channel7", "DRS_Board4_Group1_Channel8",

    "DRS_Board4_Group2_Channel0", "DRS_Board4_Group2_Channel1", "DRS_Board4_Group2_Channel2",
    "DRS_Board4_Group2_Channel3", "DRS_Board4_Group2_Channel4", "DRS_Board4_Group2_Channel5",
    "DRS_Board4_Group2_Channel6", "DRS_Board4_Group2_Channel7", "DRS_Board4_Group2_Channel8",

    "DRS_Board4_Group3_Channel0", "DRS_Board4_Group3_Channel1", "DRS_Board4_Group3_Channel2",
    "DRS_Board4_Group3_Channel3", "DRS_Board4_Group3_Channel4", "DRS_Board4_Group3_Channel5",
    "DRS_Board4_Group3_Channel6", "DRS_Board4_Group3_Channel7", "DRS_Board4_Group3_Channel8",

    "DRS_Board5_Group0_Channel0", "DRS_Board5_Group0_Channel1", "DRS_Board5_Group0_Channel2",
    "DRS_Board5_Group0_Channel3", "DRS_Board5_Group0_Channel4", "DRS_Board5_Group0_Channel5",
    "DRS_Board5_Group0_Channel6", "DRS_Board5_Group0_Channel7", "DRS_Board5_Group0_Channel8",

    "DRS_Board5_Group1_Channel0", "DRS_Board5_Group1_Channel1", "DRS_Board5_Group1_Channel2",
    "DRS_Board5_Group1_Channel3", "DRS_Board5_Group1_Channel4", "DRS_Board5_Group1_Channel5",
    "DRS_Board5_Group1_Channel6", "DRS_Board5_Group1_Channel7", "DRS_Board5_Group1_Channel8",

    "DRS_Board5_Group2_Channel0", "DRS_Board5_Group2_Channel1", "DRS_Board5_Group2_Channel2",
    "DRS_Board5_Group2_Channel3", "DRS_Board5_Group2_Channel4", "DRS_Board5_Group2_Channel5",
    "DRS_Board5_Group2_Channel6", "DRS_Board5_Group2_Channel7", "DRS_Board5_Group2_Channel8",

    "DRS_Board5_Group3_Channel0", "DRS_Board5_Group3_Channel1", "DRS_Board5_Group3_Channel2",
    "DRS_Board5_Group3_Channel3", "DRS_Board5_Group3_Channel4", "DRS_Board5_Group3_Channel5",
    "DRS_Board5_Group3_Channel6", "DRS_Board5_Group3_Channel7", "DRS_Board5_Group3_Channel8",

    "DRS_Board6_Group0_Channel0", "DRS_Board6_Group0_Channel1", "DRS_Board6_Group0_Channel2",
    "DRS_Board6_Group0_Channel3", "DRS_Board6_Group0_Channel4", "DRS_Board6_Group0_Channel5",
    "DRS_Board6_Group0_Channel6", "DRS_Board6_Group0_Channel7", "DRS_Board6_Group0_Channel8",

    "DRS_Board6_Group1_Channel0", "DRS_Board6_Group1_Channel1", "DRS_Board6_Group1_Channel2",
    "DRS_Board6_Group1_Channel3", "DRS_Board6_Group1_Channel4", "DRS_Board6_Group1_Channel5",
    "DRS_Board6_Group1_Channel6", "DRS_Board6_Group1_Channel7", "DRS_Board6_Group1_Channel8",

    "DRS_Board6_Group2_Channel0", "DRS_Board6_Group2_Channel1", "DRS_Board6_Group2_Channel2",
    "DRS_Board6_Group2_Channel3", "DRS_Board6_Group2_Channel4", "DRS_Board6_Group2_Channel5",
    "DRS_Board6_Group2_Channel6", "DRS_Board6_Group2_Channel7", "DRS_Board6_Group2_Channel8",

    "DRS_Board6_Group3_Channel0", "DRS_Board6_Group3_Channel1", "DRS_Board6_Group3_Channel2",
    "DRS_Board6_Group3_Channel3", "DRS_Board6_Group3_Channel4", "DRS_Board6_Group3_Channel5",
    "DRS_Board6_Group3_Channel6", "DRS_Board6_Group3_Channel7", "DRS_Board6_Group3_Channel8",

    "DRS_Board7_Group0_Channel0", "DRS_Board7_Group0_Channel1", "DRS_Board7_Group0_Channel2",
    "DRS_Board7_Group0_Channel3", "DRS_Board7_Group0_Channel4", "DRS_Board7_Group0_Channel5",
    "DRS_Board7_Group0_Channel6", "DRS_Board7_Group0_Channel7", "DRS_Board7_Group0_Channel8",

    "DRS_Board7_Group1_Channel0", "DRS_Board7_Group1_Channel1", "DRS_Board7_Group1_Channel2",
    "DRS_Board7_Group1_Channel3", "DRS_Board7_Group1_Channel4", "DRS_Board7_Group1_Channel5",
    "DRS_Board7_Group1_Channel6", "DRS_Board7_Group1_Channel7", "DRS_Board7_Group1_Channel8",

    "DRS_Board7_Group2_Channel0", "DRS_Board7_Group2_Channel1", "DRS_Board7_Group2_Channel2",
    "DRS_Board7_Group2_Channel3", "DRS_Board7_Group2_Channel4", "DRS_Board7_Group2_Channel5",
    "DRS_Board7_Group2_Channel6", "DRS_Board7_Group2_Channel7", "DRS_Board7_Group2_Channel8",

    "DRS_Board7_Group3_Channel0", "DRS_Board7_Group3_Channel1", "DRS_Board7_Group3_Channel2",
    "DRS_Board7_Group3_Channel3", "DRS_Board7_Group3_Channel4", "DRS_Board7_Group3_Channel5",
    "DRS_Board7_Group3_Channel6", "DRS_Board7_Group3_Channel7", "DRS_Board7_Group3_Channel8",
]




# ===========================================================
# 1) ALL CHANNELS (you can give any subset from this list)
# ===========================================================


# ===========================================================
# 2) Polynomial rising-edge t50 extractor
# ===========================================================
def fit_polynomial_t50(time, waveform, ten_idx, ninety_idx):
    print(f"     [polyfit] Fit region indices: {ten_idx} → {ninety_idx}")

    x = time[ten_idx:ninety_idx]
    y = waveform[ten_idx:ninety_idx]

    if len(x) < 4:
        print("     [polyfit] ERROR: Not enough points.")
        return np.nan

    # Fit cubic polynomial
    coeffs = np.polyfit(x, y, deg=3)
    poly = np.poly1d(coeffs)

    # Solve for 50%
    half_val = 0.5 * np.max(y)
    print(f"     [polyfit] Half-height = {half_val:.3f}")

    roots = np.roots(poly - half_val)
    real_roots = roots[np.isreal(roots)].real

    if len(real_roots) == 0:
        print("     [polyfit] ERROR: No real t50 root.")
        return np.nan

    t50 = real_roots[0]
    print(f"     [polyfit] t50 extracted = {t50:.3f}")

    return t50



SAMPLE_TIME_NS = 0.2  
def process_event_medium(event_idx, wf):
    """
    Compute t50 for one waveform (medium-tight criteria).
    Always returns t50 in ns.
    """
    baseline_samples = 200
    baseline = np.median(wf[:baseline_samples])
    baseline_std = 1.4826 * np.median(np.abs(wf[:baseline_samples] - baseline))
    wf_bs = wf - baseline

    wf_smooth = medfilt(wf_bs, kernel_size=5)
    threshold = 2.5 * baseline_std
    peaks = np.where(wf_smooth > threshold)[0]

    if len(peaks) == 0:
        rising_region = np.arange(len(wf_bs))
    else:
        groups = np.split(peaks, np.where(np.diff(peaks) != 1)[0] + 1)
        rising_region = np.concatenate(groups)

    peak_value = np.max(wf_smooth[rising_region])

    ten_candidates = np.where(wf_smooth[rising_region] >= 0.10 * peak_value)[0]
    ninety_candidates = np.where(wf_smooth[rising_region] >= 0.90 * peak_value)[0]

    ten_idx = rising_region[ten_candidates[0]] if len(ten_candidates) else rising_region[0]
    ninety_idx = rising_region[ninety_candidates[0]] if len(ninety_candidates) else rising_region[-1]

    if ten_idx >= ninety_idx:
        ninety_idx = min(ten_idx + 1, len(wf_bs)-1)

    t_poly = np.arange(ten_idx, ninety_idx+1)
    y_poly = wf_bs[ten_idx:ninety_idx+1]

    if len(t_poly) < 4:
        t50_idx = ten_idx + (ninety_idx - ten_idx)/2
    else:
        coeffs = np.polyfit(t_poly, y_poly, 3)
        a,b,c,d = coeffs
        half_height = 0.5 * peak_value
        roots = np.roots([a,b,c,d - half_height])
        real_roots = [r.real for r in roots if np.isreal(r) and ten_idx <= r.real <= ninety_idx]
        t50_idx = real_roots[0] if real_roots else ten_idx + (ninety_idx - ten_idx)/2

    t50_ns = t50_idx * SAMPLE_TIME_NS
    print(f"     [process_event_medium] Event {event_idx}: t50 = {t50_ns:.3f} ns")
    return t50_ns

import uproot
import numpy as np
import matplotlib.pyplot as plt

# Assuming ALL_CHANNELS and process_event_medium() are already defined

def t50_vs_channel(root_file, channels=None, event_indices=[0,1]):
    """
    Compute t50 for selected events and plot t50 vs channel.
    """
    tree = uproot.open(root_file)["EventTree"]

    if channels is None:
        channels = ALL_CHANNELS

    # Store t50 values for each event per channel
    t50_data = {event_idx: [] for event_idx in event_indices}

    for ch in channels:
        if ch not in tree.keys():
            print(f"Channel not found: {ch}")
            for event_idx in event_indices:
                t50_data[event_idx].append(np.nan)
            continue

        wf_array = tree[ch].array(library="np")
        for event_idx in event_indices:
            if event_idx >= len(wf_array):
                t50_data[event_idx].append(np.nan)
            else:
                wf = wf_array[event_idx]
                t50 = process_event_medium(event_idx, wf)
                t50_data[event_idx].append(t50)

    # Make the plot
    plt.figure(figsize=(14,6))
    for event_idx in event_indices:
        plt.plot(channels, t50_data[event_idx], marker='o', linestyle='-', label=f"Event {event_idx}")

    plt.xticks(rotation=90)
    plt.xlabel("Channel")
    plt.ylabel("t50 (ns)")
    plt.title("t50 vs Channel")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("t50_vs_channel.pdf")


if __name__ == "__main__":
    file_path = "/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1468_250927145556_TimingDAQ.root"
    t50_vs_channel(file_path, channels=ALL_CHANNELS, event_indices=[0])
