import uproot
import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import os

# ---------------------------
# User settings
# ---------------------------
file_path = "/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1468_250927145556_TimingDAQ.root"


#ALL_CHANNELS = ["DRS_Board0_Group3_Channel6", "DRS_Board0_Group3_Channel7"]
ALL_CHANNELS = ["DRS_Board0_Group3_Channel6",
                "DRS_Board1_Group3_Channel6",
                "DRS_Board2_Group3_Channel6",
                "DRS_Board3_Group3_Channel6",
                "DRS_Board0_Group3_Channel7",
                "DRS_Board1_Group3_Channel7",
                "DRS_Board2_Group3_Channel7",
                "DRS_Board3_Group3_Channel7",]

# ===========================================================
# 2) Polynomial rising-edge t50 extractor
# ===========================================================
def fit_polynomial_t50(time, waveform, ten_idx, ninety_idx):
    print(f" [polyfit] Fit region indices: {ten_idx} → {ninety_idx}")

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


# ===========================================================
# 3) Process a single event (baseline, peak, t50)
# ===========================================================
from scipy.signal import medfilt
import numpy as np

# def process_event(event_idx, wf):
#     """
#     Process one waveform and compute t50 using Akshat’s exact polynomial rise-fit method.
#     Returns: t50 or np.nan
#     """

#     # ----------------------------------------------------
#     # Baseline subtraction
#     # ----------------------------------------------------
#     baseline_samples = 200
#     baseline = np.median(wf[:baseline_samples])
#     baseline_std = 1.4826 * np.median(np.abs(wf[:baseline_samples] - baseline))
#     wf_bs = wf - baseline

#     # ----------------------------------------------------
#     # Smoothing for peak-detection only
#     # ----------------------------------------------------
#     wf_bs_smooth = medfilt(wf_bs, kernel_size=5)

#     # ----------------------------------------------------
#     # Peak detection
#     # ----------------------------------------------------
#     threshold = 3 * baseline_std
#     peaks = np.where(wf_bs_smooth > threshold)[0]
#     if len(peaks) == 0:
#         print(f"Event {event_idx:04d}: No peak above threshold")
#         return np.nan

#     # contiguous groups of peaks
#     groups = np.split(peaks, np.where(np.diff(peaks) != 1)[0] + 1)
#     if len(groups) != 1:
#         print(f"Event {event_idx:04d}: Multiple clusters — skipping")
#         return np.nan

#     rising_region = groups[0]
#     peak_value = np.max(wf_bs_smooth[rising_region])

#     # ----------------------------------------------------
#     # 10% – 90% region detection
#     # ----------------------------------------------------
#     try:
#         ten_rel = np.where(wf_bs_smooth[rising_region] >= 0.10 * peak_value)[0][0]
#         ninety_rel = np.where(wf_bs_smooth[rising_region] >= 0.90 * peak_value)[0][0]
#     except Exception:
#         print(f"Event {event_idx:04d}: Bad 10–90 region")
#         return np.nan

#     ten_idx = rising_region[0] + ten_rel
#     ninety_idx = rising_region[0] + ninety_rel

#     if ten_idx >= ninety_idx:
#         print(f"Event {event_idx:04d}: Invalid window (ten >= ninety)")
#         return np.nan

#     # ----------------------------------------------------
#     # Extract RAW waveform points for polynomial fit
#     # ----------------------------------------------------
#     t = np.arange(len(wf_bs))
#     t_poly = t[ten_idx:ninety_idx+1]
#     y_poly = wf_bs[ten_idx:ninety_idx+1]

#     if len(t_poly) < 6:
#         print(f"Event {event_idx:04d}:  Too few points ({len(t_poly)}) for poly fit")
#         return np.nan

#     # ----------------------------------------------------
#     # 3rd-order polynomial fit
#     # ----------------------------------------------------
#     coeffs = np.polyfit(t_poly, y_poly, 3)
#     a, b, c, d = coeffs

#     half_height = 0.5 * peak_value

#     # Solve cubic for half-height
#     roots = np.roots([a, b, c, d - half_height])
#     real_roots = [
#         r.real for r in roots
#         if np.isreal(r) and ten_idx <= r.real <= ninety_idx
#     ]

#     if len(real_roots) == 0:
#         print(f"Event {event_idx:04d}: No valid t50 root")
#         return np.nan

#     t50 = real_roots[0]
#     print(f"Event {event_idx:04d}: ✔ t50 = {t50:.2f}")

#     return t50


def process_event_loose(event_idx, wf):
    """
    Process one waveform and compute t50 with a looser criterion.
    Returns: t50 or np.nan if completely unusable.
    """

    # ----------------------------------------------------
    # Baseline subtraction
    # ----------------------------------------------------
    baseline_samples = 200
    baseline = np.median(wf[:baseline_samples])
    baseline_std = 1.4826 * np.median(np.abs(wf[:baseline_samples] - baseline))
    wf_bs = wf - baseline

    # ----------------------------------------------------
    # Smoothing for peak detection
    # ----------------------------------------------------
    wf_bs_smooth = medfilt(wf_bs, kernel_size=5)

    # ----------------------------------------------------
    # Peak detection (looser: 2 * baseline_std)
    # ----------------------------------------------------
    threshold = 3 * baseline_std
    peaks = np.where(wf_bs_smooth > threshold)[0]

    if len(peaks) == 0:
        # No peaks, fallback: use the whole waveform as rising region
        rising_region = np.arange(len(wf_bs))
        peak_value = np.max(wf_bs_smooth)
    else:
        # Merge all contiguous clusters into one rising region
        groups = np.split(peaks, np.where(np.diff(peaks) != 1)[0] + 1)
        rising_region = np.concatenate(groups)
        peak_value = np.max(wf_bs_smooth[rising_region])

    # ----------------------------------------------------
    # 10% – 90% region detection (looser)
    # ----------------------------------------------------
    ten_candidates = np.where(wf_bs_smooth[rising_region] >= 0.10 * peak_value)[0]
    ninety_candidates = np.where(wf_bs_smooth[rising_region] >= 0.90 * peak_value)[0]

    if len(ten_candidates) == 0:
        ten_idx = rising_region[0]
    else:
        ten_idx = rising_region[ten_candidates[0]]

    if len(ninety_candidates) == 0:
        ninety_idx = rising_region[-1]
    else:
        ninety_idx = rising_region[ninety_candidates[0]]

    # Safety check
    if ten_idx >= ninety_idx:
        ninety_idx = ten_idx + 1
        if ninety_idx >= len(wf_bs):
            ninety_idx = len(wf_bs) - 1

    # ----------------------------------------------------
    # Polynomial fit (3rd order)
    # ----------------------------------------------------
    t_poly = np.arange(ten_idx, ninety_idx + 1)
    y_poly = wf_bs[ten_idx:ninety_idx + 1]

    if len(t_poly) < 4:
        # Too few points, return midpoint as fallback
        t50 = ten_idx + (ninety_idx - ten_idx) / 2
        return t50

    coeffs = np.polyfit(t_poly, y_poly, 3)
    a, b, c, d = coeffs
    half_height = 0.5 * peak_value

    roots = np.roots([a, b, c, d - half_height])
    real_roots = [r.real for r in roots if np.isreal(r) and ten_idx <= r.real <= ninety_idx]

    if len(real_roots) == 0:
        # Looser fallback: midpoint
        t50 = ten_idx + (ninety_idx - ten_idx) / 2
    else:
        t50 = real_roots[0]

    real_t50 =  0.2 * t50  # convert to ns assuming 0.2 ns sampling
    return real_t50


def process_event_tighter(event_idx, wf):
    """
    Process one waveform and compute t50 with slightly tighter criteria.
    Returns: t50 or np.nan if completely unusable.
    """

    # ----------------------------------------------------
    # Baseline subtraction
    # ----------------------------------------------------
    baseline_samples = 200
    baseline = np.median(wf[:baseline_samples])
    baseline_std = 1.4826 * np.median(np.abs(wf[:baseline_samples] - baseline))
    wf_bs = wf - baseline

    # ----------------------------------------------------
    # Smoothing for peak detection
    # ----------------------------------------------------
    wf_bs_smooth = medfilt(wf_bs, kernel_size=5)

    # ----------------------------------------------------
    # Peak detection (tighter: 3 × baseline_std)
    # ----------------------------------------------------
    threshold = 3 * baseline_std
    peaks = np.where(wf_bs_smooth > threshold)[0]

    if len(peaks) == 0:
        print(f"Event {event_idx:04d}: ❌ No peak above threshold")
        return np.nan

    # Split into contiguous groups
    groups = np.split(peaks, np.where(np.diff(peaks) != 1)[0] + 1)
    
    # Take only the group with the **highest peak**
    peak_values = [np.max(wf_bs_smooth[g]) for g in groups]
    best_group_idx = np.argmax(peak_values)
    rising_region = groups[best_group_idx]
    peak_value = peak_values[best_group_idx]

    # ----------------------------------------------------
    # 10% – 90% region detection (tighter)
    # ----------------------------------------------------
    ten_candidates = np.where(wf_bs_smooth[rising_region] >= 0.10 * peak_value)[0]
    ninety_candidates = np.where(wf_bs_smooth[rising_region] >= 0.90 * peak_value)[0]

    if len(ten_candidates) == 0 or len(ninety_candidates) == 0:
        print(f"Event {event_idx:04d}: ❌ Cannot define 10–90% region")
        return np.nan

    ten_idx = rising_region[ten_candidates[0]]
    ninety_idx = rising_region[ninety_candidates[0]]

    if ten_idx >= ninety_idx:
        print(f"Event {event_idx:04d}: ❌ Invalid 10–90% window")
        return np.nan

    # ----------------------------------------------------
    # Polynomial fit (3rd order)
    # ----------------------------------------------------
    t_poly = np.arange(ten_idx, ninety_idx + 1)
    y_poly = wf_bs[ten_idx:ninety_idx + 1]

    if len(t_poly) < 6:
        print(f"Event {event_idx:04d}: ❌ Too few points for poly fit")
        return np.nan

    coeffs = np.polyfit(t_poly, y_poly, 3)
    a, b, c, d = coeffs
    half_height = 0.5 * peak_value

    roots = np.roots([a, b, c, d - half_height])
    real_roots = [r.real for r in roots if np.isreal(r) and ten_idx <= r.real <= ninety_idx]

    if len(real_roots) == 0:
        print(f"Event {event_idx:04d}: ❌ No valid t50 root")
        return np.nan

    t50 = real_roots[0]
    print(f"Event {event_idx:04d}: ✔ t50 = {t50:.2f}")
    real_t50 =  0.2 * t50  # convert to ns assuming 0.2 ns sampling
    return real_t50


SAMPLE_TIME_NS = 0.2  
def process_event_medium(event_idx, wf):
    SAMPLE_TIME_NS = 0.2
    baseline_samples = 200
    baseline = np.median(wf[:baseline_samples])
    baseline_std = 1.4826 * np.median(np.abs(wf[:baseline_samples] - baseline))
    wf_bs = wf - baseline

    # flip negative MCP pulse
    wf_bs_flip = -wf_bs

    wf_smooth = medfilt(wf_bs_flip, kernel_size=5)
    threshold = 2.5 * baseline_std
    peaks = np.where(wf_smooth > threshold)[0]

    if len(peaks) == 0:
        rising_region = np.arange(len(wf_bs_flip))
    else:
        groups = np.split(peaks, np.where(np.diff(peaks) != 1)[0] + 1)
        rising_region = np.concatenate(groups)

    peak_value = np.max(wf_smooth[rising_region])

    ten_candidates = np.where(wf_smooth[rising_region] >= 0.10 * peak_value)[0]
    ninety_candidates = np.where(wf_smooth[rising_region] >= 0.90 * peak_value)[0]

    ten_idx = rising_region[ten_candidates[0]] if len(ten_candidates) else rising_region[0]
    ninety_idx = rising_region[ninety_candidates[0]] if len(ninety_candidates) else rising_region[-1]

    if ten_idx >= ninety_idx:
        ninety_idx = min(ten_idx + 1, len(wf_bs_flip)-1)

    t_poly = np.arange(ten_idx, ninety_idx+1)
    y_poly = wf_bs_flip[ten_idx:ninety_idx+1]

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
    return t50_ns

# ===========================================================
# 4) Plot histogram + save PDF
# ===========================================================
def save_t50_pdf(channel, t50_values, outdir="HG-Dream_timing"):
    channel_dir = os.path.join(outdir, channel, "t50_hists")
    os.makedirs(outdir, exist_ok=True)
    pdf_path = outdir + f"/{channel}_t50_hist.pdf"
    print(f"   [plot] Saving t50 PDF for {channel} → {outdir}")
    plt.figure(figsize=(8, 5))
    clean = t50_values[~np.isnan(t50_values)]
    plt.hist(clean, bins=60, alpha=0.7, histtype='step', color='blue')
    plt.title(f"t50 Distribution — {channel}")
    plt.xlabel("t50 (ns)")
    plt.ylabel("Events")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()


# Save a single waveform plot with baseline & fit overlay
def save_waveform_plot(channel, event_idx, wf, time, ten_idx, ninety_idx, t50, outdir="HG-Dream_timing"):
    channel_dir = os.path.join(outdir, channel, "waveforms")
    os.makedirs(channel_dir, exist_ok=True)
    pdf_path = os.path.join(channel_dir, f"{channel}_event{event_idx:04d}.pdf")

    plt.figure(figsize=(10, 4))
    plt.plot(time, -wf, label="Baseline-subtracted", alpha=0.7)
    plt.axvspan(time[ten_idx], time[ninety_idx], color="yellow", alpha=0.3, label="10–90% region")
    if not np.isnan(t50):
        plt.axvline(t50, color="red", linestyle="--", label=f"t50 = {t50:.2f} ns")
    plt.title(f"{channel} — Event {event_idx}")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()


# ===========================================================
# 5) Process one *channel* fully
# ===========================================================
def process_channel(tree, channel, max_events=7000, output_dir="HG-Dream_timing"):
    """
    Process all events for a given channel, compute t50 if possible, 
    and save waveform plots and t50 histogram.
    """

    print("\n=====================================================")
    print(f"   PROCESSING CHANNEL: {channel}")
    print("=====================================================")

    branch = channel
    if branch not in tree.keys():
        print(f"   [ERROR] Branch not found: {branch}")
        return

    wf_array = tree[branch].array(library="np")
    n_events = min(max_events, len(wf_array))
    print(f"   Total events = {len(wf_array)}, using = {n_events}")

    t50_list = []
    time = np.arange(len(wf_array[0])) * 0.2  # assuming 0.2 ns sampling

    for i in range(n_events):
        wf = wf_array[i]

        # ----------------------------
        # Baseline subtraction
        # ----------------------------
        baseline_samples = 200
        baseline = np.median(wf[:baseline_samples])
        wf_bs = wf - baseline

        # ----------------------------
        # Compute t50 (if possible)
        # ----------------------------
        t50 = process_event_medium(i, wf)
        t50_list.append(t50)

        # ----------------------------
        # Determine 10–90% region for plotting
        # ----------------------------
        try:
            wf_smooth = medfilt(wf_bs, kernel_size=5)
            peaks = np.where(wf_smooth > 3 * 1.4826 * np.median(np.abs(wf[:baseline_samples] - baseline)))[0]

            if len(peaks) >= 1:
                groups = np.split(peaks, np.where(np.diff(peaks) != 1)[0] + 1)
                rising_region = groups[0]
                ten_rel = np.where(wf_smooth[rising_region] >= 0.10 * np.max(wf_smooth[rising_region]))[0][0]
                ninety_rel = np.where(wf_smooth[rising_region] >= 0.90 * np.max(wf_smooth[rising_region]))[0][0]
                ten_idx = rising_region[0] + ten_rel
                ninety_idx = rising_region[0] + ninety_rel
            else:
                ten_idx, ninety_idx = 0, len(wf_bs)-1
        except Exception:
            ten_idx, ninety_idx = 0, len(wf_bs)-1

        # ----------------------------
        # Save waveform plot
        # ----------------------------
        save_waveform_plot(channel, i, wf_bs, time, ten_idx, ninety_idx, t50, output_dir)

    # ----------------------------
    # Save t50 histogram
    # ----------------------------
    save_t50_pdf(channel, np.array(t50_list), output_dir)



# ===========================================================
# 6) MAIN controller function
# ===========================================================
def run_all_channels(root_file, channel_subset=None, max_events=200, output_dir="HG-Dream_timing"):
    print("\n========== Loading ROOT File ==========")
    tree = uproot.open(root_file)["EventTree"]

    if channel_subset is None:
        channel_subset = ALL_CHANNELS

    print("\n========== Channels to Process ==========")
    for ch in channel_subset:
        print("   ", ch)

    for ch in channel_subset:
        process_channel(tree, ch, max_events=max_events, output_dir=output_dir)


# ===========================================================
# 7) Example run
# ===========================================================
if __name__ == "__main__":
    file_path = "/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1468_250927145556_TimingDAQ.root"
    
    run_all_channels(
        root_file=file_path,
        channel_subset=ALL_CHANNELS,
        max_events=50000,
        output_dir="MCPWaveform_deltat_analysis"
    )
