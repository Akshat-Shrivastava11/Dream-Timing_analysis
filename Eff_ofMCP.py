#!/usr/bin/env python3
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import os

# ============================================================
# User configurable paths
# ============================================================

INPUT_FILE = "/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1468_250927145556_TimingDAQ.root"
FAIL_DIR = "fail_events"
os.makedirs(FAIL_DIR, exist_ok=True)

# ============================================================
# Pure threshold-based channel firing
# ============================================================

def channel_fired(wf, kernel_size=5, threshold_factor=2.5):
    """
    Returns True if the channel has ANY sample above (baseline + N*sigma).
    No timing extraction. No T50.
    """
    wf_flip = -wf
    wf_smooth = medfilt(wf_flip, kernel_size=kernel_size)

    baseline = np.mean(wf_smooth[:50])
    sigma = np.std(wf_smooth[:50])
    threshold = baseline + threshold_factor * sigma

    return np.any(wf_smooth > threshold)


# ============================================================
# Load File & Waveforms
# ============================================================

print("Opening ROOT file...")
root_file = uproot.open(INPUT_FILE)
tree = root_file["EventTree"]

waveforms = {}
for b in range(4):
    waveforms[b] = {}
    for ch in (6, 7):
        branch = f"DRS_Board{b}_Group3_Channel{ch}"
        waveforms[b][ch] = tree[branch].array(library="np")

total_events = len(waveforms[0][6])
print(f"Total events detected: {total_events}")

# ============================================================
# MCP Efficiency Calculation
# ============================================================

N_BOARDS = 4
CHANNELS = [6, 7]

global_fail_events = []  # only for events with ≥1 fired board

print("Processing events...")

for iev in range(total_events):

    board_fired = {}

    # Decide if each board fired
    for b in range(N_BOARDS):
        fired_ch6 = channel_fired(waveforms[b][6][iev])
        fired_ch7 = channel_fired(waveforms[b][7][iev])

        board_fired[b] = fired_ch6 and fired_ch7

    n_fired = sum(board_fired.values())

    # If NO board fired → skip this event entirely
    if n_fired == 0:
        continue

    # Event is counted
    event_failed = (n_fired != N_BOARDS)
    global_fail_events.append(event_failed)

    # ---------------------------------------------------------
    # Plot failing events (using Matplotlib gallery style)
    # ---------------------------------------------------------
    if event_failed:
        plt.style.use("ggplot")   # Apply gallery-inspired style

        fig, axes = plt.subplots(4, 2, figsize=(12, 14), sharex=True)
        fig.suptitle(f"Event {iev}: At least one board failed", fontsize=16)

        for b in range(N_BOARDS):
            for j, ch in enumerate(CHANNELS):
                ax = axes[b][j]
                wf = waveforms[b][ch][iev]
                ax.plot(wf, linewidth=1.2)
                ax.set_title(f"Board {b} Ch{ch} — {'FIRED' if board_fired[b] else 'FAILED'}")

        plt.tight_layout()
        plt.savefig(f"{FAIL_DIR}/event_{iev}.png")
        plt.close()


# ============================================================
# Compute Efficiency
# ============================================================

N_counted = len(global_fail_events)
N_fail = sum(global_fail_events)
N_pass = N_counted - N_fail

efficiency = N_pass / N_counted if N_counted > 0 else 0.0

print("\n============= MCP Efficiency =============")
print(f"Events processed              : {total_events}")
print(f"Events with ≥1 fired board    : {N_counted}")
print(f"Events all 4 boards fired     : {N_pass}")
print(f"Events with failures          : {N_fail}")
print(f"Efficiency                    : {efficiency:.6f}")
print("==========================================\n")

# ============================================================
# Save efficiency to text file
# ============================================================

with open("mcp_efficiency.txt", "w") as f:
    f.write("MCP Efficiency Summary\n")
    f.write("======================\n")
    f.write(f"Events processed           : {total_events}\n")
    f.write(f"Events counted (≥1 fired)  : {N_counted}\n")
    f.write(f"Events all boards fired    : {N_pass}\n")
    f.write(f"Events with failures       : {N_fail}\n")
    f.write(f"Efficiency                 : {efficiency:.6f}\n")

print("Saved efficiency results to mcp_efficiency.txt")
print(f"Failing-event plots saved in: {FAIL_DIR}/")
