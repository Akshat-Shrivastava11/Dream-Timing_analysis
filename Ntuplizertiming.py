"""
================================================================================
 TRUE TIMING RECONSTRUCTION — MULTIPROCESSING (MEMORY-SAFE)
================================================================================

Boards processed:
  • Boards 0–3 only

Channel conventions:
  • Channel 8  : trigger channel (t_trigfit)
  • Group 3    : MCP group
  • Group 3 Ch6, Ch7 : MCP timing channels

Definitions:

  Δt_MCP(b) = t_fit(b,3,6) − t_fit(b,3,7)

  ttrue(b,g,c) = t_fit(b,g,c) − t_trigfit(b,g)

  trueminusdeltat(b,g,c) = ttrue(b,g,c) − Δt_MCP(b)

Output ROOT contains:
  • deltat_MCP_BoardX
  • ttrue_BoardX_GroupY_ChannelZ
  • trueminusdeltat_BoardX_GroupY_ChannelZ

Units assumed: ns
================================================================================
"""

import uproot
import numpy as np
import matplotlib.pyplot as plt              # required by user (unused)
from scipy.stats import norm                 # required by user (unused)
from multiprocessing import Pool, cpu_count
from matplotlib.backends.backend_pdf import PdfPages  # required by user (unused)
from tqdm import tqdm
import time
import os                                    # required by user (unused)

# ================= USER SETTINGS =================
INPUT_FILE  = "/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1468_250927145556_TimingDAQ.root"
TREE_NAME   = "EventTree"
OUTPUT_FILE = "/lustre/research/hep/akshriva/Dream-Timing/TRUE-HGtiming/run1468_250927145556_TimingDAQ_afteranalysis.root"

NG = 4
NC = 9
TRUE_BOARDS = range(4)
# =================================================


# ================= HELPERS =================
def br(b, g, c, var="LP2_50"):
    return f"DRS_Board{b}_Group{g}_Channel{c}_{var}"

def valid_channel(g, c):
    if c == 8:
        return False
    if g == 3 and c in (6, 7):
        return False
    return True

def board_branchlist(b):
    needed = set()

    # MCP Δt
    needed.add(br(b, 3, 6))
    needed.add(br(b, 3, 7))

    # Triggers
    for g in range(NG):
        needed.add(br(b, g, 8))

    # Detector channels
    for g in range(NG):
        for c in range(NC):
            if valid_channel(g, c):
                needed.add(br(b, g, c))

    return sorted(needed)


# ================= WORKER =================
def process_board(b):
    """
    Worker computes:
      • Δt_MCP(b)
      • ttrue(b,g,c)
      • trueminusdeltat(b,g,c)
    """
    with uproot.open(INPUT_FILE) as f:
        tree = f[TREE_NAME]
        arrays = tree.arrays(board_branchlist(b), library="np")

    # MCP Δt
    ch6 = br(b, 3, 6)
    ch7 = br(b, 3, 7)
    if ch6 not in arrays or ch7 not in arrays:
        return {}, {}, {}

    deltat_mcp = arrays[ch6] - arrays[ch7]

    ttrue = {}
    trueminusdeltat = {}

    for g in range(NG):
        trig = br(b, g, 8)
        if trig not in arrays:
            continue

        t_trig = arrays[trig]

        for c in range(NC):
            if not valid_channel(g, c):
                continue

            ch = br(b, g, c)
            if ch not in arrays:
                continue

            ttrue_val = arrays[ch] - t_trig
            ttrue[(b, g, c)] = ttrue_val
            trueminusdeltat[(b, g, c)] = ttrue_val - deltat_mcp

    return {b: deltat_mcp}, ttrue, trueminusdeltat


# ================= MAIN =================
def main():
    t0 = time.time()
    print("=" * 90)
    print(" TRUE TIMING RECONSTRUCTION (MULTIPROCESSING)")
    print("=" * 90)

    # ---------- EVENT COUNT ----------
    print("\n[1/4] Reading ROOT file (minimal)")
    with uproot.open(INPUT_FILE) as f:
        tree = f[TREE_NAME]
        probe = br(0, 0, 8)
        key = probe if probe in tree.keys() else tree.keys()[0]
        n_events = len(tree[key].array(library="np"))

    print(f"  → Events : {n_events:,}")
    print(f"  → Boards : {list(TRUE_BOARDS)}")

    # ---------- MULTIPROCESSING ----------
    print("\n[2/4] Computing derived quantities")
    nproc = min(cpu_count(), len(TRUE_BOARDS))
    print(f"  → Using {nproc} processes")

    deltat_mcp = {}
    ttrue = {}
    trueminusdeltat = {}

    with Pool(nproc) as pool:
        for d_b, t_b, tm_b in tqdm(
            pool.imap_unordered(process_board, TRUE_BOARDS),
            total=len(TRUE_BOARDS),
            desc="Boards"
        ):
            deltat_mcp.update(d_b)
            ttrue.update(t_b)
            trueminusdeltat.update(tm_b)

    print(f"  → Δt_MCP boards            : {len(deltat_mcp)}")
    print(f"  → ttrue channels           : {len(ttrue)}")
    print(f"  → trueminusdeltat channels : {len(trueminusdeltat)}")

    # ---------- WRITE ROOT ----------
    print("\n[3/4] Writing output ROOT file")
    out = {}

    # Δt per board
    for b, arr in deltat_mcp.items():
        out[f"deltat_MCP_Board{b}"] = arr

    # ttrue
    for (b, g, c), arr in ttrue.items():
        out[f"ttrue_Board{b}_Group{g}_Channel{c}"] = arr

    # trueminusdeltat
    for (b, g, c), arr in trueminusdeltat.items():
        out[f"trueminusdeltat_Board{b}_Group{g}_Channel{c}"] = arr

    with uproot.recreate(OUTPUT_FILE) as fout:
        fout[TREE_NAME] = out

    # ---------- DONE ----------
    print("\n[4/4] Done")
    print(f"  → Output file : {OUTPUT_FILE}")
    print(f"  → Branches    : {len(out)}")
    print(f"  → Runtime     : {time.time() - t0:.1f} s")
    print("=" * 90)


if __name__ == "__main__":
    main()
