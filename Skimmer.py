"""
================================================================================
 TRUE TIMING RECONSTRUCTION — MULTIPROCESSING (MEMORY-SAFE)
================================================================================

Boards processed:
  • Boards 0–3 only

Channel conventions:
  • Channel 8        : trigger channel (per group)
  • Group 3 Ch6, Ch7 : MCP channels
  • MCP reference    : Group 3, Channel 7 (US MCP)

Definitions (applied uniformly to ALL channels):

  t_final(b,g,c) =
      ( t_fit(b,g,c) − t_trig(b,g) )
    − ( t_fit(b,3,7) − t_trig(b,g) )

Notes:
  • The SAME trigger (Group g, Channel 8) is used in both subtractions.
  • MCP correction is board-local and uses Group 3 MCP7.
  • No cross-group trigger mixing is performed.
  • Trigger and MCP channels are included and reduce to zero by construction.
  • Uniform output is produced for downstream analysis.

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
OUTPUT_FILE = "/lustre/research/hep/akshriva/Dream-Timing/TRUE-HGtiming/skimmed_files/run1468_250927145556_TimingDAQ_postanalysis_allchannels.root"

NG = 4
NC = 9
TRUE_BOARDS = range(4)
# =================================================


# ================= HELPERS =================
def br(b, g, c, var="LP2_50"):
    return f"DRS_Board{b}_Group{g}_Channel{c}_{var}"

def board_branchlist(b):
    needed = set()
    for g in range(NG):
        for c in range(NC):
            needed.add(br(b, g, c))
    return sorted(needed)


# ================= WORKER =================
def process_board(b):
    """
    Worker computes:
      • t_final(b,g,c) for ALL channels, ALL groups

    MCP reference:
      • Board b, Group 3, Channel 7

    IMPORTANT:
      • The SAME trigger (Group g, Channel 8) is used for both
        channel and MCP7 subtraction.
    """
    with uproot.open(INPUT_FILE) as f:
        tree = f[TREE_NAME]
        arrays = tree.arrays(board_branchlist(b), library="np")

    t_final = {}

    # MCP7 raw timing (no trigger subtraction yet)
    mcp_ref = br(b, 3, 7)
    if mcp_ref not in arrays:
        return {}

    t_mcp7 = arrays[mcp_ref]

    for g in range(NG):
        trig = br(b, g, 8)
        if trig not in arrays:
            continue

        t_trig = arrays[trig]

        # MCP7 aligned to SAME trigger as channel
        mcp_term = t_mcp7 - t_trig

        for c in range(NC):
            ch = br(b, g, c)
            if ch not in arrays:
                continue

            # (t_fit - t_trig_g) - (t_mcp7 - t_trig_g)
            t_final_val = (arrays[ch] - t_trig) - mcp_term
            t_final[(b, g, c)] = t_final_val

    return t_final


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

    t_final = {}

    with Pool(nproc) as pool:
        for t_b in tqdm(
            pool.imap_unordered(process_board, TRUE_BOARDS),
            total=len(TRUE_BOARDS),
            desc="Boards"
        ):
            t_final.update(t_b)

    print(f"  → tfinal channels : {len(t_final)}")

    # ---------- WRITE ROOT ----------
    print("\n[3/4] Writing output ROOT file")
    out = {}

    for (b, g, c), arr in t_final.items():
        out[f"tfinal_Board{b}_Group{g}_Channel{c}"] = arr

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
