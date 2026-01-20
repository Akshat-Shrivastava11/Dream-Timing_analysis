#!/usr/bin/env python3
import uproot
import numpy as np
import matplotlib.pyplot as plt              # required by user (unused)
from scipy.stats import norm                 # required by user (unused)
from multiprocessing import Pool, cpu_count
from matplotlib.backends.backend_pdf import PdfPages  # required by user (unused)
from tqdm import tqdm
import time
import os
import argparse

# ================= CONFIG =================
NG = 4
NC = 9
TRUE_BOARDS = range(4)
TREE_DEFAULT = "EventTree"

def br(b, g, c, var="LP2_50"):
    return f"DRS_Board{b}_Group{g}_Channel{c}_{var}"

# ---------------- PARTICLE SELECTION ----------------
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
        "muon":     {"TTUMuonVeto": True,  "PSD": False},
        "pion":     {"TTUMuonVeto": False, "PSD": False, "Cer474": True, "Cer519": True, "Cer537": True},
        "electron": {"TTUMuonVeto": False, "PSD": True,  "Cer474": True, "Cer519": True, "Cer537": True},
        "proton":   {"TTUMuonVeto": False, "PSD": False, "Cer474": False,"Cer519": False,"Cer537": False},
    }
    return selections.get(particle_type.lower(), {})


CORRECTION_FACTORS = {
    "1040": {
        "DRS_Board0_Group0_Channel8": 9.545,
        "DRS_Board0_Group1_Channel8": 9.545,
        "DRS_Board0_Group2_Channel8": 3.1889,
        "DRS_Board0_Group3_Channel8": 3.1889,
        "DRS_Board1_Group0_Channel8": 8.7275,
        "DRS_Board1_Group1_Channel8": 8.7275,
        "DRS_Board1_Group2_Channel8": 3.3646,
        "DRS_Board1_Group3_Channel8": 3.3646,
        "DRS_Board2_Group0_Channel8": 9.9585,
        "DRS_Board2_Group1_Channel8": 9.9585,
        "DRS_Board2_Group2_Channel8": 6.283,
        "DRS_Board2_Group3_Channel8": 6.283,
        "DRS_Board3_Group0_Channel8": 11.085,
        "DRS_Board3_Group1_Channel8": 11.085,
        "DRS_Board3_Group2_Channel8": 7.5975,
        "DRS_Board3_Group3_Channel8": 7.5975,
        "DRS_Board4_Group0_Channel8": 8.482,
        "DRS_Board4_Group1_Channel8": 8.482,
        "DRS_Board4_Group2_Channel8": 8.2775,
        "DRS_Board4_Group3_Channel8": 8.2775,
        "DRS_Board5_Group0_Channel8": 11.2745,
        "DRS_Board5_Group1_Channel8": 11.2745,
        "DRS_Board5_Group2_Channel8": 8.3525,
        "DRS_Board5_Group3_Channel8": 8.3525,
    },
    "default": {
        "DRS_Board0_Group0_Channel8": 0.0,
        "DRS_Board0_Group1_Channel8": 0.0,
        "DRS_Board0_Group2_Channel8": -0.1405,
        "DRS_Board0_Group3_Channel8": -0.1405,
        "DRS_Board1_Group0_Channel8": -0.2964,
        "DRS_Board1_Group1_Channel8": -0.2964,
        "DRS_Board1_Group2_Channel8": -0.1078,
        "DRS_Board1_Group3_Channel8": -0.1078,
        "DRS_Board2_Group0_Channel8": -0.1697,
        "DRS_Board2_Group1_Channel8": -0.1697,
        "DRS_Board2_Group2_Channel8": -0.1068,
        "DRS_Board2_Group3_Channel8": -0.1068,
        "DRS_Board3_Group0_Channel8": -0.2385,
        "DRS_Board3_Group1_Channel8": -0.2385,
        "DRS_Board3_Group2_Channel8": -0.1485,
        "DRS_Board3_Group3_Channel8": -0.1485,
        "DRS_Board4_Group0_Channel8": -1.638,
        "DRS_Board4_Group1_Channel8": -1.638,
        "DRS_Board4_Group2_Channel8": -1.735,
        "DRS_Board4_Group3_Channel8": -1.735,
        "DRS_Board5_Group0_Channel8": -1.588,
        "DRS_Board5_Group1_Channel8": -1.588,
        "DRS_Board5_Group2_Channel8": -1.563,
        "DRS_Board5_Group3_Channel8": -1.563,
        "DRS_Board6_Group0_Channel8": -1.514,
        "DRS_Board6_Group1_Channel8": -1.514,
        "DRS_Board6_Group2_Channel8": -1.655,
        "DRS_Board6_Group3_Channel8": -1.655,
        "DRS_Board7_Group0_Channel8": -1.733,
        "DRS_Board7_Group1_Channel8": -1.733,
        "DRS_Board7_Group2_Channel8": -0.8045,
        "DRS_Board7_Group3_Channel8": -0.8045,
    },
}

def trig_key(b: int, g: int) -> str:
    return f"DRS_Board{b}_Group{g}_Channel8"

def get_trig_correction(b: int, g: int, cf_key: str) -> float:
    d = CORRECTION_FACTORS.get(cf_key, CORRECTION_FACTORS["default"])
    return float(d.get(trig_key(b, g), 0.0))


def read_service_value(arrays: dict, service: str, method: str) -> np.ndarray:
    # try exact first, then common suffix naming
    if service in arrays:
        return arrays[service]
    k2 = f"{service}_{method}"
    if k2 in arrays:
        return arrays[k2]
    raise KeyError(f"Missing service branch for {service} (tried: {service}, {k2})")

def fired_from_threshold(values: np.ndarray, thr: float) -> np.ndarray:
    # negative threshold: negative pulses, fired if more negative than thr
    if thr < 0:
        return values < thr
    return values > thr

def build_particle_mask(arrays: dict, particle: str) -> np.ndarray:
    sel = get_particle_selection(particle)
    if not particle or particle.lower() in ("none", "all", "any") or len(sel) == 0:
        # all events pass
        first = next(iter(arrays.values()))
        return np.ones_like(first, dtype=bool)

    first = next(iter(arrays.values()))
    mask = np.ones_like(first, dtype=bool)

    for det, req_fired in sel.items():
        _, _, thr, method = get_service_drs_cut(det)
        v = read_service_value(arrays, det, method)
        fired = fired_from_threshold(v, thr)
        mask &= (fired if req_fired else ~fired)

    return mask




# ---------------- BRANCH LIST ----------------
def board_branchlist(b, services=None):
    needed = set()
    for g in range(NG):
        for c in range(NC):
            needed.add(br(b, g, c))
    # ensure MCP ref exists
    needed.add(br(b, 3, 7))

    if services:
        for s in services:
            needed.add(s)
            _, _, _, method = get_service_drs_cut(s)
            needed.add(f"{s}_{method}")

    return sorted(needed)

# ---------------- WORKER ----------------
def process_board(job):
    """
    Implements exactly:
      t_final(b,g,c) =
          ( t_fit(b,g,c) - t_trig(b,g) )
        - ( t_fit(b,3,7) - t_trig(b,g) )

    where the SAME trigger t_trig(b,g) is used in both subtractions.

    MCP reference timing is always Board b, Group 3, Channel 7.
    """
    b, input_file, tree_name, particle, use_abs, add_cf, cf_key = job

    # PID branches needed for event-level mask
    services = list(get_particle_selection(particle).keys()) if particle else []

    with uproot.open(input_file) as f:
        tree = f[tree_name]
        arrays = tree.arrays(board_branchlist(b, services=services), library="np")

    # event mask (particle selection)
    mask = build_particle_mask(arrays, particle)

    # MCP reference raw timing (no trigger subtraction yet)
    mcp_ref = br(b, 3, 7)
    if mcp_ref not in arrays:
        return {}

    t_mcp7 = arrays[mcp_ref][mask]
    t_final = {}

    for g in range(NG):
        trig = br(b, g, 8)
        if trig not in arrays:
            continue

        t_trig = arrays[trig][mask]

        # optional correction factors added to trigger (group-dependent)
        corr = get_trig_correction(b, g, cf_key) if add_cf else 0.0
        t_trig_corr = t_trig + corr

        # MCP term uses SAME trigger as channel group g
        mcp_term = t_mcp7 - t_trig_corr

        for c in range(NC):
            ch = br(b, g, c)
            if ch not in arrays:
                continue

            # (t_ch - t_trig_g) - (t_mcp7 - t_trig_g)
            tf = (arrays[ch][mask] - t_trig_corr) - mcp_term

            if use_abs:
                tf = np.abs(tf)

            t_final[(b, g, c)] = tf

    return t_final


# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser(description="TRUE timing reconstruction with optional particle selection.")
    parser.add_argument("-i", "--input", default='/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1468_250927145556_TimingDAQ.root',  help="Input ROOT file")
    parser.add_argument("-o", "--outdir", default='TRUE-HGtiming/skimmed_files',  help="Output directory")
    parser.add_argument("--tree", default=TREE_DEFAULT, help=f"Tree name (default: {TREE_DEFAULT})")
    parser.add_argument("-p", "--particle", default="none",
                        help="Particle selection: muon, pion, electron, proton, none (default: none)")
    parser.add_argument("--abs", action="store_true", help="Store |tfinal| instead of signed tfinal")
    parser.add_argument("--add_correctionfactors", action="store_true",
                    help="Add trigger correction factors.")
    parser.add_argument("--cf_key", default="default",
                    help="Correction set key (e.g. 'default', '1040').")

    args = parser.parse_args()

    input_file = args.input
    outdir = args.outdir
    tree_name = args.tree
    particle = args.particle
    use_abs = args.abs

    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(outdir, f"{base}_postaskim_allchannels.root")

    t0 = time.time()
    print("=" * 90)
    print(" TRUE TIMING RECONSTRUCTION (MULTIPROCESSING)")
    print("=" * 90)
    print(f"Input    : {input_file}")
    print(f"Tree     : {tree_name}")
    print(f"Particle : {particle}")
    print(f"Abs      : {use_abs}")
    print(f"Outdir   : {outdir}")
    print(f"Output   : {output_file}")

    # event count
    with uproot.open(input_file) as f:
        tree = f[tree_name]
        probe = br(0, 0, 8)
        key = probe if probe in tree.keys() else tree.keys()[0]
        n_events = len(tree[key].array(library="np"))
    print(f"\nEvents (raw): {n_events:,}")
    print(f"Boards      : {list(TRUE_BOARDS)}")

    # multiprocessing
    nproc = min(cpu_count(), len(TRUE_BOARDS))
    print(f"\nUsing {nproc} processes\n")

    t_final = {}
    #work = [(b, input_file, tree_name, particle, use_abs) for b in TRUE_BOARDS]
    work = [
    (b, input_file, tree_name, particle, use_abs,
     args.add_correctionfactors, args.cf_key)
    for b in TRUE_BOARDS
]

    with Pool(nproc) as pool:
        for t_b in tqdm(pool.imap_unordered(process_board, work),
                        total=len(work), desc="Boards"):
            t_final.update(t_b)

    print(f"\n→ tfinal channels (written): {len(t_final)}")

    # write root
    out = {f"tfinal_Board{b}_Group{g}_Channel{c}": arr
           for (b, g, c), arr in t_final.items()}

    with uproot.recreate(output_file) as fout:
        fout[tree_name] = out

    print("\nDone")
    print(f"Branches: {len(out)}")
    print(f"Runtime : {time.time() - t0:.1f} s")
    print("=" * 90)

if __name__ == "__main__":
    main()
