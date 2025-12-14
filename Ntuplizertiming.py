import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from multiprocessing import Pool, cpu_count
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import time
import os

# ================= USER SETTINGS =================
INPUT_FILE  = "/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1468_250927145556_TimingDAQ.root"
TREE_NAME   = "EventTree"
OUTPUT_FILE = "/lustre/research/hep/akshriva/Dream-Timing/TRUE-HGtiming/run1468_250927145556_TimingDAQut_postTrue.root"

NB = 7   # boards 0–6 (Board 7 ignored)
NG = 4   # groups 0–3
NC = 9   # channels 0–8
# =================================================


# ================= IGNORE BOARD 7 =================
IGNORE_PREFIXES = [
    "DRS_Board7_Group0_Channel",
    "DRS_Board7_Group1_Channel",
    "DRS_Board7_Group2_Channel",
    "DRS_Board7_Group3_Channel",
]


def is_ignored(branch):
    return any(branch.startswith(p) for p in IGNORE_PREFIXES)


# ================= HELPERS =================
def br(b, g, c, var):
    return f"DRS_Board{b}_Group{g}_Channel{c}_{var}"


def valid_channel(g, c):
    if c == 8:
        return False            # trigger
    if g == 3 and c in (6, 7):
        return False            # MCP channels
    return True


# ================= PDF WORKERS =================
def make_deltat_pdf(args):
    b, data = args
    mu, sigma = norm.fit(data)

    plt.figure(figsize=(6,4))
    plt.hist(data, bins=200, density=True, alpha=0.7)
    x = np.linspace(data.min(), data.max(), 1000)
    plt.plot(x, norm.pdf(x, mu, sigma), "r-")
    plt.xlabel("Δt MCP [ns]")
    plt.ylabel("Density")
    plt.title(f"MCP Δt Board {b}\nμ={mu:.3f}, σ={sigma:.3f}")
    plt.tight_layout()
    plt.savefig(f"deltat_MCP_board{b}.pdf")
    plt.close()


def make_ttrue_page(args):
    (b, g, c), data = args

    plt.figure(figsize=(6,4))
    plt.hist(data, bins=200, histtype="step", linewidth=1.5)
    plt.xlabel("t_true [ns]")
    plt.ylabel("Events")
    plt.title(f"t_true Board {b} Group {g} Channel {c}")
    plt.tight_layout()

    fname = f"tmp_ttrue_b{b}_g{g}_c{c}.pdf"
    plt.savefig(fname)
    plt.close()
    return fname


# ================= MAIN =================
def main():

    t0 = time.time()
    print("="*90)
    print(" TRUE TIMING RECONSTRUCTION (FINAL, UPROOT + MULTIPROCESSING)")
    print("="*90)

    # ---------- READ ROOT ----------
    print("\n[1/6] Reading ROOT file")
    with uproot.open(INPUT_FILE) as f:
        tree = f[TREE_NAME]
        all_arrays = tree.arrays(library="np")

    print(f"  → Total branches read: {len(all_arrays)}")

    # ---------- FILTER BOARD 7 ----------
    print("\n[2/6] Removing Board 7 branches")
    arrays = {
        k: v for k, v in all_arrays.items()
        if not is_ignored(k)
    }

    print(f"  → Branches kept   : {len(arrays)}")
    print(f"  → Branches removed: {len(all_arrays) - len(arrays)}")

    n_events = len(next(iter(arrays.values())))
    print(f"  → Events: {n_events:,}")
    print(f"  → Time elapsed: {time.time() - t0:.1f} s")

    # ---------- MCP Δt ----------
    print("\n[3/6] Computing MCP Δt per board")
    deltat_mcp = {}

    for b in tqdm(range(NB), desc="Boards (MCP Δt)"):
        sig = br(b,3,6,"LP2_50")
        trg = br(b,3,7,"LP2_50")

        if sig not in arrays or trg not in arrays:
            print(f"  !! Board {b}: MCP branches missing, skipping")
            continue

        deltat_mcp[b] = arrays[sig] - arrays[trg]

    print(f"  → MCP reference built for {len(deltat_mcp)} boards")

    # ---------- t_true ----------
    print("\n[4/6] Computing t_true for all valid channels")
    ttrue = {}

    for b in tqdm(range(NB), desc="Boards (t_true)"):
        if b not in deltat_mcp:
            continue

        for g in range(NG):
            trig = br(b,g,8,"LP2_50")
            if trig not in arrays:
                continue

            t_trig = arrays[trig]

            for c in range(NC):
                if not valid_channel(g, c):
                    continue

                ch = br(b,g,c,"LP2_50")
                if ch not in arrays:
                    continue

                ttrue[(b,g,c)] = (
                    arrays[ch]
                    - t_trig
                    + deltat_mcp[b]
                )

    print(f"  → t_true channels computed: {len(ttrue)}")
    print(f"  → Time elapsed: {time.time() - t0:.1f} s")

    # ---------- WRITE ROOT ----------
    #print("\n[5/6] Writing output ROOT file")
    #out_branches = dict(arrays)
    print("\n[5/6] Writing output ROOT file (derived quantities only)")

    out_branches = {}

    # MCP Δt
    for b, arr in deltat_mcp.items():
        out_branches[f"deltat_MCP_Board{b}"] = arr

    # t_true
    for (b,g,c), arr in ttrue.items():
        out_branches[f"ttrue_Board{b}_Group{g}_Channel{c}"] = arr

    with uproot.recreate(OUTPUT_FILE) as fout:
        fout[TREE_NAME] = out_branches

    print(f"  → ROOT file written: {OUTPUT_FILE}")
    print(f"  → Branches written: {len(out_branches)}")

    for b, arr in deltat_mcp.items():
        out_branches[f"deltat_MCP_Board{b}"] = arr

    for (b,g,c), arr in ttrue.items():
        out_branches[f"ttrue_Board{b}_Group{g}_Channel{c}"] = arr

    with uproot.recreate(OUTPUT_FILE) as fout:
        fout[TREE_NAME] = out_branches

    print(f"  → ROOT file written: {OUTPUT_FILE}")

    # # ---------- PDFs ----------
    # print("\n[6/6] Producing PDFs (multiprocessing)")
    # nproc = min(cpu_count(), 8)
    # print(f"  → Using {nproc} processes")

    # with Pool(nproc) as pool:
    #     list(tqdm(
    #         pool.imap_unordered(make_deltat_pdf, deltat_mcp.items()),
    #         total=len(deltat_mcp),
    #         desc="Δt MCP PDFs"
    #     ))

    # with Pool(nproc) as pool:
    #     tmp_files = list(tqdm(
    #         pool.imap_unordered(make_ttrue_page, ttrue.items()),
    #         total=len(ttrue),
    #         desc="t_true PDFs"
    #     ))

    # print("  → Merging t_true PDFs")
    # with PdfPages("ttrue_all_channels.pdf") as pdf:
    #     for f in sorted(tmp_files):
    #         img = plt.imread(f)
    #         plt.figure(figsize=(6,4))
    #         plt.imshow(img)
    #         plt.axis("off")
    #         pdf.savefig()
    #         plt.close()
    #         os.remove(f)

    # print("\n" + "="*90)
    # print("DONE")
    # print(f"Total runtime: {time.time() - t0:.1f} s")
    # print("Outputs:")
    # print(f"  ROOT : {OUTPUT_FILE}")
    # print("  PDFs : deltat_MCP_boardX.pdf")
    # print("         ttrue_all_channels.pdf")
    # print("="*90)


if __name__ == "__main__":
    main()
