import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count

# ================= USER SETTINGS =================
ANA_FILE  = "/lustre/research/hep/akshriva/Dream-Timing/TRUE-HGtiming/skimmed_files/run1468_250927145556_TimingDAQ_postanalysis_allchannels.root"
TREE_NAME = "EventTree"

OUTDIR = "./TRUE-HGtiming/SkimmedResults"
os.makedirs(OUTDIR, exist_ok=True)

BOARDS = range(4)   # change to range(4) when ready
NG = 4
NC = 9

NBINS = 300
XLIM_TFINAL = (0.0, 30.0)
CUT_MIN = 1.0       # ignore |tfinal| < 1 ns
# =================================================


# ================= WORKER =================
def plot_board(b):
    """
    Plot all groups for a single board.
    Runs in its own process.
    """
    colors = plt.cm.tab10.colors

    with uproot.open(ANA_FILE) as f:
        tree = f[TREE_NAME]

        # -------- preload all branches for this board --------
        data = {
            (g, c): tree[f"tfinal_Board{b}_Group{g}_Channel{c}"].array(library="np")
            for g in range(NG)
            for c in range(NC)
            if f"tfinal_Board{b}_Group{g}_Channel{c}" in tree.keys()
        }

    pdf_path = f"{OUTDIR}/Board{b}_tfinal_byGroup.pdf"
    bin_edges = np.linspace(XLIM_TFINAL[0], XLIM_TFINAL[1], NBINS + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    with PdfPages(pdf_path) as pdf:
        for g in range(NG):

            fig, ax = plt.subplots(figsize=(7.5, 5))

            for c in range(NC):

                # -------- skip trigger --------
                if c == 8:
                    continue

                # -------- skip MCP channels in Group 3 --------
                if g == 3 and c in (6, 7):
                    continue

                if (g, c) not in data:
                    continue

                arr = data[(g, c)]
                if arr.size < 100:
                    continue

                # -------- absolute value + low-value mask --------
                arr_abs = np.abs(arr)
                arr_abs = arr_abs[arr_abs >= CUT_MIN]

                if arr_abs.size == 0:
                    continue

                hist, _ = np.histogram(arr_abs, bins=bin_edges)

                ax.plot(
                    bin_centers,
                    hist,
                    linewidth=1.2,
                    linestyle="-",
                    color=colors[c % len(colors)],
                    label=f"C{c}",
                    antialiased=False
                )

            ax.set_xlabel(
                r"$|(t_{\mathrm{fit}}^{ch}-t_{\mathrm{trig}}^{g})"
                r"-(t_{\mathrm{fit}}^{\mathrm{MCP7}}-t_{\mathrm{trig}}^{g})|$ [ns]"
            )
            ax.set_ylabel("Events")
            ax.set_title(f"Board {b} — Group {g} (|tfinal|, > {CUT_MIN} ns)")
            ax.set_xlim(*XLIM_TFINAL)
            # ax.set_yscale("log")  # enable if desired

            # KEEP minor ticks
            ax.minorticks_on()
            ax.tick_params(axis="both", which="major", length=6)
            ax.tick_params(axis="both", which="minor", length=3)

            ax.legend(
                title="Channel",
                fontsize=7,
                ncol=3,
                frameon=False
            )

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return b


# ================= MAIN =================
def main():
    print("Generating |tfinal| resolution plots (fast, multiprocessing)")
    print(f'Loading ANA file : "{ANA_FILE}"')
    print(f'Saving to        : "{OUTDIR}/"')

    nproc = min(cpu_count(), len(BOARDS))
    print(f"Using {nproc} processes")

    with Pool(nproc) as pool:
        for b in tqdm(
            pool.imap_unordered(plot_board, BOARDS),
            total=len(BOARDS),
            desc="Boards"
        ):
            print(f"  → Saved Board {b} PDFs")

    print("Done.")


if __name__ == "__main__":
    main()
