import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import os

# ================= USER SETTINGS =================
ORIG_FILE = "/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1468_250927145556_TimingDAQ.root"
ANA_FILE  = "/lustre/research/hep/akshriva/Dream-Timing/TRUE-HGtiming/run1468_250927145556_TimingDAQ_afteranalysis.root"

TREE_NAME = "EventTree"

OUTDIR = "./TRUE-HGtimingt/board_plots_by_group_lessbins"
os.makedirs(OUTDIR, exist_ok=True)

BOARDS = range(4)
NG = 4
NC = 9

NBINS = 1000

XLIM_CORR = (-0.5, 180)
XLIM_ORIG = (-0.5, 140)
# =================================================


# ================= HELPERS =================
def br(b, g, c, var="LP2_50"):
    return f"DRS_Board{b}_Group{g}_Channel{c}_{var}"


# ================= MAIN =================
def main():

    print("Generating board/group timing overlays (all channels included)")

    # Fixed, repeatable color cycle
    colors = plt.cm.tab10.colors

    with uproot.open(ORIG_FILE) as f_orig, uproot.open(ANA_FILE) as f_ana:
        tree_orig = f_orig[TREE_NAME]
        tree_ana  = f_ana[TREE_NAME]

        for b in tqdm(BOARDS, desc="Boards"):

            # ============================================================
            # Corrected timing: |trueminusdeltat| (linear scale)
            # ============================================================
            pdf_corr = f"{OUTDIR}/Board{b}_trueminusdeltat_byGroup.pdf"
            with PdfPages(pdf_corr) as pdf:

                for g in range(NG):

                    fig, ax = plt.subplots(figsize=(7.5, 5))

                    for c in range(NC):

                        key = f"trueminusdeltat_Board{b}_Group{g}_Channel{c}"
                        if key not in tree_ana.keys():
                            continue

                        arr = np.abs(tree_ana[key].array(library="np"))

                        ax.hist(
                            arr,
                            bins=NBINS,
                            histtype="step",
                            linewidth=1.2,
                            color=colors[c % len(colors)],
                            label=f"C{c}"
                        )

                    ax.set_xlabel("abs(t₍fit₎ᶜʰ − t₍fit₎ᵗʳᶦᵍ − Δt₍MCP₎) [ns]")
                    ax.set_ylabel("Events")
                    ax.set_title(f"Board {b} — Group {g} (Corrected)")
                    ax.set_xlim(*XLIM_CORR)

                    ax.minorticks_on()
                    ax.tick_params(axis="both", which="major", length=6)
                    ax.tick_params(axis="both", which="minor", length=3)

                    ax.legend(
                        title="Channel",
                        fontsize=8,
                        ncol=3,
                        frameon=False
                    )

                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

            # ============================================================
            # Original timing: |LP2_50| (LOG Y-SCALE)
            # ============================================================
            pdf_orig = f"{OUTDIR}/Board{b}_LP2_50_byGroup.pdf"
            with PdfPages(pdf_orig) as pdf:

                for g in range(NG):

                    fig, ax = plt.subplots(figsize=(7.5, 5))

                    for c in range(NC):

                        key = br(b, g, c)
                        if key not in tree_orig.keys():
                            continue

                        arr = np.abs(tree_orig[key].array(library="np"))

                        ax.hist(
                            arr,
                            bins=NBINS,
                            histtype="step",
                            linewidth=1.2,
                            color=colors[c % len(colors)],
                            label=f"C{c}"
                        )

                    #ax.set_xlabel("|LP2_50| [ns]")
                    ax.set_xlabel("abs(t50₍fit₎ᶜʰ [ns]")
                    ax.set_ylabel("Events")
                    ax.set_title(f"Board {b} — Group {g} (Original)")
                    
                    ax.set_xlim(*XLIM_ORIG)

                    # LOG SCALE FOR ORIGINAL LP2_50
                    ax.set_yscale("log")

                    ax.minorticks_on()
                    ax.tick_params(axis="both", which="major", length=6)
                    ax.tick_params(axis="both", which="minor", length=3)

                    ax.legend(
                        title="Channel",
                        fontsize=8,
                        ncol=3,
                        frameon=False
                    )

                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

            print(f"  → Saved Board {b} PDFs")

    print("Done.")


if __name__ == "__main__":
    main()
