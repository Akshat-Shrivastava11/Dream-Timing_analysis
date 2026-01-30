#!/usr/bin/env python3
import os
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ================= USER SETTINGS =================
#ANA_FILE = "TRUE-HGtiming/skimmed_files/run1355_250924165834_TimingDAQ_postaskim_allchannels_newmethod.root"
ANA_FILE = "TRUE-HGtiming/skimmed_files/run1513_250928194230_TimingDAQ_postaskim_allchannels_newmethod.root"
TREE_NAME = "EventTree"

OUTDIR = "./TRUE-HGtiming/3mmplots_histonly/90deg_calibration"
os.makedirs(OUTDIR, exist_ok=True)

BOARDS = [0, 1, 2, 3]
NG = 4
NC = 9

NBINS = 200
CUT_MIN = 1.0
MIN_ENTRIES = 200
MIN_RAW = 500


# ---------------- CHANNEL MASKS ----------------
def _base_ok(g, c):
    if c == 8:
        return False
    if g == 3 and c in (6, 7):
        return False
    return True

def _ok(g, c, parity):
    if not _base_ok(g, c):
        return False
    return (c % 2 == 1) if parity == "odd" else (c % 2 == 0)


def _xlabel():
    return r"$|t_{\mathrm{final}}|$ [ns]"


def _prep(arr, xlim):
    if arr.size < MIN_RAW:
        return None
    arr = np.abs(arr)
    arr = arr[arr >= CUT_MIN]
    if arr.size < MIN_ENTRIES:
        return None
    arr = arr[(arr >= xlim[0]) & (arr <= xlim[1])]
    if arr.size < 50:
        return None
    return arr


# ---------------- 2×2 BOARDS ----------------
def make_boards(parity, label, xlim):
    out = f"{OUTDIR}/HISTONLY_{label}_Boards_vertical.pdf"
    bins = np.linspace(xlim[0], xlim[1], NBINS + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    with uproot.open(ANA_FILE) as f, PdfPages(out) as pdf:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        fig, axes = plt.subplots(2, 2, figsize=(8, 14), sharex=True)
        axes = axes.flatten()

        for ax, b in zip(axes, BOARDS):
            any_ok = False
            for g in range(NG):
                for c in range(NC):
                    if not _ok(g, c, parity):
                        continue
                    k = f"tfinal_Board{b}_Group{g}_Channel{c}"
                    if k not in keys:
                        continue

                    arr = _prep(tree[k].array(library="np"), xlim)
                    if arr is None:
                        continue

                    mu, sig = arr.mean(), arr.std()
                    h, _ = np.histogram(arr, bins=bins)
                    ax.step(centers, h, where="mid",
                            label=f"B{b}G{g}C{c} μ={mu:.2f} σ={sig:.2f}")
                    any_ok = True

            ax.set_title(f"{label} — Board {b}")
            ax.set_xlim(*xlim)
            ax.set_ylabel("Events")
            ax.legend(fontsize=6, ncol=2, frameon=False)

        for ax in axes:
            ax.set_xlabel(_xlabel())

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print("Saved:", out)


# ---------------- 16 SUBPLOTS ----------------
def make_16(parity, label, xlim):
    out = f"{OUTDIR}/HISTONLY_{label}_16Subplots_vertical.pdf"
    bins = np.linspace(xlim[0], xlim[1], NBINS + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    layout = []
    for g in range(NG):
        layout.append((0, g, 2, g))
    for g in range(NG):
        layout.append((1, g, 3, g))

    with uproot.open(ANA_FILE) as f, PdfPages(out) as pdf:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        fig, axes = plt.subplots(8, 2, figsize=(9, 28), sharex=True)

        for r, (bL, gL, bR, gR) in enumerate(layout):
            for cidx, (b, g) in enumerate([(bL, gL), (bR, gR)]):
                ax = axes[r, cidx]
                any_ok = False

                for ch in range(NC):
                    if not _ok(g, ch, parity):
                        continue
                    k = f"tfinal_Board{b}_Group{g}_Channel{ch}"
                    if k not in keys:
                        continue

                    arr = _prep(tree[k].array(library="np"), xlim)
                    if arr is None:
                        continue

                    mu, sig = arr.mean(), arr.std()
                    h, _ = np.histogram(arr, bins=bins)
                    ax.step(centers, h, where="mid",
                            label=f"C{ch} μ={mu:.2f} σ={sig:.2f}")
                    any_ok = True

                ax.set_title(f"{label} — B{b}G{g}")
                ax.set_xlim(*xlim)
                ax.set_ylabel("Events")
                ax.legend(fontsize=6, ncol=2, frameon=False)

        for ax in axes[-1]:
            ax.set_xlabel(_xlabel())

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print("Saved:", out)


# ---------------- MAIN ----------------
def main():
    make_boards("odd",  "SCI", (7.0, 14))
    make_16(   "odd",  "SCI", (7.0, 14))

    make_boards("even", "CER", (7, 14))
    make_16(   "even", "CER", (7,14))

    print("All done.")

if __name__ == "__main__":
    main()