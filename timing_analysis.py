import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count
from scipy.optimize import curve_fit
import csv

# ================= USER SETTINGS =================
ANA_FILE  = "/lustre/research/hep/akshriva/Dream-Timing/TRUE-HGtiming/skimmed_files/run1468_250927145556_TimingDAQ_postanalysis_allchannels.root"
TREE_NAME = "EventTree"

OUTDIR = "./TRUE-HGtiming/SkimmedResults"
os.makedirs(OUTDIR, exist_ok=True)

BOARDS = range(4)
NG = 4
NC = 9

NBINS = 300
XLIM_TFINAL = (0.0, 30.0)
CUT_MIN = 1.0
# =================================================


# ================= FIT MODEL =================
def folded_gaussian(x, A, mu, sigma, B):
    return A * (
        np.exp(-0.5 * ((x - mu) / sigma) ** 2) +
        np.exp(-0.5 * ((x + mu) / sigma) ** 2)
    ) + B


# ================= WORKER =================
def plot_board(b):

    colors = plt.cm.tab10.colors
    fit_results = []

    with uproot.open(ANA_FILE) as f:
        tree = f[TREE_NAME]

        data = {
            (g, c): tree[f"tfinal_Board{b}_Group{g}_Channel{c}"].array(library="np")
            for g in range(NG)
            for c in range(NC)
            if f"tfinal_Board{b}_Group{g}_Channel{c}" in tree.keys()
        }

    pdf_path = f"{OUTDIR}/Board{b}_tfinal_byGroup_fitonly.pdf"
    csv_path = f"{OUTDIR}/Board{b}_fit_results.csv"

    bin_edges = np.linspace(XLIM_TFINAL[0], XLIM_TFINAL[1], NBINS + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    with PdfPages(pdf_path) as pdf:
        for g in range(NG):

            fig, ax = plt.subplots(figsize=(7.5, 5))

            for c in range(NC):

                # Skip trigger
                if c == 8:
                    continue

                # Skip MCP channels
                if g == 3 and c in (6, 7):
                    continue

                if (g, c) not in data:
                    continue

                arr = data[(g, c)]
                if arr.size < 500:
                    continue

                arr_abs = np.abs(arr)
                arr_abs = arr_abs[arr_abs >= CUT_MIN]
                if arr_abs.size < 200:
                    continue

                hist, _ = np.histogram(arr_abs, bins=bin_edges)

                # ax.plot(
                #     bin_centers,
                #     hist,
                #     color=colors[c % len(colors)],
                #     lw=1.2,
                #     label=f"C{c}",
                #     antialiased=False
                # )

                # ---------- FIT ----------
                try:
                    peak_x = bin_centers[np.argmax(hist)]
                    sigma0 = np.std(arr_abs[(arr_abs > peak_x - 2) & (arr_abs < peak_x + 2)])
                    sigma0 = max(sigma0, 0.3)

                    p0 = [hist.max(), peak_x, sigma0, np.median(hist[:10])]

                    popt, pcov = curve_fit(
                        folded_gaussian,
                        bin_centers,
                        hist,
                        p0=p0,
                        bounds=([0, 0, 0.05, 0], [np.inf, 30, 10, np.inf]),
                        maxfev=20000
                    )

                    A, mu, sigma, B = popt

                    xfit = np.linspace(*XLIM_TFINAL, 800)
                    yfit = folded_gaussian(xfit, *popt)

                    ax.plot(
                        xfit,
                        yfit,
                        color=colors[c % len(colors)],
                        ls="-",
                        lw=1.5
                    )

                    fit_results.append({
                        "Board": b,
                        "Group": g,
                        "Channel": c,
                        "mu_ns": mu,
                        "sigma_ns": sigma
                    })

                except RuntimeError:
                    continue

            ax.set_xlabel(
                r"$|(t_{\mathrm{fit}}^{ch}-t_{\mathrm{trig}}^{g})"
                r"-(t_{\mathrm{fit}}^{\mathrm{MCP7}}-t_{\mathrm{trig}}^{g})|$ [ns]"
            )
            ax.set_ylabel("Events")
            ax.set_title(f"Board {b} — Group {g} (|tfinal| > {CUT_MIN} ns)")
            ax.set_xlim(*XLIM_TFINAL)

            ax.minorticks_on()
            ax.tick_params(axis="both", which="major", length=6)
            ax.tick_params(axis="both", which="minor", length=3)

            ax.legend(fontsize=7, ncol=3, frameon=False)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    # ---------- WRITE CSV ----------
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Board", "Group", "Channel", "mu_ns", "sigma_ns"]
        )
        writer.writeheader()
        writer.writerows(fit_results)

    return b


# ================= MAIN =================
def main():
    print("Generating |tfinal| resolution plots + folded Gaussian fits")

    nproc = min(cpu_count(), len(BOARDS))
    with Pool(nproc) as pool:
        for b in tqdm(pool.imap_unordered(plot_board, BOARDS),
                      total=len(BOARDS),
                      desc="Boards"):
            print(f"  → Board {b} done")

    print("Done.")


if __name__ == "__main__":
    main()
