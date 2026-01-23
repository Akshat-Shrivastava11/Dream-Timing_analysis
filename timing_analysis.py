import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count
from scipy.optimize import curve_fit

# ================= USER SETTINGS =================
ANA_FILE = "TRUE-HGtiming/skimmed_files/run1355_250924165834_TimingDAQ_postaskim_allchannels_newmethod.root"
TREE_NAME = "EventTree"

OUTDIR = "./TRUE-HGtiming/SkimmedResults/positron80GeV_1355_newmethod"
os.makedirs(OUTDIR, exist_ok=True)

BOARDS = range(4)
NG = 4
NC = 9

NBINS = 300
XLIM_TFINAL = (5.0, 20.0)   # plot/fit range for |tfinal|
CUT_MIN = 1.0               # ignore |tfinal| < CUT_MIN
MIN_ENTRIES = 200           # after cuts on |tfinal|
MIN_RAW = 500               # raw entries before abs/cut
# =================================================


# ================= FIT MODEL =================
def folded_gaussian(x, A, mu, sigma, B):
    return A * (
        np.exp(-0.5 * ((x - mu) / sigma) ** 2) +
        np.exp(-0.5 * ((x + mu) / sigma) ** 2)
    ) + B


# def _fit_channel(bin_centers, hist, arr_abs):
#     """
#     Fit folded Gaussian to histogram of |tfinal|.
#     Returns popt=(A,mu,sigma,B) or None.
#     Uses Poisson-like weights for stability.
#     """
#     if np.sum(hist) < MIN_ENTRIES:
#         return None

#     peak_x = bin_centers[np.argmax(hist)]
#     core = arr_abs[(arr_abs > peak_x - 2) & (arr_abs < peak_x + 2)]
#     sigma0 = np.std(core) if core.size > 20 else np.std(arr_abs)
#     sigma0 = max(float(sigma0), 0.3)

#     p0 = [float(hist.max()), float(peak_x), float(sigma0), float(np.median(hist[:10]))]
#     yerr = np.sqrt(np.maximum(hist, 1.0))

#     try:
#         popt, _ = curve_fit(
#             folded_gaussian,
#             bin_centers,
#             hist,
#             p0=p0,
#             sigma=yerr,
#             absolute_sigma=True,
#             bounds=([0, 0, 0.05, 0], [np.inf, 30, 10, np.inf]),
#             maxfev=40000
#         )
#         return popt
#     except Exception:
#         return None
def _fit_channel(bin_centers, hist, arr_abs):
    """
    Robust folded-Gaussian fit:
      1) find peak bin
      2) estimate sigma0 from a tight core region
      3) fit only within [peak - W*sigma0, peak + W*sigma0]
      4) fix baseline B from sidebands (optional but recommended)
    Returns popt=(A,mu,sigma,B) or None.
    """
    if np.sum(hist) < MIN_ENTRIES:
        return None

    # only consider bins with content for stability
    nonzero = hist > 0
    if nonzero.sum() < 10:
        return None

    # peak location from histogram
    peak_idx = np.argmax(hist)
    peak_x = float(bin_centers[peak_idx])

    # robust sigma estimate from data near peak (core ±2 ns)
    core = arr_abs[(arr_abs > peak_x - 2.0) & (arr_abs < peak_x + 2.0)]
    if core.size < 50:
        core = arr_abs
    # MAD-based sigma (robust)
    med = np.median(core)
    mad = np.median(np.abs(core - med))
    sigma0 = 1.4826 * mad if mad > 0 else np.std(core)
    sigma0 = max(float(sigma0), 0.25)

    # Fit window: restrict to local region around the peak
    W = 3.0  # 3 sigma window is usually enough
    lo = max(XLIM_TFINAL[0], peak_x - W * sigma0)
    hi = min(XLIM_TFINAL[1], peak_x + W * sigma0)

    fit_mask = (bin_centers >= lo) & (bin_centers <= hi)
    # also require some statistics in the fit window
    if np.sum(hist[fit_mask]) < MIN_ENTRIES * 0.6 or fit_mask.sum() < 8:
        return None

    x = bin_centers[fit_mask]
    y = hist[fit_mask]

    # Baseline from sidebands (fix B)
    # Take first/last 10 bins of the FULL histogram as sidebands
    side_n = 10
    side_vals = np.r_[hist[:side_n], hist[-side_n:]]
    Bfix = float(np.median(side_vals))

    # Fit A, mu, sigma with fixed B
    def model_fixedB(xx, A, mu, sigma):
        return folded_gaussian(xx, A, mu, sigma, Bfix)

    # initial guesses
    A0 = float(max(y.max() - Bfix, 1.0))
    mu0 = peak_x
    p0 = [A0, mu0, sigma0]

    # weights (Poisson)
    yerr = np.sqrt(np.maximum(y, 1.0))

    try:
        popt3, _ = curve_fit(
            model_fixedB,
            x, y,
            p0=p0,
            sigma=yerr,
            absolute_sigma=True,
            bounds=([0, 0, 0.05], [np.inf, 30, 10]),
            maxfev=40000
        )
        A, mu, sigma = popt3
        return (float(A), float(mu), float(sigma), float(Bfix))
    except Exception:
        return None


def _channel_ok(g, c):
    # Skip trigger
    if c == 8:
        return False
    # Skip MCP channels
    if g == 3 and c in (6, 7):
        return False
    return True


def _xlabel():
    return (
        r"$|(t_{\mathrm{fit}}^{ch}-t_{\mathrm{trig}}^{g})"
        r"-(t_{\mathrm{fit}}^{\mathrm{MCP7}}-t_{\mathrm{trig}}^{3})|$ [ns]"
    )


# ================= WORKER: PER-BOARD PDFs =================
def plot_board(b):
    colors = plt.cm.tab10.colors

    with uproot.open(ANA_FILE) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        # Load all available channels for this board
        data = {}
        for g in range(NG):
            for c in range(NC):
                if not _channel_ok(g, c):
                    continue
                k = f"tfinal_Board{b}_Group{g}_Channel{c}"
                if k in keys:
                    data[(g, c)] = tree[k].array(library="np")

    # Outputs per board
    pdf_hist_only = f"{OUTDIR}/Board{b}_tfinal_byGroup_hist_only_zoomedin.pdf"
    pdf_hist_fit  = f"{OUTDIR}/Board{b}_tfinal_byGroup_hist_plus_fit_zoomedin.pdf"
    pdf_gaus_only = f"{OUTDIR}/Board{b}_tfinal_byGroup_gaussians_only_BGClegend.pdf"

    bin_edges = np.linspace(XLIM_TFINAL[0], XLIM_TFINAL[1], NBINS + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    xlabel = _xlabel()

    with PdfPages(pdf_hist_only) as pdfH, PdfPages(pdf_hist_fit) as pdfF, PdfPages(pdf_gaus_only) as pdfG:
        for g in range(NG):
            figH, axH = plt.subplots(figsize=(7.5, 5))
            figF, axF = plt.subplots(figsize=(7.5, 5))
            figG, axG = plt.subplots(figsize=(7.5, 5))

            any_fit_g = False

            for c in range(NC):
                if not _channel_ok(g, c):
                    continue
                if (g, c) not in data:
                    continue

                arr = data[(g, c)]
                if arr.size < MIN_RAW:
                    continue

                arr_abs = np.abs(arr)
                arr_abs = arr_abs[arr_abs >= CUT_MIN]
                if arr_abs.size < MIN_ENTRIES:
                    continue

                hist, _ = np.histogram(arr_abs, bins=bin_edges)
                col = colors[c % len(colors)]

                # HIST ONLY
                axH.step(bin_centers, hist, where="mid", lw=1.2, color=col, label=f"C{c}")

                # HIST + FIT
                axF.step(bin_centers, hist, where="mid", lw=1.0, color=col, alpha=0.75, label=f"C{c}")

                # FIT
                popt = _fit_channel(bin_centers, hist, arr_abs)
                if popt is None:
                    continue

                A, mu, sigma, B = popt
                xfit = np.linspace(*XLIM_TFINAL, 800)
                yfit = folded_gaussian(xfit, *popt)

                axF.plot(xfit, yfit, color=col, lw=1.8,
                         label=f"C{c} fit: μ={mu:.2f}, σ={sigma:.2f}")

                # GAUSSIANS ONLY (legend must be BGC)
                any_fit_g = True
                axG.plot(xfit, yfit, lw=1.4, color=col, label=f"B{b}G{g}C{c}")

            # Style HIST ONLY
            axH.set_xlabel(xlabel)
            axH.set_ylabel("Events")
            axH.set_title(f"Board {b} — Group {g} (|tfinal| > {CUT_MIN} ns) — HIST ONLY")
            axH.set_xlim(*XLIM_TFINAL)
            axH.minorticks_on()
            axH.tick_params(axis="both", which="major", length=6)
            axH.tick_params(axis="both", which="minor", length=3)
            axH.legend(fontsize=7, ncol=4, frameon=False)
            figH.tight_layout()
            pdfH.savefig(figH)
            plt.close(figH)

            # Style HIST + FIT
            axF.set_xlabel(xlabel)
            axF.set_ylabel("Events")
            axF.set_title(f"Board {b} — Group {g} (|tfinal| > {CUT_MIN} ns) — HIST + FIT")
            axF.set_xlim(*XLIM_TFINAL)
            axF.minorticks_on()
            axF.tick_params(axis="both", which="major", length=6)
            axF.tick_params(axis="both", which="minor", length=3)
            axF.legend(fontsize=7, ncol=2, frameon=False)
            figF.tight_layout()
            pdfF.savefig(figF)
            plt.close(figF)

            # Style GAUSSIANS ONLY
            axG.set_xlabel(xlabel)
            axG.set_ylabel("Arbitrary units (fit)")
            axG.set_title(f"Folded-Gaussian curves only — Board {b}, Group {g} (BGC legend)")
            axG.set_xlim(*XLIM_TFINAL)
            axG.minorticks_on()
            axG.tick_params(axis="both", which="major", length=6)
            axG.tick_params(axis="both", which="minor", length=3)
            if any_fit_g:
                axG.legend(fontsize=7, ncol=3, frameon=False)
            else:
                axG.text(0.5, 0.5, "No successful fits", ha="center", va="center",
                         transform=axG.transAxes)
            figG.tight_layout()
            pdfG.savefig(figG)
            plt.close(figG)

    return b


# ================= FILE-LEVEL: GAUSSIANS ONLY (MULTI-PAGE: one page per board) =================
def make_allboards_gaussians_only_multipage():
    pdf_path = f"{OUTDIR}/ALLBOARDS_gaussians_only_BGClegend_multipage.pdf"

    bin_edges = np.linspace(XLIM_TFINAL[0], XLIM_TFINAL[1], NBINS + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    xlabel = _xlabel()

    with uproot.open(ANA_FILE) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        with PdfPages(pdf_path) as pdf:
            for b in BOARDS:
                fig, ax = plt.subplots(figsize=(11, 7.5))
                any_fit = False

                for g in range(NG):
                    for c in range(NC):
                        if not _channel_ok(g, c):
                            continue
                        k = f"tfinal_Board{b}_Group{g}_Channel{c}"
                        if k not in keys:
                            continue

                        arr = tree[k].array(library="np")
                        if arr.size < MIN_RAW:
                            continue

                        arr_abs = np.abs(arr)
                        arr_abs = arr_abs[arr_abs >= CUT_MIN]
                        if arr_abs.size < MIN_ENTRIES:
                            continue

                        hist, _ = np.histogram(arr_abs, bins=bin_edges)
                        popt = _fit_channel(bin_centers, hist, arr_abs)
                        if popt is None:
                            continue

                        any_fit = True
                        xfit = np.linspace(*XLIM_TFINAL, 800)
                        yfit = folded_gaussian(xfit, *popt)
                        ax.plot(xfit, yfit, lw=1.1, label=f"B{b}G{g}C{c}")

                ax.set_xlabel(xlabel)
                ax.set_ylabel("Arbitrary units (fit)")
                ax.set_title(f"Folded-Gaussian curves only — Board {b} (BGC legend)")
                ax.set_xlim(*XLIM_TFINAL)
                ax.minorticks_on()
                ax.tick_params(axis="both", which="major", length=6)
                ax.tick_params(axis="both", which="minor", length=3)

                if any_fit:
                    ax.legend(fontsize=6, ncol=5, frameon=False)
                else:
                    ax.text(0.5, 0.5, "No successful fits for this board",
                            ha="center", va="center", transform=ax.transAxes)

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Saved: {pdf_path}")


# ================= FILE-LEVEL: GAUSSIANS ONLY (SINGLE PAGE: all boards on one plot) =================
def make_allboards_gaussians_only_singlepage():
    pdf_path = f"{OUTDIR}/ALLBOARDS_gaussians_only_BGClegend_SINGLEPAGE.pdf"

    bin_edges = np.linspace(XLIM_TFINAL[0], XLIM_TFINAL[1], NBINS + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    xlabel = _xlabel()

    with uproot.open(ANA_FILE) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        with PdfPages(pdf_path) as pdf:
            fig, ax = plt.subplots(figsize=(12, 8))
            any_fit = False

            for b in BOARDS:
                for g in range(NG):
                    for c in range(NC):
                        if not _channel_ok(g, c):
                            continue
                        k = f"tfinal_Board{b}_Group{g}_Channel{c}"
                        if k not in keys:
                            continue

                        arr = tree[k].array(library="np")
                        if arr.size < MIN_RAW:
                            continue

                        arr_abs = np.abs(arr)
                        arr_abs = arr_abs[arr_abs >= CUT_MIN]
                        if arr_abs.size < MIN_ENTRIES:
                            continue

                        hist, _ = np.histogram(arr_abs, bins=bin_edges)
                        popt = _fit_channel(bin_centers, hist, arr_abs)
                        if popt is None:
                            continue

                        any_fit = True
                        xfit = np.linspace(*XLIM_TFINAL, 800)
                        yfit = folded_gaussian(xfit, *popt)
                        ax.plot(xfit, yfit, lw=1.0, label=f"B{b}G{g}C{c}")

            ax.set_xlabel(xlabel)
            ax.set_ylabel("Arbitrary units (fit)")
            ax.set_title("Folded-Gaussian curves only — ALL BOARDS (BGC legend)")
            ax.set_xlim(*XLIM_TFINAL)
            ax.minorticks_on()
            ax.tick_params(axis="both", which="major", length=6)
            ax.tick_params(axis="both", which="minor", length=3)

            if any_fit:
                ax.legend(fontsize=6, ncol=6, frameon=False)
            else:
                ax.text(0.5, 0.5, "No successful fits",
                        ha="center", va="center", transform=ax.transAxes)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved: {pdf_path}")

def make_allboards_hist_only_multipage():
    """
    File-level HIST ONLY:
      - multi-page PDF: one page per board
      - overlays all channels (g,c) for that board
      - uses |tfinal| and CUT_MIN, XLIM_TFINAL, NBINS
    """
    pdf_path = f"{OUTDIR}/ALLBOARDS_hist_only_multipage.pdf"

    bin_edges = np.linspace(XLIM_TFINAL[0], XLIM_TFINAL[1], NBINS + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    xlabel = _xlabel()

    with uproot.open(ANA_FILE) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        with PdfPages(pdf_path) as pdf:
            for b in BOARDS:
                fig, ax = plt.subplots(figsize=(11, 7.5))
                any_hist = False

                for g in range(NG):
                    for c in range(NC):
                        if not _channel_ok(g, c):
                            continue

                        k = f"tfinal_Board{b}_Group{g}_Channel{c}"
                        if k not in keys:
                            continue

                        arr = tree[k].array(library="np")
                        if arr.size < MIN_RAW:
                            continue

                        arr_abs = np.abs(arr)
                        arr_abs = arr_abs[arr_abs >= CUT_MIN]
                        if arr_abs.size < MIN_ENTRIES:
                            continue

                        hist, _ = np.histogram(arr_abs, bins=bin_edges)
                        any_hist = True

                        ax.step(bin_centers, hist, where="mid", lw=1.0, label=f"B{b}G{g}C{c}")

                ax.set_xlabel(xlabel)
                ax.set_ylabel("Events")
                ax.set_title(f"HIST ONLY — Board {b} (all channels)")
                ax.set_xlim(*XLIM_TFINAL)
                ax.minorticks_on()
                ax.tick_params(axis="both", which="major", length=6)
                ax.tick_params(axis="both", which="minor", length=3)

                if any_hist:
                    ax.legend(fontsize=6, ncol=6, frameon=False)
                else:
                    ax.text(0.5, 0.5, "No channels passed cuts",
                            ha="center", va="center", transform=ax.transAxes)

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Saved: {pdf_path}")


def make_allboards_hist_only_singlepage():
    """
    File-level HIST ONLY:
      - single-page PDF: ALL boards, all channels on one plot
      - overlays all histograms with BGC legend
    Warning: can get visually crowded.
    """
    pdf_path = f"{OUTDIR}/ALLBOARDS_hist_only_SINGLEPAGE.pdf"

    bin_edges = np.linspace(XLIM_TFINAL[0], XLIM_TFINAL[1], NBINS + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    xlabel = _xlabel()

    with uproot.open(ANA_FILE) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        with PdfPages(pdf_path) as pdf:
            fig, ax = plt.subplots(figsize=(12, 8))
            any_hist = False

            for b in BOARDS:
                for g in range(NG):
                    for c in range(NC):
                        if not _channel_ok(g, c):
                            continue

                        k = f"tfinal_Board{b}_Group{g}_Channel{c}"
                        if k not in keys:
                            continue

                        arr = tree[k].array(library="np")
                        if arr.size < MIN_RAW:
                            continue

                        arr_abs = np.abs(arr)
                        arr_abs = arr_abs[arr_abs >= CUT_MIN]
                        if arr_abs.size < MIN_ENTRIES:
                            continue

                        hist, _ = np.histogram(arr_abs, bins=bin_edges)
                        any_hist = True

                        ax.step(bin_centers, hist, where="mid", lw=0.9, label=f"B{b}G{g}C{c}")

            ax.set_xlabel(xlabel)
            ax.set_ylabel("Events")
            ax.set_title("HIST ONLY — ALL BOARDS (all channels)")
            ax.set_xlim(*XLIM_TFINAL)
            ax.minorticks_on()
            ax.tick_params(axis="both", which="major", length=6)
            ax.tick_params(axis="both", which="minor", length=3)

            if any_hist:
                ax.legend(fontsize=6, ncol=7, frameon=False)
            else:
                ax.text(0.5, 0.5, "No channels passed cuts",
                        ha="center", va="center", transform=ax.transAxes)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved: {pdf_path}")


def _channel_ok_even(g, c):
    # keep your original vetoes
    if not _channel_ok(g, c):
        return False
    # even channels only
    return (c % 2 == 0)


def make_evenchannels_fitcentered_perboard(b, W=3.0):
    """
    Per board, one PDF with 4 pages (one per group).
    Even channels only. Hist + fit, with x-limits centered around fitted peaks.
    """
    colors = plt.cm.tab10.colors
    pdf_path = f"{OUTDIR}/Board{b}_evenChannels_fitCentered_hist_plus_fit.pdf"

    bin_edges = np.linspace(XLIM_TFINAL[0], XLIM_TFINAL[1], NBINS + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    xlabel = _xlabel()

    with uproot.open(ANA_FILE) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        # preload arrays for even channels only (this board)
        data = {}
        for g in range(NG):
            for c in range(NC):
                if not _channel_ok_even(g, c):
                    continue
                k = f"tfinal_Board{b}_Group{g}_Channel{c}"
                if k in keys:
                    data[(g, c)] = tree[k].array(library="np")

    with PdfPages(pdf_path) as pdf:
        for g in range(NG):
            fig, ax = plt.subplots(figsize=(7.5, 5))

            # --- first pass: fit each even channel to determine dynamic x-range ---
            fit_params = {}  # c -> (A,mu,sigma,B)
            mus = []
            lo_list = []
            hi_list = []

            for c in range(NC):
                if not _channel_ok_even(g, c):
                    continue
                if (g, c) not in data:
                    continue

                arr = data[(g, c)]
                if arr.size < MIN_RAW:
                    continue

                arr_abs = np.abs(arr)
                arr_abs = arr_abs[arr_abs >= CUT_MIN]
                if arr_abs.size < MIN_ENTRIES:
                    continue

                hist, _ = np.histogram(arr_abs, bins=bin_edges)
                popt = _fit_channel(bin_centers, hist, arr_abs)
                if popt is None:
                    continue

                A, mu, sigma, B = popt
                fit_params[c] = popt
                mus.append(mu)
                lo_list.append(mu - W * sigma)
                hi_list.append(mu + W * sigma)

            # dynamic x-range (fallback to global if no fits)
            if len(lo_list) > 0:
                xlo = max(XLIM_TFINAL[0], float(min(lo_list)))
                xhi = min(XLIM_TFINAL[1], float(max(hi_list)))
                # avoid degenerate windows
                if (xhi - xlo) < 1.0:
                    m = float(np.median(mus))
                    xlo = max(XLIM_TFINAL[0], m - 1.0)
                    xhi = min(XLIM_TFINAL[1], m + 1.0)
            else:
                xlo, xhi = XLIM_TFINAL

            # --- second pass: plot hist + fit for even channels within that zoom ---
            any_drawn = False
            for c in range(NC):
                if not _channel_ok_even(g, c):
                    continue
                if (g, c) not in data:
                    continue

                arr = data[(g, c)]
                if arr.size < MIN_RAW:
                    continue

                arr_abs = np.abs(arr)
                arr_abs = arr_abs[arr_abs >= CUT_MIN]
                if arr_abs.size < MIN_ENTRIES:
                    continue

                hist, _ = np.histogram(arr_abs, bins=bin_edges)
                col = colors[c % len(colors)]

                ax.step(bin_centers, hist, where="mid", lw=1.0, color=col, alpha=0.75, label=f"C{c}")
                any_drawn = True

                if c in fit_params:
                    popt = fit_params[c]
                    A, mu, sigma, B = popt
                    xfit = np.linspace(xlo, xhi, 600)
                    yfit = folded_gaussian(xfit, *popt)
                    ax.plot(xfit, yfit, color=col, lw=1.8, label=f"C{c} fit: μ={mu:.2f}, σ={sigma:.2f}")

            ax.set_xlabel(xlabel)
            ax.set_ylabel("Events")
            ax.set_title(f"Board {b} — Group {g} — EVEN channels only — fit-centered zoom")
            ax.set_xlim(xlo, xhi)
            ax.minorticks_on()
            ax.tick_params(axis="both", which="major", length=6)
            ax.tick_params(axis="both", which="minor", length=3)

            if any_drawn:
                ax.legend(fontsize=7, ncol=2, frameon=False)
            else:
                ax.text(0.5, 0.5, "No even-channel histograms passed cuts",
                        ha="center", va="center", transform=ax.transAxes)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved: {pdf_path}")

# ================= MAIN =================
def main():
    print("Generating per-board PDFs (hist-only + hist+fit + gaussians-only-by-group). No CSV.")

    nproc = min(cpu_count(), len(list(BOARDS)))
    with Pool(nproc) as pool:
        for b in tqdm(pool.imap_unordered(plot_board, BOARDS),
                      total=len(list(BOARDS)),
                      desc="Boards"):
            print(f"  → Board {b} done")

    print("Generating file-level Gaussian-only PDFs (multi-page + single-page).")
    make_allboards_gaussians_only_multipage()
    make_allboards_gaussians_only_singlepage()

    print("Generating file-level HIST-only PDFs (multi-page + single-page).")
    make_allboards_hist_only_multipage()
    make_allboards_hist_only_singlepage()

    print("Generating EVEN-channel fit-centered zoom PDFs (one per board).")
    for b in BOARDS:
        make_evenchannels_fitcentered_perboard(b, W=3.0)


    

    print("Done.")


if __name__ == "__main__":
    main()
