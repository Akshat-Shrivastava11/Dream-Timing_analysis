#!/usr/bin/env python3
import os
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy.optimize import curve_fit

# ================= USER SETTINGS =================
ANA_FILE = "TRUE-HGtiming/skimmed_files/run1513_250928194230_TimingDAQ_postaskim_allchannels_newmethod.root"
TREE_NAME = "EventTree"

OUTDIR = "./TRUE-HGtiming/SkimmedResults/90degreeRuns_tfinal_analysis"
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
SQRT2PI = np.sqrt(2.0 * np.pi)

def folded_gaussian_counts(x, N, mu, sigma, B, binw):
    """
    Expected COUNTS PER BIN at positions x for a folded Gaussian on |t|.
    N = total signal yield (approximately counts above baseline)
    B = baseline counts/bin
    binw = histogram bin width (ns)
    """
    g1 = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (SQRT2PI * sigma)
    g2 = np.exp(-0.5 * ((x + mu) / sigma) ** 2) / (SQRT2PI * sigma)
    pdf = g1 + g2
    return N * binw * pdf + B

def _fit_channel(bin_centers, hist, arr_abs, bin_edges):
    """
    Fit yield-normalized folded Gaussian in counts/bin:
      y = N*binw*pdf_folded(x; mu,sigma) + B
    Returns popt=(N,mu,sigma,B) or None.
    """
    if np.sum(hist) < MIN_ENTRIES:
        return None

    nonzero = hist > 0
    if nonzero.sum() < 10:
        return None

    peak_idx = int(np.argmax(hist))
    peak_x = float(bin_centers[peak_idx])

    core = arr_abs[(arr_abs > peak_x - 2.0) & (arr_abs < peak_x + 2.0)]
    if core.size < 50:
        core = arr_abs

    med = np.median(core)
    mad = np.median(np.abs(core - med))
    sigma0 = 1.4826 * mad if mad > 0 else np.std(core)
    sigma0 = max(float(sigma0), 0.20)

    W = 3.0
    lo = max(XLIM_TFINAL[0], peak_x - W * sigma0)
    hi = min(XLIM_TFINAL[1], peak_x + W * sigma0)

    fit_mask = (bin_centers >= lo) & (bin_centers <= hi)
    if np.sum(hist[fit_mask]) < MIN_ENTRIES * 0.6 or fit_mask.sum() < 8:
        return None

    x = bin_centers[fit_mask]
    y = hist[fit_mask]

    binw = float(bin_edges[1] - bin_edges[0])

    side_n = 10
    side_vals = np.r_[hist[:side_n], hist[-side_n:]]
    B0 = float(np.median(side_vals))

    N0 = float(max(np.sum(y - B0), 1.0))
    yerr = np.sqrt(np.maximum(y, 1.0))

    def model(xx, N, mu, sigma, B):
        return folded_gaussian_counts(xx, N, mu, sigma, B, binw)

    p0 = [N0, peak_x, sigma0, B0]

    bounds_lo = [0.0, 0.0, 0.05, 0.0]
    bounds_hi = [np.inf, 30.0, 3.0, np.inf]

    try:
        popt, _ = curve_fit(
            model, x, y,
            p0=p0,
            sigma=yerr,
            absolute_sigma=True,
            bounds=(bounds_lo, bounds_hi),
            maxfev=60000
        )
        N, mu, sigma, B = popt
        return (float(N), float(mu), float(sigma), float(B))
    except Exception:
        return None

# ================= CHANNEL SELECTION =================
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

# ================= IO HELPERS =================
def _binning():
    bin_edges = np.linspace(XLIM_TFINAL[0], XLIM_TFINAL[1], NBINS + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return bin_edges, bin_centers

def _prep_arr(arr):
    """abs + CUT_MIN + MIN_ENTRIES check; returns cleaned arr_abs or None."""
    if arr is None or arr.size < MIN_RAW:
        return None
    arr_abs = np.abs(arr)
    arr_abs = arr_abs[arr_abs >= CUT_MIN]
    if arr_abs.size < MIN_ENTRIES:
        return None
    return arr_abs

def _load_board_data(tree, keys, b):
    """
    Load all available arrays for a board into dict:
      data[(g,c)] = np.array
    """
    data = {}
    for g in range(NG):
        for c in range(NC):
            if not _channel_ok(g, c):
                continue
            k = f"tfinal_Board{b}_Group{g}_Channel{c}"
            if k in keys:
                data[(g, c)] = tree[k].array(library="np")
    return data

# ================= EXISTING: PER-BOARD PDFs (BY GROUP, CHANNELS OVERLAID) =================
def plot_board(b):
    colors = plt.cm.tab10.colors

    bin_edges, bin_centers = _binning()
    xlabel = _xlabel()

    with uproot.open(ANA_FILE) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())
        data = _load_board_data(tree, keys, b)

    pdf_hist_only = f"{OUTDIR}/Board{b}_tfinal_byGroup_hist_only_zoomedin.pdf"
    pdf_hist_fit  = f"{OUTDIR}/Board{b}_tfinal_byGroup_hist_plus_fit_zoomedin.pdf"
    pdf_gaus_only = f"{OUTDIR}/Board{b}_tfinal_byGroup_gaussians_only_BGClegend.pdf"

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

                arr_abs = _prep_arr(data[(g, c)])
                if arr_abs is None:
                    continue

                hist, _ = np.histogram(arr_abs, bins=bin_edges)
                col = colors[c % len(colors)]

                axH.step(bin_centers, hist, where="mid", lw=1.2, color=col, label=f"C{c}")
                axF.step(bin_centers, hist, where="mid", lw=1.0, color=col, alpha=0.75, label=f"C{c}")

                popt = _fit_channel(bin_centers, hist, arr_abs, bin_edges)
                if popt is None:
                    continue

                N, mu, sigma, B = popt
                xfit = np.linspace(*XLIM_TFINAL, 800)
                binw = float(bin_edges[1] - bin_edges[0])
                yfit = folded_gaussian_counts(xfit, N, mu, sigma, B, binw)

                axF.plot(xfit, yfit, color=col, lw=1.8,
                         label=f"C{c} fit: μ={mu:.2f}, σ={sigma:.2f}")

                any_fit_g = True
                axG.plot(xfit, yfit, lw=1.4, color=col, label=f"B{b}G{g}C{c}")

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

# ================= NEW: PER-BOARD PDFs (BY CHANNEL, MODES/GROUPS OVERLAID) =================
def plot_board_bychannel_modes_overlay(b):
    """
    For each channel C (page), overlay all modes/groups G0..G3 for that board+channel.
    Produces: hist-only, hist+fit, gaussians-only PDFs.
    """
    colors = plt.cm.tab10.colors  # we will color by group
    bin_edges, bin_centers = _binning()
    xlabel = _xlabel()

    with uproot.open(ANA_FILE) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())
        data = _load_board_data(tree, keys, b)

    pdf_hist_only = f"{OUTDIR}/Board{b}_BYCHANNEL_modesOverlay_hist_only.pdf"
    pdf_hist_fit  = f"{OUTDIR}/Board{b}_BYCHANNEL_modesOverlay_hist_plus_fit.pdf"
    pdf_gaus_only = f"{OUTDIR}/Board{b}_BYCHANNEL_modesOverlay_gaussians_only.pdf"

    # channel list present for this board
    channels_present = sorted({c for (g, c) in data.keys()})
    # keep only allowed channels (safety)
    channels_present = [c for c in channels_present if _channel_ok(0, c)]  # ok doesn't depend on g except MCP veto already handled

    with PdfPages(pdf_hist_only) as pdfH, PdfPages(pdf_hist_fit) as pdfF, PdfPages(pdf_gaus_only) as pdfG:
        for c in channels_present:
            figH, axH = plt.subplots(figsize=(7.5, 5))
            figF, axF = plt.subplots(figsize=(7.5, 5))
            figG, axG = plt.subplots(figsize=(7.5, 5))

            any_hist = False
            any_fit  = False

            for g in range(NG):
                if not _channel_ok(g, c):
                    continue
                if (g, c) not in data:
                    continue

                arr_abs = _prep_arr(data[(g, c)])
                if arr_abs is None:
                    continue

                hist, _ = np.histogram(arr_abs, bins=bin_edges)
                any_hist = True

                col = colors[g % len(colors)]
                axH.step(bin_centers, hist, where="mid", lw=1.2, color=col, label=f"G{g}")
                axF.step(bin_centers, hist, where="mid", lw=1.0, color=col, alpha=0.75, label=f"G{g}")

                popt = _fit_channel(bin_centers, hist, arr_abs, bin_edges)
                if popt is None:
                    continue

                any_fit = True
                N, mu, sigma, B = popt
                xfit = np.linspace(*XLIM_TFINAL, 800)
                binw = float(bin_edges[1] - bin_edges[0])
                yfit = folded_gaussian_counts(xfit, N, mu, sigma, B, binw)

                axF.plot(xfit, yfit, color=col, lw=1.8,
                         label=f"G{g} fit: μ={mu:.2f}, σ={sigma:.2f}")

                axG.plot(xfit, yfit, color=col, lw=1.4, label=f"B{b}C{c}G{g}")

            # HIST ONLY page
            axH.set_xlabel(xlabel)
            axH.set_ylabel("Events")
            axH.set_title(f"Board {b} — Channel {c}: modes/groups overlaid — HIST ONLY")
            axH.set_xlim(*XLIM_TFINAL)
            axH.minorticks_on()
            axH.tick_params(axis="both", which="major", length=6)
            axH.tick_params(axis="both", which="minor", length=3)
            if any_hist:
                axH.legend(fontsize=8, ncol=4, frameon=False)
            else:
                axH.text(0.5, 0.5, "No groups passed cuts", ha="center", va="center", transform=axH.transAxes)
            figH.tight_layout()
            pdfH.savefig(figH)
            plt.close(figH)

            # HIST + FIT page
            axF.set_xlabel(xlabel)
            axF.set_ylabel("Events")
            axF.set_title(f"Board {b} — Channel {c}: modes/groups overlaid — HIST + FIT")
            axF.set_xlim(*XLIM_TFINAL)
            axF.minorticks_on()
            axF.tick_params(axis="both", which="major", length=6)
            axF.tick_params(axis="both", which="minor", length=3)
            if any_hist:
                axF.legend(fontsize=8, ncol=2, frameon=False)
            else:
                axF.text(0.5, 0.5, "No groups passed cuts", ha="center", va="center", transform=axF.transAxes)
            figF.tight_layout()
            pdfF.savefig(figF)
            plt.close(figF)

            # GAUSSIANS ONLY page
            axG.set_xlabel(xlabel)
            axG.set_ylabel("Arbitrary units (fit)")
            axG.set_title(f"Board {b} — Channel {c}: folded-Gaussian curves only (modes overlaid)")
            axG.set_xlim(*XLIM_TFINAL)
            axG.minorticks_on()
            axG.tick_params(axis="both", which="major", length=6)
            axG.tick_params(axis="both", which="minor", length=3)
            if any_fit:
                axG.legend(fontsize=8, ncol=3, frameon=False)
            else:
                axG.text(0.5, 0.5, "No successful fits", ha="center", va="center", transform=axG.transAxes)
            figG.tight_layout()
            pdfG.savefig(figG)
            plt.close(figG)

    print(f"Saved: {pdf_hist_only}")
    print(f"Saved: {pdf_hist_fit}")
    print(f"Saved: {pdf_gaus_only}")

# ================= FILE-LEVEL: GAUSSIANS ONLY (MULTI-PAGE: one page per board) =================
def make_allboards_gaussians_only_multipage():
    pdf_path = f"{OUTDIR}/ALLBOARDS_gaussians_only_BGClegend_multipage.pdf"

    bin_edges, bin_centers = _binning()
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

                        arr_abs = _prep_arr(tree[k].array(library="np"))
                        if arr_abs is None:
                            continue

                        hist, _ = np.histogram(arr_abs, bins=bin_edges)
                        popt = _fit_channel(bin_centers, hist, arr_abs, bin_edges)
                        if popt is None:
                            continue

                        any_fit = True
                        xfit = np.linspace(*XLIM_TFINAL, 800)
                        N, mu, sigma, B = popt
                        binw = float(bin_edges[1] - bin_edges[0])
                        yfit = folded_gaussian_counts(xfit, N, mu, sigma, B, binw)
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

    bin_edges, bin_centers = _binning()
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

                        arr_abs = _prep_arr(tree[k].array(library="np"))
                        if arr_abs is None:
                            continue

                        hist, _ = np.histogram(arr_abs, bins=bin_edges)
                        popt = _fit_channel(bin_centers, hist, arr_abs, bin_edges)
                        if popt is None:
                            continue

                        any_fit = True
                        xfit = np.linspace(*XLIM_TFINAL, 800)
                        N, mu, sigma, B = popt
                        binw = float(bin_edges[1] - bin_edges[0])
                        yfit = folded_gaussian_counts(xfit, N, mu, sigma, B, binw)
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

# ================= FILE-LEVEL: HIST ONLY (MULTI-PAGE: one page per board) =================
def make_allboards_hist_only_multipage():
    pdf_path = f"{OUTDIR}/ALLBOARDS_hist_only_multipage.pdf"

    bin_edges, bin_centers = _binning()
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

                        arr_abs = _prep_arr(tree[k].array(library="np"))
                        if arr_abs is None:
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

# ================= FILE-LEVEL: HIST ONLY (SINGLE PAGE: all boards on one plot) =================
def make_allboards_hist_only_singlepage():
    pdf_path = f"{OUTDIR}/ALLBOARDS_hist_only_SINGLEPAGE.pdf"

    bin_edges, bin_centers = _binning()
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

                        arr_abs = _prep_arr(tree[k].array(library="np"))
                        if arr_abs is None:
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

# ================= NEW: FILE-LEVEL "MODES ONLY" PLOTS (4 curves total) =================
def make_allboards_modes_only_hist_singlepage():
    """
    One plot with just modes (groups) of all channels:
      - concatenate arrays across ALL boards and ALL channels per group g
      - overlay 4 histograms (G0..G3)
    """
    pdf_path = f"{OUTDIR}/ALLBOARDS_MODES_ONLY_hist_only_SINGLEPAGE.pdf"

    bin_edges, bin_centers = _binning()
    xlabel = _xlabel()
    colors = plt.cm.tab10.colors

    with uproot.open(ANA_FILE) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        group_arrays = {g: [] for g in range(NG)}

        for b in BOARDS:
            for g in range(NG):
                for c in range(NC):
                    if not _channel_ok(g, c):
                        continue
                    k = f"tfinal_Board{b}_Group{g}_Channel{c}"
                    if k not in keys:
                        continue
                    arr_abs = _prep_arr(tree[k].array(library="np"))
                    if arr_abs is None:
                        continue
                    group_arrays[g].append(arr_abs)

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(10, 7))
        any_hist = False

        for g in range(NG):
            if len(group_arrays[g]) == 0:
                continue
            allg = np.concatenate(group_arrays[g])
            hist, _ = np.histogram(allg, bins=bin_edges)
            any_hist = True
            ax.step(bin_centers, hist, where="mid", lw=1.4, color=colors[g % len(colors)], label=f"G{g}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Events")
        ax.set_title("MODES ONLY — overlay groups (G0–G3), all boards + all channels")
        ax.set_xlim(*XLIM_TFINAL)
        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", length=6)
        ax.tick_params(axis="both", which="minor", length=3)

        if any_hist:
            ax.legend(fontsize=10, ncol=4, frameon=False)
        else:
            ax.text(0.5, 0.5, "No groups passed cuts", ha="center", va="center", transform=ax.transAxes)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved: {pdf_path}")

def make_allboards_modes_only_gaussians_singlepage():
    """
    Same as above, but plot only the folded-Gaussian fit curves for each group g,
    using the group-concatenated distribution.
    """
    pdf_path = f"{OUTDIR}/ALLBOARDS_MODES_ONLY_gaussians_only_SINGLEPAGE.pdf"

    bin_edges, bin_centers = _binning()
    xlabel = _xlabel()
    colors = plt.cm.tab10.colors

    with uproot.open(ANA_FILE) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        group_arrays = {g: [] for g in range(NG)}

        for b in BOARDS:
            for g in range(NG):
                for c in range(NC):
                    if not _channel_ok(g, c):
                        continue
                    k = f"tfinal_Board{b}_Group{g}_Channel{c}"
                    if k not in keys:
                        continue
                    arr_abs = _prep_arr(tree[k].array(library="np"))
                    if arr_abs is None:
                        continue
                    group_arrays[g].append(arr_abs)

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(10, 7))
        any_fit = False

        for g in range(NG):
            if len(group_arrays[g]) == 0:
                continue
            allg = np.concatenate(group_arrays[g])
            hist, _ = np.histogram(allg, bins=bin_edges)

            popt = _fit_channel(bin_centers, hist, allg, bin_edges)
            if popt is None:
                continue

            any_fit = True
            N, mu, sigma, B = popt
            xfit = np.linspace(*XLIM_TFINAL, 800)
            binw = float(bin_edges[1] - bin_edges[0])
            yfit = folded_gaussian_counts(xfit, N, mu, sigma, B, binw)

            ax.plot(xfit, yfit, lw=2.0, color=colors[g % len(colors)],
                    label=f"G{g}: μ={mu:.2f}, σ={sigma:.2f}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Arbitrary units (fit)")
        ax.set_title("MODES ONLY — folded-Gaussian curves from group-concatenated data (G0–G3)")
        ax.set_xlim(*XLIM_TFINAL)
        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", length=6)
        ax.tick_params(axis="both", which="minor", length=3)

        if any_fit:
            ax.legend(fontsize=10, ncol=2, frameon=False)
        else:
            ax.text(0.5, 0.5, "No successful fits", ha="center", va="center", transform=ax.transAxes)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved: {pdf_path}")

# ================= EVEN-CHANNEL FIT-CENTERED (UNCHANGED) =================
def _channel_ok_even(g, c):
    if not _channel_ok(g, c):
        return False
    return (c % 2 == 0)

def make_evenchannels_fitcentered_perboard(b, W=3.0):
    colors = plt.cm.tab10.colors
    pdf_path = f"{OUTDIR}/Board{b}_evenChannels_fitCentered_hist_plus_fit.pdf"

    bin_edges, bin_centers = _binning()
    xlabel = _xlabel()

    with uproot.open(ANA_FILE) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

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

            fit_params = {}
            mus = []
            lo_list = []
            hi_list = []

            for c in range(NC):
                if not _channel_ok_even(g, c):
                    continue
                if (g, c) not in data:
                    continue

                arr_abs = _prep_arr(data[(g, c)])
                if arr_abs is None:
                    continue

                hist, _ = np.histogram(arr_abs, bins=bin_edges)
                popt = _fit_channel(bin_centers, hist, arr_abs, bin_edges)
                if popt is None:
                    continue

                N, mu, sigma, B = popt
                fit_params[c] = popt
                mus.append(mu)
                lo_list.append(mu - W * sigma)
                hi_list.append(mu + W * sigma)

            if len(lo_list) > 0:
                xlo = max(XLIM_TFINAL[0], float(min(lo_list)))
                xhi = min(XLIM_TFINAL[1], float(max(hi_list)))
                if (xhi - xlo) < 1.0:
                    m = float(np.median(mus))
                    xlo = max(XLIM_TFINAL[0], m - 1.0)
                    xhi = min(XLIM_TFINAL[1], m + 1.0)
            else:
                xlo, xhi = XLIM_TFINAL

            any_drawn = False
            for c in range(NC):
                if not _channel_ok_even(g, c):
                    continue
                if (g, c) not in data:
                    continue

                arr_abs = _prep_arr(data[(g, c)])
                if arr_abs is None:
                    continue

                hist, _ = np.histogram(arr_abs, bins=bin_edges)
                col = colors[c % len(colors)]

                ax.step(bin_centers, hist, where="mid", lw=1.0, color=col, alpha=0.75, label=f"C{c}")
                any_drawn = True

                if c in fit_params:
                    N, mu, sigma, B = fit_params[c]
                    xfit = np.linspace(xlo, xhi, 600)
                    binw = float(bin_edges[1] - bin_edges[0])
                    yfit = folded_gaussian_counts(xfit, N, mu, sigma, B, binw)
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
    print("Generating per-board PDFs (by GROUP, channels overlaid): hist-only + hist+fit + gaussians-only-by-group.")
    nproc = min(cpu_count(), len(list(BOARDS)))
    with Pool(nproc) as pool:
        for b in tqdm(pool.imap_unordered(plot_board, BOARDS),
                      total=len(list(BOARDS)),
                      desc="Boards (by group)"):
            print(f"  → Board {b} done (by group)")

    print("Generating NEW per-board PDFs (by CHANNEL, modes/groups overlaid).")
    for b in BOARDS:
        plot_board_bychannel_modes_overlay(b)

    print("Generating file-level Gaussian-only PDFs (multi-page + single-page).")
    make_allboards_gaussians_only_multipage()
    make_allboards_gaussians_only_singlepage()

    print("Generating file-level HIST-only PDFs (multi-page + single-page).")
    make_allboards_hist_only_multipage()
    make_allboards_hist_only_singlepage()

    print("Generating NEW file-level MODES-ONLY PDFs (4 curves total).")
    make_allboards_modes_only_hist_singlepage()
    make_allboards_modes_only_gaussians_singlepage()

    print("Generating EVEN-channel fit-centered zoom PDFs (one per board).")
    for b in BOARDS:
        make_evenchannels_fitcentered_perboard(b, W=3.0)

    print("Done.")

if __name__ == "__main__":
    main()
