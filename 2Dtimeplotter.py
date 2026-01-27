#!/usr/bin/env python3
import os
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

# ================= USER SETTINGS =================
ANA_FILE = "TRUE-HGtiming/skimmed_files/run1355_250924165834_TimingDAQ_postaskim_allchannels_newmethod.root"
TREE_NAME = "EventTree"

OUTDIR = "./TRUE-HGtiming/SkimmedResults/positron80GeV_1355_newmethod/timeheatmaps"
os.makedirs(OUTDIR, exist_ok=True)

BOARDS = range(4)
NG = 4
NC = 9

NBINS = 300
XLIM_TFINAL = (5.0, 20.0)   # fit range for |tfinal|
CUT_MIN = 1.0               # ignore |tfinal| < CUT_MIN
MIN_ENTRIES = 200           # after abs + cut
MIN_RAW = 500               # raw entries before abs/cut

# Plot cosmetics
ANNOTATE = True             # write values in cells
SHOW_METHOD = False         # if True, annotate small "F" for fit, "M" for median fallback
CMAP = "viridis"            # any matplotlib colormap name
# =================================================


# ================= MODEL (used only to compute mu) =================
def folded_gaussian(x, A, mu, sigma, B):
    return A * (
        np.exp(-0.5 * ((x - mu) / sigma) ** 2) +
        np.exp(-0.5 * ((x + mu) / sigma) ** 2)
    ) + B


def fit_folded_gaussian_mu(bin_centers, hist, arr_abs):
    """
    Robust folded-Gaussian fit (fixed baseline from sidebands).
    Returns (mu, sigma) or None.
    """
    if np.sum(hist) < MIN_ENTRIES:
        return None

    nonzero = hist > 0
    if nonzero.sum() < 10:
        return None

    peak_idx = int(np.argmax(hist))
    peak_x = float(bin_centers[peak_idx])

    # robust sigma guess from core around the peak
    core = arr_abs[(arr_abs > peak_x - 2.0) & (arr_abs < peak_x + 2.0)]
    if core.size < 50:
        core = arr_abs

    med = float(np.median(core))
    mad = float(np.median(np.abs(core - med)))
    sigma0 = 1.4826 * mad if mad > 0 else float(np.std(core))
    sigma0 = max(sigma0, 0.25)

    # fit window: 3 sigma around peak
    W = 3.0
    lo = max(XLIM_TFINAL[0], peak_x - W * sigma0)
    hi = min(XLIM_TFINAL[1], peak_x + W * sigma0)

    fit_mask = (bin_centers >= lo) & (bin_centers <= hi)
    if np.sum(hist[fit_mask]) < MIN_ENTRIES * 0.6 or fit_mask.sum() < 8:
        return None

    x = bin_centers[fit_mask]
    y = hist[fit_mask]

    # baseline from sidebands
    side_n = 10
    side_vals = np.r_[hist[:side_n], hist[-side_n:]]
    Bfix = float(np.median(side_vals))

    def model_fixedB(xx, A, mu, sigma):
        return folded_gaussian(xx, A, mu, sigma, Bfix)

    A0 = float(max(y.max() - Bfix, 1.0))
    p0 = [A0, peak_x, sigma0]

    yerr = np.sqrt(np.maximum(y, 1.0))

    try:
        popt, _ = curve_fit(
            model_fixedB,
            x, y,
            p0=p0,
            sigma=yerr,
            absolute_sigma=True,
            bounds=([0, 0, 0.05], [np.inf, 30, 10]),
            maxfev=40000
        )
        _, mu, sigma = popt
        return (float(mu), float(sigma))
    except Exception:
        return None


# ================= CHANNEL VETOES =================
def channel_ok(g, c):
    if c == 8:            # trigger channel
        return False
    if g == 3 and c in (6, 7):  # MCP channels veto (your choice)
        return False
    return True


# ================= MAPPING (from your screenshots) =================
# Each row: (left_cell, right_cell) where cell is (g,c). None = 53x special cell (skip)
EVEN_ROWS = [
    ((0, 2), (0, 0)),
    ((0, 6), (0, 4)),
    ((1, 2), (1, 0)),
    ((1, 6), (1, 4)),
    ((2, 2), (2, 0)),
    ((2, 6), (2, 4)),
    ((3, 2), (3, 0)),
    (None,  (3, 4)),
]

ODD_ROWS = [
    ((0, 3), (0, 1)),
    ((0, 7), (0, 5)),
    ((1, 3), (1, 1)),
    ((1, 7), (1, 5)),
    ((2, 3), (2, 1)),
    ((2, 7), (2, 5)),
    ((3, 3), (3, 1)),
    (None,  (3, 5)),
]


def compute_channel_mu(arr, bin_edges, bin_centers):
    """
    Return (value, method) where value is mu (fit) or median fallback.
    method: 'fit' | 'median' | 'nan'
    """
    if arr is None or arr.size < MIN_RAW:
        return (np.nan, "nan")

    arr_abs = np.abs(arr)
    arr_abs = arr_abs[arr_abs >= CUT_MIN]
    if arr_abs.size < MIN_ENTRIES:
        return (np.nan, "nan")

    hist, _ = np.histogram(arr_abs, bins=bin_edges)
    fit = fit_folded_gaussian_mu(bin_centers, hist, arr_abs)
    if fit is not None:
        mu, _sigma = fit
        return (mu, "fit")

    return (float(np.median(arr_abs)), "median")


def build_board_grid_mu(tree, keys, b, bin_edges, bin_centers):
    """
    Returns:
      grid_mu: shape (8 rows, 4 cols) = [even_left, even_right, odd_left, odd_right]
      grid_method: same shape, strings: fit|median|nan
    """
    grid_mu = np.full((8, 4), np.nan, dtype=float)
    grid_method = np.full((8, 4), "", dtype=object)

    # helper to read branch if exists
    def read_tfinal(g, c):
        if not channel_ok(g, c):
            return None
        k = f"tfinal_Board{b}_Group{g}_Channel{c}"
        if k not in keys:
            return None
        return tree[k].array(library="np")

    # EVEN in cols 0-1
    for r, (left, right) in enumerate(EVEN_ROWS):
        for cc, cell in enumerate([left, right]):  # cc=0->col0, cc=1->col1
            if cell is None:
                grid_mu[r, cc] = np.nan
                grid_method[r, cc] = "nan"
                continue
            g, c = cell
            arr = read_tfinal(g, c)
            val, method = compute_channel_mu(arr, bin_edges, bin_centers)
            grid_mu[r, cc] = val
            grid_method[r, cc] = method

    # ODD in cols 2-3
    for r, (left, right) in enumerate(ODD_ROWS):
        for cc, cell in enumerate([left, right]):  # cc=0->col2, cc=1->col3
            col = 2 + cc
            if cell is None:
                grid_mu[r, col] = np.nan
                grid_method[r, col] = "nan"
                continue
            g, c = cell
            arr = read_tfinal(g, c)
            val, method = compute_channel_mu(arr, bin_edges, bin_centers)
            grid_mu[r, col] = val
            grid_method[r, col] = method

    return grid_mu, grid_method


def draw_board_heatmap(ax, grid, method_grid, board_id, vmin, vmax):
    im = ax.imshow(grid, origin="upper", cmap=CMAP, vmin=vmin, vmax=vmax, aspect="auto")

    # ticks
    ax.set_xticks([0, 1, 2, 3], ["even-L", "even-R", "odd-L", "odd-R"])
    ax.set_yticks(range(8), [str(i) for i in range(8)])
    ax.set_ylabel("Row (top→bottom)")
    ax.set_title(f"Board {board_id}: z = μ(|tfinal|) [ns]")

    # grid lines
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which="minor", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # annotate
    if ANNOTATE:
        for r in range(8):
            for c in range(4):
                val = grid[r, c]
                if np.isfinite(val):
                    s = f"{val:.2f}"
                    if SHOW_METHOD:
                        m = method_grid[r, c]
                        s = s + ("F" if m == "fit" else ("M" if m == "median" else ""))
                    ax.text(c, r, s, ha="center", va="center", fontsize=8, color="white")
    return im


def main():
    # binning for fit computation
    bin_edges = np.linspace(XLIM_TFINAL[0], XLIM_TFINAL[1], NBINS + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    with uproot.open(ANA_FILE) as f:
        tree = f[TREE_NAME]
        keys = set(tree.keys())

        # --- build all boards first to get consistent color scale ---
        board_grids = {}
        board_methods = {}
        all_vals = []

        for b in BOARDS:
            grid_mu, grid_method = build_board_grid_mu(tree, keys, b, bin_edges, bin_centers)
            board_grids[b] = grid_mu
            board_methods[b] = grid_method
            all_vals.append(grid_mu[np.isfinite(grid_mu)])

        all_vals = np.concatenate(all_vals) if len(all_vals) else np.array([])
        if all_vals.size == 0:
            raise RuntimeError("No finite μ values were produced. Check branch names / cuts.")

        vmin = float(np.nanpercentile(all_vals, 2))
        vmax = float(np.nanpercentile(all_vals, 98))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

        # --- per-board PDFs ---
        for b in BOARDS:
            pdf_path = os.path.join(OUTDIR, f"tfinal_heatmap_mu_board{b}.pdf")
            with PdfPages(pdf_path) as pdf:
                fig, ax = plt.subplots(figsize=(7.5, 6.5))
                im = draw_board_heatmap(ax, board_grids[b], board_methods[b], b, vmin, vmax)
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("μ(|tfinal|) [ns]")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
            print(f"Saved: {pdf_path}")

        # --- mosaic PDF: (0,2) top row; (1,3) bottom row ---
        mosaic_path = os.path.join(OUTDIR, "tfinal_heatmap_mu_mosaic.pdf")
        with PdfPages(mosaic_path) as pdf:
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))

            layout = [[0, 2],
                      [1, 3]]

            ims = []
            for i in range(2):
                for j in range(2):
                    b = layout[i][j]
                    im = draw_board_heatmap(
                        axes[i, j],
                        board_grids[b],
                        board_methods[b],
                        b,
                        vmin, vmax
                    )
                    ims.append(im)

            # one shared colorbar
            cbar = fig.colorbar(ims[0], ax=axes.ravel().tolist(), shrink=0.9)
            cbar.set_label("μ(|tfinal|) [ns]")

            fig.suptitle("tfinal heatmaps (μ of |tfinal|) — Board layout: [0 2; 1 3]", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

        print(f"Saved: {mosaic_path}")


if __name__ == "__main__":
    main()
