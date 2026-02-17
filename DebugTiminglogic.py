# #!/usr/bin/env python3
# import argparse
# import os
# import numpy as np
# import uproot
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from scipy.optimize import curve_fit

# # ================= HELPER FUNCTIONS =================
# def parse_code(code_str):
#     """Parses '104' into (Board=1, Group=0, Channel=4)."""
#     b = int(code_str[0])
#     g = int(code_str[1])
#     c = int(code_str[2])
#     return b, g, c

# def get_branch(b, g, c):
#     return f"DRS_Board{b}_Group{g}_Channel{c}_LP2_50"

# def gaussian(x, amp, mean, sigma):
#     return amp * np.exp(-(x - mean)**2 / (2 * sigma**2))

# def get_time_arrays(tree, branches):
#     """Safely loads multiple branches."""
#     data = {}
#     keys = set(tree.keys())
#     for name, br in branches.items():
#         if br not in keys:
#             print(f"[ERROR] Missing branch: {br}")
#             return None
#     arrays = tree.arrays(list(branches.values()), library="np")
#     for name, br in branches.items():
#         data[name] = arrays[br]
#     return data

# def fit_and_plot_peak(ax, data, label, color):
#     """
#     Plots histogram and fits Gaussian to the main signal bump (ignoring 0).
#     """
#     # 1. Select only "Signal" (> 10 ns) to ignore the huge noise spike at 0
#     signal = data[data > 10]
    
#     # Binning focused on the relevant range (0 to 200ns)
#     bins = np.linspace(0, 200, 201) # 1ns per bin
    
#     counts, edges, _ = ax.hist(data, bins=bins, color=color, alpha=0.6, label=label)
#     ax.hist(data, bins=bins, color=color, histtype='step', lw=1.5)

#     if len(signal) < 50:
#         ax.text(0.5, 0.5, "No Signal Found (>10ns)", transform=ax.transAxes, ha='center')
#         return

#     # 2. Find Peak statistics (Mean/RMS of the bump)
#     mu_signal = np.mean(signal)
#     std_signal = np.std(signal)
    
#     # 3. Simple Gaussian Fit to the bump
#     # Find mode of signal
#     counts_sig, edges_sig = np.histogram(signal, bins=100, range=(10, 180))
#     peak_idx = np.argmax(counts_sig)
#     peak_val = 0.5 * (edges_sig[peak_idx] + edges_sig[peak_idx+1])
    
#     # Fit window: +/- 20ns around peak
#     mask_fit = (edges[:-1] > peak_val - 20) & (edges[:-1] < peak_val + 20)
#     x_fit = 0.5 * (edges[1:] + edges[:-1])
    
#     try:
#         p0 = [counts.max(), peak_val, 10.0]
#         popt, _ = curve_fit(gaussian, x_fit[mask_fit], counts[mask_fit], p0=p0, maxfev=2000)
        
#         # Plot Fit
#         x_plot = np.linspace(0, 200, 500)
#         ax.plot(x_plot, gaussian(x_plot, *popt), 'r-', lw=2, 
#                 label=f'Fit: $\mu$={popt[1]:.2f} ns, $\sigma$={abs(popt[2]):.2f} ns')
        
#         # Draw vertical line at mean
#         ax.axvline(popt[1], color='red', linestyle='--', alpha=0.8)
        
#     except:
#         ax.axvline(peak_val, color='black', linestyle='--', label=f'Peak ~ {peak_val:.1f} ns')

#     ax.legend(loc="upper right", fontsize=12)
#     ax.set_xlabel("Arrival Time [ns]", fontsize=12)
#     ax.set_ylabel("Events", fontsize=12)
#     ax.grid(True, alpha=0.3)

# # ================= PLOTTING LOGIC =================
# def plot_separate_pages(pdf, rl, data, code_str):
#     """
#     Creates 4 separate pages, one for each component.
#     """
#     components = [
#         ('t_ch',   f"1. Probe Channel {code_str}", 'blue'),
#         ('t_trig', "2. Local Trigger (Ch8)", 'orange'),
#         ('t_mcp',  "3. Reference MCP (G3 Ch7)", 'green'),
#         ('t_ref',  "4. Reference Trigger (G3 Ch8)", 'red')
#     ]

#     for key, title, color in components:
#         fig, ax = plt.subplots(figsize=(11, 8)) # Landscape-ish for better detail
        
#         # Plot Main Histogram
#         fit_and_plot_peak(ax, data[key], title, color)
        
#         fig.suptitle(f"Run: {rl} | {title}", fontsize=16, weight='bold')
#         fig.tight_layout()
#         pdf.savefig(fig)
#         plt.close(fig)

# # ================= MAIN =================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("files", nargs="+", help="Input ROOT files")
#     parser.add_argument("--channel", "-c", required=True, help="Channel (e.g., 104)")
#     parser.add_argument("--tree", default="EventTree")
#     args = parser.parse_args()

#     # Identify columns
#     b, g, c = parse_code(args.channel)
    
#     branches = {
#         't_ch':   get_branch(b, g, c),
#         't_trig': get_branch(b, g, 8),
#         't_mcp':  get_branch(b, 3, 7),
#         't_ref':  get_branch(b, 3, 8)
#     }

#     print(f"--- Inspecting Channel {args.channel} Peaks ---")
    
#     out_name = f"DEBUG_PEAKS_SEPARATE_Channel{args.channel}.pdf"
    
#     with PdfPages(out_name) as pdf:
#         for fpath in args.files:
#             print(f"Processing {os.path.basename(fpath)}...")
#             try:
#                 with uproot.open(fpath) as f:
#                     if args.tree not in f: continue
#                     data = get_time_arrays(f[args.tree], branches)
#                     if data:
#                         plot_separate_pages(pdf, os.path.basename(fpath), data, args.channel)
#             except Exception as e:
#                 print(f"Error: {e}")

#     print(f"Saved: {out_name}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import argparse
import os
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

# ================= HELPER FUNCTIONS =================
def parse_code(code_str):
    b, g, c = int(code_str[0]), int(code_str[1]), int(code_str[2])
    return b, g, c

def get_branch(b, g, c):
    return f"DRS_Board{b}_Group{g}_Channel{c}_LP2_50"

def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-(x - mean)**2 / (2 * sigma**2))

def fit_and_plot(ax, data, color, label, title, xlim):
    """
    Plots histogram within 'xlim' and fits Gaussian to the highest peak found there.
    """
    # 1. Histogram within the requested range
    bins = np.linspace(xlim[0], xlim[1], 200) 
    centers = 0.5 * (bins[1:] + bins[:-1])
    
    counts, edges, _ = ax.hist(data, bins=bins, color=color, alpha=0.5, label=label)
    ax.hist(data, bins=bins, color=color, histtype='step', lw=1.5)
    
    # 2. Smart Peak Finding
    if np.sum(counts) > 0:
        peak_idx = np.argmax(counts)
        peak_val = centers[peak_idx]
        max_counts = counts[peak_idx]
        
        # 3. Fit Window: +/- 2.0 ns around the peak
        mask_fit = (centers > peak_val - 2.0) & (centers < peak_val + 2.0)
        
        if np.sum(mask_fit) > 4:
            try:
                p0 = [max_counts, peak_val, 0.4] 
                popt, _ = curve_fit(gaussian, centers[mask_fit], counts[mask_fit], p0=p0, maxfev=5000)
                
                mu = popt[1]
                sig = abs(popt[2])
                
                # Plot Fit
                x_curve = np.linspace(peak_val - 5, peak_val + 5, 200)
                ax.plot(x_curve, gaussian(x_curve, *popt), 'k--', lw=2, 
                        label=f'Fit: $\mu$={mu:.1f}, $\sigma$={sig:.3f} ns')
            except:
                ax.text(0.5, 0.5, "Fit Failed", transform=ax.transAxes)

    ax.set_xlim(xlim)
    ax.set_title(title)
    ax.set_xlabel("Time [ns]")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

# ================= MAIN LOGIC =================
def plot_operations(pdf, rl, arrays, b, g, c):
    
    raw_ch   = arrays[get_branch(b, g, c)]
    raw_trig = arrays[get_branch(b, g, 8)]
    raw_mcp  = arrays[get_branch(b, 3, 7)]
    raw_ref  = arrays[get_branch(b, 3, 8)]

    # Clean
    mask = (raw_ch > 20) & (raw_trig > 20) & (raw_mcp > 20) & (raw_ref > 20)
    if np.sum(mask) < 50: return

    # Compute
    dt_local = raw_ch[mask] - raw_trig[mask]
    dt_ref   = raw_mcp[mask] - raw_ref[mask]
    t_final  = dt_local - dt_ref
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 14))
    fig.suptitle(f"Run {rl} | Channel {b}{g}{c}\nValid Events: {np.sum(mask)}", fontsize=16)

    # Panel 1: Local (Wide View)
    fit_and_plot(axes[0], dt_local, 'purple', 
                 r"$\Delta t_{local}$", 
                 "1. Local Term (-100 to 100)", 
                 xlim=(-100, 100))

    # Panel 2: Ref (Wide View)
    fit_and_plot(axes[1], dt_ref, 'brown', 
                 r"$\Delta t_{ref}$", 
                 "2. Reference Term (-100 to 100)", 
                 xlim=(-100, 100))

    # Panel 3: Final (Focused View)
    fit_and_plot(axes[2], t_final, 'blue', 
                 r"$t_{final}$", 
                 "3. Final Result (-25 to 30)", 
                 xlim=(-14, -12))  # <--- REQUESTED RANGE

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Input ROOT files")
    parser.add_argument("--channel", "-c", required=True, help="Channel (e.g. 104)")
    parser.add_argument("--tree", default="EventTree")
    args = parser.parse_args()

    b, g, c = parse_code(args.channel)
    branches = [
        get_branch(b, g, c),
        get_branch(b, g, 8),
        get_branch(b, 3, 7),
        get_branch(b, 3, 8)
    ]

    out_name = f"OPERATIONS_FOCUSED_Channel_evenmore{args.channel}.pdf"
    with PdfPages(out_name) as pdf:
        for fpath in args.files:
            try:
                with uproot.open(fpath) as f:
                    if args.tree not in f: continue
                    arrays = f[args.tree].arrays(branches, library="np")
                    plot_operations(pdf, os.path.basename(fpath), arrays, b, g, c)
            except Exception as e:
                print(f"Error {fpath}: {e}")

    print(f"Saved: {out_name}")

if __name__ == "__main__":
    main()