import uproot
import os 
import numpy as np
import matplotlib.pyplot as plt
import os
file_path = "/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1468_250927145556_TimingDAQ.root" #chris file path post timing pion run 
#branch = "DRS_Board0_Group0_Channel0_LP2_50"
#branch = "DRS_Board0_Group0_Channel8_LP2_50"
branch = 'DRS_Board0_Group0_Channel1_LP2_50'
# Output directory for PDF
outdir = "tpeak_histograms"
os.makedirs(outdir, exist_ok=True)

def plot_branch_histogram(file_path, branch, outdir="histograms", bins=80):
    """
    Loads a branch from a ROOT file, cleans values, plots histogram, and saves a PDF.
    """

    # Make output directory
    os.makedirs(outdir, exist_ok=True)

    # Open tree
    tree = uproot.open(file_path)["EventTree"]

    # Load branch into numpy
    vals = tree[branch].array(library="np")
    print(f"Loaded {len(vals)} entries from branch: {branch}")

    # Clean values
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]

    # Plot histogram
    plt.figure(figsize=(8,5))
    plt.hist(vals, bins=bins, histtype="step", alpha=0.6)
    plt.xlabel(branch)
    plt.ylabel("Entries")
    plt.title(f"Histogram of {branch}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save PDF
    pdf_path = os.path.join(outdir, f"{branch}.pdf")
    plt.savefig(pdf_path)
    plt.close()

    print("Saved PDF:", pdf_path)


ALL_CHANNELS = [
    "DRS_Board0_Group0_Channel0", "DRS_Board0_Group0_Channel1", "DRS_Board0_Group0_Channel2",
    "DRS_Board0_Group0_Channel3", "DRS_Board0_Group0_Channel4", "DRS_Board0_Group0_Channel5",
    "DRS_Board0_Group0_Channel6", "DRS_Board0_Group0_Channel7", "DRS_Board0_Group0_Channel8",
   
]
output_dir = "Chris_t50"                   # directory for PDFs

for channel in ALL_CHANNELS:
    branch_name = f"{channel}_LP2_50"
    print(f"Processing branch: {branch_name}")
    plot_branch_histogram(file_path, branch_name, outdir=output_dir, bins=80)