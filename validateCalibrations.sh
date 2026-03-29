#!/bin/bash
#SBATCH --job-name=val_calib
#SBATCH --output=/lustre/research/hep/akshriva/Dream-Timing/logs/calib_%j.out
#SBATCH --error=/lustre/research/hep/akshriva/Dream-Timing/logs/calib_%j.err
#SBATCH --time=04:00:00            # Max wall time (HH:MM:SS) - adjust as needed
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=4          # CPU cores per task 
#SBATCH --mem=8G                   # Total memory per node
#SBATCH --partition=nocona         # TTU primary compute partition

# Source your bashrc so the node recognizes custom aliases/functions
source ~/.bashrc

# Activate the environment
aienv

# Navigate into your working directory
cd /lustre/research/hep/akshriva/Dream-Timing/
aienv
# Execute the script
python3 validateCalibrations.py