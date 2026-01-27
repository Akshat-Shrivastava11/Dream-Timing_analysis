#!/bin/bash
#SBATCH -J timingSkim
#SBATCH -p quanah
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 08:00:00
#SBATCH --array=0-355%40
#SBATCH -o /lustre/research/hep/akshriva/Dream-Timing/jobs_submission/logs/%x_%A_%a.out
#SBATCH -e /lustre/research/hep/akshriva/Dream-Timing/jobs_submission/logs/%x_%A_%a.err

set -euo pipefail

# =====================
# Paths
# =====================
INDIR="/lustre/research/hep/yofeng/HG-DREAM/CERN/ROOT"
OUTDIR="/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitskims"
JOBDIR="/lustre/research/hep/akshriva/Dream-Timing/jobs_submission"
LOGDIR="${JOBDIR}/logs"

SCRIPT="/lustre/research/hep/akshriva/Dream-Timing/Fituplizertiming.py"
CONFIG="/lustre/research/hep/cmadrid/TimingDAQ/DRS_Service_TTU_2025_Sep.config"
TREE="EventTree"

CHUNK=500
NPROC=4

mkdir -p "${LOGDIR}" "${OUTDIR}"

cd /lustre/research/hep/akshriva/Dream-Timing

# =====================
# Activate environment
# =====================
source /lustre/work/akshriva/aienv/bin/activate

# =====================
# File list (stable)
# =====================
mapfile -t FILES < <(ls -1 ${INDIR}/run*_converted.root | sort)

NFILES=${#FILES[@]}
TASK=${SLURM_ARRAY_TASK_ID}

if [[ ${TASK} -ge ${NFILES} ]]; then
  echo "Invalid task id ${TASK}, NFILES=${NFILES}"
  exit 2
fi

INFILE="${FILES[$TASK]}"
BASENAME="$(basename "${INFILE}")"
RUNID="${BASENAME%%_converted.root}"
OUTFILE="${RUNID}_converted_timingskim.root"

echo "=============================="
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Array Task ID : ${TASK}"
echo "Input file    : ${INFILE}"
echo "Output file   : ${OUTDIR}/${OUTFILE}"
echo "CPUs          : ${SLURM_CPUS_PER_TASK}"
echo "=============================="

python3 "${SCRIPT}" \
  -i "${INFILE}" \
  -o "${OUTFILE}" \
  --outdir "${OUTDIR}" \
  --config "${CONFIG}" \
  --tree "${TREE}" \
  --nproc "${NPROC}" \
  --chunk "${CHUNK}" \
  --verbose
