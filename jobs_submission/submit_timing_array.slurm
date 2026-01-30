#!/bin/bash
#SBATCH -J timingSkim
#SBATCH -p nocona
#SBATCH -A default
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 06:00:00
#SBATCH --array=0-355%40
#SBATCH -o /lustre/research/hep/akshriva/Dream-Timing/jobs_submission/logs/%x_%A_%a.out
#SBATCH -e /lustre/research/hep/akshriva/Dream-Timing/jobs_submission/logs/%x_%A_%a.err

# ---- critical: prevent "unbound variable" crashes from conda deactivate hooks
set +u
set -eo pipefail

WORKDIR="/lustre/research/hep/akshriva/Dream-Timing"
IN_DIR="/lustre/research/hep/yofeng/HG-DREAM/CERN/ROOT"
OUT_DIR="/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples"
CFG="/lustre/research/hep/cmadrid/TimingDAQ/DRS_Service_TTU_2025_Sep.config"
TREE="EventTree"

mkdir -p "${OUT_DIR}"
mkdir -p "/lustre/research/hep/akshriva/Dream-Timing/jobs_submission/logs"

cd "${WORKDIR}"

echo "=== SLURM DIAG ==="
echo "JobID: ${SLURM_JOB_ID}  TaskID: ${SLURM_ARRAY_TASK_ID}"
echo "Node : $(hostname)"
echo "CWD  : $(pwd)"
echo "CPUs : ${SLURM_CPUS_PER_TASK}"
echo "=================="

# --- conda activation (prefix env) ---
# Defensive: some site envs/module wrappers set nounset later; keep it off.
set +u

# If any conda deactivate hook expects this variable, define it to avoid crash.
export CONDA_BACKUP_ROOTSYS=""

# Source conda function definition
source /lustre/work/akshriva/miniconda3/etc/profile.d/conda.sh

# Activate your prefix environment
conda activate /lustre/work/akshriva/aienv

echo "Python: $(which python)"
python -V
python -c "import uproot, awkward, numpy; print('Imports OK')"

# ---- select file by array index ----
mapfile -t FILES < <(ls -1 ${IN_DIR}/run*_converted.root | sort)
NFILES="${#FILES[@]}"
echo "Found ${NFILES} input files"

IDX="${SLURM_ARRAY_TASK_ID}"
if [ "${IDX}" -ge "${NFILES}" ]; then
  echo "[FATAL] Task index ${IDX} >= NFILES ${NFILES}"
  exit 2
fi

INFILE="${FILES[$IDX]}"
BASE="$(basename "${INFILE}")"
RUNID="${BASE%%_converted.root}"
OUTFILE="${RUNID}_converted_timingskim.root"
OUTPATH="${OUT_DIR}/${OUTFILE}"

echo "Input : ${INFILE}"
echo "Output: ${OUTPATH}"

# Skip if already done
if [ -s "${OUTPATH}" ]; then
  echo "[SKIP] Output exists: ${OUTPATH}"
  exit 0
fi

# ---- tmpdir: prefer node-local scratch if available ----
if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "${SLURM_TMPDIR:-}" ]; then
  TMPDIR="${SLURM_TMPDIR}/.tmp_ntuple_${SLURM_JOB_ID}_${IDX}"
else
  TMPDIR="${OUT_DIR}/.tmp_ntuple_${SLURM_JOB_ID}_${IDX}"
fi
mkdir -p "${TMPDIR}"
echo "Tmpdir: ${TMPDIR}"

# ---- run ----
python3 Fituplizertiming.py \
  -i "${INFILE}" \
  -o "${OUTFILE}" \
  --outdir "${OUT_DIR}" \
  --config "${CFG}" \
  --tree "${TREE}" \
  --nproc "${SLURM_CPUS_PER_TASK}" \
  --chunk 500 \
  --tmpdir "${TMPDIR}" \
  --verbose \
  --keep_all_originals

echo "[DONE] ${OUTPATH}"
