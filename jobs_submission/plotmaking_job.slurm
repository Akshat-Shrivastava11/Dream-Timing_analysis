#!/bin/bash
#SBATCH -J timingHist3mm
#SBATCH -p nocona
#SBATCH -A default
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 06:00:00
#SBATCH --array=0-999%40
#SBATCH -o /lustre/research/hep/akshriva/Dream-Timing/jobs_submission/logs/%x_%A_%a.out
#SBATCH -e /lustre/research/hep/akshriva/Dream-Timing/jobs_submission/logs/%x_%A_%a.err

# ---- critical: prevent "unbound variable" crashes from conda deactivate hooks
set +u
set -eo pipefail

WORKDIR="/lustre/research/hep/akshriva/Dream-Timing"
IN_DIR="/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples"
OUT_BASE="/lustre/research/hep/akshriva/Dream-Timing/TimingHistPlots3mm"
SCRIPT="timinghistplots3mm.py"

mkdir -p "${OUT_BASE}"
mkdir -p "/lustre/research/hep/akshriva/Dream-Timing/jobs_submission/logs"

cd "${WORKDIR}"

echo "=== SLURM DIAG ==="
echo "JobID: ${SLURM_JOB_ID}  TaskID: ${SLURM_ARRAY_TASK_ID}"
echo "Node : $(hostname)"
echo "CWD  : $(pwd)"
echo "CPUs : ${SLURM_CPUS_PER_TASK}"
echo "=================="

# --- conda activation (prefix env) ---
set +u
export CONDA_BACKUP_ROOTSYS=""
source /lustre/work/akshriva/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/work/akshriva/aienv

echo "Python: $(which python)"
python -V
python -c "import uproot, awkward, numpy, matplotlib; print('Imports OK')"

# ---- build file list (adapt pattern if needed) ----
# If your files are .root directly in IN_DIR (common):
mapfile -t FILES < <(find "${IN_DIR}" -maxdepth 1 -type f -name "*.root" | sort)

# If they are nested or have a stricter naming pattern, swap the find line with e.g.:
# mapfile -t FILES < <(find "${IN_DIR}" -type f -name "run*_postaskim*.root" | sort)

NFILES="${#FILES[@]}"
echo "Found ${NFILES} input ROOT files in ${IN_DIR}"

# Optional: local dry-run (NOT a slurm mode) -> bash script --dry-run
if [ "${1:-}" = "--dry-run" ]; then
  echo "[DRY-RUN] First 10 files:"
  printf '  %s\n' "${FILES[@]:0:10}"
  echo "[DRY-RUN] Suggested array: 0-$((NFILES-1))"
  exit 0
fi

IDX="${SLURM_ARRAY_TASK_ID}"
if [ "${IDX}" -ge "${NFILES}" ]; then
  echo "[FATAL] Task index ${IDX} >= NFILES ${NFILES}"
  exit 2
fi

INFILE="${FILES[$IDX]}"
BASE="$(basename "${INFILE}")"
STEM="${BASE%.root}"

# Per-input output folder (keeps outputs separated)
OUTDIR="${OUT_BASE}/${STEM}"
mkdir -p "${OUTDIR}"

# Sentinel file so reruns skip finished inputs
DONE_SENTINEL="${OUTDIR}/.DONE"
if [ -f "${DONE_SENTINEL}" ]; then
  echo "[SKIP] Already completed: ${INFILE}"
  exit 0
fi

# ---- tmpdir: prefer node-local scratch if available ----
if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "${SLURM_TMPDIR:-}" ]; then
  TMPDIR="${SLURM_TMPDIR}/.tmp_hist_${SLURM_JOB_ID}_${IDX}"
else
  TMPDIR="${OUTDIR}/.tmp_hist_${SLURM_JOB_ID}_${IDX}"
fi
mkdir -p "${TMPDIR}"
echo "Tmpdir: ${TMPDIR}"

echo "Input : ${INFILE}"
echo "Outdir: ${OUTDIR}"

# ---- run ----
# If timinghistplots3mm.py supports --outdir, add it here.
# Otherwise it will write wherever the script defaults (often CWD),
# so consider updating the python script to accept --outdir.
python3 "${SCRIPT}" \
  --ana-file "${INFILE}" \
  > "${OUTDIR}/stdout.log" 2> "${OUTDIR}/stderr.log"

# Mark done if successful
touch "${DONE_SENTINEL}"

echo "[DONE] ${INFILE}"
