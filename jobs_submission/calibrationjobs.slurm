#!/bin/bash
#SBATCH -J calibFlushMean
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

# =========================
#  Calibrate+flush selected ROOT files
#  Runs:
#    python3 calibrate_flush_runs.py --calib-stat mean --test {INFILE} --outdir {OUTDIR}
# =========================

# Prevent "unbound variable" crashes from conda deactivate hooks
set +u
set -eo pipefail

WORKDIR="/lustre/research/hep/akshriva/Dream-Timing"
IN_DIR="/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples"
OUT_BASE="/lustre/research/hep/akshriva/Dream-Timing/CalibrateFlushRuns"
SCRIPT="calibrate_flush_runs.py"
CALIB_STAT="mean"   # mean | mode

LOGDIR="/lustre/research/hep/akshriva/Dream-Timing/jobs_submission/logs"
mkdir -p "${OUT_BASE}" "${LOGDIR}"

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

# -------------------------
# Explicit list of files to run on (EDIT THIS LIST)
# -------------------------
FILES=(
  "${IN_DIR}/run1499_250928094112_converted_timingskim.root"
  "${IN_DIR}/run1499_250928103402_converted_timingskim.root"
  "${IN_DIR}/run1500_250928104710_converted_timingskim.root"
  "${IN_DIR}/run1501_250928105227_converted_timingskim.root"
  "${IN_DIR}/run1502_250928113749_converted_timingskim.root"
  "${IN_DIR}/run1504_250928133854_converted_timingskim.root"
  "${IN_DIR}/run1505_250928141538_converted_timingskim.root"
  "${IN_DIR}/run1506_250928143030_converted_timingskim.root"
  "${IN_DIR}/run1506_250928145724_converted_timingskim.root"
  "${IN_DIR}/run1507_250928152053_converted_timingskim.root"
  "${IN_DIR}/run1507_250928160030_converted_timingskim.root"
  "${IN_DIR}/run1508_250928161049_converted_timingskim.root"
  "${IN_DIR}/run1509_250928164817_converted_timingskim.root"
  "${IN_DIR}/run1510_250928172949_converted_timingskim.root"
  "${IN_DIR}/run1511_250928180741_converted_timingskim.root"
  "${IN_DIR}/run1512_250928183645_converted_timingskim.root"
  "${IN_DIR}/run1512_250928185722_converted_timingskim.root"
  "${IN_DIR}/run1513_250928192918_converted_timingskim.root"
  "${IN_DIR}/run1513_250928194230_converted_timingskim.root"
  "${IN_DIR}/run1514_250928200533_converted_timingskim.root"
  "${IN_DIR}/run1515_250928202053_converted_timingskim.root"
)

NFILES="${#FILES[@]}"
echo "Selected NFILES=${NFILES}"

# Optional: local dry-run -> bash this_script.sh --dry-run
if [ "${1:-}" = "--dry-run" ]; then
  echo "[DRY-RUN] Files:"
  printf '  (%2d) %s\n' $(seq 0 $((NFILES-1))) "${FILES[@]}" 2>/dev/null || true
  echo "[DRY-RUN] Suggested array: 0-$((NFILES-1))"
  exit 0
fi

IDX="${SLURM_ARRAY_TASK_ID}"
if [ "${IDX}" -ge "${NFILES}" ]; then
  echo "[FATAL] Task index ${IDX} >= NFILES ${NFILES}"
  echo "Set your SBATCH --array to: 0-$((NFILES-1)) (or reduce it)."
  exit 2
fi

INFILE="${FILES[$IDX]}"
if [ ! -f "${INFILE}" ]; then
  echo "[FATAL] Missing input file: ${INFILE}"
  exit 3
fi

BASE="$(basename "${INFILE}")"
STEM="${BASE%.root}"

OUTDIR="${OUT_BASE}/${STEM}"
mkdir -p "${OUTDIR}"

DONE_SENTINEL="${OUTDIR}/.DONE"
if [ -f "${DONE_SENTINEL}" ]; then
  echo "[SKIP] Already completed: ${INFILE}"
  exit 0
fi

# tmpdir: prefer node-local scratch
if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "${SLURM_TMPDIR:-}" ]; then
  TMPDIR="${SLURM_TMPDIR}/.tmp_calib_${SLURM_JOB_ID}_${IDX}"
else
  TMPDIR="${OUTDIR}/.tmp_calib_${SLURM_JOB_ID}_${IDX}"
fi
mkdir -p "${TMPDIR}"
echo "Tmpdir: ${TMPDIR}"

echo "Input : ${INFILE}"
echo "Outdir: ${OUTDIR}"

# -------------------------
# Run
# -------------------------
python3 "${SCRIPT}" \
  --calib-stat "${CALIB_STAT}" \
  --test "${INFILE}" \
  --outdir "${OUTDIR}" \
  > "${OUTDIR}/stdout.log" 2> "${OUTDIR}/stderr.log"

touch "${DONE_SENTINEL}"
echo "[DONE] ${INFILE}"
