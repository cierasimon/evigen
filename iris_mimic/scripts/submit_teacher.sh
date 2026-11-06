#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs data/mimic
PY=${PY:-python}
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
"$PY" -c 'import iris,torch,lancedb,transformers; print("Preflight OK")'
BASELINE_JOB=${BASELINE_JOB:?Set BASELINE_JOB to a successful baseline job ID}
QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3-4B}
# This scoring job waits until the baseline has successfully built the LanceDB table.
SCORE_ID=$(sbatch --parsable --dependency=afterok:${BASELINE_JOB} --job-name=mimic-qwen-score --partition=gpu-common --gres=gpu:1 --cpus-per-task=4 --mem=40G --time=08:00:00 --output=logs/qwen-%j.out --error=logs/qwen-%j.err --wrap="cd $PWD && export PYTHONPATH=$PWD/src && export HF_HOME=/work/cs884/hf-cache && $PY scripts/score_teacher.py --model $QWEN_MODEL")
echo "Submitted Qwen scoring job: $SCORE_ID"
for L in 0.1 0.5 1.0; do
  ID=$(sbatch --parsable --dependency=afterok:${SCORE_ID} --job-name="mimic-l${L}" --partition=gpu-common --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=03:00:00 --output="logs/teacher-${L}-%j.out" --error="logs/teacher-${L}-%j.err" --wrap="cd $PWD && export PYTHONPATH=$PWD/src && $PY scripts/train_teacher.py --lambda-teacher $L --output data/mimic/teacher-l${L}.pt")
  echo "Submitted lambda=$L job: $ID"
done
