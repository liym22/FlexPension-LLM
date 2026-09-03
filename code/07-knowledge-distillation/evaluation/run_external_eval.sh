#!/bin/bash
set -euo pipefail

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/load_env.sh"
: "${SERVER_EXTERNAL_DATASET_DIR:?Set SERVER_EXTERNAL_DATASET_DIR in .env}"

# Run inference, format validation, and metric evaluation for four external surveys.
# Resume support skips subtasks that already contain the expected number of valid results.
#
# Usage:
#   bash run_external_eval.sh
#   ADAPTERS_PATH=/path/to/checkpoint TEMPERATURE=0.0 bash run_external_eval.sh

INFERENCE_DIR="${SCRIPT_DIR}/../inference"
EXTERNAL_DIR="${SERVER_EXTERNAL_DATASET_DIR}"
OUTPUT_ROOT="${SERVER_OUTPUT_DIR}/external_results"

MODEL_PATH="${MODEL_PATH:-"${SERVER_MODEL_PATH}"}"
if [[ -z "${ADAPTERS_PATH:-}" ]]; then
  ADAPTERS_ROOT="${SERVER_OUTPUT_DIR}/v2-data100"
  if [[ -d "${ADAPTERS_ROOT}" ]]; then
    ADAPTERS_PATH=$(find "${ADAPTERS_ROOT}" -type d -name 'checkpoint-*' | sort -V | tail -n 1)
  fi
fi
if [[ -z "${ADAPTERS_PATH:-}" || ! -d "${ADAPTERS_PATH}" ]]; then
  echo "ERROR: no adapter checkpoint found; set ADAPTERS_PATH explicitly" >&2
  exit 1
fi
TEMPERATURE="${TEMPERATURE:-"0.5"}"
EXPECTED_SAMPLES=500

# 1. Activate the evaluation environment.
source "${SERVER_CONDA_PATH}/bin/activate" ${SERVER_CONDA_ENV}

# 2. Runtime variables
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export HF_HOME="${SERVER_BASE}/.cache_dir/huggingface"
export MODELSCOPE_CACHE="${SERVER_BASE}/.cache_dir/modelscope"
export TMPDIR="${SERVER_BASE}/.tmp"

mkdir -p "${OUTPUT_ROOT}/infer_results" \
         "${OUTPUT_ROOT}/format_validation" \
         "${OUTPUT_ROOT}/evaluation_results"

# A subtask is complete when its result file contains the expected valid responses.
is_complete() {
  local result_path="$1"
  [[ ! -f "${result_path}" ]] && return 1
  local count
  count=$(python - "${result_path}" <<'PY'
import json, sys
count = 0
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if str(obj.get("response", "")).strip():
            count += 1
print(count)
PY
)
  [[ "${count}" -ge "${EXPECTED_SAMPLES}" ]]
}

# Run one subtask: inference, format validation, then metric evaluation.
run_subtask() {
  local dataset="$1"   # Test dataset path.
  local tag="$2"       # Subtask identifier and output filename prefix.

  local result_path="${OUTPUT_ROOT}/infer_results/${tag}.jsonl"
  local log_file="${OUTPUT_ROOT}/infer_results/${tag}.log"
  local validate_dir="${OUTPUT_ROOT}/format_validation/${tag}"
  local eval_dir="${OUTPUT_ROOT}/evaluation_results/${tag}"

  # Skip completed subtasks when resuming.
  if is_complete "${result_path}"; then
    echo "[Skip] ${tag}: complete results found (>= ${EXPECTED_SAMPLES}); skipping"
    return 0
  fi

  echo "[Run ] ${tag}: starting inference..."
  # Remove an incomplete result before rerunning the subtask.
  rm -f "${result_path}"

  NPROC_PER_NODE=8 swift infer \
    --model            "${MODEL_PATH}" \
    --adapters         "${ADAPTERS_PATH}" \
    --val_dataset      "${dataset}" \
    --infer_backend    transformers \
    --seed             42 \
    --max_batch_size   8 \
    --write_batch_size 100 \
    --add_non_thinking_prefix true \
    --result_path      "${result_path}" \
    --enable_thinking  false \
    --max_new_tokens   2048 \
    --temperature      "${TEMPERATURE}" \
    --torch_dtype      bfloat16 \
    2>&1 | tee -a "${log_file}" \
  || { echo "[Error] ${tag}: inference failed; skipping remaining steps"; return 0; }

  echo "[Val ] ${tag}: validating format..."
  python "${INFERENCE_DIR}/validate_infer_format.py" \
    --input      "${result_path}" \
    --output-dir "${validate_dir}" \
    || echo "[Warn] ${tag}: format validation failed"

  echo "[Eval] ${tag}: evaluating metrics..."
  python "${SCRIPT_DIR}/eval_batch_infer.py" \
    --result-path  "${result_path}" \
    --ground-truth "${dataset}" \
    --output-dir   "${eval_dir}" \
    || echo "[Warn] ${tag}: evaluation failed"

  echo "[Done] ${tag}: complete"
}

# Run the four reported external datasets sequentially.
DATASETS=(
  test_2017_new
  test_cfps_new
  test_chip_new
  test_clds_new
)

echo "====== Starting external evaluation: ${#DATASETS[@]} subtasks ======"
echo "Model: ${MODEL_PATH}"
echo "Adapter: ${ADAPTERS_PATH}"
echo "Temperature: ${TEMPERATURE}"
echo "Output root: ${OUTPUT_ROOT}"
echo "========================================================"

for name in "${DATASETS[@]}"; do
  run_subtask "${EXTERNAL_DIR}/${name}.jsonl" "${name}"
done

echo "====== Completed all ${#DATASETS[@]} subtasks ======"
