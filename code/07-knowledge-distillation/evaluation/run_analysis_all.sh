#!/bin/bash
set -euo pipefail

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/load_env.sh"

# Run all three robustness analyses:
# 1) Data scaling: latest checkpoint from each training subset.
# 2) Training dynamics: every checkpoint from the full-data run.
# 3) Inference sensitivity: multiple temperatures with the final full-data checkpoint.

OUTPUT_ROOT="${SERVER_OUTPUT_DIR}"
INFERENCE_DIR="${SCRIPT_DIR}/../inference"
MODEL_PATH="${SERVER_MODEL_PATH}"
TEST_DATASET="${SERVER_DATASET_DIR}/test.jsonl"
TOTAL_TEST_SAMPLES=$(python - "${TEST_DATASET}" <<'PY' || echo "0"
import sys

path = sys.argv[1]
count = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            count += 1
print(count)
PY
)

INFER_DIR="${OUTPUT_ROOT}/infer_results"
VALIDATE_DIR="${OUTPUT_ROOT}/format_validation"
EVAL_DIR="${OUTPUT_ROOT}/evaluation_results"

mkdir -p "${INFER_DIR}" "${VALIDATE_DIR}" "${EVAL_DIR}"

is_infer_complete() {
  local result_path="$1"
  if [[ ! -f "${result_path}" ]]; then
    return 1
  fi

  local count
  count=$(python - "${result_path}" <<'PY' || echo "0"
import json
import sys

path = sys.argv[1]
count = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        resp = obj.get("response")
        if resp is not None and str(resp).strip() != "":
            count += 1
print(count)
PY
)

  if [[ "${count}" -ge "${TOTAL_TEST_SAMPLES}" ]]; then
    return 0
  fi
  return 1
}

run_pipeline() {
  local adapters_path="$1"
  local tag="$2"
  local temperature="$3"

  local result_path="${INFER_DIR}/${tag}.jsonl"
  local log_file="${INFER_DIR}/${tag}.log"

  if is_infer_complete "${result_path}"; then
    echo "[Skip] Completed results found for ${tag} (${result_path})"
    return 0
  fi

  MODEL_PATH="${MODEL_PATH}" \
  ADAPTERS_PATH="${adapters_path}" \
  TEST_DATASET="${TEST_DATASET}" \
  RESULT_PATH="${result_path}" \
  LOG_FILE="${log_file}" \
  TEMPERATURE="${temperature}" \
    bash "${INFERENCE_DIR}/run_batch_infer.sh" || {
      echo "[Error] Inference failed for ${tag}! Skipping evaluation."
      return 0
    }

  python "${INFERENCE_DIR}/validate_infer_format.py" \
    --input "${result_path}" \
    --output-dir "${VALIDATE_DIR}/${tag}" \
    || echo "[Warning] Validation failed for ${tag}"

  python "${SCRIPT_DIR}/eval_batch_infer.py" \
    --result-path "${result_path}" \
    --ground-truth "${TEST_DATASET}" \
    --output-dir "${EVAL_DIR}/${tag}" \
    || echo "[Warning] Eval failed for ${tag}"
}

latest_checkpoint() {
  local adapters_root="$1"
  local latest
  latest=$(ls -1d "${adapters_root}"/v*/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
  if [[ -z "${latest}" ]]; then
    echo ""
  else
    echo "${latest}"
  fi
}

# 1) Data Scaling Analysis
for train_tag in data10 data25 data50 data75 data100; do
  adapters_root="${OUTPUT_ROOT}/v2-${train_tag}"
  if [[ ! -d "${adapters_root}" ]]; then
    echo "[Skip] Missing adapters dir: ${adapters_root}"
    continue
  fi

  latest_ckpt=$(latest_checkpoint "${adapters_root}")
  if [[ -z "${latest_ckpt}" ]]; then
    echo "[Skip] No checkpoint in ${adapters_root}"
    continue
  fi

  ckpt_name=$(basename "${latest_ckpt}")
  tag="data_scaling_${train_tag}_${ckpt_name}"
  run_pipeline "${latest_ckpt}" "${tag}" "0.5"
done

# 2) Training Dynamics & Convergence Analysis
adapters_root_100="${OUTPUT_ROOT}/v2-data100"
if [[ -d "${adapters_root_100}" ]]; then
  checkpoints=$(ls -1d "${adapters_root_100}"/v*/checkpoint-* 2>/dev/null | sort -V)
  if [[ -n "${checkpoints}" ]]; then
    for ckpt in ${checkpoints}; do
      ckpt_name=$(basename "${ckpt}")
      tag="training_dynamics_data100_${ckpt_name}"
      run_pipeline "${ckpt}" "${tag}" "0.5"
    done
  else
    echo "[Skip] No checkpoints in ${adapters_root_100}"
  fi
else
  echo "[Skip] Missing adapters dir: ${adapters_root_100}"
fi

# 3) Inference Robustness & Sensitivity Analysis
if [[ -d "${adapters_root_100}" ]]; then
  latest_ckpt_100=$(latest_checkpoint "${adapters_root_100}")
  if [[ -n "${latest_ckpt_100}" ]]; then
    ckpt_name=$(basename "${latest_ckpt_100}")
    for temperature in 0.0 0.2 0.5 0.9; do
      tag="robustness_data100_${ckpt_name}_t${temperature}"
      run_pipeline "${latest_ckpt_100}" "${tag}" "${temperature}"
    done
  else
    echo "[Skip] No checkpoint in ${adapters_root_100}"
  fi
else
  echo "[Skip] Missing adapters dir: ${adapters_root_100}"
fi
