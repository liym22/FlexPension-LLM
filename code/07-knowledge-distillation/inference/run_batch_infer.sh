#!/bin/bash
set -euo pipefail

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/load_env.sh"

# 1. Activate the inference environment.
source "${SERVER_CONDA_PATH}/bin/activate" "${SERVER_CONDA_ENV}"

# 2. Configure runtime and GPU memory behavior.
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export HF_HOME="${SERVER_BASE}/.cache_dir/huggingface"
export MODELSCOPE_CACHE="${SERVER_BASE}/.cache_dir/modelscope"
export TMPDIR="${SERVER_BASE}/.tmp"

# 3. Configurable parameters
MODEL_PATH=${MODEL_PATH:-"${SERVER_MODEL_PATH}"}
if [[ -z "${ADAPTERS_PATH:-}" ]]; then
    ADAPTERS_ROOT="${SERVER_OUTPUT_DIR}/v2-data100"
    if [[ -d "${ADAPTERS_ROOT}" ]]; then
        ADAPTERS_PATH=$(find "${ADAPTERS_ROOT}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)
    fi
fi
if [[ -z "${ADAPTERS_PATH:-}" || ! -d "${ADAPTERS_PATH}" ]]; then
    echo "ERROR: no adapter checkpoint found; set ADAPTERS_PATH explicitly" >&2
    exit 1
fi
TEST_DATASET=${TEST_DATASET:-"${SERVER_DATASET_DIR}/test.jsonl"}
RESULT_PATH=${RESULT_PATH:-"${SERVER_OUTPUT_DIR}/batch_infer_results.jsonl"}
LOG_FILE=${LOG_FILE:-"${SERVER_OUTPUT_DIR}/batch_infer.log"}
TEMPERATURE=${TEMPERATURE:-"0.5"}

mkdir -p "$(dirname "${RESULT_PATH}")" "$(dirname "${LOG_FILE}")"

# Remove a prior result so reruns do not append duplicate records.
rm -f "${RESULT_PATH}"

echo "Starting batched parallel inference with the 35B model..." | tee -a "${LOG_FILE}"

# 4. Run batch inference with a 2,048-token limit and configurable temperature.
# Use --val_dataset to select the batch-inference path explicitly.
NPROC_PER_NODE=8 swift infer \
    --model "${MODEL_PATH}" \
    --adapters "${ADAPTERS_PATH}" \
    --val_dataset "${TEST_DATASET}" \
    --infer_backend transformers \
    --seed 42 \
    --max_batch_size 8 \
    --write_batch_size 100 \
    --add_non_thinking_prefix true \
    --result_path "${RESULT_PATH}" \
    --enable_thinking false \
    --max_new_tokens 2048 \
    --temperature "${TEMPERATURE}" \
    --torch_dtype bfloat16 2>&1 | tee -a "${LOG_FILE}"

echo "Batch inference complete. See the result path printed by Swift." | tee -a "${LOG_FILE}"
