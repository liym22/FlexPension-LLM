#!/bin/bash
set -euo pipefail
# Correct-only ablation experiment evaluation script

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/load_env.sh"

# 1. Activate the evaluation environment.
source "${SERVER_CONDA_PATH}/bin/activate" "${SERVER_CONDA_ENV}"

# 2. Configurable parameters
RESULT_PATH=${RESULT_PATH:-"${SERVER_OUTPUT_DIR}/infer_results/correct_only_results.jsonl"}
GROUND_TRUTH=${GROUND_TRUTH:-"${SERVER_DATASET_DIR}/correct_only/test.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"${SERVER_OUTPUT_DIR}/evaluation_results"}

# 3. Verify that inference results exist.
if [ ! -f "${RESULT_PATH}" ]; then
    echo "[Error] Inference results file not found: ${RESULT_PATH}"
    exit 1
fi

echo "[Info] Evaluating correct_only experiment results..."
echo "Inference results: ${RESULT_PATH}"
echo "Test dataset: ${GROUND_TRUTH}"

# 4. Run evaluation.
python "${SCRIPT_DIR}/eval_batch_infer.py" \
    --result-path "${RESULT_PATH}" \
    --ground-truth "${GROUND_TRUTH}" \
    --output-dir "${OUTPUT_DIR}"

echo "[Done] Evaluation complete."
