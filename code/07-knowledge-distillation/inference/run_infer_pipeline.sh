#!/bin/bash
set -euo pipefail

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/load_env.sh"

# Unified inference, format validation, and evaluation pipeline.
# Example:
#   bash run_infer_pipeline.sh \
#     --adapters /path/to/checkpoint-1998 \
#     --tag train100_ckpt1998 \
#     --output-root ${SERVER_OUTPUT_DIR}

EVALUATION_DIR="${SCRIPT_DIR}/../evaluation"
TEST_DATASET="${SERVER_DATASET_DIR}/test.jsonl"

MODEL_PATH="${SERVER_MODEL_PATH}"
ADAPTERS_PATH=""
OUTPUT_ROOT="${SERVER_OUTPUT_DIR}"
TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_PATH="$2"; shift 2 ;;
    --adapters)
      ADAPTERS_PATH="$2"; shift 2 ;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --tag)
      TAG="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "${ADAPTERS_PATH}" || -z "${TAG}" ]]; then
  echo "Usage: --adapters <path> --tag <name> [--model <path>] [--output-root <dir>]"
  exit 1
fi

INFER_DIR="${OUTPUT_ROOT}/infer_results"
VALIDATE_DIR="${OUTPUT_ROOT}/format_validation"
EVAL_DIR="${OUTPUT_ROOT}/evaluation_results"

RESULT_PATH="${INFER_DIR}/${TAG}.jsonl"
LOG_FILE="${INFER_DIR}/${TAG}.log"

mkdir -p "${INFER_DIR}" "${VALIDATE_DIR}" "${EVAL_DIR}"

# 1) Inference
MODEL_PATH="${MODEL_PATH}" \
ADAPTERS_PATH="${ADAPTERS_PATH}" \
TEST_DATASET="${TEST_DATASET}" \
RESULT_PATH="${RESULT_PATH}" \
LOG_FILE="${LOG_FILE}" \
  bash "${SCRIPT_DIR}/run_batch_infer.sh"

# 2) Format validation
python "${SCRIPT_DIR}/validate_infer_format.py" \
  --input "${RESULT_PATH}" \
  --output-dir "${VALIDATE_DIR}/${TAG}"

# 3) Evaluation
python "${EVALUATION_DIR}/eval_batch_infer.py" \
  --result-path "${RESULT_PATH}" \
  --ground-truth "${TEST_DATASET}" \
  --output-dir "${EVAL_DIR}/${TAG}"
