#!/bin/bash
set -euo pipefail

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/load_env.sh"

# Run inference, validation, and evaluation with the latest checkpoint.
# Example:
#   bash run_latest_ckpt_infer.sh --train-tag data100 --output-root ${SERVER_OUTPUT_DIR}

OUTPUT_ROOT="${SERVER_OUTPUT_DIR}"

TRAIN_TAG=""
MODEL_PATH="${SERVER_MODEL_PATH}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-tag)
      TRAIN_TAG="$2"; shift 2 ;;
    --model)
      MODEL_PATH="$2"; shift 2 ;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "${TRAIN_TAG}" ]]; then
  echo "Usage: --train-tag <data25|data50|data75|data100|data10> [--model <path>] [--output-root <dir>]"
  exit 1
fi

ADAPTERS_ROOT="${OUTPUT_ROOT}/v2-${TRAIN_TAG}"
if [[ ! -d "${ADAPTERS_ROOT}" ]]; then
  echo "Missing adapters directory: ${ADAPTERS_ROOT}"
  exit 1
fi

LATEST_CKPT=$(ls -1d "${ADAPTERS_ROOT}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
if [[ -z "${LATEST_CKPT}" ]]; then
  echo "No checkpoint found under: ${ADAPTERS_ROOT}"
  exit 1
fi

CKPT_NAME=$(basename "${LATEST_CKPT}")
TAG="train${TRAIN_TAG}_${CKPT_NAME}"

bash "${SCRIPT_DIR}/run_infer_pipeline.sh" \
  --model "${MODEL_PATH}" \
  --adapters "${LATEST_CKPT}" \
  --output-root "${OUTPUT_ROOT}" \
  --tag "${TAG}"
