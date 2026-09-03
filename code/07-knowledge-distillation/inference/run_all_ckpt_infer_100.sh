#!/bin/bash
set -euo pipefail

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/load_env.sh"

# Run inference for every checkpoint from the full-data training run.
# Example:
#   bash run_all_ckpt_infer_100.sh --output-root ${SERVER_OUTPUT_DIR}

OUTPUT_ROOT="${SERVER_OUTPUT_DIR}"
MODEL_PATH="${SERVER_MODEL_PATH}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_PATH="$2"; shift 2 ;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

ADAPTERS_ROOT="${OUTPUT_ROOT}/v2-data100"
if [[ ! -d "${ADAPTERS_ROOT}" ]]; then
  echo "Missing adapters directory: ${ADAPTERS_ROOT}"
  exit 1
fi

CHECKPOINTS=$(ls -1d "${ADAPTERS_ROOT}"/checkpoint-* 2>/dev/null | sort -V)
if [[ -z "${CHECKPOINTS}" ]]; then
  echo "No checkpoints found under: ${ADAPTERS_ROOT}"
  exit 1
fi

for CKPT in ${CHECKPOINTS}; do
  CKPT_NAME=$(basename "${CKPT}")
  TAG="train100_${CKPT_NAME}"

  bash "${SCRIPT_DIR}/run_infer_pipeline.sh" \
    --model "${MODEL_PATH}" \
    --adapters "${CKPT}" \
    --output-root "${OUTPUT_ROOT}" \
    --tag "${TAG}"
done
