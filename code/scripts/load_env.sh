#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Error: .env not found at ${ENV_FILE}" >&2
    echo "Copy .env.example to .env and configure the required paths." >&2
    return 1 2>/dev/null || exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

required_vars=(
    SERVER_BASE
    SERVER_CONDA_PATH
    SERVER_CONDA_ENV
    SERVER_MODEL_PATH
    SERVER_DATASET_DIR
    SERVER_OUTPUT_DIR
)

missing_vars=()
for variable_name in "${required_vars[@]}"; do
    if [[ -z "${!variable_name:-}" ]]; then
        missing_vars+=("${variable_name}")
    fi
done

if (( ${#missing_vars[@]} > 0 )); then
    echo "Error: missing required environment variables:" >&2
    printf '  - %s\n' "${missing_vars[@]}" >&2
    return 1 2>/dev/null || exit 1
fi
