#!/bin/bash
set -euo pipefail
# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/load_env.sh"

# 1. Run training scripts sequentially.
# Each train_*.sh script owns its environment and log configuration.

# 100%
bash "${SCRIPT_DIR}/training/train_100.sh"

# 75%
bash "${SCRIPT_DIR}/training/train_75.sh"

# 50%
bash "${SCRIPT_DIR}/training/train_50.sh"

# 25%
bash "${SCRIPT_DIR}/training/train_25.sh"

# 10%
bash "${SCRIPT_DIR}/training/train_10.sh"
