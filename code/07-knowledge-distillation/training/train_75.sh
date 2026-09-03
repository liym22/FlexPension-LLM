#!/bin/bash
set -euo pipefail
# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/load_env.sh"

# 1. Activate the training environment.
source "${SERVER_CONDA_PATH}/bin/activate" "${SERVER_CONDA_ENV}"

# 2. Configure the GPU environment and memory allocator.
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# 3. Store logs in persistent storage.
LOG_DIR=${LOG_DIR:-"${SERVER_OUTPUT_DIR}"}
mkdir -p "${LOG_DIR}"
LOG_FILE=${LOG_FILE:-"${LOG_DIR}/full_train_75.log"}

# 4. Start ms-swift training with the Transformers backend.
DATASET_PATH=${DATASET_PATH:-"${SERVER_DATASET_DIR}/train_75.jsonl"}
VAL_DATASET_PATH=${VAL_DATASET_PATH:-"${SERVER_DATASET_DIR}/val.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"${SERVER_OUTPUT_DIR}/v2-data75"}

NPROC_PER_NODE=8 swift sft \
    --model ${SERVER_MODEL_PATH} \
    --dataset "${DATASET_PATH}" \
    --val_dataset "${VAL_DATASET_PATH}" \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --lr_scheduler_type cosine \
    --warmup_steps 0 \
    --weight_decay 0.1 \
    --optim adamw_torch_fused \
    --bf16 true \
    --gradient_accumulation_steps 2 \
    --output_dir "${OUTPUT_DIR}" \
    --eval_steps 200 \
    --save_steps 100 \
    --save_total_limit 20 \
    --logging_steps 5 \
    --seed 42 \
    --data_seed 42 \
    --max_length 4096 \
    --attn_impl sdpa \
    --deepspeed zero3 \
    --gradient_checkpointing true \
    2>&1 | tee -a "${LOG_FILE}"
