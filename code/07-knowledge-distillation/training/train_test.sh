#!/bin/bash
set -euo pipefail
# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/load_env.sh"

# 1. Configure the GPU environment and memory allocator.
export CUDA_VISIBLE_DEVICES=0,4
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# 2. Start a one-epoch ms-swift smoke run with the Transformers backend.
NPROC_PER_NODE=2 swift sft \
    --model ${SERVER_MODEL_PATH} \
    --dataset "${SERVER_DATASET_DIR}/train.jsonl#100" \
    --val_dataset "${SERVER_DATASET_DIR}/val.jsonl#20" \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --gradient_accumulation_steps 2 \
    --output_dir "${SERVER_OUTPUT_DIR}/test_output" \
    --eval_steps 10 \
    --save_steps 10 \
    --logging_steps 2 \
    --max_length 4096 \
    --max_steps 20 \
    --attn_impl flash_attn \
    --deepspeed zero3 \
    --gradient_checkpointing true
