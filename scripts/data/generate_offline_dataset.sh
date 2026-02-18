#!/bin/bash

export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 
export NCCL_P2P_LEVEL=NVL
export LOGLEVEL=INFO
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=86400 # 24 hour timeout, necessary for long checkpoint save calls that may take longer than the default 8 minutes
export RAY_gcs_rpc_server_reconnect_timeout_s=3600
N_GPUS_PER_NODE=4
N_CPUS_PER_NODE=16
TEMP_DIR="/tmp/"
# Getting the node names
MODEL_NAME="danieldritter/OAPL-DeepCoder-Round1"
DATASET_PATH="path/to/dataset.parquet"
TEMP=0.6
TOP_P=0.95
NUM_RESPONSES=2
BATCH_SIZE=300 # set to large value and inference engine will handle batching internally
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=65536
SEED=1
OUTPUT_DIR="${DATASET_FOLDER}/deepcoder_test_run2"
mkdir -p "$OUTPUT_DIR"
FILTER_THRESHOLDS="0,1,5,10"
INCLUDE_CHAT_TEMPLATE=true

ABILITY="code"
TENSOR_MODEL_PARALLEL_SIZE=2
MAX_NUM_BATCHED_TOKENS=131072 # model max for deepseek 
echo "Generating offline dataset..."
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH" 
echo "Output: $OUTPUT_DIR"
python ./verl/trainer/generate_offline_dataset.py \
    model.path="$MODEL_NAME" \
    data.dataset_path="$DATASET_PATH" \
    data.ability="$ABILITY" \
    data.num_responses=$NUM_RESPONSES \
    data.batch_size=$BATCH_SIZE \
    data.output_dir="$OUTPUT_DIR" \
    data.include_chat_template=$INCLUDE_CHAT_TEMPLATE \
    data.filter_all_correct=true \
    data.filter_all_incorrect=true \
    "data.filter_correct_thresholds=\"${FILTER_THRESHOLDS}\"" \
    "data.filter_incorrect_thresholds=\"${FILTER_THRESHOLDS}\"" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    rollout.temperature=$TEMP \
    rollout.top_p=$TOP_P \
    rollout.prompt_length=$MAX_PROMPT_LENGTH \
    rollout.response_length=$MAX_RESPONSE_LENGTH \
    rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
    rollout.gpu_memory_utilization=0.9 \
    rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor.ulysses_sequence_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
    rollout.enforce_eager=true \
    rollout.free_cache_engine=true \
    ray_init.num_cpus=$N_CPUS_PER_NODE \
    ray_init.temp_dir="$TEMP_DIR" \
    rollout.seed=$SEED

echo "Dataset generation complete!"