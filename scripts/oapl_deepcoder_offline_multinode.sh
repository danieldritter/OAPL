#!/bin/bash

# Multi-node Ray Setup
export NCCL_DEBUG=INFO
export NCCL_SHM_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 # https://github.com/bigscience-workshop/bigscience/blob/7ccf7e42577fe71e88cf8bed3b9ca965c7afb8f7/train/tr11-176B-ml/tr11-176B-ml.slurm#L207C1-L207C35
export NCCL_P2P_LEVEL=NVL # https://github.com/huggingface/accelerate/issues/314#issuecomment-1565259831
export LOGLEVEL=INFO
# Increasing default timeouts. Defaults were often hit in multi-node training.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=86400
export RAY_gcs_rpc_server_reconnect_timeout_s=3600
N_GPUS_PER_NODE=4
CPUS_PER_NODE=16
TEMP_DIR="/tmp/"
# Getting the node names
NODES_ARRAY=( "node1" "node2" "..." )
echo "Nodes in the job: ${NODES_ARRAY[@]}"
head_node=${NODES_ARRAY[0]}
head_node_ip=$(hostname --ip-address)
head_node_port=( $( comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1 ) )
echo "Nodes in the job: ${#NODES_ARRAY[@]}"
echo "Head node IP: $head_node_ip"
echo "Head node port: $head_node_port"
# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

ip_head=$head_node_ip:$head_node_port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$head_node_port \
    --num-cpus "${CPUS_PER_NODE}" --num-gpus $N_GPUS_PER_NODE --temp-dir $TEMP_DIR --block &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=${#NODES_ARRAY[@]}

for ((i = 1; i <= worker_num; i++)); do
    node_i=${NODES_ARRAY[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${CPUS_PER_NODE}" --num-gpus $N_GPUS_PER_NODE --temp-dir $TEMP_DIR --block &
    sleep 5
done

# Generation parameters
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=32000
MAX_VAL_ROLLOUT_LENGTH=65536
TOKENIZER_MODEL_MAX_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
BATCH_SIZE=256
MINIBATCH_SIZE=256
BATCH_SIZE_PER_GPU=1
TEMP=0.6
TOP_P=0.95
TRAIN_FILES="['path/to/train.parquet']"
TEST_FILES="['path/to/test.parquet']"
K_EVAL=4
BETA2=0.001
BETA1=1.0
SEED=42
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
EXPERIMENT_NAME="deepcoder"
CONVERT_TO_V_STAR=True
LOGGERS="['console','wandb']"
VAL_BEFORE_TRAIN=True
RUN_OFFLINE=True
USE_PRECOMPUTED_VALUES=True
TOTAL_EPOCHS=4
SHUFFLE=True
NUM_CHECKPOINTS_TO_KEEP=null  # None = keep all checkpoints
SAVE_FREQ="epoch"
TEST_FREQ="epoch"
MAX_NUM_BATCHED_TOKENS=131072 # model max for deepseek 

# LR Scheduler Params
LR_SCHEDULER="cosine"
MIN_LR_RATIO=0.05
LR_WARMUP_STEPS_RATIO=0.02

# optimizer params
LR=1e-6
OPTIMIZER_TYPE="decoupled_adamw"
OPT_BETA1=0.9
OPT_BETA2=0.95
OPT_EPS=1e-10
WEIGHT_DECAY=1e-8
TENSOR_MODEL_PARALLEL_SIZE=2
SAVE_CONTENTS="['hf_model']"
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    python -m verl.trainer.main_apo \
    data.run_offline=$RUN_OFFLINE \
    data.use_precomputed_values=$USE_PRECOMPUTED_VALUES \
    data.tokenizer_model_max_length=$TOKENIZER_MODEL_MAX_LENGTH \
    algorithm.beta2=$BETA2 \
    algorithm.adv_estimator=gae \
    data.seed=$SEED \
    data.shuffle=$SHUFFLE \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$TEST_FILES" \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    actor_rollout_ref.model.path=$MODEL_NAME \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.checkpoint.save_contents=$SAVE_CONTENTS \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.actor.optim.optimizer_type=$OPTIMIZER_TYPE \
    actor_rollout_ref.actor.optim.beta1=$OPT_BETA1 \
    actor_rollout_ref.actor.optim.beta2=$OPT_BETA2 \
    actor_rollout_ref.actor.optim.eps=$OPT_EPS \
    actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY \
    actor_rollout_ref.actor.optim.warmup_style=$LR_SCHEDULER \
    actor_rollout_ref.actor.optim.min_lr_ratio=$MIN_LR_RATIO \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$LR_WARMUP_STEPS_RATIO \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINIBATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_RESPONSE_LENGTH + MAX_PROMPT_LENGTH)) \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.val_kwargs.n=$K_EVAL \
    actor_rollout_ref.rollout.val_kwargs.temperature=$TEMP \
    actor_rollout_ref.rollout.val_kwargs.top_p=$TOP_P \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=$TEMP \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.response_length=$MAX_VAL_ROLLOUT_LENGTH \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    critic.convert_to_v_star=$CONVERT_TO_V_STAR \
    critic.beta1=$BETA1 \
    critic.model.path=$CRITIC_MODEL_NAME \
    critic.model.use_remove_padding=False \
    critic.ppo_micro_batch_size_per_gpu=$BATCH_SIZE_PER_GPU \
    trainer.logger=$LOGGERS \
    trainer.project_name='OAPL' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=${#NODES_ARRAY[@]} \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.resume_mode='auto' \
    ray_init.num_cpus=null \
    ray_init.temp_dir=$TEMP_DIR \
    trainer.max_actor_ckpt_to_keep=$NUM_CHECKPOINTS_TO_KEEP