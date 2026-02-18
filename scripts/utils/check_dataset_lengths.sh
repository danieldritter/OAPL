#!/bin/bash

# This script checks the lengths of dataset prompts relative to a given model 

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DATASET_FOLDER="/n/netscratch/kdbrantley_lab/Lab/dritter/datasets"
DATASET_FILES="${DATASET_FOLDER}/deepcoder_test_run/correct_0_incorrect_0/train.parquet"
PROMPT_KEY="prompt"
RESPONSE_KEY="response"
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=65536
python scripts/utils/check_dataset_lengths.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_files $DATASET_FILES \
    --prompt_key $PROMPT_KEY \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --response_key $RESPONSE_KEY \
    --max_response_length $MAX_RESPONSE_LENGTH


