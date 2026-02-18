#!/bin/bash

DATASET_PATH="/path/to/offline_dataset.parquet"
# OUTPUT_PATH is last part of dataset path after dataset dir, excluding the file extension 
OUTPUT_PATH="./passatk_evals/${DATASET_PATH#$DATASET_DIR/}2"
echo "Evaluating dataset at path: ${DATASET_PATH}"
echo "Output will be saved to: ${OUTPUT_PATH}"
python ./scripts/evaluation/offline_dataset_passatk.py \
    --dataset_paths "${DATASET_PATH}" \
    --output_path "$OUTPUT_PATH"