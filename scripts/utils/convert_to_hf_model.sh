#!/bin/bash
# Converts a model checkpoint to hf format from fsdp
BACKEND="fsdp"
LOCAL_DIR="checkpoint_dir"
OUTPUT_DIR="${LOCAL_DIR}/huggingface"
python scripts/utils/convert_to_hf_model.py merge \
    --local_dir $LOCAL_DIR \
    --backend $BACKEND \
    --target_dir $OUTPUT_DIR