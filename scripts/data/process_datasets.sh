#!/bin/bash
# leaving out codeforces since we only eval on lcb
DATASET_DIR="."
python preprocess/data_preprocess/deepcoder.py \
    --local_dir "${DATASET_DIR}/deepcoder_test_run" \
    --exclude_codeforces