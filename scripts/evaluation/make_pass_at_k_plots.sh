#!/bin/bash

# Set your base directory here
BASE_DIR="./passatk_evals"
RESULT_FILES="result_csv1.csv,result_csv2.csv"
OUTPUT_DIR="${BASE_DIR}/deepcoder_test"
LABELS="Model 1,Model 2"
python scripts/evaluation/make_pass_at_k_plot.py \
    --result_files "${RESULT_FILES}" \
    --output_dir "${OUTPUT_DIR}" \
    --title "Pass@K for Base Model (LCB Eval)" \
    --labels "${LABELS}" \
    --legend_title "Model"

