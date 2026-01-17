#!/usr/bin/env bash

set -x
export CUDA_VISIBLE_DEVICES="0,1,2,3"

MODELS=(
    './AGGC/GRPO/checkpoints/verl_AGGC_math/qwen2.5_3b_AGGC/hf'
)

PROMPT_TYPE="qwen25-math-cot"
MAX_TOKENS_PER_CALL="4096"
SPLIT="test"
NUM_TEST_SAMPLE=-1
DATA_NAMES="math"
IFS=',' read -ra DATASETS <<< "$DATA_NAMES"
ALL_EXIST=true

for MODEL_DIR in "${MODELS[@]}"; do

    if [ -d "$MODEL_DIR" ]; then
        MODEL_NAME_OR_PATH="$MODEL_DIR"
        OUTPUT_DIR="$MODEL_DIR/eval"
        
        echo "Processing model: $MODEL_NAME_OR_PATH"
        echo "Output directory: $OUTPUT_DIR"
        

        mkdir -p "$OUTPUT_DIR"
        
        TOKENIZERS_PARALLELISM=false \
        python3 -u math_eval.py \
            --model_name_or_path "${MODEL_NAME_OR_PATH}" \
            --data_name "${DATA_NAMES}" \
            --output_dir "${OUTPUT_DIR}" \
            --split "${SPLIT}" \
            --prompt_type "${PROMPT_TYPE}" \
            --num_test_sample "${NUM_TEST_SAMPLE}" \
            --seed 0 \
            --temperature 0 \
            --n_sampling 1 \
            --top_p 1 \
            --start 0 \
            --end -1 \
            --use_vllm \
            --save_outputs \
            --max_tokens_per_call "${MAX_TOKENS_PER_CALL}" \
            --overwrite
        
        echo "Completed evaluation for: $MODEL_NAME_OR_PATH"
        echo "----------------------------------------"
    fi
done

echo "All evaluations completed!" 
