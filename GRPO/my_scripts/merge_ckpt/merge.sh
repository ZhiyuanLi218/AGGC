
#!/bin/bash

# Set the root directory containing model checkpoints
# Please modify this path as needed 
ROOT_DIR="./AGGC/GRPO/checkpoints/verl_AGGC_gsm8k"
# Iterate over each model directory in the root directory
for model_dir in "$ROOT_DIR"/*; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")

        step_dir=$(find "$model_dir" -maxdepth 1 -type d -name "global_step_*" | head -n 1)
        
        if [ -z "$step_dir" ]; then
            echo "Warning: No global_step directory found in $model_dir, skipping..."
            continue
        fi

        checkpoint_dir="$step_dir/actor"
        target_dir="$model_dir/hf"
        if [ -d "$target_dir" ] && [ "$(ls -A $target_dir)" ]; then
            echo "Warning: $target_dir is not empty, skipping..."
            continue
        fi

        echo "Processing model: $model_name"
        echo "Checkpoint dir: $checkpoint_dir"
        echo "Target dir: $target_dir"

        python3 -m verl.model_merger merge \
            --backend "fsdp" \
            --local_dir "$checkpoint_dir" \
            --target_dir "$target_dir"
            
        echo "Finished processing $model_name"
        echo "--------------------------------"
    fi
done


