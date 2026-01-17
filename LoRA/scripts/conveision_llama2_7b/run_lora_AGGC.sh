#!/usr/bin/env bash

BASE_MODEL="meta-llama/Llama-2-7b-hf"
OUTPUT_PATH="./AGGC/output/math/Llama-2-7B-r128_AGGC"
DATA_PATH="./AGGC/LoRA/pissa-dataset"

deepspeed --master_port=${MASTER_PORT} --include=localhost:0,1,2,3 train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $BASE_MODEL \
    --full_finetune False \
    --bf16 \
    --init_weights True \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --lora_rank 128 \
    --lora_alpha 128 \
    --lora_dropout 0 \
    --data_path $DATA_PATH \
    --sub_task metamath:100000 \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --log_level info \
    --log_level_replica warning \
    --logging_strategy steps \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --merge True \
    --max_grad_norm 0.0 \
    --per_module_norm True \
    --per_module_apply_to lora_only \
    --per_module_auto_bounds True \
    --per_module_ema_beta 0.95 \
    --per_module_low_mult 0.95 \
    --per_module_high_mult 0.6 \
    --per_module_min_norm 0.01 \
    --per_module_raise_small False \
    --per_module_sync_dist True \
    --per_module_bounds_schedule linear \
    --per_module_high_mult_late 0.7 \
    --per_module_low_mult_late 0.98 \
    --per_module_bounds_switch_ratio 0.4 \
    --per_module_bounds_transition_ratio 0.3 \
    --per_module_max_up_scale 1.1 \
    --per_module_max_down_scale 0.9
