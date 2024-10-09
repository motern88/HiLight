deepspeed train_GPUs.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path model_zoo/LLM/gemma/gemma-2b-it\
    --vision_tower_aux model_zoo/Long-CLIP/longclip-B.pt \
    --vision_tower model_zoo/CLIP-ViP/lr1e-5_ft_2332_table1k_120000.pt \
    --data_path /root/autodl-tmp/Valley-703k/valley_train.json \
    --val_path /root/autodl-tmp/Valley-703k/valley_val.json \
    --video_folder /root/autodl-tmp/Valley-703k/videospkl \
    --val_folder /root/autodl-tmp/Valley-703k/videospkl \
    --only_save_token_mining True \
    --only_tune_token_mining True \
    --save_token_mining_w_path "/root/autodl-tmp/TokenMining" \
    --group_by_length True \
    --output_dir "/root/autodl-tmp/HiLight-2B-FT" \
    --num_train_epochs 20 \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 5 \
    --gradient_accumulation_steps 1 \
    --use_cpu False \
    --fp16 True \
    --evaluation_strategy="steps" \
    --eval_steps 500 \
    --save_strategy "steps" \
    --save_steps 117125 \
    --save_total_limit 10 \
    --logging_steps 100 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.002 \
    --lr_scheduler_type "cosine" \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --logging_dir "/root/tf-logs"

    # --token_mining_w_path "model_zoo/TokenMining/token_mining.bin" \
    # --data_path /root/autodl-tmp/VideoInstruct-100K/only_test_dataloader_Train.json \
    # --data_path /root/autodl-tmp/VideoInstruct-100K/VideoInstruct100K_Train.json \
    # --video_folder ./data/MiniGemini-Finetune/VideoInstruct-100K/Video-ChatGPT  \
    # --num_train_epochs 1 \
    # --vision_tower_aux model_zoo/Long-CLIP/longclip-B.pt \
    # --vision_tower model_zoo/CLIP-ViP/lr1e-5_ft_2332_table1k_120000.pt \
    # --vision_tower_aux model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup \
    # --fp16 True \

    # --data_path /root/autodl-tmp/Valley-703k/valley_train.json \
    # --val_path /root/autodl-tmp/Valley-703k/valley_val.json \
    # --video_folder /root/autodl-tmp/Valley-703k/videospkl \
    # --val_folder /root/autodl-tmp/Valley-703k/videospkl \
    