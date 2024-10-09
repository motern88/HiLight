deepspeed train_GPUs_lora.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path model_zoo/LLM/gemma/gemma-2b-it\
    --vision_tower_aux model_zoo/Long-CLIP/longclip-B.pt \
    --vision_tower model_zoo/CLIP-ViP/pretrain_clipvip_base_16.pt \
    --data_path /root/autodl-tmp/VideoInstruct-100K/VideoInstruct100K_Train.json \
    --val_path /root/autodl-tmp/VideoInstruct-100K/VideoInstruct100K_Eval.json \
    --video_folder /root/autodl-tmp/Videospkl \
    --val_folder /root/autodl-tmp/Videospkl \
    --only_save_token_mining False \
    --only_tune_token_mining False \
    --use_lora True \
    --save_token_mining_w_path "model_zoo/TokenMining" \
    --group_by_length True \
    --output_dir  "output/HiLight-2B-LoraFT" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 5 \
    --gradient_accumulation_steps 1 \
    --use_cpu False \
    --fp16 True \
    --evaluation_strategy="steps" \
    --eval_steps 20 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.002 \
    --lr_scheduler_type "cosine" \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 5 \
    --report_to wandb \
    --logging_dir "/root/tf-logs/runs"

    # --output_dir  "/root/autodl-tmp/HiLight-2B-FT" \
    # --token_mining_w_path "model_zoo/TokenMining/token_mining.bin"
    # --data_path ./data/MiniGemini-Finetune/VideoInstruct-100K/VideoInstruct100K_Train.json \
    # --video_folder ./data/MiniGemini-Finetune/VideoInstruct-100K/Video-ChatGPT  \
    # --num_train_epochs 1 \
    # --vision_tower_aux model_zoo/LongCLIP/longclip-B.pt \
    # --vision_tower model_zoo/CLIP-ViP/lr1e-5_ft_2332_table1k_120000.pt \
    # --vision_tower_aux model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup \
    # --fp16 True \