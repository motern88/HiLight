CUDA_VISIBLE_DEVICES=0 \
python train_GPUs.py \
    --model_name_or_path model_zoo/LLM/gemma/gemma-2b-it\
    --version gemma \
    --data_path ./data/MiniGemini-Finetune/only_test_dataloader_Train.json \
    --video_folder ./data/MiniGemini-Finetune/only_test_dataloader \
    --only_save_token_mining True \
    --save_token_mining_w_path "model_zoo/TokenMining/TokenMining.pt" \
    --token_mining_w_path "model_zoo/TokenMining/TokenMining.pt" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --group_by_length True \
    --output_dir "./work_dirs/Mini-Gemini-2B-FT" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --use_cpu False \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 3 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True

    # --data_path ./data/MiniGemini-Finetune/VideoInstruct-100K/VideoInstruct100K_Train.json \
    # --video_folder ./data/MiniGemini-Finetune/VideoInstruct-100K/Video-ChatGPT  \
    # --num_train_epochs 1 \
    # --vision_tower_aux model_zoo/LongCLIP/longclip-B.pt \
    # --vision_tower model_zoo/CLIP-ViP/lr1e-5_ft_2332_table1k_120000.pt \
    # --vision_tower_aux model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup \