import torch
import torch.nn as nn

# 读取.bin文件中的权重字典
weights_dict = torch.load("VideoGPT-internvideo2-projector.bin", map_location=torch.device('cpu'))

# HiLight:
# 'model.token_mining.query_projector.0.weight', 'model.token_mining.query_projector.0.bias',
# 'model.token_mining.query_projector.1.weight', 'model.token_mining.query_projector.1.bias',
# 'model.token_mining.key_projector.0.weight', 'model.token_mining.key_projector.0.bias',
# 'model.token_mining.key_projector.1.weight', 'model.token_mining.key_projector.1.bias',
# 'model.token_mining.val_projector.0.weight', 'model.token_mining.val_projector.0.bias',
# 'model.token_mining.val_projector.1.weight', 'model.token_mining.val_projector.1.bias',
# 'model.token_mining.LongCLIP_projector.0.weight', 'model.token_mining.LongCLIP_projector.0.bias',
# 'model.token_mining.CLIP_VIP_projector.0.weight', 'model.token_mining.CLIP_VIP_projector.0.bias',
# 'model.token_mining.tokenizer_projector.0.weight', 'model.token_mining.tokenizer_projector.0.bias',
# 'model.token_mining.tokenizer_projector.1.weight', 'model.token_mining.tokenizer_projector.1.bias',
# 'model.token_mining.tokenizer_projector.3.weight', 'model.token_mining.tokenizer_projector.3.bias'

# VideoGPT-clip-projector:
# 'model.mm_projector.0.weight', 'model.mm_projector.0.bias',
# 'model.mm_projector.2.weight', 'model.mm_projector.2.bias'

# VideoGPT-internvideo2-projector:
# 'model.mm_projector.0.weight', 'model.mm_projector.0.bias',
# 'model.mm_projector.2.weight', 'model.mm_projector.2.bias'

# MiniGemini:
# 'model.mm_projector.0.weight', 'model.mm_projector.0.bias',
# 'model.mm_projector.2.weight', 'model.mm_projector.2.bias',
# 'model.vlm_uni_query_projector.0.weight', 'model.vlm_uni_query_projector.0.bias',
# 'model.vlm_uni_query_projector.1.weight', 'model.vlm_uni_query_projector.1.bias',
# 'model.vlm_uni_aux_projector.0.weight', 'model.vlm_uni_aux_projector.0.bias',
# 'model.vlm_uni_aux_projector.1.weight', 'model.vlm_uni_aux_projector.1.bias',
# 'model.vlm_uni_val_projector.0.weight', 'model.vlm_uni_val_projector.0.bias',
# 'model.vlm_uni_val_projector.1.weight', 'model.vlm_uni_val_projector.1.bias'
# 打印权重字典中的键（即层的名称）
print(weights_dict.keys())