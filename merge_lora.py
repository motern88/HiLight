import os.path
import re

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, GenerationConfig
from peft import PeftModel
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
from hilight.model.language_model.hilight_gemma import HiLightGemmaForCausalLM
import transformers

@dataclass
class ModelArguments:
    # 定义模型相关的参数，如模型名称、版本、是否冻结骨干网络等
    model_name_or_path: Optional[str] = field(default="model_zoo/LLM/gemma/gemma-2b-it")
    # 注意HiLight-2B中config里配置的mm_vision_tower优先级更高
    vision_tower: Optional[str] = field(default="model_zoo/CLIP-ViP/pretrain_clipvip_base_16.pt")  # 视觉塔（vision_tower）的名称或路径。视觉塔是模型中用于处理视觉信息的部分
    vision_tower_aux: Optional[str] = field(default="model_zoo/Long-CLIP/longclip-B.pt")  # 辅助视觉塔（auxiliary vision tower）的名称或路径
    token_mining_w_path:Optional[str] = field(default="model_zoo/TokenMining/原始权重_valley_20epoch.bin") # 加载token_mining权重的路径
    lora_path: Optional[str] = field(default="/root/autodl-tmp/HiLight-2B-LoraFT/VT1K-1epoch")
    output_path: Optional[str] = field(default="/root/autodl-tmp/HiLight-2B-LoRA-Merge-stage2-VT1K-1epoch")

def load_pretrained_model(model_args):
    # 实例化模型
    model = HiLightGemmaForCausalLM.from_pretrained(
        "/root/autodl-tmp/HiLight-2B-LoRA-Merge-stage2-VI100K-3epoch",
        cache_dir="/root/autodl-tmp/HiLight-2B-LoRA-Merge-stage2-VI100K-3epoch",
        attn_implementation=None,
        torch_dtype=None,
    )
    print("-模型配置加载完成！-")
    # 实例化和Gemma一致的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "model_zoo/LLM/gemma/tokenizer",
        cache_dir="model_zoo/LLM/gemma/tokenizer",
        model_max_length=512,
        padding_side="right",
    )

    # # 初始化视觉模块
    # model.get_model().initialize_vision_modules(
    #     model_args=model_args,
    #     fsdp=None # inference_args.fsdp
    # )
    # print("-视觉塔权重加载完成-")

    # token_mining初始化，如果存在训练好的token_mining权重，会直接加载，这里加上一层判断选择是否加载
    model.get_model().initialize_uni_modules(
        model_args=model_args
    )
    print("-Hilight_TokenMining .pt文件权重加载完成-")

    return tokenizer, model


def update_state_dict(state_dict):
    # 创建一个新的state_dict来存储更新后的键和对应的权重值
    new_state_dict = {}

    # 遍历原始state_dict中的所有键和权重值
    for key, value in state_dict.items():
        # 打印原始键
        # print(key)
        
        # 检查键中是否有重复的".model"
        if re.search(r'\.model\.model', key):
            # 使用正则表达式替换掉重复的".model"
            new_key = re.sub(r'\.model\.model', '.model', key)
            # 如果需要，进一步替换其他模式，例如去除"base_model."
            if "base_model." in new_key:
                new_key = re.sub(r'base_model\.', '', new_key)
            
            # 打印更新后的键
            print(f"Updated key: {new_key}")
            
            # 将更新后的键和对应的权重值存入新的state_dict
            new_state_dict[new_key] = value
        else:
            # 如果没有重复的".model"，则直接使用原始键
            new_state_dict[key] = value

    return new_state_dict


def main(model_args):

    base_tokenizer, base_model = load_pretrained_model(model_args)

    state_dict = base_model.state_dict()
    print("/打印base_model的权重/")
    for key in state_dict:
        # print(key)
        if "layers.17.self_attn.q_proj.weight" in key:
            print("layers.17.self_attn.q_proj.weight:")
            print(state_dict[key])
        if 'tokenizer_projector.1.weight' in key:
            print("tokenizer_projector.1.weight:")
            print(state_dict[key])
    
    lora_model = PeftModel.from_pretrained(base_model, model_args.lora_path, torch_dtype=None)
    
    state_dict = lora_model.state_dict()
    # 修改state_dict权重字段
    state_dict = update_state_dict(state_dict)
    # 使用更新后的state_dict来更新lora_model的权重
    lora_model.load_state_dict(state_dict, strict=False)
    
    print("/打印lora的权重/")
    for key in state_dict:
        # 打印特定权重值
        if "layers.17.self_attn.q_proj.lora_A.default.weight" in key:  # base_model.model.model.layers.17.self_attn.q_proj.lora_A.default.weight
            print("layers.17.self_attn.q_proj.lora_A.default.weight:")
            print(state_dict[key])
        if 'tokenizer_projector.1.weight' in key:
            print("tokenizer_projector.1.weight:")
            print(state_dict[key])
            
    print("应用lora中")
    model = lora_model.merge_and_unload()

    state_dict = model.state_dict()
    print("/打印应用lora后的权重/")
    for key in state_dict:
        if "layers.17.self_attn.q_proj.weight" in key:
            print("layers.17.self_attn.q_proj.weight:")
            print(state_dict[key])
        if 'tokenizer_projector.1.weight' in key:
            print("tokenizer_projector.1.weight:")
            print(state_dict[key])

    print("保存merge模型中")
    model.save_pretrained(model_args.output_path) # output_dir
    print(f"merge模型已保存在{model_args.output_path}")

    # base_tokenizer.save_pretrained(model_args.output_path)


if __name__ == "__main__":
    # 解析参数
    parser = transformers.HfArgumentParser(ModelArguments)
    # 解析不同模块的参数
    model_args, = parser.parse_args_into_dataclasses()  # TODO:此处,号是为了让model_args顺利解包，使得TokenMining顺利加载
    main(model_args)