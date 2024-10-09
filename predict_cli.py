import argparse
import torch

from hilight.constants import VIDEO_TOKEN_INDEX, DEFAULT_VIDEO_TOKEN
from hilight.conversation import conv_templates, SeparatorStyle
from hilight.mm_utils import tokenizer_video_token

from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field

import transformers
from transformers import TextStreamer
from transformers import (
    AutoTokenizer,
)

from hilight.model.processor.hilight_video_processor import load_video
from hilight.model.language_model.hilight_gemma import HiLightGemmaForCausalLM

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def load_pretrained_model(model_args,inference_args):
    # 实例化模型
    model = HiLightGemmaForCausalLM.from_pretrained(
        "/root/autodl-tmp/Gemma-2B",
        cache_dir="/root/autodl-tmp/Gemma-2B",  # HiLight-2B-LoRA-Merge # HiLight-2B
        attn_implementation=None,
        torch_dtype=(torch.bfloat16 if inference_args.bf16 else None),
    )
    print("-模型配置加载完成！-")

    # 实例化和Gemma一致的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "model_zoo/LLM/gemma/tokenizer",
        cache_dir="model_zoo/LLM/gemma/tokenizer",
        model_max_length=512,
        padding_side="right",
    )

    # 初始化视觉模块
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=None # inference_args.fsdp
    )
    print("-视觉塔权重加载完成-")
    # 获取初始化后的视觉塔模块
    vision_tower = model.get_vision_tower()
    vision_tower_aux = model.get_vision_tower_aux()

    # 根据训练参数将视觉塔模块转换到相应的数据类型和设备
    vision_tower.to(inference_args.device)
    vision_tower_aux.to(inference_args.device)

    # token_mining初始化，如果存在训练好的token_mining权重，会直接加载，这里加上一层判断选择是否加载
    model.get_model().initialize_uni_modules(
        model_args=model_args
    )
    print("-Hilight_TokenMining .pt文件权重加载完成-")

    model.eval()
    # 将模型移动到指定设备
    model.to(inference_args.device)

    return tokenizer, model

def main(model_args, inference_args):
    # 禁用PyTorch的初始化
    disable_torch_init()

    tokenizer, model = load_pretrained_model(model_args, inference_args)

    # 设置推断对话模板
    conv = conv_templates["gemma"].copy()
    roles = conv.roles

    if inference_args.videos_file is not None:
        videos = []
        # 如果args.videos_file包含逗号（,），则将其分割成一个列表，每个元素是一个图像文件路径。
        if ',' in inference_args.videos_file:
            videos = inference_args.videos_file.split(',')
        # 如果没有逗号，将args.image_file作为一个单独的元素添加到images列表中。
        else:
            videos = [inference_args.videos_file]

        # 初始化一个空列表video_convert，用于存放加载后的图像对象。
        video_convert = []

        # 遍历videos列表，对每个视频文件加载，返回视频特征添加在image_convert列表中
        for _video in videos:
            video = load_video(_video)
            video = video.to(inference_args.device)
            video_convert.append(video)
        video_tensor = video_convert

    # 如果没有指定视频文件，则将videos和video_tensor设置为None
    else:
        videos = None
        video_tensor = None


    while True:
        try:
            inp = input(f"{roles[0]}: ")  # 获取用户输入
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")  # 如果输入为空，则退出
            break
        # 打印提示信息
        print(f"{roles[1]}: ", end="")

        # 根据是否有视频输入，构造并添加消息到对话模板
        if videos is not None:
            # first message
            inp = (DEFAULT_VIDEO_TOKEN + '\n')*len(videos) + inp
            conv.append_message(conv.roles[0], inp)
            videos = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        # 获取对话模板中的提示
        prompt = conv.get_prompt()

        # 增加视频分割字符串
        # 如果prompt字符串中DEFAULT_VIDEO_TOKEN出现的次数大于或等于2，
        if prompt.count(DEFAULT_VIDEO_TOKEN) >= 2:
            # 初始化一个空字符串final_str，用来存放最终处理后的字符串。
            final_str = ''
            # 使用DEFAULT_VIDEO_TOKEN作为分隔符，将prompt字符串分割成一个列表sent_split。
            sent_split = prompt.split(DEFAULT_VIDEO_TOKEN)
            # 遍历sent_split列表中的每个子字符串_sub_sent，索引为_idx
            for _idx, _sub_sent in enumerate(sent_split):
                # 如果当前索引_idx等于sent_split列表长度减1，即最后一个元素，
                if _idx == len(sent_split) - 1:
                    # 将这个子字符串直接添加到final_str的末尾。
                    final_str = final_str + _sub_sent
                # 否则，将子字符串、图像标签（格式为'Video 1:'、'Video 2:'等）和DEFAULT_VIDEO_TOKEN添加到final_str的末尾。
                else:
                    final_str = final_str + _sub_sent + f'Image {_idx + 1}:' + DEFAULT_VIDEO_TOKEN
            prompt = final_str

        # 将提示转换为输入ID，并添加到模型中
        input_ids = tokenizer_video_token(prompt, tokenizer, VIDEO_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).to(model.device)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,  # 文本的编码形式，通常是分词后的结果转换为模型能够理解的ID格式。
                videos=video_tensor,  # 视频tensor
                do_sample=True if inference_args.temperature > 0 else False,  # 是否在文本生成过程中进行采样
                temperature=inference_args.temperature,  # 温度参数
                max_new_tokens=inference_args.max_new_tokens,  # 最大新生成的token数
                bos_token_id=tokenizer.bos_token_id,  # 序列开始的标记ID Begin of sequence token
                eos_token_id=tokenizer.eos_token_id,  # 序列结束的标记ID End of sequence token
                pad_token_id=tokenizer.pad_token_id,  # 填充标记ID Pad token
                streamer=streamer,  # 一个TextStreamer对象，用于流式处理生成的文本，以便实时显示输出
                use_cache=True)  # 是否在生成过程中使用缓存，可以加快重复生成的速度，但可能会影响生成结果的多样性

        # 将生成的文本ID解码为文本，并更新对话模板
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        conv.messages[-1][-1] = outputs

@dataclass
class ModelArguments:
    # 定义模型相关的参数，如模型名称、版本、是否冻结骨干网络等
    model_name_or_path: Optional[str] = field(default="model_zoo/LLM/gemma/gemma-2b-it")
    # 注意HiLight-2B中config里配置的mm_vision_tower优先级更高
    vision_tower: Optional[str] = field(default="model_zoo/CLIP-ViP/pretrain_clipvip_base_16.pt")  # 视觉塔（vision_tower）的名称或路径。视觉塔是模型中用于处理视觉信息的部分
    vision_tower_aux: Optional[str] = field(default="model_zoo/Long-CLIP/longclip-B.pt")  # 辅助视觉塔（auxiliary vision tower）的名称或路径
    token_mining_w_path:Optional[str] = field(default="model_zoo/TokenMining/原始权重_valley_20epoch.bin") # 加载token_mining权重的路径

@dataclass
class InferenceArguments:
    temperature: float = field(default=0.1)  # 模型温度系数
    max_new_tokens: int = field(default=128)  # 最大新生成token数
    videos_file: str = field(default="data/inference/videoA.mp4")  # 存储输入视频文件路径或是包含视频文件路径的列表
    device: str = field(default="cuda")
    bf16: bool = field(default=False)

if __name__ == "__main__":
    # 解析参数
    parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments))
    # 解析不同模块的参数
    model_args, inference_args = parser.parse_args_into_dataclasses()
    main(model_args,inference_args)
