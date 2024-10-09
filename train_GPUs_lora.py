import os
import torch
import logging
import copy
import random
from torch.utils.data import Dataset
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
from hilight.train.hilight_trainer import HiLightTrainer
from hilight.mm_utils import tokenizer_video_token
from hilight import conversation as conversation_lib
from hilight.constants import IGNORE_INDEX, DEFAULT_VIDEO_TOKEN

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    set_seed,
)

from hilight.model.processor.hilight_video_processor import load_video
from hilight.model.language_model.hilight_gemma import HiLightGemmaForCausalLM
import deepspeed
import pickle

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"


local_rank = None

def rank0_print(*args):
    # 如果当前进程的local_rank为0（主进程），则打印参数
    if local_rank == 0:
        print(*args)
# -----------------------------------------------------------------------------
# 下：初始化model和data参数

# 初始化参数
@dataclass
class ModelArguments:
    # 定义模型相关的参数，如模型名称、版本、是否冻结骨干网络等
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    # 注意HiLight-2B中config里配置的mm_vision_tower优先级更高
    vision_tower: Optional[str] = field(default="model_zoo/CLIP-ViP/lr1e-5_ft_2332_table1k_120000.pt")  # 视觉塔（vision_tower）的名称或路径。视觉塔是模型中用于处理视觉信息的部分
    vision_tower_aux: Optional[str] = field(default="model_zoo/Long-CLIP/longclip-B.pt")  # 辅助视觉塔（auxiliary vision tower）的名称或路径
    save_token_mining_w_path: Optional[str] = field(default=None) # 保存token_mining权重的路径
    token_mining_w_path:Optional[str] = field(default=None) # 加载token_mining权重的路径
    only_tune_token_mining: bool = field(default=True)  # 是否只优化token_mining冻结其他参数
    use_lora: bool = field(default=False) # 是否开启lora

@dataclass
class DataArguments:
    # 训练数据的路径
    data_path: str = field(default="data/HiLight-Finetune/only_test_dataloader_Train.json",
                           metadata={"help": "训练数据集json路径"})
    val_path: str = field(default="data/HiLight-Finetune/only_test_dataloader_Train.json",
                          metadata={"help": "验证数据集json路径"})
    lazy_preprocess: bool = False  # 是否采用延迟预处理
    is_multimodal: bool = True  # 是否为多模态数据
    video_folder: Optional[str] = field(default=None)  # 视频数据的文件夹路径
    val_folder: Optional[str] = field(default=None)  # 验证集数据文件夹路径

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="./output")  # 输出目录
    per_device_train_batch_size: int = field(default=2)  # 每个设备的批次大小
    num_train_epochs: int = field(default=5)
    max_steps:int = field(default=-1)  # 最大训练步数
    use_cpu: bool = field(default=False)  # 不使用 CUDA
    group_by_length: bool = field(default=True)  # 设置长度分组采样器
    model_max_length: int = field(
        default=1024,
        metadata={
            "help":
            "模型能处理的最大序列长度。默认值为 512。序列将被右填充（可能被截断）, Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    only_save_token_mining: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)  # 是否移除未使用的列,必要参数!!!
    # freeze_mm_mlp_adapter: bool = field(default=False)  # 是否冻结多模态混合层（mm_mlp_adapter）的参数
    # mm_projector_lr: Optional[float] = None  # 多模态投影器的学习率

# 上：初始化model和data参数
# -----------------------------------------------------------------------------
# 下：data process

def _mask_targets(target, tokenized_lens, speakers):
    # 初始化当前索引为第一个分词长度，即跳过对话头部的分词
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    # 从第二个分词长度开始处理，因为第一个已经在初始化当前索引时使用
    tokenized_lens = tokenized_lens[1:]
    # 将目标数组中从开始到当前索引的部分设置为IGNORE_INDEX，即遮蔽对话头部
    target[:cur_idx] = IGNORE_INDEX
    # 遍历剩余的分词长度和说话者
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        # 如果说话者是"human"，则遮蔽该说话者的话语部分
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
            # 更新当前索引为下一阶段的开始位置
        cur_idx += tokenized_len

def _add_speaker_and_signal(header, source, get_conversation=True):
    """
    为每一轮对话添加发言者和开始/结束信号。
    """
    BEGIN_SIGNAL = "### "  # 开始信号
    END_SIGNAL = "\n"  # 结束信号
    conversation = header  # 初始化对话字符串
    for sentence in source:
        # print("sentence:",sentence)
        # 获取发言者信息
        from_str = sentence["from"]
        # 根据发言者信息设置对应的角色
        if from_str.lower() == "human":
            from_str = "user"
        elif from_str.lower() == "gpt":
            from_str = "model"
        else:
            from_str = 'unknown'
        # 将发言者信息和发言内容拼接，并添加到对话字符串中
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        # 如果需要，更新整个对话字符串
        if get_conversation:
            conversation += sentence["value"]
    # 在对话字符串末尾添加开始信号
    conversation += BEGIN_SIGNAL
    # 返回处理后的对话字符串
    return conversation

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """对字符串列表进行分词。"""
    # 对每个文本字符串进行分词，并将结果存储在tokenized_list列表中
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",  # 返回PyTorch张量
            padding="longest",  # 所有序列填充到最长序列的长度
            max_length=tokenizer.model_max_length,  # 设置序列的最大长度
            truncation=True,  # 截断超出最大长度的序列
        ) for text in strings
    ]
    # 提取每个分词结果的input_ids，并存储在input_ids和labels列表中
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    # 计算每个序列中非填充token的数量，并存储在input_ids_lens和labels_lens列表中
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    # 返回一个包含input_ids、labels、input_ids_lens和labels_lens的字典
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

# 定义一个函数，用于预处理数据
def preprocess(
    sources: Sequence[str],  # 输入数据源
    tokenizer: transformers.PreTrainedTokenizer,  # 传入的预训练分词器
    has_image: bool = True,  # 是否包含图像
    prompt: str = None,  # 可选的提示文本
    refine_prompt: bool = False,  # 是否细化提示
) -> Dict:
    """
    给定一个数据源列表，每个数据源都是一个对话列表。这个转换过程包括以下步骤：

    1.在每个句子的开头添加信号'### '，并以结束信号'\n'结束；
    2.将多个对话串联在一起；
    3.对串联后的对话进行分词处理；
    4.制作目标对话的深度拷贝。用IGNORE_INDEX遮蔽人类话语。
    """
    # 添加结束符号并拼接
    conversations = []
    for source in sources:
        # 创建对话头部，这里仅使用两个换行符作为分隔
        header = f"\n\n"
        # 向对话中添加说话者标签和信号
        conversation = _add_speaker_and_signal(header, source)
        # 将生成的对话添加到对话列表中
        # print("conversation ", conversation)
        conversations.append(conversation)
        # print("conversations ", conversations)
    # tokenize 对话数据
    # 对串联的对话进行分词处理
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    # 获取分词后得到的输入id列表
    input_ids = conversations_tokenized["input_ids"]
    # 深拷贝输入id作为目标
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        # print("source:",source)
        # 对每个对话的头部和内容进行分词，获取分词长度列表
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        # print("tokenized_lens,input_ids:",_tokenize_fn([header] + [s["value"] for s in source],tokenizer))
        # 获取对话中每个句子的说话者
        speakers = [sentence["from"] for sentence in source]
        # 使用说话者信息和分词长度对目标列表进行掩码
        _mask_targets(target, tokenized_lens, speakers)

    # 返回一个字典，包含处理好的输入id和掩码后的目标
    # print("input_ids after preprocess",input_ids)
    # print("labels after preprocess:", targets)
    return dict(input_ids=input_ids, labels=targets)

# 定义函数preprocess_gemma，用于预处理对话数据
def preprocess_gemma(
        sources,  # 源对话数据列表
        tokenizer: transformers.PreTrainedTokenizer,  # 使用的分词器
        has_image: bool = False  # 是否包含图像数据的标志
) -> Dict:
    """
    预处理对话数据，适用于GEMMA风格的模型。
    """
    # 复制默认对话模板
    conv = conversation_lib.default_conversation.copy()
    # 定义角色映射，用于区分人类和GPT的发言
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # print(f"preprocess_gemma中sources:{sources}")
    # 应用提示模板并构建对话列表
    conversations = []
    for i, source in enumerate(sources):
        # 如果源数据的第一个元素不是来自人类的，则跳过它
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        # 清空消息列表
        conv.messages = []
        # 遍历源数据中的每个句子
        for j, sentence in enumerate(source):
            # 获取角色并添加消息到对话
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        # 将构建的对话添加到列表中
        conversations.append(conv.get_prompt())

    # 对对话进行分词
    if has_image:
        # 如果有图像，对每个对话使用tokenizer_video_token进行分词
        input_ids = torch.stack([tokenizer_video_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        # 否则，直接使用tokenizer对对话进行分词
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # 克隆输入ID以创建目标序列
    targets = input_ids.clone()

    # 确保分隔符风格是GEMMA
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # 为目标序列添加遮罩
    sep = "<start_of_turn>" + conv.sep + conv.roles[1] + "\n"
    for conversation, target in zip(conversations, targets):
        # 计算目标序列中的有效长度
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        # 分割对话为多个轮次
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        # 遮罩第一个位置
        target[:cur_len] = IGNORE_INDEX
        # 遍历每个轮次
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            # 分割轮次为指令和响应
            parts = rou.split(sep)
            if len(parts) != 2:  # 如果分割后的部分组成不一致，则打印警告并跳出循环
                print(f"WARNING: parts!=: {parts}")
                break
            # 重新连接指令和分隔符
            parts[0] += sep

            # 如果有图像，计算每个部分的长度和指令长度
            if has_image:
                round_len = len(tokenizer_video_token(rou, tokenizer))
                instruction_len = len(tokenizer_video_token(parts[0], tokenizer)) - 1  # 排除<bos>标记
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # 排除<bos>标记

            # 遮罩指令部分
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            # 更新当前长度
            cur_len += round_len
        # 遮罩剩余部分
        target[cur_len:] = IGNORE_INDEX

        # 如果当前长度小于模型最大长度，并且与有效长度不一致，则警告并遮罩整个目标序列
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    # 返回包含输入ID和目标序列的字典
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

# 预处理多模态数据的函数，
def preprocess_multimodal(sources: Sequence[str], is_multimodal) -> Dict:
    # 如果不是多模态数据，则直接返回原始数据
    if not is_multimodal:
        return sources

    # 对于多模态数据，进行预处理
    for source in sources:
        for sentence in source:
            # 如果句子中包含默认图像标记，则进行替换和格式化
            if DEFAULT_VIDEO_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN, '').strip()
                sentence['value'] = DEFAULT_VIDEO_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                # TODO: 可能需要增加这段
                # # 如果版本包含"mmtag"，则使用<Image>标签包裹图像标记
                # if "mmtag" in conversation_lib.default_conversation.version:
                #     sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN, '<Video>' + DEFAULT_VIDEO_TOKEN + '</Video>')
    return sources

import glob
# 用于查找文件的函数
def find_files_with_name(folder_path, file_name_without_extension):
    # 构建用于匹配的模式，例如 "file_name_without_extension.*"
    pattern = os.path.join(folder_path, f"{file_name_without_extension}.*")

    # 使用 glob.glob 搜索匹配的文件
    file_paths = glob.glob(pattern)

    # 返回找到的第一个文件的后缀
    if file_paths:
        return os.path.splitext(os.path.basename(file_paths[0]))[1]
    else:
        return None

# 上：data process
# -----------------------------------------------------------------------------
# 下：Dataset 和 DataClollator

class LazySupervisedDataset(Dataset):
    """惰性加载的监督式微调数据集类"""

    def __init__(self, data_path: str, video_folder: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 ):
        super(LazySupervisedDataset, self).__init__()
        self.video_folder = video_folder
        # 记录日志信息，表明正在加载数据
        logging.warning("LazySupervisedDataset加载数据中...")
        # 打开指定路径的文件，并使用json模块加载数据
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("懒加载模式，不在初始化读取全部数据...")
        # 记录日志信息，表明正在格式化输入数据，但在惰性模式下跳过此步骤
        # 保存分词器和加载的数据字典到类的实例变量中
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def __len__(self):
        # 定义获取数据集长度的方法，返回数据字典的长度
        return len(self.list_data_dict)

    # 定义一个属性，用于获取每个样本的模态长度
    @property
    def modality_lengths(self):
        length_list = []  # 初始化一个列表，用于存储模态长度
        for sample in self.list_data_dict:  # 遍历数据字典列表
            # 计算当前样本中所有对话的长度总和
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            # 如果样本中包含'video'键，则使用当前长度；否则，使用负长度
            cur_len = cur_len if ('video' in sample) else -cur_len
            length_list.append(cur_len)  # 将计算出的长度添加到列表中
        return length_list  # 返回模态长度列表

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # print(f"Fetching item {i}")

        # 按索引获取数据集中单个样本
        sources = self.list_data_dict[i]
        # 确保索引i是单个数据样本而非列表
        if isinstance(i, int):
            sources = [sources]
        # 断言保证只有一个数据样本，因为不应该被包裹在列表中
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # 从加载的数据中提取对话内容
        # conversations_sources = [example["conversations"] for example in sources]
        # print("conversations_sources: ",conversations_sources)
        # 使用预处理函数处理对话内容，并使用分词器进行编码
        # data_dict = preprocess_gemma(conversations_sources, self.tokenizer, has_image=True)
        # print("data_dict:",data_dict)

        # 初始化用于表示该样本的有效性标志，默认为True
        # is_valid = True

        # 如果数据样本中包含视频信息，则加载视频特征
        # print("sources[0]",sources[0])
        if 'video' in sources[0]:
            # print("这条样本包含视频！")
            video_file = self.list_data_dict[i]['video']  # 获取视频文件路径
            video_file = video_file.replace('.pkl', '')  # 将读取到的.pkl后缀删除，只保留文件名
            video_folder = self.video_folder  # "data/MiniGemini-Finetune/only_test_dataloader"
            pkl = False  # 不使用pkl方式读取
            if pkl:
                file_extension = ".pkl"  # 使用pkl方式
            else:
                # 调用函数返回匹配到的文件后缀，没匹配到则为None
                file_extension = find_files_with_name(video_folder, video_file)  # 例如'.mkv'或None
            # 打开视频文件并加载特征
            try:
                with open(f"{video_folder}/{video_file}{file_extension}", "rb") as video_path:
                    # print("loading:",video_path)
                    try:
                        if pkl:
                            features = pickle.load(video_path)  # 使用pickle读取已经抽好的特征  # 使用pkl方式
                            features = torch.from_numpy(features) # ndarray 2 tensor
                        else:
                            features = load_video(video_path) # 使用load_video读取原始视频
                        # print("video_features",features.shape)  # torch.Size([1, 12, 3, 224, 224])
                    except Exception as error:
                        features = None  # 设置为一个默认值
                        # print("video_features", features)
            except FileNotFoundError:
                # 如果找不到视频文件，则将有效性标志设置为False
                features = None  # 设置为一个默认值，具体取决于你的需求
                # print("video_features", features)

            # 将source样本中文本数据进行处理，用图像标记替换原始的<video>字符
            sources = preprocess_multimodal(
                sources = copy.deepcopy([e["conversations"] for e in sources]),
                is_multimodal = True) # TODO:当前写死为多模态数据

        # 如果默认版本以"gemma"开头，则调用preprocess_gemma函数
        if conversation_lib.default_conversation.version.startswith("gemma"):
            # print("调用preprocess_gemma函数处理sources")
            data_dict = preprocess_gemma(sources, self.tokenizer, has_image=True)
        # 如果索引i是整数，则说明获取的是单个样本，相应地调整data_dict
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # 如果数据中存在视频，则将视频特征添加到返回的字典中
        if 'video' in self.list_data_dict[i]:
            data_dict["video"] = features
            # print("data Dataset has video __getitem__",data_dict)  # 有input_ids,labels,video

        # 返回包含input_ids、labels、（可选）video 的特征的字典
        return data_dict

class LazyValidationDataset(LazySupervisedDataset):
    def __init__(self, data_path: str, video_folder: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 ):
        super(LazySupervisedDataset, self).__init__()
        self.video_folder = video_folder
        logging.warning("LazyValidationDataset加载数据中...")
        list_data_dict = json.load(open(data_path, "r"))
        logging.warning("懒加载模式，不在初始化读取全部数据...")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def __len__(self):
        # 定义获取数据集长度的方法，返回数据字典的长度
        return len(self.list_data_dict)

    # 定义一个属性，用于获取每个样本的模态长度
    @property
    def modality_lengths(self):
        length_list = []  # 初始化一个列表，用于存储模态长度
        for sample in self.list_data_dict:  # 遍历数据字典列表
            # 计算当前样本中所有对话的长度总和
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            # 如果样本中包含'video'键，则使用当前长度；否则，使用负长度
            cur_len = cur_len if ('video' in sample) else -cur_len
            length_list.append(cur_len)  # 将计算出的长度添加到列表中
        return length_list  # 返回模态长度列表

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # print(f"Fetching item {i}")
        # 按索引获取数据集中单个样本
        sources = self.list_data_dict[i]
        # 确保索引i是单个数据样本而非列表
        if isinstance(i, int):
            sources = [sources]
        # 断言保证只有一个数据样本，因为不应该被包裹在列表中
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # 从加载的数据中提取对话内容
        # conversations_sources = [example["conversations"] for example in sources]
        # print("conversations_sources: ",conversations_sources)
        # 使用预处理函数处理对话内容，并使用分词器进行编码
        # data_dict = preprocess_gemma(conversations_sources, self.tokenizer, has_image=True)
        # print("data_dict:",data_dict)

        # 初始化用于表示该样本的有效性标志，默认为True
        # is_valid = True

        # 如果数据样本中包含视频信息，则加载视频特征
        # print("sources[0]",sources[0])
        if 'video' in sources[0]:
            # print("这条样本包含视频！")
            video_file = self.list_data_dict[i]['video']  # 获取视频文件路径
            video_file = video_file.replace('.pkl', '')  # 将读取到的.pkl后缀删除，只保留文件名
            video_folder = self.video_folder  # "data/MiniGemini-Finetune/only_test_dataloader"
            pkl = True # 不使用pkl方式读取
            if pkl:
                file_extension = ".pkl"  # 使用pkl方式
            else:
                # 调用函数返回匹配到的文件后缀，没匹配到则为None
                file_extension = find_files_with_name(video_folder, video_file)  # 例如'.mkv'或None

            try:
                with open(f"{video_folder}/{video_file}{file_extension}", "rb") as video_path:
                    # print("loading:",video_path)
                    try:
                        if pkl:
                            features = pickle.load(video_path)  # 使用pickle读取已经抽好的特征  # 使用pkl方式
                            features = torch.from_numpy(features) # ndarray 2 tensor
                        else:
                            features = load_video(video_path) # 使用load_video读取原始视频
                        # print("video_features",features.shape)  # torch.Size([1, 12, 3, 224, 224])
                    except Exception as error:
                        features = None  # 设置为一个默认值
                        print(f"{video_folder}/{video_file}{file_extension} load failed")
                        # print("video_features", features)
            except FileNotFoundError:
                # 如果找不到视频文件，则将有效性标志设置为False
                features = None  # 设置为一个默认值，具体取决于你的需求
                print(f"{video_folder}/{video_file}{file_extension} not found")

            # 将source样本中文本数据进行处理，用图像标记替换原始的<video>字符
            sources = preprocess_multimodal(
                sources = copy.deepcopy([e["conversations"] for e in sources]),
                is_multimodal = True
            ) # FIXME:当前写死为多模态数据

        # 如果默认版本以"gemma"开头，则调用preprocess_gemma函数
        if conversation_lib.default_conversation.version.startswith("gemma"):
            # print("调用preprocess_gemma函数处理sources")
            data_dict = preprocess_gemma(sources, self.tokenizer, has_image=True)
        # 如果索引i是整数，则说明获取的是单个样本，相应地调整data_dict
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # 如果数据中存在视频，则将视频特征添加到返回的字典中
        if 'video' in self.list_data_dict[i]:
            data_dict["video"] = features
            # print("data Dataset has video __getitem__",data_dict)  # 有input_ids,labels,video

        # 返回包含input_ids、labels、（可选）video 的特征的字典
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """为监督式微调准备数据的类."""

    tokenizer: transformers.PreTrainedTokenizer  # 预设分词器，用于对输入序列进行编码和填充

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # print("instances",instances) # input_ids,video,is_valid
        # print("DataCollator前显存占用：", torch.cuda.memory_summary())  # Allocated memory ： 9584 MiB

        # 从输入的实例中提取input_ids和labels，并将它们转换为元组形式
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        # 使用pad_sequence方法对input_ids进行填充，以创建批次数据的第一个维度
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,  # 指定batch size为第一个维度
            padding_value=self.tokenizer.pad_token_id)  # 使用分词器的pad_token_id作为填充值
        # 使用pad_sequence方法对labels进行填充，以创建批次数据的第二个维度
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)  # 通常使用-100作为labels的填充值
        # 创建一个字典，包含input_ids、labels和attention_mask
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),  # 注意力掩码，不等于pad_token_id的位置为True
        )

        # 如果输入实例中包含视频信息,则将视频特征转换为张量并添加到批次字典中
        if 'video' in instances[0]:
            # print("instances[1]",instances[1])
            # features = [torch.tensor(instance['video']) for instance in instances]
            features = [instance['video'] for instance in instances]
            # print("features.shape",features[0].shape)
            if all(x is not None and x.shape == features[0].shape for x in features):
                # 如果视频特征的形状都相同，则将它们堆叠成一个批次的张量
                batch['videos'] = torch.stack(features)
            else:
                # 如果形状不同，则直接将视频特征列表添加到批次字典中
                batch['videos'] = features
        # 返回一个包含input_ids、labels、attention_mask和（如果存在）视频特征的字典

        # print("DataCollator后显存占用：", torch.cuda.memory_summary())  # Allocated memory ： 9584 MiB
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """为监督式微调创建数据集和整理器."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          video_folder=data_args.video_folder)
    eval_dataset = LazyValidationDataset(tokenizer=tokenizer,
                                         data_path=data_args.val_path,
                                         video_folder=data_args.val_folder)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # 返回一个字典，包含训练数据集、评估数据集和数据整理器
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


# 上：Dataset 和 DataClollator
# -----------------------------------------------------------------------------
# 下：保存模型

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    # 检查参数对象是否有ds_id属性，这是DeepSpeed用来跟踪参数状态的标识
    if hasattr(param, "ds_id"):
        # 如果参数状态是ZeroParamStatus.NOT_AVAILABLE，即参数不可用
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            # 如果ignore_status为False，即不应该忽略状态
            if not ignore_status:
                # 记录警告日志，指出参数状态不是可用的
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        # 使用GatheredParameters上下文管理器来处理参数
        with zero.GatheredParameters([param]):
            # 将参数的数据从原始计算图中分离，转换到CPU，并创建一个副本
            param = param.data.detach().cpu().clone()
    # 如果参数对象没有ds_id属性，或者参数是可用的
    else:
        # 直接将参数的数据从原始计算图中分离，转换到CPU，并创建一个副本
        param = param.detach().cpu().clone()

    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    # 从命名参数中筛选出匹配keys_to_match列表中的键的参数
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    # 对每个参数应用maybe_zero_3函数，可能将其设置为零，并将参数转移到CPU
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """收集模型状态并保存到磁盘"""

    # 如果训练器参数中指定了only_save_token_mining，则只保存token_mining
    if getattr(trainer.args, "only_save_token_mining", False):
        # 定义要保存的适配器相关的键
        keys_to_match = ['token_mining']

        # 获取适配器的状态并可能将其设置为零
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        # 保存模型配置
        trainer.model.config.save_pretrained(output_dir)

        # 获取输出目录的当前文件夹名称和父文件夹
        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        # 如果是主进程，则保存适配器权重
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            # 如果输出目录以'checkpoint-'开头，则创建mm_projector文件夹
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "token_mining")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            # 否则直接保存
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'token_mining.bin'))
        return

    # 如果训练器使用了DeepSpeed，则同步CUDA并保存模型
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    # 获取模型的状态字典
    state_dict = trainer.model.state_dict()
    # 如果需要保存
    if trainer.args.should_save:
        # 创建一个CPU状态字典
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        # 删除原始状态字典
        del state_dict
        # 调用训练器的_save方法保存模型
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

# 检查各参数的 requires_grad 属性，保证只训练token_mining部分
def check_requires_grad(model):
    for name, param in model.named_parameters():
        if "token_mining" in name:
            assert param.requires_grad == True
        else:
            assert param.requires_grad == False

# 上：保存模型
# -----------------------------------------------------------------------------
# 下：Lora微调所需函数

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['token_mining','vision_tower']  # ,'token_mining',
    for name, module in model.named_modules():
        # 除了视觉塔和tokenmining之外的全都
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            # print(name)
            # names = name.split('.')
            # print(names)
            # lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            lora_module_names.add(name)

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    # print(lora_module_names)
    return list(lora_module_names)

# 上：Lora微调所需函数
# -----------------------------------------------------------------------------
# 下：训练设置

def train():
    # 定义全局变量 local_rank
    global local_rank

    # 解析参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # 解析不同模块的参数
    model_args, data_args, training_args= parser.parse_args_into_dataclasses()  # HfArgumentParser 类中的一个方法,将解析的命令行参数转换为数据类的实例
    # 设置全局变量 local_rank 的值为 training_args 中的 local_rank
    local_rank = training_args.local_rank

    # 实例化模型
    model = HiLightGemmaForCausalLM.from_pretrained(
        "work_dirs/HiLight-2B",
        cache_dir="work_dirs/HiLight-2B",
        attn_implementation=None,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    rank0_print("-模型配置加载完成！-")

    # 实例化和Gemma一致的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "model_zoo/LLM/gemma/tokenizer",
        cache_dir="model_zoo/LLM/gemma/tokenizer",
        model_max_length=512,
        padding_side="right",
    )

    # # 使用梯度检查点技术
    # if training_args.gradient_checkpointing:
    #     # 不管哪个分支都实现了相同的事:使用get_input_embeddings().register_forward_hook()函数获取模型make_inputs_require_grad
    #     if hasattr(model, "enable_input_require_grads"):
    #         # model.enable_gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})  # 报错：模型没有这个方法
    #         model.enable_input_require_grads()
    #     else:
    #         def make_inputs_require_grad(module, input, output):
    #             output.requires_grad_(True)
    #         model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # 初始化视觉模块
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )
    rank0_print("-视觉塔权重加载完成-")
    # 获取初始化后的视觉塔模块
    vision_tower = model.get_vision_tower()
    vision_tower_aux = model.get_vision_tower_aux()

    # 根据训练参数将视觉塔模块转换到相应的数据类型和设备
    vision_tower.to(training_args.device)
    vision_tower_aux.to(training_args.device)

    # token_mining初始化，如果存在训练好的token_mining权重，会直接加载，这里加上一层判断选择是否加载
    model.get_model().initialize_uni_modules(
        model_args=model_args
    )
    rank0_print("-Hilight_TokenMining .pt文件权重加载完成-")

    # 将模型移动到指定设备
    model.to(training_args.device)
    # total_params = sum(param.numel() for param in model.parameters())
    # print(f"Total number of parameters: {total_params}")  # 总参数量：275,0610,946

    # 如果需要调整TokenMining，则锁定模型其他部分的梯度，只对tokenmining部分的参数进行梯度更新
    if model_args.only_tune_token_mining:
        model.requires_grad_(False)
        for p in model.get_model().token_mining.parameters():
            p.requires_grad = True
            
        # 在模型配置完成后，检查参数的 requires_grad 属性
        check_requires_grad(model)

    if model_args.use_lora:
        model.enable_input_require_grads() # 放开embedding层梯度！ 很重要
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=64,
            lora_alpha=32,
            target_modules=find_all_linear_names(model),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    # 实例化trainer
    trainer = HiLightTrainer(
        model=model,
        args=training_args,
        # callbacks=None, # 不使用TensorBoardCallback 防止因为tensorboard报错
        tokenizer=tokenizer,  # 分词器
        train_dataset=LazySupervisedDataset(tokenizer=tokenizer,
                                            data_path=data_args.data_path,
                                            video_folder=data_args.video_folder),
        data_collator=DataCollatorForSupervisedDataset(tokenizer)
        # **data_module
    )

    # 这里打印每个 batch 的信息
    # for batch in trainer.get_train_dataloader():
    #     print("Train batch:", batch)
    #     break

    # model.print_trainable_parameters()
    # trainable params: 78,446,592 || all params: 2,826,958,338 || trainable%: 2.774946872952466

    # 用于混合精度中自动将模型权重转换为半精度，防止和半精度数据计算时产生错误
    with torch.autocast("cuda"):
        trainer.train()

    # 单独保存token mining权重
    if training_args.only_save_token_mining and model_args.save_token_mining_w_path is not None:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                        output_dir=model_args.save_token_mining_w_path)
        rank0_print("-单独保存token mining权重-")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 防止抛出tokenizer并行警告
    train()


