from transformers import Trainer
import torch
import math
from typing import Iterator, List, Optional
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler
from transformers.tokenization_utils_base import BatchEncoding
# from transformers.trainer_pt_utils import get_length_grouped_indices
from transformers.trainer import has_length
from transformers.utils import is_datasets_available
import os
import json
import logging


class HiLightTrainer(Trainer):
    # 获取训练采样器
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # 如果没有提供训练数据集或者训练数据集没有长度信息，则不返回任何采样器
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_length:
            lengths = self.train_dataset.modality_lengths
            # 获取数据集中video的内容是否可用
            valid_flags = [sample.get('video') is not None for sample in self.train_dataset]
            # print("valid_flags:", valid_flags)
            # print("lengths:", lengths)
            return ValidLengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                valid_flags=valid_flags,
                group_by_modality=True, # TODO:暂时内部写死
            )
        
    def log(self, logs: dict):
        # Convert any torch.Tensor values to scalars before logging
        scalar_logs = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in logs.items()}
        super().log(scalar_logs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        try:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        except (TypeError, ValueError) as e:
            logging.warning(f"Skipping sample due to missing video features: {e}")
            return None, None, None

    # def create_optimizer_and_scheduler(self, num_training_steps: int):
    #     # 重写此方法以设置自定义优化器和调度器
    #     if self.optimizer is None:
    #         self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
    #     if self.lr_scheduler is None:
    #         self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #             self.optimizer,
    #             T_max=num_training_steps,
    #             eta_min=0.00001
    #         )

class ValidLengthGroupedSampler(Sampler):
    r"""
    采样器，以一种方式对数据集的特征进行分组，使具有大致相同长度的特征聚在一起，同时保持一定的随机性。
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        valid_flags: Optional[List[bool]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        """
        Args:
            batch_size (int): 每个批次的样本数。
            world_size (int): 分布式训练的进程数。
            lengths (List[int], optional): 数据集中每个特征的长度列表，默认为 None。
            generator (torch.Generator, optional): 生成随机数的生成器，默认为 None。
            group_by_modality (bool, optional): 是否根据模态分组，默认为 False。
               """
        if lengths is None:
            raise ValueError("必须提供长度。")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths # dataset of lengths(所有数据，包含无效的)
        self.valid_flags = valid_flags
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.valid_flags, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.valid_flags, self.batch_size, self.world_size, generator=self.generator)
        # print("按长度分组indices", indices)
        return iter(indices)
    
def get_modality_length_grouped_indices(lengths, valid_flags, batch_size, world_size, generator=None):
    # 我们需要使用 torch 来生成随机数，因为分布式采样器会设置 torch 的随机种子
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # print("所有样本均属于同一模态!")
        # 所有样本均属于同一模态
        return get_length_grouped_indices(lengths, valid_flags, batch_size, world_size, generator=generator)
    # TODO:以下按照lengths长度分组，但未实现valid_flags有效性分组
    # 将正负长度的索引和长度分开
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    # 对正负长度的索引进行分组后，将其打乱顺序以增加随机性
    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    # 计算每个超级批次的大小
    megabatch_size = world_size * batch_size
    # 将打乱后的索引划分成超级批次
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    # 获取最后一个超级批次
    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]

    # 将最后一个超级批次合并，形成附加的批次
    additional_batch = last_mm + last_lang
    # 去除最后一个超级批次，并打乱超级批次的顺序
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # 如果存在附加的批次，则添加到超级批次列表中
    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    # 返回展平后的超级批次列表
    return [i for megabatch in megabatches for i in megabatch]

def get_length_grouped_indices(lengths, valid_flags, batch_size, world_size, generator=None, merge=True):
    # 我们需要使用 torch 来生成随机数，因为分布式采样器会设置 torch 的随机种子。
    # 使用 torch.randperm 生成随机索引，确保每个进程在分布式训练中有不同的索引
    indices = torch.randperm(len(lengths), generator=generator).tolist()
    # print("indices",indices)
    # 筛选有效索引
    valid_indices = [i for i in indices if valid_flags[i]]
    # print("valid_indices",valid_indices)
    # 计算每个超级批次的大小
    megabatch_size = world_size * batch_size
    # 将索引按照超级批次大小划分成若干超级批次
    megabatches = [valid_indices[i : i + megabatch_size] for i in range(0, len(valid_indices), megabatch_size)]
    # print("megabatches",megabatches)
    # 对每个超级批次按照特征长度降序排序
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    # print("megabatches",megabatches)
    # 如果指定合并超级批次，则对其进行分组以增加随机性
    if merge:
        megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    # 展平超级批次列表并返回
    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    将索引列表分割成大致相等长度的 `chunks` 个块。
    """

    if len(indices) % num_chunks != 0:
        # 如果索引数不能被 num_chunks 整除，则进行普通分割
        return [indices[i::num_chunks] for i in range(num_chunks)]

    # 计算每个块中的索引数
    num_indices_per_chunk = len(indices) // num_chunks

    # 初始化块列表和块长度列表
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        # 找到最短的块，并将当前索引添加到该块中
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks
