#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2024 Yanwei Li
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import transformers


from .multimodal_encoder.builder import build_vision_tower, build_vision_tower_aux

from hilight.constants import (IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN,
                             DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)

IS_NEW_TRANSFORMERS = transformers.__version__ >= "4.34.0"

from .multimodal_projector.token_mining import UniLMProjectors

# Train.py中model.get_model().initialize_vision_modules()使用此类时，此类初始化视觉塔的构建
class HiLightMetaModel:

    def __init__(self, config):
        super(HiLightMetaModel, self).__init__(config)

        # 如果配置对象中存在 mm_vision_tower 属性，则构建视觉模块，并延迟加载
        if hasattr(config, "mm_vision_tower"):
            # print("HiLightMetaModel类初始化时，build_vision_tower")
            self.vision_tower = build_vision_tower(config, delay_load=True)  # True

        # 如果配置对象中存在 mm_vision_tower_aux 属性，则构建辅助视觉模块，并延迟加载
        if hasattr(config, "mm_vision_tower_aux"):
            self.vision_tower_aux = build_vision_tower_aux(config, delay_load=True)

    # 获取主要视觉模块
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    # 获取辅助视觉模块
    def get_vision_tower_aux(self):
        vision_tower_aux = getattr(self, 'vision_tower_aux', None)
        if type(vision_tower_aux) is list:
            vision_tower_aux = vision_tower_aux[0]
        return vision_tower_aux

    # 初始化视觉模块
    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        vision_tower_aux = model_args.vision_tower_aux

        self.config.mm_vision_tower = vision_tower
        self.config.mm_vision_tower_aux = vision_tower_aux

        # 如果主要视觉模块不存在，则构建主要视觉模块
        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        # 如果存在辅助视觉模块，则构建辅助视觉模块
        if vision_tower_aux is not None:
            if self.get_vision_tower_aux() is None:
                vision_tower_aux = build_vision_tower_aux(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.vision_tower_aux = [vision_tower_aux]
                else:
                    self.vision_tower_aux = vision_tower_aux
            else:
                if fsdp is not None and len(fsdp) > 0:
                    vision_tower_aux = self.vision_tower_aux[0]
                else:
                    vision_tower_aux = self.vision_tower_aux
                vision_tower_aux.load_model()

    # 初始化Token Mining模块
    def initialize_uni_modules(self, model_args):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 初始化token_mining模块
        self.token_mining = UniLMProjectors(hidden_size=768,hidden_size_aux=512,output_hidden_size=2048)
        self.token_mining.to(device=device)
        print("-Hilight_TokenMining 模块在initialize_uni_modules函数初始化-")

        # 获取是否加载token_mining权重
        token_mining_w_path = getattr(model_args, "token_mining_w_path", None)  # TODO:确保配置文件添加了token_mining_w_path
        if token_mining_w_path is not None:  # 如果存在权重路径则加载权重
            print(f"-检测到 Hilight_TokenMining 权重路径{token_mining_w_path}，试图加载-")
            token_mining_w_state = torch.load(token_mining_w_path)
            print("/打印tokenmining权重/")
            for key in token_mining_w_state:
                if 'tokenizer_projector.1.weight' in key:
                    print(f"需要加载的权重层名称: {key}")
            if ".pt" in token_mining_w_path:
                try:
                    # 如果是torch保存方式保存的.pt文件则直接可以加载
                    self.token_mining.load_state_dict(token_mining_w_state)
                    print("-Hilight_TokenMining .pt文件权重加载完成-")
                except Exception:
                    print("-Hilight_TokenMining 无法加载.pt文件-")
            if ".bin" in token_mining_w_path:
                try:
                    # 如果是HF保存方式保存的.bin文件权重需要进行修剪
                    token_mining_w_state = {k.replace('model.token_mining.', ''): v for k, v in
                                            token_mining_w_state.items()}  # 修剪权重字段
                    self.token_mining.load_state_dict(token_mining_w_state)
                    print("-Hilight_TokenMining .bin文件权重加载完成-")
                except Exception as e:
                    print(f"-Hilight_TokenMining 无法加载.pt文件也无法加载HF的.bin文件：{e}-")

        print("/打印tokenmining加载后的特定权重/")
        for name, param in self.token_mining.named_parameters():
            if 'tokenizer_projector.1.weight' in name:  # token_mining.tokenizer_projector.1.weight/model.layers.17.mlp.down_proj.weight
                print(f"实际加载的权重层名称: {name}")
                print(f"实际加载的权重值: {param.data}")


    
class HiLightMetaForCausalLM(ABC):
    @abstractmethod  # 定义一个抽象方法，必须由子类实现
    def get_model(self):
        pass  # 子类需要实现这个方法，返回模型实例

    # 获取模型的图像处理部分（视觉塔）
    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    # 获取模型的辅助图像处理部分
    def get_vision_tower_aux(self):
        return self.get_model().get_vision_tower_aux()

    # 编码图像数据
    def encode_videos(self, videos):
        # 如果图像为多个样本组成的batch，则按batch拆分成列表
        # print("视频特征",images.shape)
        # images = images.to(dtype=torch.float32)
        # assert images.dtype == torch.float32, "请确视觉特征输入为float32"
        # print("images.dtype",images.dtype)

        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.dtype}")

        # 经过视频encoder的特征
        videos_tensor = self.get_model().get_vision_tower().video_forward(videos)
        videos_aux_tensor = self.get_model().get_vision_tower_aux().video_forward(videos)  # encoder前双塔可以接受同一个视频特征
        # print("images和images_aux：",videos_tensor.shape,videos_aux_tensor.shape)
        # torch.Size([1, 2353, 768]) torch.Size([1, 12, 512])

        # assert videos_tensor.dtype == torch.float32, "请确保主塔输出为float32"
        # assert videos_aux_tensor.dtype == torch.float32, "请确保副塔输出为float32"
        videos, video_feat = self.get_model().token_mining(videos = videos_tensor, videos_aux = videos_aux_tensor)
        # print("video_feat",video_feat.shape)  # torch.Size([1, 12, 2048])
        return video_feat


    def prepare_inputs_labels_for_langmodal(self, input_ids):
        # print("prepare_inputs_labels_for_langmodal为纯文本处理new_input_embeds")
        new_input_embeds = self.get_model().embed_tokens(input_ids)
        return new_input_embeds


    # 将Vision_tower的输出处理成LLM的输入格式
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, videos=None
    ):
        # print("input_ids",input_ids.shape) # torch.Size([2, 63])
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        vision_tower = self.get_vision_tower()
        # 如果视觉模块不存在，或者没有图像数据，或者输入序列长度为1，则不执行多模态处理
        if vision_tower is None or videos is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and videos is not None and input_ids.shape[1] == 1:
                # 调整 attention_mask 大小以适应新的键值对
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                # 更新 position_ids
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            # # 生成文本对应的new_input_embeds
            # new_input_embeds = self.get_model().embed_tokens(input_ids)
            # 返回原始的输入和标签，不进行多模态处理
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # 将图像数据打包成张量格式
        if isinstance(videos, list):
            videos = torch.stack(videos, dim=0)

        # print("Images data type:", images.type())  # Images data type: torch.FloatTensor
        # for name, param in self.encode_images.named_parameters():
        #     print(
        #         f"Layer: {name}, Weight type: {param.data.type()}, Bias type: {param.data.type() if param.ndim > 1 else 'N/A'}")
        # 对图像进行编码，得到图像特征
        image_features = self.encode_videos(videos)

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.

        # 如果 attention_mask、position_ids 或 labels 为 None，则创建相应的占位张量
        _labels = labels # torch.Size([B, Length]
        _position_ids = position_ids # None
        _attention_mask = attention_mask  # torch.Size([B, Length]
        # print("labels:",labels.shape,"attention_mask:",attention_mask.shape,)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # 使用 attention_mask 移除填充项 -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # 初始化存储新输入嵌入和标签的列表
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # 遍历每个批次的输入
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 计算图像标记的数量
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # print("样本中图像标记数量",num_images)  # tensor(1)
            if num_images == 0:
                # 如果没有图像标记，则直接添加图像特征和对应的标签
                cur_image_features = image_features[cur_image_idx]
                # print("cur_input_ids",cur_input_ids)
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            # TODO:从此开始的这段代码写的太好了，黄金代码！实现任意位置任意数量的图像标记替换
            # 样本中存在图像标记时，找到图像标记的索引
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            # image_token_indices [-1, 第一个标记位置, 第二个标记位置, ... , 第N个标记位置, 总序列长度] 例如  [-1, 15, 54]

            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            # 根据图像标记将输入序列分割
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            # cur_input_ids_noim列表中有N+1个tensor，对应按每个图像标记前后分割得到的序列
            # cur_labels_noim与cur_input_ids_noim一致
            # 记录分割后序列的形状
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            # split_sizes 例如[15, 38] 图像标记前后两段序列长度分别为15，38

            # print("cur_input_ids_noim",cur_input_ids_noim)
            # 将分割后的文本序列拼接成一个张量经过embed得到embeds
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            # print("cur_input_embeds",cur_input_embeds.shape)  # torch.Size([53, 2048])
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            # print("cur_input_embeds_no_im",cur_input_embeds_no_im.shape)  # 过完embeds后序的张量组成的元组
            cur_new_input_embeds = []
            cur_new_labels = []
            # 初始化最大位置ID
            max_pos_id = 0
            # 遍历分割后的序列和图像特征
            for i in range(num_images + 1):
                # 过完embed后的张量组成的列表
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i]) # 过embed前的labels组成的列表
                max_pos_id += cur_input_embeds_no_im[i].shape[0]
                if i < num_images:
                    # 这里将分割好经过embed后的文本，按照一段文本一段视频的顺序追加到列表，随后在将列表中的张量拼接在一起
                    # input添加图像特征
                    cur_image_features = image_features[cur_image_idx]  # torch.Size([N, 2048])
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    # labels添加图像占位标签
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    max_pos_id += cur_image_features.shape[0]
            # print("max_pos_id", max_pos_id)  # 65  # 53文本序列+12视频特征

            # 将分割后的嵌入和标签转换为指定设备，此处完成即处理完单条样本所需的cur_new_input_embeds，cur_new_labels
            cur_new_input_embeds = [x.to(device=cur_input_embeds.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            # print("cur_new_labels",cur_new_labels.shape)

            # 将新嵌入和标签添加到列表中，用于将样本组batch
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # 如果定义了最大序列长度，则截断超长的序列
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # 确定最大序列长度，准备填充
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        # 初始化填充后的输入嵌入和标签张量
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        # 遍历序列，填充并更新张量! 这里填充是以batch中最长样本为基准进行填充的
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                # print("左填充")
                # 如果配置了左侧填充，则在左侧添加零张量
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                # print("右填充") # 一般走右填充
                # 否则在右侧添加零张量
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                # print("cur_new_embed",cur_new_embed.shape)  # torch.Size([65, 2048])
                # print("填充",torch.cat((
                #     cur_new_embed,
                #     torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                # ), dim=0).shape) # torch.Size([98, 2048]
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        # 将填充后的输入嵌入堆叠成批次张量
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # 根据原始的 attention_mask、position_ids 和 labels 更新它们
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        # print("最终new_input_embeds:",new_input_embeds.shape)  # torch.Size([2, 98, 2048])
        # print("最终new_labels:",new_labels.shape)  # torch.Size([2, 98])
        # print("最终attention_mask",attention_mask.shape)  # torch.Size([2, 98]

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
