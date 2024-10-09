import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------
# 下：CLIP-ViP模型结构定义，只保留视觉模块，参考CLIP-ViP.src.modeling.CLIP_ViP.py

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPVisionConfig

@dataclass
class CLIPOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    vision_model_output: BaseModelOutputWithPooling = None


    def to_tuple(self) -> Tuple[Any]:
        # 将 CLIPOutput 转换为元组形式
        return tuple(
            self[k] if k not in ["vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )

class CLIPVisionViPEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig, additional_vision_config=None):
        super().__init__()
        # 初始化函数，接收 CLIPVisionConfig 类型的参数 config，以及 additional_vision_config 参数
        self.config = config
        # print("config",config)
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.temporal_size = additional_vision_config.temporal_size  # 设置时序大小
        self.if_use_temporal_embed = additional_vision_config.if_use_temporal_embed  # 是否使用时序嵌入
        self.patch_size = config.patch_size # 应当为16
        # print("config.patch_size",config.patch_size)

        self.add_cls_num = additional_vision_config.add_cls_num  # 要添加的类别数量
        self.added_cls = nn.Parameter(torch.randn(self.add_cls_num, self.embed_dim))  # 添加的类别作为参数

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))  # 类别嵌入作为参数

        # 使用 Conv2d 进行图像的 patch 嵌入
        # print("embed_dim",self.embed_dim)  # embed_dim 768
        # print("patch_size",self.patch_size)  # patch_size 应当为16
        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

        # 计算 patch 和位置嵌入相关信息
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

        # 如果使用时序嵌入，初始化时序嵌入参数
        if self.if_use_temporal_embed:
            self.temporal_embedding = nn.Parameter(torch.zeros(1, self.temporal_size, self.embed_dim))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # 正向传播函数，接收图像的像素值作为输入并返回张量

        # 获取输入图像的维度信息
        B, T, C, H, W = pixel_values.shape  # T 时间步数帧数 batch=6时torch.Size([6, 12, 3, 224, 224])

        # 如果使用时序嵌入，根据输入的时序大小调整时间嵌入的形状
        if self.if_use_temporal_embed:
            if T != self.temporal_embedding.shape[1]:
                time_embed = self.temporal_embedding.transpose(1, 2)
                time_embed = F.interpolate(time_embed, size=(T), mode='linear')
                time_embed = time_embed.transpose(1, 2)
            else:
                time_embed = self.temporal_embedding
        # print("time_embed",time_embed.shape)  # torch.Size([1, 12, 768])

        # 对输入图像进行 patch 嵌入并进行形状变换
        patch_embeds = self.patch_embedding(pixel_values.reshape(-1, C, H, W))
        # print("patch_embeds",patch_embeds.shape)  # torch.Size([72, 768, 14, 14])
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)   # [B*T, H*W, C]
        # print("patch_embeds", patch_embeds.shape)  # torch.Size([72, 196, 768])
        C = patch_embeds.shape[-1]
        patch_embeds = patch_embeds.reshape(B, T, -1, C)
        # print("patch_embeds",patch_embeds.shape)  # torch.Size([6, 12, 196, 768])

        # 如果使用时序嵌入，将时间嵌入加到 patch 嵌入上
        if self.if_use_temporal_embed:
            patch_embeds = patch_embeds + time_embed.unsqueeze(2)    # [B, T, H*W, C]

        # 将位置嵌入加到 patch 嵌入上
        # self.position_embedding(self.position_ids[:, 1:]).unsqueeze(1) : torch.Size([1, 1, 196, 768])
        patch_embeds = patch_embeds + self.position_embedding(self.position_ids[:, 1:]).unsqueeze(1)
        # print("patch_embeds",patch_embeds.shape)  # torch.Size([6, 12, 196, 768])

        # 类别嵌入加上位置嵌入
        class_embeds = self.class_embedding.expand(B, 1, -1)
        class_embeds = class_embeds + self.position_embedding(self.position_ids[:, 0:1])
        # print("class_embeds",class_embeds.shape)  # torch.Size([6, 1, 768])

        # 添加的类别嵌入加上位置嵌入
        added_cls = self.added_cls.expand(B, self.add_cls_num, -1)
        added_cls = added_cls + self.position_embedding(self.position_ids[:, 0:1])
        # print("added_cls",added_cls.shape)  # torch.Size([6, 3, 768])

        # 获取 patch 嵌入的大小并进行拼接
        N, L = patch_embeds.shape[1], patch_embeds.shape[2]

        embeds = torch.cat([class_embeds, added_cls, patch_embeds.reshape(patch_embeds.shape[0], -1, patch_embeds.shape[-1])], dim=1)
        M = 1 + self.add_cls_num

        return (embeds, (M, N, L))  # [B, M+N*L, C]

class CLIPAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim 必须能被 num_heads 整除 (得到 `embed_dim`: {self.embed_dim}，`num_heads`: {self.num_heads})."
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        # 线性映射层
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_size,  # (4, 12, 196)
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """输入形状: Batch x Time x Channel"""

        if inputs_size is not None:
            return self.forward2(hidden_states, inputs_size), None
        else:
            raise Exception("CLIP-ViP中的Attention没有正确跳转到forward2——处理视频而非图像的attention")

    # 处理视频用的attention
    def forward2(self, hidden_states, inputs_size):
        M, N, L = inputs_size
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj 获取查询投影
        query_states = self.q_proj(hidden_states) * self.scale  # 查询状态
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)  # 键状态
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)  # 值状态

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        # qkv: [B*num_heads, M+N*L, head_dim]
        # in-frame attention:
        q = query_states[:, M:].reshape(-1, L, self.head_dim)  # [B*num_heads*N, L, head_dim]
        k = key_states[:, :M].repeat(1, N, 1).reshape(-1, M, self.head_dim)  # [B*num_heads*N, M, head_dim]
        k = torch.cat([k, key_states[:, M:].reshape(-1, L, self.head_dim)], dim=1)  # [B*num_heads*N, M+L, head_dim]
        v = value_states[:, :M].repeat(1, N, 1).reshape(-1, M, self.head_dim)  # [B*num_heads*N, M, head_dim]
        v = torch.cat([v, value_states[:, M:].reshape(-1, L, self.head_dim)], dim=1)  # [B*num_heads*N, M+L, head_dim]
        attn_weights = torch.bmm(q, k.transpose(1, 2))  # 注意力权重
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)  # softmax归一化
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)  # dropout
        attn_output = torch.bmm(attn_probs, v)  # 注意力输出  [B*num_heads*N, L, head_dim]
        attn_output = attn_output.view(bsz, self.num_heads, N, L, self.head_dim)  # 重塑维度
        attn_output = attn_output.permute(0, 2, 3, 1, 4)  # 维度置换
        attn_output_frames = attn_output.reshape(bsz, N * L, embed_dim)  # [B, N*L, C]

        # cls divided attention:
        q = query_states[:, :M]  # [B*num_heads, M, head_dim]
        k = key_states  # [B*num_heads, M+N*L, head_dim]
        v = value_states  # [B*num_heads, M+N*L, head_dim]
        attn_weights = torch.bmm(q, k.transpose(1, 2))  # 注意力权重
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)  # softmax归一化
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)  # dropout
        attn_output = torch.bmm(attn_probs, v)  # 注意力输出 [B*num_heads, M, head_dim]
        attn_output = attn_output.view(bsz, self.num_heads, M, self.head_dim)  # 重塑维度
        attn_output = attn_output.transpose(1, 2)  # 维度置换
        attn_output_cls = attn_output.reshape(bsz, M, embed_dim)  # [B, M, C]

        attn_output = torch.cat([attn_output_cls, attn_output_frames], dim=1)  # 拼接注意力输出
        attn_output = self.out_proj(attn_output)  # 输出投影

        return attn_output  # 返回注意力输出

class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]  # 激活函数
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)  # 第一个线性层
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)  # 第二个线性层

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)  # 第一个线性层的前向传播
        hidden_states = self.activation_fn(hidden_states)  # 应用激活函数
        hidden_states = self.fc2(hidden_states)  # 第二个线性层的前向传播
        return hidden_states  # 返回最终的隐藏状态

class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)  # 自注意力层
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)  # 第一个 LayerNorm
        self.mlp = CLIPMLP(config)  # 多层感知机（MLP）
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)  # 第二个 LayerNorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_size,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:

        # 如果 hidden_states 是一个 tuple，则表示传入了两个部分的隐藏状态
        if isinstance(hidden_states, tuple):
            residual = hidden_states  # 保留残差连接的部分
            # 对第一个隐藏状态进行 LayerNorm 处理
            hidden_states = (self.layer_norm1(hidden_states[0]), self.layer_norm1(hidden_states[1]))
            # 经过自注意力机制，得到更新的隐藏状态和注意力权重
            hidden_states, attn_weights = self.self_attn(
                hidden_states=hidden_states,
                inputs_size=inputs_size,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
            )

            # 使用残差连接，将更新的隐藏状态与原始隐藏状态相加
            hidden_states = residual + hidden_states
            residual = hidden_states  # 更新残差连接的部分

            # 对更新后的隐藏状态进行 LayerNorm 处理
            hidden_states = self.layer_norm2(hidden_states)
            # 通过多层感知机进行前馈神经网络的处理
            hidden_states = self.mlp(hidden_states)
            # 再次使用残差连接，将前馈网络处理后的隐藏状态与原始隐藏状态相加
            hidden_states = residual + hidden_states

        else:  # 如果 hidden_states 不是 tuple
            residual = hidden_states  # 保留残差连接的部分
            # 对隐藏状态进行 LayerNorm 处理
            hidden_states = self.layer_norm1(hidden_states)
            # 经过自注意力机制，得到更新的隐藏状态和注意力权重
            hidden_states, attn_weights = self.self_attn(
                hidden_states=hidden_states,
                inputs_size=inputs_size,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
            )

            # 使用残差连接，将更新的隐藏状态与原始隐藏状态相加
            hidden_states = residual + hidden_states
            residual = hidden_states  # 更新残差连接的部分
            # 对更新后的隐藏状态进行 LayerNorm 处理
            hidden_states = self.layer_norm2(hidden_states)
            # 通过多层感知机进行前馈神经网络的处理
            hidden_states = self.mlp(hidden_states)
            # 再次使用残差连接，将前馈网络处理后的隐藏状态与原始隐藏状态相加
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)  # 输出为更新后的隐藏状态
        if output_attentions:  # 如果需要输出注意力权重
            outputs += (attn_weights,)  # 添加注意力权重到输出中

        return outputs  # 返回更新后的隐藏状态及注意力权重（可选）

class CLIPEncoder(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        # 使用 ModuleList 存储多个 CLIPEncoderLayer 层
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False  # 梯度检查点默认关闭

    def forward(
        self,
        inputs_embeds,
        inputs_size = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果需要输出隐藏状态，则初始化 encoder_states 为空元组，否则置为 None
        encoder_states = () if output_hidden_states else None
        # 如果需要输出注意力，则初始化 all_attentions 为空元组，否则置为 None
        all_attentions = () if output_attentions else None
        hidden_states = inputs_embeds

        # 遍历多个 CLIPEncoderLayer 进行编码处理
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点且在训练模式下进行前向传播
                # 创建自定义的前向传播函数
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                # 使用梯度检查点进行计算
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    inputs_size,
                    attention_mask,
                    causal_attention_mask,
                )
            else:
                # 正常情况下调用 encoder_layer 进行前向传播
                layer_outputs = encoder_layer(
                    hidden_states,
                    inputs_size,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,  # output_attetnions=output_attentions
                )

            # 获取编码后的隐藏状态
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态加入到 encoder_states 中
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        # 如果不需要返回字典格式的输出，将结果以元组形式返回
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        # 如果需要返回字典格式的输出，使用 BaseModelOutput 返回结果
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

# 这个类的存在只是因为CLIPModel继承了这个类
class CLIPPreTrainedModel(PreTrainedModel):

    config_class = CLIPConfig  # CLIP 模型的配置类
    base_model_prefix = "clip"  # 基础模型前缀
    supports_gradient_checkpointing = True  # 是否支持梯度检查点
    _keys_to_ignore_on_load_missing = [r"position_ids"]  # 加载模型时要忽略的键列表

    def _init_weights(self, module):
        """初始化权重"""
        # 初始化权重的方法
        factor = self.config.initializer_factor  # 初始化因子
        # ... 其他层的权重初始化，如视觉嵌入层、注意力机制、MLP、整个模型等
        if isinstance(module, CLIPVisionViPEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, CLIPAttention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, CLIPMLP):
            factor = self.config.initializer_factor
            in_proj_std = (
                (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            )
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, CLIPModel):
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        # 设置梯度检查点的方法，针对 CLIPEncoder 类型的模块
        if isinstance(module, CLIPEncoder):
            module.gradient_checkpointing = value  # 设置模块是否使用梯度检查点

class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig, additional_vision_config=None):
        super().__init__()
        self.config = config
        # 获取嵌入维度
        embed_dim = config.hidden_size

        # 创建 CLIPVisionViPEmbeddings 实例，用于处理视觉输入的嵌入
        self.embeddings = CLIPVisionViPEmbeddings(config, additional_vision_config)
        # 对嵌入进行预层归一化
        self.pre_layrnorm = nn.LayerNorm(embed_dim)
        # 创建 CLIPEncoder 实例，用于编码嵌入
        self.encoder = CLIPEncoder(config)
        # 对编码输出进行后层归一化
        self.post_layernorm = nn.LayerNorm(embed_dim)

    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        视觉模型的前向传播函数
        Args:
            pixel_values (Optional[torch.FloatTensor]): 视觉输入的像素值
            output_attentions (Optional[bool]): 是否输出注意力权重
            output_hidden_states (Optional[bool]): 是否输出隐藏状态
            return_dict (Optional[bool]): 是否返回字典形式的输出
        Returns:
            Union[Tuple, BaseModelOutputWithPooling]: 模型的输出，可能包含最终的隐藏状态和池化输出
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 处理视觉输入的嵌入和输入尺寸
        # print("pixel_values",pixel_values.shape)  # torch.Size([6, 12, 3, 224, 224])
        hidden_states, inputs_size = self.embeddings(pixel_values)   # tuple  inputs_size (4, 12, 196)
        # print("embeddings,hidden_states",hidden_states.shape)
        # 对嵌入进行预层归一化
        hidden_states = self.pre_layrnorm(hidden_states)  # torch.Size([6, 2356, 768])
        # print(hidden_states.shape)

        # 调用 CLIPEncoder 进行编码
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            inputs_size=inputs_size,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器输出的最后隐藏状态
        last_hidden_state = encoder_outputs[0]  # torch.Size([6, 2356, 768])
        # print(hidden_states.shape)
        # 提取池化输出，使用第一个位置的特征
        pooled_output = last_hidden_state[:, 0, :]
        # 对池化输出进行后层归一化
        pooled_output = self.post_layernorm(pooled_output)  # torch.Size([6, 768])

        # 如果不返回字典，则返回一组元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果返回字典，则返回包含池化输出的 BaseModelOutputWithPooling 对象
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        ),last_hidden_state

class CLIPVisionModel(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionTransformer(config)
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 获取模型的输入嵌入层（embeddings）中的 patch_embedding
        return self.vision_model.embeddings.patch_embedding

    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        视觉模型的前向传播函数

        Args:
            pixel_values (Optional[torch.FloatTensor]): 视觉输入的像素值
            output_attentions (Optional[bool]): 是否输出注意力权重
            output_hidden_states (Optional[bool]): 是否输出隐藏状态
            return_dict (Optional[bool]): 是否返回字典形式的输出

        Returns:
            Union[Tuple, BaseModelOutputWithPooling]: 模型的输出，可能包含最终的隐藏状态和池化输出
        """
        # 调用 CLIPVisionTransformer 模型的前向传播函数
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

class CLIPModel(CLIPPreTrainedModel):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        # 检查vision encoder是否符合预期配置
        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                f"config.vision_config is expected to be of type CLIPVisionConfig but is of type {type(config.vision_config)}."
            )

        # 获取视觉配置
        vision_config = config.vision_config

        # 获取额外的视觉配置（如果存在）
        if hasattr(config, "vision_additional_config"):
            additional_vision_config = config.vision_additional_config
        else:
            additional_vision_config = None

        # 获取投影维度和视觉嵌入维度
        self.projection_dim = config.projection_dim
        self.vision_embed_dim = vision_config.hidden_size

        # 创建CLIPVisionTransformer 模型实例
        self.vision_model = CLIPVisionTransformer(vision_config, additional_vision_config)

        # 创建文本和视觉投影层
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        # 创建 logit_scale 参数
        self.logit_scale = nn.Parameter(torch.ones([]) * self.config.logit_scale_init_value)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_image_features(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            if_norm: Optional[bool] = None,
    ) -> torch.FloatTensor:

        # 使用 CLIP 模型的配置替代视觉和文本组件的某些字段（如果指定）
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 CLIPVisionModel 模型的前向传播函数
        vision_outputs, last_hidden_state = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取last_hidden_state
        # print("last_hidden_state",last_hidden_state.shape)  # 应当为([B, 2352, 768])

        # 获取池化输出
        pooled_output = vision_outputs[1]  # pooled_output
        # 将池化输出通过视觉投影层投影到指定维度
        image_features = self.visual_projection(pooled_output)

        # 是否进行归一化
        if_norm = if_norm if if_norm is not None else False
        if if_norm:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features, last_hidden_state

# 上：CLIP-ViP模型结构定义，只保留视觉模块，参考CLIP-ViP.src.modeling.CLIP_ViP.py
# ---------------------------------------------------------------------------------------
# 下：推理阶段模型使用的类定义，初始化，参考CLIP-ViP项目中inference.py

# 定义一个类来表示clip_vision_additional_config
class ClipVisionAdditionalConfig:
    def __init__(self, type, temporal_size, if_use_temporal_embed, logit_scale_init_value, add_cls_num):
        self.type = type
        self.temporal_size = temporal_size
        self.if_use_temporal_embed = if_use_temporal_embed
        self.logit_scale_init_value = logit_scale_init_value
        self.add_cls_num = add_cls_num


import json
def load_json(file_path):
    with open(file_path, 'r') as f:
        config_data = json.load(f)
    return config_data

class VidCLIP(nn.Module):
    def __init__(self):
        super(VidCLIP, self).__init__()
        # 从预训练的 CLIPConfig 加载配置
        config_data = load_json("model_zoo/CLIP-ViP/clipvip-vision_config.json")
        # print("File exists:", os.path.exists(file_path))
        clipconfig = CLIPConfig(**config_data)
        # print("clipconfig", clipconfig)
        # clipconfig = CLIPConfig(config=file_path)
        # clipconfig = CLIPConfig.from_pretrained("../../../model_zoo/CLIP-ViP/clipvip-vision_config.json")  # .from_pretrained方法无法读取本地多层上级目录的配置文件
        # 创建clip_vision_additional_config对象
        clip_vision_additional_config = ClipVisionAdditionalConfig(
            type="ViP",
            temporal_size=12,
            if_use_temporal_embed=1,
            logit_scale_init_value=4.60,
            add_cls_num=3  # cls token在此处不参与attention计算，多少都无所谓
        )
        setattr(clipconfig, "vision_additional_config", clip_vision_additional_config)
        self.vision_additional_config = clip_vision_additional_config
        # 加载配置
        # self.clipmodel = CLIPModel.from_pretrained(path, config=clipconfig)
        self.model = CLIPModel(config=clipconfig)
        # print("CLIPModel加载完成")

    # 对视频进行前向传播获取视频特征
    def forward_video(self, video):
        inputs = {"pixel_values": video,
                  "if_norm": True}
        # 获取到[B,512]和最后一层hiddenstate[B,2352,512]
        video_features, last_hidden_state = self.model.get_image_features(**inputs)
        # 只需要hiddenstate
        return last_hidden_state


# 上：推理阶段模型使用的类定义，初始化，参考CLIP-ViP项目中inference.py
# ---------------------------------------------------------------------------------------
# 下：定义最外层包装的主塔类

class CLIPViPVisionTower(nn.Module): # nn.Module类后续才能被移动张量到上面
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        # 初始化是否已加载标志
        self.is_loaded = False

        # 设置视觉塔名称、优化标志等属性
        self.vision_tower_path = vision_tower
        self.is_optimize = getattr(args, 'optimize_vision_tower', False)

        # 如果不延迟加载或者需要解冻视觉塔，则加载模型，否则仅加载配置
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            pass
            # 应当加载 clip-vip 的配置 可能只需要vision encoder only
            # self.cfg_only = CLIPModel(config="...../model_zoo/CLIP-ViP/clipvip-vision_config.json")
            # print("CLIPViPVisionTower【delay_load=True】,但是不执行延迟加载，会在使用load_model方法是一并加载配置文件和权重")
            # print("CLIPViPVisionTower模型初始化")


    # 加载模型方法
    def load_model(self):
        # 直接加载权重到当前实例上
        self.vision_tower = VidCLIP()

        # 如果是预训练权重则直接加载模型
        if "pretrain_clipvip_base_16" in self.vision_tower_path:
            loaded_state_dict = torch.load(self.vision_tower_path)
            # self.vision_tower.load_state_dict(loaded_state_dict, strict=False)
            # print("-CLIPViP 权重载入完成-")
        else:  # 非预训练权重
            # 如果自己在分布式保存的模型权重文件中字段包含module.前缀导致模型不匹配则需要一下修剪字段
            # print("-CLIPViP 正在载入非预训练权重并获取权重字段-")
            loaded_state_dict = torch.load(self.vision_tower_path)
            loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
            # print("-CLIPViP 权重字段修改完成-")
            # self.vision_tower.load_state_dict(loaded_state_dict, strict=False)
            # print("-CLIPViP 权重载入完成-")

        loaded_state_dict = {k.replace('clipmodel', 'model'): v for k, v in loaded_state_dict.items()}
        self.vision_tower.load_state_dict(loaded_state_dict, strict=False)

        # state_dict = self.vision_tower.state_dict()
        # last_layer_weight = state_dict["model.vision_model.encoder.layers.11.self_attn.v_proj.weight"]
        # print(f"vision_tower加载后权重张量: \n{last_layer_weight}")

        # # 移动到设备上
        # self.vision_tower.to(device)
        # print("主塔移动到device上")

        self.vision_tower.requires_grad_(False)

        # 设置加载标志为 True
        self.is_loaded = True


    # 视频前向传播方法
    def video_forward(self, videos):
        # 如果输入为列表，则分别对每个视频进行前向传播
        if type(videos) is list:
            video_features = []
            for video in videos:
                video_features = video  # 传入的已经是视频特征了
                # print("video_features", video_features.shape)
                video_features = self.vision_tower.forward_video(video_features)
                video_features.append(video_features)
        # 否则对单个视频进行前向传播
        else:
            video_features = videos  # 传入的已经是视频特征了
            # print("video_features", video_features.shape)
            video_features = self.vision_tower.forward_video(video_features)
        return video_features


    def forward(self, videos):
        # 如果不需要优化，则使用无梯度更新模式
        if not self.is_optimize:
            with torch.no_grad():
                video_features = self.video_forward(videos)
        # 否则使用正常模式
        else:
            video_features = self.video_forward(videos)

        return video_features

    # 返回虚拟特征属性
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    # 返回数据类型属性
    @property
    def dtype(self):
        return self.vision_tower.dtype

    # 返回设备属性
    @property
    def device(self):
        return self.vision_tower.device

    # 返回配置属性
    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    # 返回隐藏尺寸属性
    @property
    def hidden_size(self):
        return self.config.hidden_size

    # 返回补丁数量属性
    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2