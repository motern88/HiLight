from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    from transformers import AutoConfig, AutoModelForCausalLM, \
                            GemmaConfig, GemmaModel, GemmaForCausalLM
except:
    print("New model not imported. Try to update Transformers to 4.38.0 or later.")
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.generation.utils import logging

from ..hilight_arch import HiLightMetaModel, HiLightMetaForCausalLM
from ..multimodal_projector.token_mining import UniLMProjectors

logger = logging.get_logger(__name__)

# 定义一个名为MiniGeminiConfig的类，它继承自GemmaConfig类
class HiLightConfig(GemmaConfig):
    # 指定模型类型为"hilight_gemma"
    model_type = "hilight_gemma"

# 定义一个名为MiniGeminiGemmaModel的类，它继承自MiniGeminiMetaModel和GemmaModel类
class HiLightGemmaModel(HiLightMetaModel, GemmaModel):
    # 设置配置类为MiniGeminiConfig
    config_class = HiLightConfig
    
    def __init__(self, config: GemmaConfig):
        # 调用父类的初始化方法，并传入配置参数
        super(HiLightGemmaModel, self).__init__(config)
        hidden_size = 768  # CLIP-ViP的通道数
        hidden_size_aux = 512  # LongCLIP的通道数
        output_hidden_size = 2048  # 语言模型的通道数
        self.token_mining = UniLMProjectors(hidden_size, hidden_size_aux, output_hidden_size)


# 定义一个名为HilightGemmaForCausalLM的类，它继承自GemmaForCausalLM和MiniGeminiMetaForCausalLM
class HiLightGemmaForCausalLM(GemmaForCausalLM, HiLightMetaForCausalLM):
    # 设置配置类为MiniGeminiConfig
    config_class = HiLightConfig

    def __init__(self, config):
        # 调用父类的初始化方法，并传入配置参数
        super(GemmaForCausalLM, self).__init__(config)
        # 创建一个MiniGeminiGemmaModel实例
        self.model = HiLightGemmaModel(config)
        # 设置词汇表大小为配置中指定的大小
        self.vocab_size = config.vocab_size
        # 创建一个线性层，用于语言模型的头部，输入维度为隐藏层大小，输出维度为词汇表大小，不使用偏置项
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取模型实例的方法
    def get_model(self):
        return self.model
    
    # 单独保存token_mining模块的方法
    def save_token_mining_weights(self, file_path):
        torch.save(self.model.token_mining.state_dict(), file_path)

    # 用于Hilight前向过程
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        videos: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:


        # print("forward position_ids",position_ids.shape)  # torch.Size([1, 20])
        # print("forward attention_mask",attention_mask.shape)  # torch.Size([1, 20])
        # print("forward past_key_values",past_key_values.shape)  # None
        # if videos is None:
        #     print("HiLight-forward：videos不存在")

        # 如果没有传入inputs_embeds，纯文本则在此处生成
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                labels=labels,
                videos=videos
            )
        # 确保张量为float16（Hilight模型的输出在这里由32转为16）,并且允许转换不成功（因为在循环中的后续对话时常出现none，而无法也不需要对none进行精度转换）
        try:
            inputs_embeds = inputs_embeds.to(torch.float16)
        except:
            pass

        # print("forward-input_ids",input_ids)  # None
        # print("forward-inputs_embeds",inputs_embeds.shape,inputs_embeds.dtype)  # torch.Size([1, 6, 2048]),torch.float16
        # print("forward-attention_mask",attention_mask)  # tensor([[1, 1, 1, 1, 1, 1]], device='cuda:0')
        # print("forward-position_ids",position_ids)  # tensor([[0, 1, 2, 3, 4, 5]], device='cuda:0')
        # print("forward-past_key_values",past_key_values)  # None
        # print("forward-labels",labels)  # None
        # print("forward-use_cache",use_cache)  # True
        # print("forward-cache_position",cache_position)  # tensor([0, 1, 2, 3, 4, 5], device='cuda:0')
        # print("forward-output_attentions",output_attentions)  # False
        # print("forward-output_hidden_states",output_hidden_states)  # False
        # print("forward-return_dict",return_dict)  # True

        # 调用父类的前向传播方法，并传入相应的参数
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    # 生成方法，用于Hilight视频问答
    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            videos: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        # 如果inputs_embeds在kwargs中，则抛出未实现异常，因为不支持inputs_embeds
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # print("inputs", inputs)  # tensor([[     2,   1841,   2652,    575,   5642, 235336]], device='cuda:0')
        # print("position_ids", position_ids)  # None
        # print("attention_mask", attention_mask)  # None

        # 如果传入了videos，则调用prepare_inputs_labels_for_multimodal方法,生成inputs_embeds
        if videos is not None:
            # print("generate video is not None")
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=inputs,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                labels=None,
                videos=videos
            )
        else:
            # print("generate video is None")
            # TODO：发现支持视频模态输入之后对话反而不支持纯文本了，不该啊，反正先在这里加上这个分支来临时弥补一下，看看后续有没有更好的方法
            inputs_embeds = self.prepare_inputs_labels_for_langmodal(input_ids=inputs)


        # 调用父类的生成方法，并传入相应的参数
        return super().generate(
            position_ids=None,
            attention_mask=None,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    # 准备生成输入的方法
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # 从kwargs中弹出images和images_aux
        images = kwargs.pop("images", None)
        images_aux = kwargs.pop("images_aux", None)
        # 调用父类的方法准备输入，并传入相应的参数
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        # 如果传入了images或images_aux，则将它们添加到_inputs字典中
        if images is not None:
            _inputs['images'] = images
        if images_aux is not None:
            _inputs['images_aux'] = images_aux
        # 返回准备后的输入字典
        return _inputs

# 注册HiLightConfig和HiLightGemmaForCausalLM到AutoConfig和AutoModelForCausalLM中
AutoConfig.register("hilight_gemma", HiLightConfig)
AutoModelForCausalLM.register(HiLightConfig, HiLightGemmaForCausalLM)