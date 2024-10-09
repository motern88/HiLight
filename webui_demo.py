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
from transformers.trainer_utils import set_seed

from hilight.model.processor.hilight_video_processor import load_video
from hilight.model.language_model.hilight_gemma import HiLightGemmaForCausalLM

import gradio as gr


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def load_pretrained_model(model_args, device, model_weight):
    # 实例化模型
    model = HiLightGemmaForCausalLM.from_pretrained(
        model_weight, # work_dirs/HiLight-2B-LoRA-Merge-stage2-M101K-2epoch
        cache_dir=model_weight,
        attn_implementation=None,
        torch_dtype=(None),
    )
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
        fsdp=None
    )
    # 获取初始化后的视觉塔模块
    vision_tower = model.get_vision_tower()
    vision_tower_aux = model.get_vision_tower_aux()
    # 根据训练参数将视觉塔模块转换到相应的数据类型和设备
    vision_tower.to(device)
    vision_tower_aux.to(device)

    model.eval()
    model.to(device)  # 将模型移动到指定设备

    return tokenizer, model

@dataclass
class ModelArguments:
    # 定义模型相关的参数，如模型名称、版本、是否冻结骨干网络等
    model_name_or_path: Optional[str] = field(default="model_zoo/LLM/gemma/gemma-2b-it")
    # 注意HiLight-2B中config里配置的mm_vision_tower优先级更高
    vision_tower: Optional[str] = field(default="model_zoo/CLIP-ViP/pretrain_clipvip_base_16.pt")  # 视觉塔（vision_tower）的名称或路径。视觉塔是模型中用于处理视觉信息的部分
    vision_tower_aux: Optional[str] = field(default="model_zoo/Long-CLIP/longclip-B.pt")  # 辅助视觉塔（auxiliary vision tower）的名称或路径

# 重置Gradio界面的元素和状态。
def gradio_reset(history, video_list, video_path_list, chating_videos_id, history_chat_videos_id, conv):
    if history is not None:
        history = []
    if video_list is not None:
        video_list = []
    if video_path_list is not None:
        video_path_list = []
    if chating_videos_id is not None:
        chating_videos_id = []
    if history_chat_videos_id is not None:
        history_chat_videos_id = []

    conv.messages = []
    conv.offset = 0

    # 返回一系列用于更新Gradio界面的值和控件状态。
    return None, \
           conv, \
           gr.update(value=None, interactive=True, visible=True), \
           gr.update(value=None, placeholder='请先上传并读取至少一个视频', interactive=False), \
           gr.update(value="读取视频", interactive=True), \
           history, \
           video_list, \
           video_path_list, \
           gr.update(value=None,label="历史已读取视频(点击任意视频以作为当前视频进行预览)", visible=False), \
           chating_videos_id, \
           history_chat_videos_id

# 重置对话信息，但保留所有已读取视频
def chat_reset(history, chating_videos_id, history_chat_videos_id, conv):
    print("-重置对话（chat_reset）-")
    if history is not None:
        history = []
    if chating_videos_id is not None:
        chating_videos_id = []
    if history_chat_videos_id is not None:
        history_chat_videos_id = []

    conv.messages = []
    conv.offset = 0

    # chatbot, conv, text_input, history, chating_videos_id, history_chat_videos_id
    return None, \
           conv, \
           gr.update(value=None, interactive=True), \
           history, \
           chating_videos_id, \
           history_chat_videos_id


# 获取videos_file(路径)其中包含一个或多个视频，将每个视频进行处理并将视频特征添加在video_list
def load_and_process_videos(videos_file, video_list, video_path_list, device):
    # 确保传入的视频路径是字符串类型。
    assert isinstance(videos_file, str), "videos_file输入必须为包含视频路径的字符串"
    # 如果videos_file包含逗号，则将其分割成一个列表，每个元素是一个视频文件路径
    if ',' in videos_file:
        videos = videos_file.split(',')
    # 如果没有逗号，将videos_file作为一个单独的元素添加到videos列表中
    else:
        videos = [videos_file]

    video_path_list.extend(videos)

    # 遍历videos列表，对每个视频文件加载，返回视频特征添加在 video_list 列表中
    for _video in videos:
        video = load_video(_video)
        video = video.to(device)
        video_list.append(video)  # 这里video_list包含了所有videos的tensor了

    msg = "Received."  # 设置返回消息为"Received."
    return msg, video_list, video_path_list  # 返回接收消息信号和新增视频数量

# 处理视频上传逻辑。
def upload_video(video, text_input, history, video_list, video_path_list, device):
    print("--用户点击上传视频(upload_video)--")
    # 如果没有上传视频，则返回当前界面状态，不进行视频处理
    if video is None:
        print("-未发现视频，不进行任何处理-")
        return gr.update(), gr.update(), gr.update(), history, video_list, video_path_list, gr.update()

    # 加载并处理上传的视频，获取处理消息
    llm_message, video_list, video_path_list = load_and_process_videos(video, video_list, video_path_list, device)
    print(f"-新增视频已经添加,video_list当前总量：{len(video_list)},video_patch_list当前总量：{len(video_path_list)}-")

    print("-历史视频展示区域已更新-")
    # 返回一系列用于更新Gradio界面的控件状态和值 video, text_input, upload_button, history, video_list, video_path_list, videos_show
    return gr.update(interactive=True), \
        gr.update(placeholder='您可以输入英文,在文本任意位置"添加视频至对话"将视频信息插入此处', interactive=True), \
        gr.update(value="读取视频", interactive=True), \
        history, \
        video_list, \
        video_path_list, \
        gr.update(value=video_path_list, visible=True)
        # gr.update(choices=video_path_list, value=video_path_list[-1] if video_path_list else None, visible=True)


# 为prompt添加当前视频的video标记
def add_video_to_prompt(video, text_input, video_list, video_path_list, chating_videos_id, video_prefix, device):
    print("--为文本输入框添加video_prefix标记(add_video_to_prompt)--")
    if video is None:  # 如果没有上传视频，则不做任何操作
        return gr.update(), chating_videos_id, video_list, video_path_list, gr.update()

    # 如果video已经上传过了，在video_path_list中可以找到，则查找对应索引
    if video in video_path_list:
        print("-当前video组件中的视频已经存在于video_path_list中-")
        video_index = video_path_list.index(video)
    # 如果video还未上传过，没有同步在video_path_list与videos_show中，则手动执行上传并记录索引
    else:
        print("-当前video组件中的视频未被加载过-")
        llm_message, video_list, video_path_list = load_and_process_videos(video, video_list, video_path_list, device)
        print(
            f"-新增视频已经添加,video_list当前总量：{len(video_list)},video_patch_list当前总量：{len(video_path_list)}-")
        video_index = len(video_path_list) - 1

    chating_videos_id.append(video_index)
    text_input = f"{text_input} {video_prefix} "

    # text_input, chating_videos_id, video_list, video_path_list, videos_show
    return text_input, chating_videos_id, video_list, video_path_list, gr.update(value=video_path_list, visible=True)

# 将videos_show中被选择的文件添加到Video组件展示
def select_video(videos_show, evt: gr.SelectData):
    print(f"--监听到videos_show组件被选中--")
    selected_video_path = videos_show[evt.index]
    print(f"-videos_show输出{selected_video_path}-")
    return gr.update(value=selected_video_path)

# 定义一个函数gradio_ask，用于处理用户输入的消息并生成聊天机器人的回复。
def gradio_ask(user_message, chatbot, conv, video_prefix, chating_videos_id):
    print(f"--处理用户输入(gradio_ask)--")
    # 如果用户消息为空，则返回提示信息。
    if len(user_message) == 0:
        return gr.update(interactive=False, placeholder='输入文本不应该为空！'), chatbot, conv, chating_videos_id

    # 检查video_prefix出现在user_message中的次数是否等同于chating_videos_id列表的长度
    if user_message.count(video_prefix) == len(chating_videos_id):
        chatbot = chatbot + [[user_message, None]]  # 将用户消息添加到聊天机器人的聊天记录中。
        conv.append_message(conv.roles[0], user_message)  # 将用户消息添加到对话模板中。
        conv.append_message(conv.roles[1], None)
        user_message = ""  # 重置user_message
        return user_message, chatbot, conv, chating_videos_id

    else:  # user_message有误
        user_message = "输入有误，请确保视频标记添加后不再被修改！"
        chating_videos_id = []
        return user_message, chatbot, conv, chating_videos_id


# 定义一个函数gradio_answer，用于生成聊天机器人的回答。
def gradio_answer(chatbot, conv, history, tokenizer, model,
                  video_list, chating_videos_id, history_chat_videos_id, video_prefix,
                  num_beams, temperature, top_k, top_p,
                  max_new_tokens):
    print(f"--生成聊天机器人回答(gradio_answer)--")

    # 不需要获取对话历史组成上下文，get_prompt自带上下文
    # context = ""
    # # for pair in chatbot[:-1]:
    # #     context += f"user: {pair[0]}\nmodel: {pair[1]}\n"  # user/model 为gemma模板
    # for role,message in conv.messages[:-1]:
    #     context += f"{role}: {message}\n"

    # 获取当前对话的问题
    prompt = conv.get_prompt()  # 从对话模板中获得提示，而不是chatbot[-1][0]
    system_prompt = "You need to accurately identify and understand the video I gave you and answer my questions. If the video is missing key information that you cannot answer please let me know that you cannot answer, don't lie to me.My question is:"
    prompt = f"{system_prompt}{prompt}"
    print(f"-prompt:{prompt}-")

    def tokenizer_video_token(prompt, tokenizer, video_prefix, video_token_index=VIDEO_TOKEN_INDEX, return_tensors='pt'):
        # 将提示按video_prefix分割并分词
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(video_prefix)]
        # print("prompt_chunks: ", prompt_chunks)

        # 初始化输入ID列表
        input_ids = []

        # 如果第一个分词块有内容，并且第一个分词块的第一个token是bos_token_id，则将其添加到 input_ids TODO:什么时候需要自动添加bos_token_id???
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            input_ids.append(prompt_chunks[0][0])
            prompt_chunks[0] = prompt_chunks[0][1:]  # 移除已添加的bos_token_id
        # 遍历分割后的所有分词块，并在每个分词块之间插入图像标记索引
        for i, chunk in enumerate(prompt_chunks):
            input_ids.extend(chunk)  # 添加当前分词块
            if i < len(prompt_chunks) - 1:
                input_ids.append(video_token_index)  # 在分词块之间插入图像标记索引
        # 如果需要返回张量形式的结果，则转换为指定类型的张量
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')

        # 返回分词后的输入ID
        return input_ids

    # 将提示转换为输入ID，并添加到模型中
    input_ids = tokenizer_video_token(prompt, tokenizer, video_prefix).unsqueeze(0).to(model.device)
    # print("input_ids:",input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    history_chat_videos_id.extend(chating_videos_id)
    # print(f"history_chat_videos_id:{history_chat_videos_id},chating_videos_id:{chating_videos_id}")
    if len(history_chat_videos_id) > 0 :  # 当存在视频传入时
        print("-存在视频输入-")
        # video_list中取出本轮对话中涉及到的视频元素，堆叠为张量并移动到指定设备。
        selected_video_tensors = [video_list[index] for index in history_chat_videos_id]  # 根据索引取出对应的视频张量
        video_tensor = torch.stack(selected_video_tensors).to(model.device)  # 将这些张量堆叠成一个批次的张量
        # print("video_tensor.shape:",video_tensor.shape)
    else:  # 不存在视频传入时
        print("-纯文本输入-")
        video_tensor = None

    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,  # 文本的编码形式，通常是分词后的结果转换为模型能够理解的ID格式。
            videos=video_tensor,  # 视频tensor
            do_sample= True if temperature > 0 else False,  # 是否在文本生成过程中进行采样
            num_beams=num_beams, # num_beams 值越高，生成的序列更有可能是全局最优解，但也会增加计算量
            temperature=temperature,  # 温度参数
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,  # 最大新生成的token数
            bos_token_id=tokenizer.bos_token_id,  # 序列开始的标记ID Begin of sequence token
            eos_token_id=tokenizer.eos_token_id,  # 序列结束的标记ID End of sequence token
            pad_token_id=tokenizer.pad_token_id,  # 填充标记ID Pad token
            streamer=streamer,  # 一个TextStreamer对象，用于流式处理生成的文本，以便实时显示输出
            use_cache=True)  # 是否在生成过程中使用缓存，可以加快重复生成的速度，但可能会影响生成结果的多样性

    # 将生成的文本ID解码为文本，并更新对话模板
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    conv.messages[-1][-1] = outputs
    # 更新聊天机器人中聊天记录的回答。
    chatbot[-1][1] = outputs
    # 更新历史记录
    history.append((prompt, outputs))
    # 清空chating_videos_id
    chating_videos_id = []
    # chatbot, conv, history, video_list, chating_videos_id, history_chat_videos_id
    return chatbot, conv, history, video_list, chating_videos_id, history_chat_videos_id

# 自定义CSS样式
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;700&display=swap');

body {
    font-family: 'Noto Sans', sans-serif;
}

textarea, input {
    font-family: 'Noto Sans', sans-serif;
    font-size: 16px;
    color: #333333; /* 深灰色字体 */
}
"""

def main():
    print('初始化模型')
    # 参数解析
    parser = transformers.HfArgumentParser((ModelArguments))
    model_args, = parser.parse_args_into_dataclasses()

    disable_torch_init()  # 禁用PyTorch的初始化
    set_seed(42)

    # 设置网页的标题和描述。
    title = """<h1 align="center" style="font-size: 36px; color: #333;"> HiLight-2B试验演示 </h1>"""
    description = """<h3>此界面为HiLight视觉语言模型演示，支持基于视频的多轮对话。</h3>
    <h3 style="font-size: 17px; color: #666; line-height: 1.2;">您需要：</h3>
    <h3 style="font-size: 16px; color: #666; line-height: 1.2;">1.选择一个模型权重。</h3>
    <h3 style="font-size: 16px; color: #666; line-height: 1.2;">2.上传视频并点击"读取视频"，确保您读取过至少一个视频。</h3>
    <h3 style="font-size: 16px; color: #666; line-height: 1.2;">3.选择任意一个您读取过的视频，您可以在对话中输入文本(英文)的任意位置点击"添加视频到对话"将预览框中的相应视频信息添加在输入中。至此当您发送文本给模型的时候，模型可以根据您给出的对应视频做出反应。</h3>
    <h3 style="font-size: 17px; color: #666; line-height: 1.2;">注意事项：</h3>
    <h3 style="font-size: 16px; color: #666; line-height: 1.2;">1.在点击"添加视频到对话"后文本框中会自动添加视频标记，此时您不能修改或破坏它的完整性，否则模型将不能识别到您在此处插入了视频。</h3>
    <h3 style="font-size: 16px; color: #666; line-height: 1.2;">2.您不能通过直接在"历史已读取视频"中直接上传未读取的视频，您需要通过"上传视频"区域和"读取视频"按钮来正确调用模型的视频加载模块提前将视频缓存并记录 </h3>"""

    # 使用gr.Blocks()创建一个Gradio界面
    with gr.Blocks(css=custom_css) as demo:
        # 添加标题和描述的Markdown文本。
        gr.Markdown(title)
        gr.Markdown(description)

        # 创建一个水平排列的行，其中包含两个列。
        with gr.Row():
            # 第一个列，占0.5的宽度比例。
            with gr.Column(scale=0.5):  # 在这个列中添加不同的组件，如下拉菜单、图像上传、视频上传等。

                # 创建一个下拉菜单组件。用于更换不同权重
                model_weight = gr.Dropdown(choices=["FreezeTokenMining-M101K-2epoch", "FreezeTokenMining-VT1K-2epoch", "TuneTokenMining-M101K-3epoch", "TuneTokenMining-VT1K-2epoch"], label="选择模型权重", value=None)
                # 创建一个视频上传组件。
                video = gr.Video(label="上传视频", value=None)
                # 创建上传和重启按钮。
                upload_button = gr.Button(value="读取视频", interactive=True)
                add_video2prompt = gr.Button(value="添加视频至对话", interactive=True, variant="primary")
                chat_clear = gr.Button("重置对话 🔄")
                clear = gr.Button("重置界面 🔄")

                num_beams = gr.Slider(minimum=1, maximum=10, value=1, step=1, interactive=True, label="Num Beams")
                temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, interactive=True, label="Temperature")
                top_k = gr.Slider(minimum=0, maximum=5, value=1, step=1, interactive=True, label="Top_k")
                top_p = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05, interactive=True, label="Top_p")
                max_new_tokens = gr.Slider(minimum=64, maximum=2048, value=128, step=64, interactive=True, label="最大新增Token")

            # 第二个列。
            with gr.Column():  # 在这个列中添加状态组件和聊天机器人组件。

                # 设备组件
                device = gr.State(value=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                # 聊天记录状态
                history = gr.State(value=[])
                # 视频tensor列表状态，用于在聊天过程中累积所有上传的视频的特征，以便在生成文本时作为输入数据提供给模型。
                video_list = gr.State(value=[])
                # 视频路径列表状态，隐式用于记录聊天过程中累积所有上传的视频的路径，只是为了在videos_show组件中进行展示。
                video_path_list = gr.State(value=[])
                # 当前对话涉及到的视频位于视频列表中的索引
                chating_videos_id = gr.State(value=[])
                # 历史对话涉及到的视频位于视频列表中的索引
                history_chat_videos_id = gr.State(value=[])
                # 聊天机器人组件。
                chatbot = gr.Chatbot(label='HiLight-2B模型')
                # 设置推断对话模板
                conv = gr.State(value=conv_templates["gemma"].copy())
                # 视频标记
                video_prefix = gr.State(value="<video>")
                # 用户输入文本框。
                text_input = gr.Textbox(placeholder='请先选择一种模型权重', interactive=False)
                # with gr.Row():
                #     with gr.Column(scale=0.9):
                #         # 用户输入文本框。
                #         text_input = gr.Textbox(placeholder='请先选择一种模型权重', interactive=False)
                #     with gr.Column(scale=0.1):
                #         send_button = gr.Button(value="发送", variant="primary")
                # 用于存储模型和tokenizer的状态
                model_state = gr.State(value=None)
                tokenizer_state = gr.State(value=None)
                # 显示记录聊天过程中累积所有上传的视频
                videos_show = gr.Files(value=None,label="历史已读取视频(点击任意视频以作为当前视频进行预览)",visible=False)


        # 定义一个函数 model_weights，用于根据选择的 model_weight 更新不同的模型权重。
        def update_model_weights(model_weight, device):
            if isinstance(device, gr.State):
                device = device.value  # 从 gr.State 获取 torch.device ，避免直接传入gr.Srate组件引发报错

            # 清理显存
            torch.cuda.empty_cache()

            if model_weight == "FreezeTokenMining-M101K-2epoch":
                tokenizer, model = load_pretrained_model(model_args, device,
                                                         "work_dir/HiLight-2B-LoRA-Merge-stage2-M101K-2epoch")
            elif model_weight == "FreezeTokenMining-VT1K-2epoch":
                tokenizer, model = load_pretrained_model(model_args, device,
                                                         "work_dir/HiLight-2B-LoRA-Merge-stage2-VT1K-2epoch")
            elif model_weight == "TuneTokenMining-M101K-3epoch":
                tokenizer, model = load_pretrained_model(model_args, device,
                                                         "work_dir/autodl-tmp/HiLight-2B-LoRA-Merge-stage2-tune(TM+LLM)M101K-3epoch")
            elif model_weight == "TuneTokenMining-VT1K-2epoch":
                tokenizer, model = load_pretrained_model(model_args, device,
                                                         "work_dir/autodl-tmp/HiLight-2B-LoRA-Merge-stage2-tune(TM+LLM)VT1K-2epoch")
            else:  # 抛出错误未知权重
                raise ValueError("model_weight选择了未知权重！")

            # 切换权重之后，重置对话信息
            text_input = gr.update(value=None, placeholder='请先上传并读取至少一个视频', interactive=True)
            chatbot = gr.update(value=None)
            chating_videos_id = gr.update(value=[])

            # 返回更新后的模型和tokenizer状态
            return model, tokenizer, text_input, chatbot, chating_videos_id

        # 为下拉菜单组件 model_weight 设置变化时的回调函数 update_model_weights, 更新model_state和tokenizer_state和text_input。
        model_weight.change(update_model_weights,
                            [model_weight, device],
                            [model_state, tokenizer_state, text_input, chatbot, chating_videos_id])

        # 为上传按钮 upload_button 设置点击事件，对应视频的上传逻辑,同时需要更新videos_show视频展示区域。
        upload_button.click(upload_video,
                            [video, text_input, history, video_list, video_path_list, device],
                            [video, text_input, upload_button, history, video_list, video_path_list, videos_show])

        # 定义点击按钮时的操作
        add_video2prompt.click(add_video_to_prompt,
                               [video, text_input, video_list, video_path_list, chating_videos_id, video_prefix, device],
                               [text_input, chating_videos_id, video_list, video_path_list, videos_show])

        # 监听选择事件
        videos_show.select(select_video, videos_show, video)

        # 为文本输入框 text_input 设置提交事件，首先调用 gradio_ask 函数，然后调用 gradio_answer 函数。
        text_input.submit(
            gradio_ask,
            [text_input, chatbot, conv, video_prefix, chating_videos_id],
            [text_input, chatbot, conv, chating_videos_id]
        ).then(
            gradio_answer,
            [chatbot, conv, history, tokenizer_state, model_state,
             video_list, chating_videos_id, history_chat_videos_id, video_prefix,
             num_beams, temperature, top_k, top_p,
             max_new_tokens],
            [chatbot, conv, history, video_list, chating_videos_id, history_chat_videos_id]
        )
        # send_button.click(  # 为文本发送按钮设置相同的事件
        #     gradio_ask,
        #     [text_input, chatbot, video_prefix, chating_videos_id],
        #     [text_input, chatbot, chating_videos_id]
        # ).then(
        #     gradio_answer,
        #     [chatbot, history, tokenizer_state, model_state,
        #      video_list, chating_videos_id, video_prefix,
        #      num_beams, temperature, top_k, top_p,
        #      max_new_tokens],
        #     [chatbot, history, video_list, chating_videos_id]
        # )

        # 为清空按钮 clear 设置点击事件，调用 gradio_reset 函数重置界面状态。
        clear.click(gradio_reset,
                    [history, video_list, video_path_list, chating_videos_id, history_chat_videos_id, conv],
                    [chatbot, conv, video, text_input, upload_button, history, video_list, video_path_list, videos_show, chating_videos_id, history_chat_videos_id], queue=False)
        chat_clear.click(chat_reset,
                         [history, chating_videos_id, history_chat_videos_id, conv],
                         [chatbot, conv, text_input, history, chating_videos_id, history_chat_videos_id])

    # 启动 Gradio 界面，设置 share=True 允许分享链接，inbrowser=True 在浏览器中打开界面。
    demo.launch(share=True, inbrowser=True)

if __name__ == "__main__":
    main()