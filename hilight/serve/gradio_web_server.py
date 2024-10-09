import argparse
import datetime
import json
import os
import time

import gradio as gr
import requests

from hilight.conversation import (default_conversation, conv_templates,
                                        SeparatorStyle)
from hilight.constants import LOGDIR
from hilight.utils import (build_logger, server_error_msg,
    violates_moderation, moderation_msg)
import hashlib


logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "HiLight Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}

# 根据当前日期创建一个新的日志文件名，用于存储对话内容。
def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name

# 从控制器获取所有模型的列表，并根据优先级排序。
def get_model_list():
    # 向控制器发送请求以刷新工作节点信息
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    # 确保请求成功
    assert ret.status_code == 200
    # 从控制器获取模型列表
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    # 根据优先级字典对模型列表进行排序
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""

# 加载一个示例对话，根据URL参数设置下拉菜单的默认值。
def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown(visible=True)
    # 根据URL参数中的"model"设置下拉菜单的默认值
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(value=model, visible=True)
    # 返回初始对话状态和下拉菜单
    state = default_conversation.copy()
    return state, dropdown_update

# 刷新模型列表并更新下拉菜单。
def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    # 更新下拉菜单的选项和默认值
    dropdown_update = gr.Dropdown(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    # 返回更新后的对话状态和下拉菜单
    return state, dropdown_update

# 记录用户对最后一条响应的投票（点赞或点踩）。
def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    # 记录投票信息到日志文件
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

# 分别处理用户对最后一条响应的点赞和点踩操作。
def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    # 记录点赞信息
    vote_last_response(state, "upvote", model_selector, request)
    # 返回更新后的对话状态和按钮
    return ("",) + (disable_btn,) * 3
def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    # 记录点踩信息
    vote_last_response(state, "downvote", model_selector, request)
    # 返回更新后的对话状态和按钮
    return ("",) + (disable_btn,) * 3

# 记录用户对最后一条响应的举报操作。
def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3

# 处理用户请求重新生成对话内容的操作。
def regenerate(state, video_process_mode, request: gr.Request):
    # 记录日志
    logger.info(f"regenerate. ip: {request.client.host}")
    # 清除最后一条消息
    state.messages[-1][-1] = None
    # 获取上一条人类用户的消息
    prev_human_msg = state.messages[-2]
    # 如果上一条消息包含图片，更新图片处理模式
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], video_process_mode)
    # 重置跳过标志
    state.skip_next = False
    # 返回更新后的对话状态和按钮
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5

# 处理用户请求清除对话历史的操作
def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    # 重置对话状态为默认值
    state = default_conversation.copy()
    # 返回重置后的对话状态和按钮
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5

# 处理用户输入文本和视频的操作
def add_text(state, text, video, video_process_mode, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    # 检查输入是否有效
    if len(text) <= 0 and video is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None) + (
                no_change_btn,) * 5

    text = text[:1536]  # Hard cut-off
    if video is not None:
        text = text[:1200]  # Hard cut-off for videos
        if '<video>' not in text:
            # text = '<video><video></video>' + text
            text = text + '\n<video>'
        text = (text, video, video_process_mode)
        if len(state.get_videos(return_pil=True)) > 0:
            state = default_conversation.copy()
    # 如果输入有效，更新对话状态
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    # 重置跳过标志
    state.skip_next = False
    # 返回更新后的对话状态和按钮
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5

# 处理与机器学习模型的交互，生成响应并更新界面。
def http_bot(state, model_selector, temperature, top_p, max_new_tokens, use_ocr, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    # 获取当前时间戳
    start_tstamp = time.time()
    # 获取模型名称
    model_name = model_selector

    if state.skip_next:
        # 由于输入无效，跳过生成调用
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # 第一轮对话
        # 设置对话模板
        new_state = conv_templates["gemma"].copy()
        # 将上一条消息添加到新状态中
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        # 将新状态设置为当前状态
        state = new_state

    # 查询工作节点地址
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # 如果没有可用的工作节点
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    # 构造提示
    prompt = state.get_prompt()
    # 处理图像
    all_videos = state.get_videos(return_pil=True)
    all_video_hash = [hashlib.md5(video.tobytes()).hexdigest() for video in all_videos]
    # 保存视频到文件系统
    for video, hash in zip(all_videos, all_video_hash):
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_videos", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            video.save(filename)


    # 构造请求负载
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "videos": f'List of {len(state.get_videos())} videos: {all_video_hash}',
        "use_ocr": bool(use_ocr == 'Yes'),
    }
    logger.info(f"==== request ====\n{pload}")

    pload['videos'] = state.get_videos()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    # 发送请求
    try:
        # 流式输出
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=30)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    # 如果响应中包含文本
                    if 'video' not in data.keys():
                        output = data["text"][len(prompt):].strip()
                        state.messages[-1][-1] = output + "▌"
                    else:
                        output = (data["text"][len(prompt):].strip(), data["video"])
                        state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    # 如果响应中包含错误
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        # 如果请求过程中出现异常
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    if type(state.messages[-1][-1]) is not tuple:
        state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    # 记录对话日志
    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "videos": all_video_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

# 定义了页面的标题，使用Markdown的标题语法，显示为一级标题。
title_markdown = ("""
# HiLight: Motern Multi-modality Vision Language Model
""")

# 定义了服务的使用条款，使用Markdown的标题和列表语法，显示为二级标题和列表项。
tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a preview intended for motern use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
""")

# 提供了有关许可证、使用条款和隐私实践的更多信息，使用Markdown的标题和链接语法。
learn_more_markdown = ("""
### License
This model is a closed-source preview version for internal personnel use only.
""")

# 包含了CSS样式，用于设置页面中按钮的最小宽度
block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""

# 定义函数，用于构建一个交互式示例应用
def build_demo(embed_mode, cur_dir=None, concurrency_count=10):
    # 创建一个文本框组件，不显示标签，占位符文本为"Enter text and press ENTER"
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    # 使用 Gradio.Blocks 创建一个应用界面，标题为"HiLight"
    with gr.Blocks(title="HiLight", theme=gr.themes.Default(), css=block_css) as demo:
        # 初始化 Gradio.State 对象，用于管理应用状态
        state = gr.State()

        # 如果不是嵌入模式，则显示应用标题的Markdown格式文本
        if not embed_mode:
            gr.Markdown(title_markdown)

        # 创建一个包含多个组件的行
        with gr.Row():
            # 创建一个包含多个组件的列，缩放比例为3
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    # 创建一个下拉菜单组件，用于选择模型
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)

                # 创建一个用于显示和处理图像的组件
                videobox = gr.Video()
                # 创建一个单选按钮组件，用于选择非正方形图像的预处理方式
                video_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square video", visible=False)

                # 如果当前目录未指定，则使用当前文件的目录
                if cur_dir is None:
                    cur_dir = os.path.dirname(os.path.abspath(__file__))
                # 创建一个示例组件，提供预设的图像和文本输入
                gr.Examples(examples=[
                    [f"{cur_dir}/examples/monday.jpg", "Explain why this meme is funny, and generate a picture when the weekend coming."],
                    [f"{cur_dir}/examples/woolen.png", "Show me one idea of what I could make with this?"],
                    [f"{cur_dir}/examples/extreme_ironing.jpg", "What is unusual about this video?"],
                    [f"{cur_dir}/examples/waterview.jpg", "What are the things I should be cautious about when I visit here?"],
                ], inputs=[videobox, textbox])


                with gr.Accordion("Function", open=True) as parameter_row:
                    # 创建一个折叠面板组件，用于显示和选择是否使用OCR
                    use_ocr = gr.Radio(choices=['Yes', 'No'], value='Yes', interactive=True, label="Use OCR")

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)

            # 创建另一个列，缩放比例为7，用于显示聊天机器人界面
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="HiLight Chatbot",
                    height=940,
                    layout="panel",
                )
                # 创建一个行，包含文本框和提交按钮
                with gr.Row():
                    with gr.Column(scale=7):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                # 创建一个行，包含点赞、点踩、举报、重新生成和清除历史的按钮
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

        # 如果不是嵌入模式，则显示服务条款和了解更多信息的Markdown格式文本
        if not embed_mode:
            gr.Markdown(function_markdown)
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        # 创建一个不可见的JSON组件，用于传递URL参数
        url_params = gr.JSON(visible=False)

        # 注册事件监听器
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        # 为点赞按钮注册点击事件，调用upvote_last_response函数
        upvote_btn.click(
            upvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )
        # 为点踩按钮注册点击事件，调用downvote_last_response函数
        downvote_btn.click(
            downvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )
        # 为举报按钮注册点击事件，调用flag_last_response函数
        flag_btn.click(
            flag_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )

        # 为重新生成按钮注册点击事件，调用regenerate函数，然后调用http_bot函数
        regenerate_btn.click(
            regenerate,
            [state, video_process_mode],
            [state, chatbot, textbox, videobox] + btn_list
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, use_ocr],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )

        # 为清除历史按钮注册点击事件，调用clear_history函数
        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, videobox] + btn_list,
            queue=False
        )

        # 为文本框的提交事件注册add_text函数，然后调用http_bot函数
        textbox.submit(
            add_text,
            [state, textbox, videobox, video_process_mode],
            [state, chatbot, textbox, videobox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, use_ocr],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )

        # 为提交按钮注册点击事件，调用add_text函数
        submit_btn.click(
            add_text,
            [state, textbox, videobox, video_process_mode],
            [state, chatbot, textbox, videobox] + btn_list
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, use_ocr],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )

        # 根据模型列表模式，加载示例或刷新模型列表
        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                _js=get_window_url_params
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=16)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    # 构建Gradio交互式示例应用，传入是否嵌入模式和并发计数参数。
    demo = build_demo(args.embed, concurrency_count=args.concurrency_count)
    # API开放状态,launch方法启动Gradio应用，传入服务器名称、端口号和分享参数。
    demo.queue(
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )