"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
from functools import partial

from hilight.model.processor.hilight_video_processor import load_video
from hilight.constants import WORKER_HEART_BEAT_INTERVAL
from hilight.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from hilight.mm_utils import tokenizer_video_token
from hilight.constants import VIDEO_TOKEN_INDEX, DEFAULT_VIDEO_TOKEN
from transformers import TextIteratorStreamer
from threading import Thread

import io
import base64

# 定义GB常量，用于内存计算
GB = 1 << 30

# 生成一个唯一的工作节点ID
worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0  # 全局计数器，可能用于跟踪请求数量或其他统计信息

model_semaphore = None  # 定义信号量，用于控制并发访问

# 定义心跳工作函数，用于定期向控制器发送心跳信号
def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)  # 等待指定的心跳间隔
        controller.send_heart_beat()  # 发送心跳信号到控制器

def load_pretrained_model(model_args,device):
    from hilight.model.language_model.hilight_gemma import HiLightGemmaForCausalLM
    from transformers import (
        AutoTokenizer,
    )
    # 实例化模型
    model = HiLightGemmaForCausalLM.from_pretrained(
        "work_dirs/HiLight-2B",
        cache_dir="work_dirs/HiLight-2B",
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
    vision_tower.to(device)
    vision_tower_aux.to(device)

    # token_mining初始化，如果存在训练好的token_mining权重，会直接加载，这里加上一层判断选择是否加载
    model.get_model().initialize_uni_modules(
        model_args=model_args
    )
    print("-Hilight_TokenMining .pt文件权重加载完成-")

    # 将模型移动到指定设备
    model.to(device)

    return tokenizer, model

# 定义ModelWorker类，用于加载和运行模型
class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_args, device):
        # 控制器地址和工作节点地址
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        # 工作节点ID
        self.worker_id = worker_id

        self.vision_token_len = model_args.vision_token_len # 一段视频的视觉token数长度，由于load_video是12帧，因此token数长度也是12帧

        self.model_args = model_args
        # 模型路径和名称
        if self.model_args.model_path.endswith("/"):
            model_path = self.model_args.model_path[:-1]
        if self.model_args.model_name is None:
            model_paths = self.model_args.model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = self.model_args.model_name
        # 设备信息
        self.device = device
        # 日志信息
        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        # 加载预训练模型
        self.tokenizer, self.model = load_pretrained_model(model_args=model_args, device=device)

        # 如果不是no_register，则向控制器注册工作节点，并启动心跳线程
        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    # 注册到控制器的方法
    def register_to_controller(self):
        logger.info("Register to controller")
        # 构建注册请求的URL
        url = self.controller_addr + "/register_worker"
        # 准备注册数据
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        # 发送POST请求到控制器
        r = requests.post(url, json=data)
        # 断言响应状态码为200，表示注册成功
        assert r.status_code == 200

    def send_heart_beat(self):
        # 记录日志，包括模型名称、信号量状态和全局计数器值
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        # 构建心跳接收的URL
        url = self.controller_addr + "/receive_heart_beat"

        # 循环尝试发送心跳，直到成功或发生不可恢复的错误
        while True:
            try:
                # 发送POST请求，包含工作节点名称和队列长度
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=30)
                # 从响应中获取工作节点是否存在的标志
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                # 如果发生请求异常，记录错误日志
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        # 如果工作节点不存在，则重新注册到控制器
        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        # 如果信号量未初始化，则队列长度为0
        if model_semaphore is None:
            return 0
        else:
            # 计算并返回队列长度
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        # 返回包含工作节点状态信息的字典
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }
    
    def add_content(self, prompt, new_content):
        # 根据特定的标记找到提示文本的分割点
        if '[INST]' in prompt:
            split_index = prompt.rfind(' [/INST]')
        else:
            split_index = prompt.rfind('###Assistant:')
        # 分割原始提示文本为左右两部分，并在中间插入新内容
        left_prompt = prompt[:split_index]
        right_prompt = prompt[split_index:]
        prompt = left_prompt + new_content + right_prompt
        # 返回更新后的提示文本
        return prompt

    # 使用torch.inference_mode上下文管理器来优化推理性能
    @torch.inference_mode()
    def generate_stream(self, params):
        # 获取tokenizer, model, image_processor实例
        tokenizer, model = self.tokenizer, self.model
        # 获取参数中的提示
        prompt = params["prompt"]
        # 保存原始提示
        ori_prompt = prompt
        # 获取参数中的图像列表，如果存在
        videos = params.get("video", None)
        # 初始化图像token的数量
        num_video_tokens = 0


        # 如果存在视频
        if videos is not None and len(videos) > 0 :  # len(videos) = 1
            if len(videos) > 0:
                # 检查提供的图像数量是否与提示中的图像标记数量相匹配
                if len(videos) != prompt.count(DEFAULT_VIDEO_TOKEN):
                    raise ValueError("Number of videos does not match number of <video> tokens in prompt")

                # 加载图像
                video_tensor = [load_video(video) for video in videos]
                # 将张量转移到模型设备上
                video_tensor = [tensor.to(self.device) for tensor in video_tensor]

                replace_token = DEFAULT_VIDEO_TOKEN
                # 计算图像中的标记数量
                num_image_tokens = prompt.count(replace_token) * self.vision_token_len
            else:
                video_tensor = None
            video_args = {"videos": video_tensor}
        else:
            video_tensor = None
            video_args = {}

        # 设置生成参数
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False

        # 将提示转换为input_ids并准备输入
        input_ids = tokenizer_video_token(prompt, tokenizer, VIDEO_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=30)

        # 根据模型的最大位置嵌入限制和图像token的数量，调整max_new_tokens
        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        # 如果max_new_tokens小于1，返回错误信息
        if max_new_tokens < 1:
            yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
            return

        # 启动一个线程来生成文本
        thread = Thread(target=model.generate, kwargs=dict(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            **video_args
        ))
        thread.start()

        # 收集生成的文本
        generated_text = ori_prompt
        for new_text in streamer:
            generated_text += new_text
            # 如果生成的文本以停止字符串结尾，移除停止字符串
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            # 将生成的文本作为响应的一部分返回
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"
        # 清空CUDA缓存
        torch.cuda.empty_cache()

    def generate_stream_gate(self, params):
        try:
            # 尝试生成数据流
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            # 捕获值错误异常并处理
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            # 捕获 CUDA 错误异常并处理
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            # 捕获其他未知异常并处理
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    # 释放模型信号量
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    # 从请求中获取参数
    params = await request.json()
    if model_semaphore is None:
        # 如果模型信号量为空，则创建一个异步信号量
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    # 获取模型信号量
    await model_semaphore.acquire()
    # 发送心跳
    worker.send_heart_beat()
    # 生成数据流
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    # 返回带有后台任务的流式响应
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    # 获取工作状态
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 解析命令行参数
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multi-modal", action="store_true", help="Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.multi_modal:
        # 如果多模态，则记录警告信息
        logger.warning("Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")

    # 创建模型工作器
    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_base,
                         args.model_name,
                         args.load_8bit,
                         args.load_4bit,
                         args.device,
                         use_flash_attn=args.use_flash_attn)
    # 运行 FastAPI 应用
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")