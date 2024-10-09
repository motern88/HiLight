"""
A controller manages distributed workers.
It sends worker addresses to clients.
"""
import argparse
import asyncio
import dataclasses
from enum import Enum, auto
import json
import logging
import time
from typing import List, Union
import threading

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import numpy as np
import requests
import uvicorn

from hilight.constants import CONTROLLER_HEART_BEAT_EXPIRATION
from hilight.utils import build_logger, server_error_msg


# 记录controller模块的日志信息,并保存到controller.log
logger = build_logger("controller", "controller.log")


class DispatchMethod(Enum):
    # 定义一个DispatchMethod枚举类，两种工作分配方法：LOTTERY（抽奖方式）和SHORTEST_QUEUE（最短队列方式）。
    LOTTERY = auto()
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, name):
        if name == "lottery":
            return cls.LOTTERY
        elif name == "shortest_queue":
            return cls.SHORTEST_QUEUE
        else:
            raise ValueError(f"Invalid dispatch method")


# 存储工作节点的信息，包括模型名称列表、速度、队列长度、心跳检查标志和最后心跳时间
@dataclasses.dataclass
class WorkerInfo:
    model_names: List[str]
    speed: int
    queue_length: int
    check_heart_beat: bool
    last_heart_beat: str


def heart_beat_controller(controller):
    # 无限循环，每隔CONTROLLER_HEART_BEAT_EXPIRATION秒检查一次工作节点的稳定性，并移除过期的工作节点。
    while True:
        time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
        controller.remove_stable_workers_by_expiration()


class Controller:
    # 负责工作节点的注册、状态获取、工作分配等。
    def __init__(self, dispatch_method: str):
        # Dict[str -> WorkerInfo]
        # 初始化工作节点信息字典和工作分配方法
        self.worker_info = {}
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)
        # 启动心跳检查线程
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_controller, args=(self,))
        self.heart_beat_thread.start()
        # 记录初始化日志
        logger.info("Init controller")

    # 用于注册新的工作节点或更新现有工作节点的信息。
    def register_worker(self, worker_name: str, check_heart_beat: bool,
                        worker_status: dict):
        if worker_name not in self.worker_info:
            logger.info(f"Register a new worker: {worker_name}")
        else:
            logger.info(f"Register an existing worker: {worker_name}")

        if not worker_status:
            worker_status = self.get_worker_status(worker_name)
        if not worker_status:
            return False

        self.worker_info[worker_name] = WorkerInfo(
            worker_status["model_names"], worker_status["speed"], worker_status["queue_length"],
            check_heart_beat, time.time())

        logger.info(f"Register done: {worker_name}, {worker_status}")
        return True

    # 用于获取指定工作节点的状态信息。
    def get_worker_status(self, worker_name: str):
        try:
            r = requests.post(worker_name + "/worker_get_status", timeout=5)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {worker_name}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {worker_name}, {r}")
            return None

        return r.json()

    # 用于从系统中移除指定的工作节点。
    def remove_worker(self, worker_name: str):
        del self.worker_info[worker_name]

    # 用于刷新所有工作节点的信息。
    def refresh_all_workers(self):
        old_info = dict(self.worker_info)
        self.worker_info = {}

        for w_name, w_info in old_info.items():
            if not self.register_worker(w_name, w_info.check_heart_beat, None):
                logger.info(f"Remove stale worker: {w_name}")

    # 用于列出所有工作节点支持的模型名称。
    def list_models(self):
        model_names = set()

        for w_name, w_info in self.worker_info.items():
            model_names.update(w_info.model_names)

        return list(model_names)

    # 用于根据分配方法和模型名称获取工作节点的地址。
    def get_worker_address(self, model_name: str):
        # 检查当前的分配策略。
        if self.dispatch_method == DispatchMethod.LOTTERY:
            # 初始化存储工作节点名称和速度的列表。
            worker_names = []
            worker_speeds = []
            # 遍历self.worker_info中的每个工作节点及其信息。
            for w_name, w_info in self.worker_info.items():
                # 如果工作节点支持指定的模型名称，则将其名称和速度添加到列表中。
                if model_name in w_info.model_names:
                    worker_names.append(w_name)
                    worker_speeds.append(w_info.speed)
            # 将工作节点速度转换为NumPy数组，并计算总和。
            worker_speeds = np.array(worker_speeds, dtype=np.float32)
            norm = np.sum(worker_speeds)
            # 如果所有工作节点的速度之和接近于0，则没有可用的工作节点，返回空字符串。
            if norm < 1e-4:
                return ""
            # 将工作节点速度归一化
            worker_speeds = worker_speeds / norm
            if True:  # 直接返回地址Directly return address
                # 直接根据归一化的速度随机选择一个工作节点，并返回其名称。
                pt = np.random.choice(np.arange(len(worker_names)),
                    p=worker_speeds)
                worker_name = worker_names[pt]
                return worker_name

            # 如果需要在返回前检查工作节点状态，则进入以下循环。
            # 在循环中，随机选择一个工作节点，检查其状态。
            # 如果状态检查成功，则返回该工作节点名称。
            # 如果状态检查失败，则将该工作节点从列表中移除，并更新速度列表。
            # 如果所有工作节点都被移除，则返回空字符串
            while True:
                pt = np.random.choice(np.arange(len(worker_names)),
                                      p=worker_speeds)
                worker_name = worker_names[pt]

                if self.get_worker_status(worker_name):
                    break
                else:
                    self.remove_worker(worker_name)
                    worker_speeds[pt] = 0
                    norm = np.sum(worker_speeds)
                    if norm < 1e-4:
                        return ""
                    worker_speeds = worker_speeds / norm
                    continue
            return worker_name
        # 如果分配策略是SHORTEST_QUEUE（最短队列）。
        elif self.dispatch_method == DispatchMethod.SHORTEST_QUEUE:
            worker_names = []
            worker_qlen = []
            # 遍历self.worker_info中的每个工作节点及其信息。
            for w_name, w_info in self.worker_info.items():
                # 如果工作节点支持指定的模型名称，则将其名称和处理能力/队列长度添加到列表中。
                if model_name in w_info.model_names:
                    worker_names.append(w_name)
                    worker_qlen.append(w_info.queue_length / w_info.speed)
            # 如果没有工作节点支持指定的模型名称，则返回空字符串。
            if len(worker_names) == 0:
                return ""
            # 计算并找到具有最短队列的工作节点的索引。
            min_index = np.argmin(worker_qlen)
            w_name = worker_names[min_index]
            # 更新该工作节点的队列长度。
            self.worker_info[w_name].queue_length += 1
            logger.info(f"names: {worker_names}, queue_lens: {worker_qlen}, ret: {w_name}")
            # 返回具有最短队列的工作节点名称
            return w_name
        else:
            # 如果分配策略不是LOTTERY或SHORTEST_QUEUE，则抛出异常
            raise ValueError(f"Invalid dispatch method: {self.dispatch_method}")

    # 用于接收并处理工作节点的心跳信号。
    def receive_heart_beat(self, worker_name: str, queue_length: int):
        if worker_name not in self.worker_info:
            logger.info(f"Receive unknown heart beat. {worker_name}")
            return False

        self.worker_info[worker_name].queue_length = queue_length
        self.worker_info[worker_name].last_heart_beat = time.time()
        logger.info(f"Receive heart beat. {worker_name}")
        return True

    # 用于移除由于长时间未发送心跳而过期的工作节点。
    def remove_stable_workers_by_expiration(self):
        # 计算当前时间与心跳信号过期时间的差值，得到过期阈值
        expire = time.time() - CONTROLLER_HEART_BEAT_EXPIRATION
        # 初始化一个列表，用于存储需要被移除的工作节点名称。
        to_delete = []
        # 遍历self.worker_info中的每个工作节点及其信息。
        for worker_name, w_info in self.worker_info.items():
            # 检查工作节点是否需要发送心跳信号，以及其最后一次心跳信号是否已过期。
            if w_info.check_heart_beat and w_info.last_heart_beat < expire:
                # 如果已过期，则将该工作节点名称添加到to_delete列表中。
                to_delete.append(worker_name)

        # 遍历to_delete列表，移除其中的每个工作节点
        for worker_name in to_delete:
            # 调用remove_worker方法从系统中移除指定的工作节点。
            self.remove_worker(worker_name)

    # 用于处理生成数据流的请求。
    def worker_api_generate_stream(self, params):
        # 根据传入的模型名称参数，调用get_worker_address方法获取对应工作节点的地址。
        worker_addr = self.get_worker_address(params["model"])
        # 如果获取的工作节点地址无效，则记录日志并返回错误信息。
        if not worker_addr:
            logger.info(f"no worker: {params['model']}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            # 生成一个包含错误信息的响应，并以流的形式返回
            yield json.dumps(ret).encode() + b"\0"

        # 尝试通过工作节点地址调用worker_generate_stream接口，并设置超时时间为5秒。
        try:
            response = requests.post(worker_addr + "/worker_generate_stream",
                json=params, stream=True, timeout=5)
            # 遍历响应中的每个数据块。
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                # 如果当前数据块不为空，则将其作为流的一部分发送。
                if chunk:
                    yield chunk + b"\0"
        # 如果请求过程中发生异常（例如超时或连接失败），则记录日志并返回错误信息。
        except requests.exceptions.RequestException as e:
            logger.info(f"worker timeout: {worker_addr}")
            ret = {
                "text": server_error_msg,  # 服务器错误消息
                "error_code": 3,  # 错误代码
            }
            # 生成一个包含错误信息的响应，并以流的形式返回。
            yield json.dumps(ret).encode() + b"\0"

    # 让控制器充当工作节点，以实现分层管理。这可以用于连接孤立的子网络。
    # 用于获取当前工作节点的状态信息，这个功能可以用于层级管理，使得Controller可以作为一个工作节点连接到子网络中。
    def worker_api_get_status(self):
        model_names = set()
        speed = 0
        queue_length = 0

        # 遍历self.worker_info中的每个工作节点名称
        for w_name in self.worker_info:
            # 调用get_worker_status方法获取当前工作节点的状态信息
            worker_status = self.get_worker_status(w_name)
            # 如果获取的状态信息不为空，则进行处理。
            if worker_status is not None:
                # 将获取到的模型名称添加到集合中，以确保模型名称的唯一性
                model_names.update(worker_status["model_names"])
                # 累加所有工作节点的速度。
                speed += worker_status["speed"]
                # 累加所有工作节点的队列长度。
                queue_length += worker_status["queue_length"]

        return {
            "model_names": list(model_names),
            "speed": speed,  # 所有工作节点速度的总和。
            "queue_length": queue_length,  # 所有工作节点队列长度的总和。
        }


app = FastAPI()


@app.post("/register_worker")
async def register_worker(request: Request):
    '''
    当接收到POST请求时，它会从请求的JSON数据中提取worker_name、check_heart_beat和可选的worker_status字段，
    并将它们传递给controller.register_worker方法以完成注册。
    '''
    data = await request.json()  # 从请求中获取JSON数据
    controller.register_worker(  # 调用controller中的register_worker
        data["worker_name"], data["check_heart_beat"],
        data.get("worker_status", None))  # 使用请求中的worker_name和check_heart_beat，以及可选的worker_status


@app.post("/refresh_all_workers")
async def refresh_all_workers():
    '''
    用于刷新所有工作节点的状态。
    当接收到POST请求时，它会调用controller.refresh_all_workers方法，该方法负责更新工作节点的信息。
    '''
    models = controller.refresh_all_workers()   # 调用controller中的refresh_all_workers方法


@app.post("/list_models")
async def list_models():
    '''
    用于列出所有可用的模型。
    当接收到POST请求时，它会调用controller.list_models方法，并将返回的模型列表作为响应返回。
    '''
    models = controller.list_models()
    return {"models": models}


@app.post("/get_worker_address")
async def get_worker_address(request: Request):
    '''
    用于获取特定模型的工作节点地址。
    当接收到POST请求时，它会从请求的JSON数据中提取model字段，
    并调用controller.get_worker_address方法以获取对应的工作节点地址。
    '''
    data = await request.json()
    addr = controller.get_worker_address(data["model"])
    return {"address": addr}


@app.post("/receive_heart_beat")
async def receive_heart_beat(request: Request):
    '''
    用于接收工作节点的心跳信号。
    当接收到POST请求时，它会从请求的JSON数据中提取worker_name和queue_length字段，
    并将它们传递给controller.receive_heart_beat方法以更新工作节点的状态
    '''
    data = await request.json()
    exist = controller.receive_heart_beat(
        data["worker_name"], data["queue_length"])
    return {"exist": exist}


@app.post("/worker_generate_stream")
async def worker_api_generate_stream(request: Request):
    '''
    用于生成一个数据流。
    当接收到POST请求时，它会从请求的JSON数据中提取参数，并调用controller.worker_api_generate_stream方法。
    该方法返回一个可迭代对象，StreamingResponse用于将这个可迭代对象作为流式数据发送给客户端。
    '''
    params = await request.json()
    generator = controller.worker_api_generate_stream(params)
    return StreamingResponse(generator)


@app.post("/worker_get_status")
async def worker_api_get_status(request: Request):
    '''
    用于获取工作节点的状态。
    当接收到POST请求时，它会调用controller.worker_api_get_status方法，并将其返回的结果作为响应返回。
    '''
    return controller.worker_api_get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21001)
    parser.add_argument("--dispatch-method", type=str, choices=[
        "lottery", "shortest_queue"], default="shortest_queue")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    controller = Controller(args.dispatch_method)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
