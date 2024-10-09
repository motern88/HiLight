import torch
from torchvision import transforms

import numpy as np
import decord
from decord import VideoReader
from decord import cpu, gpu
decord.bridge.set_bridge("torch")

def init_transform_dict_simple(video_res=(240, 320),
                        input_res=(224, 224),
                        randcrop_scale=(0.8, 1.0),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.48145466, 0.4578275, 0.40821073),
                        norm_std=(0.26862954, 0.26130258, 0.27577711)):
    # 定义图像归一化的变换
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    # 定义简化版的图像变换，包括尺寸调整、中心裁剪和归一化
    transform_dict = {
        'test': transforms.Compose([
            transforms.Resize(input_res, antialias=True, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_res),
            normalize,
        ])
    }
    return transform_dict

def get_sample_idx(total_frame_num):
    # 均匀采样12帧
    return np.linspace(0, total_frame_num-1, 12).astype(int)  ######### 均匀采样12帧
    # return np.linspace(0, total_frame_num - 1, 12).astype(int)

# 返回CLIP-ViP可以直接输入的video_features
def load_video(vis_path):
    # 使用VideoReader类加载视频，假设存在VideoReader类并且有cpu函数
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)

    # 获取样本帧的索引
    frame_idx = get_sample_idx(total_frame_num)
    # print("frame_idx",frame_idx) # [ 0  8 16 24 32 40 48 56 64 72 80 89]
    # 从视频中获取采样的图像帧数据
    img_array = vr.get_batch(frame_idx) # (n_clips*num_frm, H, W, 3)
    # print("img_array.shape",img_array.shape) # (12, 1080, 1920, 3)
    # print("img_array.type()", type(img_array)) # <class 'torch.Tensor'>

    # 调整数据维度，将通道维度移动到正确的位置，并进行归一化
    img_array = img_array.permute(0, 3, 1, 2).float() / 255.
    # 根据配置初始图像转换方法
    transform = init_transform_dict_simple(video_res=[240, 320],
                                           input_res=[224, 224])["test"]  # mode
    # 应用图像转换
    img_array = transform(img_array) # [clips*num_frm, C, H_crop, W_crop]

    return img_array