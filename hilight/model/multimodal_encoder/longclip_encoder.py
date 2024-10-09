import torch
import torch.nn as nn
from .longclip_model import longclip

class LongCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower_aux, args, delay_load=False):
        super().__init__()
        # 初始化是否已加载标志
        self.is_loaded = False

        # 设置视觉塔名称、优化标志等属性
        self.vision_tower_path = vision_tower_aux
        self.is_optimize = getattr(args, 'optimize_vision_tower_aux', False)

        # 如果不延迟加载或者需要解冻视觉塔，则加载模型，否则仅加载配置
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower_aux', False):
            self.load_model()
        else:
            pass
            # print("LongCLIPVisionTower【delay_load=True】,但是不执行延迟加载，会在使用load_model方法是一并加载配置文件和权重")
            # print("LongCLIPVisionTower模型初始化的时候加载权重")


    # 加载模型方法
    def load_model(self):
        # 直接加载权重到当前实例上
        # 此处self.image_processor没有被使用过，仅仅因为直接调用longclip而顺带返回了实例化后的一个图像处理器
        self.vision_tower, self.image_processor  = longclip.load(self.vision_tower_path,device="cpu") # 如果以cuda加载不会加载float32的权重
        # print("-LongCLIP 权重载入完成-")

        # # 移动到设备上
        # self.vision_tower.to(device)
        # print("副塔移动到device上")

        self.vision_tower.requires_grad_(False)

        # 设置加载标志为 True
        self.is_loaded = True

    # 图像前向传播方法，待改成视频前向传播
    def video_forward(self, videos):
        # 如果输入为列表，则分别对每个视频进行前向传播
        if type(videos) is list:
            video_features = []
            for video in videos:
                # video_features = load_video(vis_path=video)  # ([1, 12, 3, 224, 224])
                video_features = video  # 传入的已经是视频特征了
                # video_features = video_features.to(device=device)
                # 使用 .unbind(1) 拆分第二个维度（索引为1）
                frame_tensors = video_features.unbind(1)
                encoded_images = []
                for frame_tensor in frame_tensors:  # ([1, 3, 224, 224])
                    encoded_image = self.vision_tower.encode_image(frame_tensor)  # # torch.Size([1, 512])
                    encoded_images.append(encoded_image)
                # 将编码后的图像拼接为视频特征
                video_features = torch.stack(encoded_images, dim=1)  # ([1, 12, 512])
        # 否则对单个视频进行前向传播
        else:
            # video_features = load_video(vis_path=videos) # torch.float32
            video_features = videos # 传入的已经是视频特征了
            # video_features = video_features.to(device=device)
            # print(video_features.shape)  # torch.Size([1, 12, 3, 224, 224])
            # 使用 .unbind(1) 拆分第二个维度（索引为1）
            frame_tensors = video_features.unbind(1)
            # print("frame_tensors", frame_tensors.dtype)
            encoded_images = []
            for frame_tensor in frame_tensors:
                # print(frame_tensor.shape)  # torch.Size([1, 3, 224, 224])
                # ("frame_tensor", frame_tensor.dtype) # torch.float32
                encoded_image = self.vision_tower.encode_image(frame_tensor)
                # print("encoded_image", encoded_image.dtype) # torch.float16
                # print(encoded_image.shape)  # torch.Size([1, 512])
                encoded_images.append(encoded_image)
            # 将编码后的图像拼接为视频特征
            video_features = torch.stack(encoded_images, dim=1)
            # print(video_features.shape)  # torch.Size([1, 12, 512])
            # print("video_features", video_features.dtype)

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