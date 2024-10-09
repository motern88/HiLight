import os
from .clipvip_encoder import CLIPViPVisionTower
from .longclip_encoder import LongCLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    # 获取视觉塔模型路径和图像处理器路径
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    # 如果视觉塔模型路径不存在，则抛出异常
    if not os.path.exists(vision_tower):
        raise ValueError(f'找不到视觉主塔: {vision_tower}')
    return CLIPViPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

def build_vision_tower_aux(vision_tower_cfg, **kwargs):
    # 获取辅助视觉塔模型路径
    vision_tower_aux = getattr(vision_tower_cfg, 'mm_vision_tower_aux', getattr(vision_tower_cfg, 'vision_tower_aux', None))

    # 如果辅助视觉塔模型路径不存在，则抛出异常
    if not os.path.exists(vision_tower_aux):
        raise ValueError(f'找不到视觉副塔: {vision_tower_aux}')

    return LongCLIPVisionTower(vision_tower_aux, args=vision_tower_cfg, **kwargs)
