CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

# 日志目录，默认为当前目录
LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100  # 忽略的索引值
IMAGE_TOKEN_INDEX = -200  # 图像标记的索引值
VIDEO_TOKEN_INDEX = -200  # 视频标记索引值暂定和图像一致
PREDICT_TOKEN_INDEX = -300  # 预测标记的索引值
DEFAULT_IMAGE_TOKEN = "<image>"  # 默认的图像标记
DEFAULT_VIDEO_TOKEN = "<video>"   # 默认的视频标记
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"  # 默认的图像补丁标记
DEFAULT_IM_START_TOKEN = "<im_start>"  # 默认的图像开始标记
DEFAULT_IM_END_TOKEN = "<im_end>"  # 默认的图像结束标记
IMAGE_PLACEHOLDER = "<image-placeholder>"  # 图像占位符标记
DEFAULT_PREDICT_TOKEN = "<predict>"  # 默认的预测标记

# 描述性提示语，用于生成描述性文本
DESCRIPT_PROMPT = [
    "Describe this image thoroughly.",
    "Provide a detailed description in this picture.",
    "Detail every aspect of what's in this picture.",
    "Explain this image with precision and detail.",
    "Give a comprehensive description of this visual.",
    "Elaborate on the specifics within this image.",
    "Offer a detailed account of this picture's contents.",
    "Describe in detail what this image portrays.",
    "Break down this image into detailed descriptions.",
    "Provide a thorough description of the elements in this image."]