from PIL import Image
from io import BytesIO
import base64

import torch
from transformers import StoppingCriteria
from hilight.constants import IMAGE_TOKEN_INDEX


# 从base64编码的字符串加载图像。
def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


# 将PIL图像对象扩展到正方形
def expand2square(pil_img, background_color):
    # 获取图像的宽度和高度
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        # 创建一个新图像，大小为宽度，背景颜色为指定颜色
        result = Image.new(pil_img.mode, (width, width), background_color)
        # 将原始图像粘贴到新图像的中心
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        # 创建一个新图像，大小为高度，背景颜色为指定颜色
        result = Image.new(pil_img.mode, (height, height), background_color)
        # 将原始图像粘贴到新图像的中心
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

# 处理一系列图像。
def process_images(images, image_processor, model_cfg):
    # 从模型配置中获取图像宽高比设置
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    # 如果设置为'pad'，则执行填充操作
    if image_aspect_ratio == 'pad':
        for image in images:
            # 将图像转换为RGB并扩展为正方形
            image = expand2square(image.convert('RGB'), tuple(int(x*255) for x in image_processor.image_mean))
            # 预处理图像并获取像素值
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        # 如果不是'pad'，则直接调用图像处理器处理图像
        return image_processor(images, return_tensors='pt')['pixel_values']
    # 确保所有处理后的图像形状一致
    if all(x.shape == new_images[0].shape for x in new_images):
        # 将图像列表堆叠为一个张量
        new_images = torch.stack(new_images, dim=0)
    return new_images

# 使用分词器将文本提示中的'video>'标记替换为特定的图像标记。
def tokenizer_video_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    # 将提示按'<video>'分割并分词
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<video>')]
    # print("prompt_chunks: ", prompt_chunks)

    # 插入分隔符
    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        # print("第一个分词块以bos_token_id开始!")
        # 如果第一个分词块以bos_token_id开始，则设置偏移量为1
        offset = 1
        # 添加第一个分词块的第一个token
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        # 将分词块添加到输入ID列表中
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            # 如果指定返回张量，则返回
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids  # 返回输入ID列表

# 从模型路径中获取模型名称。
def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    # 如果最后一个路径组件以'checkpoint-'开头
    if model_paths[-1].startswith('checkpoint-'):
        # 返回倒数第二个和最后一个路径组件的组合
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        # 否则，只返回最后一个路径组件
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    # 初始化关键词停止标准类。
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            # 获取当前关键词的ID
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                # 如果第一个token是bos_token_id，则去掉
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                # 更新最大关键词长度
                self.max_keyword_len = len(cur_keyword_ids)
            # 将关键词ID添加到列表中
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        # 输入ID列表的长度
        self.start_len = input_ids.shape[1]

    # 检查一批输出ID中是否包含任何关键词。
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 计算偏移量
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        # 将关键词ID移动到输出ID的设备
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            # 获取截断的输出ID
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            # 如果截断的输出ID与关键词ID相等
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        # 解码输出ID
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    # 检查单个输出ID中是否包含任何关键词。
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        # 遍历输出ID张量的每一行
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)