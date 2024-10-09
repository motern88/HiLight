import dataclasses
from enum import auto, Enum
from typing import List, Tuple
import base64
from io import BytesIO
from PIL import Image

# 定义一个枚举类型SeparatorStyle，包含不同的分隔符风格
class SeparatorStyle(Enum):
    """不同的分隔符风格"""
    SINGLE = auto()  # 单分隔符风格
    TWO = auto()  # 双分隔符风格
    MPT = auto()  # MPT分隔符风格
    PLAIN = auto()  # 纯文本风格
    LLAMA_2 = auto()  # LLAMA_2分隔符风格
    GEMMA = auto()  # GEMMA分隔符风格


@dataclasses.dataclass  # 使用dataclasses.dataclass装饰器，用于保存对话历史
class Conversation:
    """保存所有对话历史的类"""
    system: str  # 系统消息
    roles: List[str]  # 角色列表
    messages: List[List[str]]  # 消息列表，每个消息是一个包含角色和文本的列表
    offset: int  # 偏移量
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE  # 分隔符风格，默认为SINGLE
    sep: str = "###"  # 默认单分隔符
    sep2: str = None  # 默认双分隔符，可为空
    version: str = "Unknown"  # 版本信息，默认为"Unknown"

    skip_next: bool = False  # 是否跳过下一个消息

    # 获取格式化后的提示信息
    def get_prompt(self):
        messages = self.messages
        # 如果消息列表不为空且第一个消息包含元组，则复制消息列表
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            # 如果版本中包含'mmtag'，则对消息列表进行修改
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        # 根据分隔符风格生成格式化后的提示信息
        if self.sep_style == SeparatorStyle.SINGLE:
            # 单分隔符风格处理
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            # 双分隔符风格处理
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            # MPT分隔符风格处理
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            # LLAMA_2分隔符风格处理
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "第一个消息不应为空"
                    assert role == self.roles[0], "第一个消息应来自用户"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.GEMMA:
            # GEMMA分隔符风格处理
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += "<start_of_turn>" + role + "\n" + message + "<end_of_turn>\n" + seps[i % 2]
                else:
                    ret += "<start_of_turn>" + role + "\n"
        elif self.sep_style == SeparatorStyle.PLAIN:
            # 纯文本风格处理
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            # 如果分隔符风格无效，则抛出异常
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    # 添加新消息到对话历史
    def append_message(self, role, message):
        self.messages.append([role, message])

    # 处理图像的方法
    def process_image(self, image, image_process_mode, return_pil=False, image_format='PNG', max_len=1344, min_len=672):
        if image_process_mode == "Pad":
            # 如果处理模式为"Pad"，则将图像扩展为正方形
            def expand2square(pil_img, background_color=(122, 116, 104)):
                width, height = pil_img.size
                if width == height:
                    return pil_img  # 如果已经是正方形，则直接返回
                elif width > height:
                    # 如果宽度大于高度，则创建新图像，并将原图像粘贴到中心
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    # 如果高度大于宽度，则创建新图像，并将原图像粘贴到中心
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image)  # 调用expand2square函数处理图像
        elif image_process_mode in ["Default", "Crop"]:
            pass  # 如果处理模式为"Default"或"Crop"，则不进行处理
        elif image_process_mode == "Resize":
            # 如果处理模式为"Resize"，则将图像调整为固定大小(336x336像素)
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
        # 根据图像的最大尺寸进行调整，确保图像的最长边不超过max_len，最短边不小于min_len
        if max(image.size) > max_len:
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw  # 计算图像的宽高比
            shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))  # 计算最短边的长度
            longest_edge = int(shortest_edge * aspect_ratio)  # 计算最长边的长度
            W, H = image.size  # 获取当前图像的宽度和高度
            if H > W:  # 如果高度大于宽度，则调整图像尺寸
                H, W = longest_edge, shortest_edge
            else:
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))  # 调整图像大小
        # 根据return_pil参数的值决定返回什么
        if return_pil:
            return image
        else:
            # 如果return_pil为False，则将图像保存到BytesIO对象，并返回其base64编码字符串
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str  # 返回base64编码字符串

    # 从对话历史中提取图像，并根据需要处理它们
    def get_images(self, return_pil=False):
        images = []
        # 遍历对话历史中从偏移量offset开始的消息
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:  # 每隔一个消息（用户消息），检查是否包含图像
                if type(msg) is tuple:
                    # 如果消息是元组类型，解包消息和图像及其处理模式
                    msg, image, image_process_mode = msg
                    # 处理图像，并根据return_pil参数决定返回类型
                    image = self.process_image(image, image_process_mode, return_pil=return_pil)
                    images.append(image)
        return images

    # 将对话历史转换为适用于Gradio Chatbot的格式
    def to_gradio_chatbot(self):
        ret = []
        # 遍历对话历史中从偏移量offset开始的消息
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:  # 每隔一个消息（用户消息），处理图像
                if type(msg) is tuple:
                    # 如果消息是元组类型，解包消息和图像及其处理模式
                    msg, image, image_process_mode = msg
                    # 处理图像，指定默认处理模式，并设置return_pil为False
                    img_b64_str = self.process_image(
                        image, "Default", return_pil=False,
                        image_format='JPEG')
                    # 将图像转换为Base64编码的HTML img标签，并与消息拼接
                    img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])  # 将处理后的消息和None作为元组添加到列表中
                else:
                    ret.append([msg, None])
            else:  # 如果是系统消息
                if type(msg) is tuple and len(msg) == 2:
                    # 如果消息是元组类型，解包消息和图像的Base64编码字符串
                    msg, img_b64_str = msg
                    # 将图像的Base64编码字符串转换为HTML img标签，并拼接到消息中
                    img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                    msg = msg.strip() + img_str
                ret[-1][-1] = msg  # 更新列表中最后一个元素的第二个值
        return ret

    # 复制当前对话对象，创建一个新的对话对象副本。
    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    # 将对话对象转换为字典格式。
    def dict(self):
        # 检查是否有图像，如果有，则返回包含图像的字典；否则返回不包含图像的字典
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                # 提取消息和图像
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            # 直接使用原始消息列表
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

# 创建一个名为conv_vicuna_v0的对话对象
conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

# 助手提供有用、详细且礼貌的回答来回答用户的问题
conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

# 要求回答尽可能有帮助，同时确保回答是安全的，不包含有害、不道德、种族主义、性别歧视、有毒、危险或非法内容
conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

# 类似于 conv_llama_2，特别强调了助手能够理解用户提供的视觉内容，并使用自然语言协助用户完成各种任务。
conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

# 使用 <|im_end|> 作为分隔符，适用于基于大型语言模型（LLM）的AI助手和用户之间的对话。
conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

# 一个简单的对话模板，没有设置系统描述和角色，使用换行符 \n 作为分隔符。
conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

# 对话模板与 conv_llava_plain 类似，但它使用了 ### 作为分隔符，并且有一个特定的版本号 "v0"。
conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

# 对话模板特别用于处理包含视觉内容的对话，视觉内容将以 <Image> 标签的格式提供。
conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

# 话模板与 conv_llava_v0 类似，但使用了不同的版本号 "v1" 和分隔符风格。
conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

# 针对用户的问题提供有用、详细且礼貌的回答。此模板使用了两个分隔符风格，其中 sep 为单个空格，sep2 为 ""
conv_vicuna_imgsp_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="imgsp_v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

# 基础模板，没有特定的系统描述和角色定义。它使用简单的换行符 \n 作为分隔符，适用于不需要复杂分隔的对话场景。
conv_llava_plain_guided = Conversation(
    system="",
    roles=("", ""),
    version="plain_guided",
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

# 对话模板与 conv_llama_2 类似，但它特别强调了助手能够理解用户提供的视觉内容，并使用自然语言协助用户完成各种任务。此模板同样使用了两个分隔符风格，其中 sep 为单个空格，sep2 为 ""，并且为对话指定了 "v1_mmtag" 版本号。
conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

# 对话模板使用两个分隔符风格，sep 为单个空格，sep2 为 "<|endoftext|>"。
conv_phi_2 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="phi2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|endoftext|>",
)

# 对话模板与 conv_llama_2 类似，使用了 "llama_v2" 作为版本号。
conv_mistral_instruct = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

# 对话模板使用逗号 , 和 "" 作为分隔符，适用于 "gemma" 版本的对话场景。
conv_gemma = Conversation(
    system="",
    roles=("user", "model"),
    version="gemma",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.GEMMA,
    sep="",
    sep2="<eos>",
)

# 对话模板使用 <|im_end|> 作为分隔符，适用于直接回答问题的对话场景。
conv_chatml_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

# 设置默认对话模板为conv_vicuna_v1实例
default_conversation = conv_gemma

# 定义一个包含多个对话模板的字典
conv_templates = {
# 定义默认模板，它与v0版本和vicuna_v1模板相同
    "default": conv_gemma,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "phi_2": conv_phi_2,
    "gemma": conv_gemma,
    "llama_2": conv_llama_2,
    "imgsp_v1": conv_vicuna_imgsp_v1,
    "plain_guided": conv_llava_plain_guided,
    "mistral_instruct": conv_mistral_instruct,
    "chatml_direct": conv_chatml_direct,
    "mistral_direct": conv_chatml_direct,
    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "llava_v0": conv_llava_v0,
    "v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,

    "mpt": conv_mpt,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())