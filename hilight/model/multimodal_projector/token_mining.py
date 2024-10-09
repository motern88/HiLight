import torch
import torch.nn as nn
from torch.nn import init

# 定义模型结构
class UniLMProjectors(nn.Module):
    def __init__(self, hidden_size, hidden_size_aux, output_hidden_size):
        '''
        hidden_size = 768  CLIP-ViP的通道数
        hidden_size_aux = 512  LongCLIP的通道数
        output_hidden_size = 2048  语言模型的通道数
        '''
        super(UniLMProjectors, self).__init__()
        # LongCLIP的特征将作为query
        self.query_projector = nn.Sequential(
            nn.LayerNorm(hidden_size_aux),
            nn.Linear(hidden_size_aux, hidden_size_aux)
        )
        # CLIP-VIP的特征将作为key和val
        self.key_projector = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size_aux) # 768，512
        )
        self.val_projector = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size_aux) # 768，512
        )
        # 将LongCLIP的每个token都扩展成32个token
        self.LongCLIP_projector = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )
        # CLIP-VIP的projector
        self.CLIP_VIP_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        # 意图将512维度的特征映射到2048维度的特征，与文本对齐
        self.tokenizer_projector = nn.Sequential(
            nn.LayerNorm(hidden_size_aux),
            nn.Linear(hidden_size_aux, output_hidden_size),
            nn.ReLU(),
            nn.Linear(output_hidden_size, output_hidden_size),
            nn.ReLU()
        )

        self._init_weights() # 如果需要使用初始化方法，则打开这行代码，使用下面的_init_weights方法进行权重初始化

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                init.ones_(module.weight)  # 权重设为1
                init.zeros_(module.bias)  # 方差设为0

    def forward(self, videos, videos_aux):
        # videos:(1,2352,768), videos_aux:(1,12,512)
        # 12帧与2352patch做cross attention
        # 主塔videos来自CLIP-ViP,副塔videos_aux来自LongCLIP

        # 先对特征进行projector
        videos = self.CLIP_VIP_projector(videos)

        # 将 videos_aux 维度从 (1, 12, 512) 调整为 (1, 12, 1, 512)
        videos_aux = videos_aux.unsqueeze(2)  # (1, 12, 512) -> (1, 12, 1, 512)
        # 将 videos_aux 维度从 (1, 12, 1, 512) 调整为 (1 * 12 * 512, 1)
        B, N, _, D = videos_aux.shape
        videos_aux = videos_aux.view(B * N * D, 1)
        # 通过 Linear 层进行处理，将 (B * N * D, 1) -> (B * N * D, 32)
        videos_aux = self.LongCLIP_projector(videos_aux)
        # 将处理后的 videos_aux 维度调整回 (1, 12, 32, 512)
        videos_aux = videos_aux.view(B, N, 32, D)
        # 再调整为 (1, 12 * 32, 512)
        videos_aux = videos_aux.view(B, N * 32, D)

        # 计算查询、辅助和值的嵌入表示
        embed_query = self.query_projector(videos_aux)
        # print(embed_query.shape)  # torch.Size([1, 12, 512])
        embed_key = self.key_projector(videos)
        # print(embed_key.shape)  # torch.Size([1, 2352, 512])
        embed_value = self.val_projector(videos)
        # print(embed_value.shape)  # torch.Size([1, 2352, 512])

        # 计算注意力分数
        embed_att = embed_query @ (embed_key.transpose(-1, -2) / (embed_key.shape[-1] ** 0.5))
        # print("embed_att",embed_att.shape)  # torch.Size([1, 12, 2352])

        # 处理NaN值
        embed_att = embed_att.nan_to_num()

        # 应用softmax并计算特征
        embed_feat = (embed_att.softmax(-1) @ embed_value) # .mean(2)

        # 意图将512维度的特征映射到2048维度的特征，与llm的文本维度一致
        embed_feat = self.tokenizer_projector(embed_feat)

        return videos, embed_feat

if __name__ == '__main__':
    # 初始化模型
    hidden_size = 768  # CLIP-ViP的通道数
    hidden_size_aux = 512  # LongCLIP的通道数
    output_hidden_size = 2048  # 语言模型的通道数
    model = UniLMProjectors(hidden_size, hidden_size_aux, output_hidden_size)

    # 随机张量
    B = 1
    Patch = 2353
    N_Frame = 12
    videos = torch.randn(B, Patch, hidden_size)
    videos_aux = torch.randn(B, N_Frame, hidden_size_aux)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    videos = videos.to(device)
    videos_aux = videos_aux.to(device)

    videos, embed_feat = model(videos, videos_aux)
    print("embed_feat.shape", embed_feat.shape)
    # embed_feat.shape torch.Size([1, 12, 2048])

    # 获取模型的所有参数
    params = list(model.parameters())
    # 计算参数总量
    total_params = sum(p.numel() for p in params)
    print(f'Total number of parameters: {total_params}')  # 2,105,856

    # # 保存权重
    # # 获取模型的 state_dict，即模型的参数
    # state_dict = model.state_dict()
    # # 保存 state_dict 到 .pt 文件
    # torch.save(state_dict, 'TokenMining.pt')