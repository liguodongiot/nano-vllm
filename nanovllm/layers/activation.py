import torch
from torch import nn
import torch.nn.functional as F

# 激活函数
class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile # 被PyTorch编译器优化执行效
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将输入张量在最后一个维度上分割为两部分x和y，每个部分占输入张量最后一维的一半
        x, y = x.chunk(2, -1)
        
        # 对分割后的x部分（gate）应用SiLU激活函数，然后将其与分割后的y部分（up）进行逐元素相乘
        # 这个操作允许模块在特征空间中引入非线性特性，同时保持特征维度不变
        return F.silu(x) * y
