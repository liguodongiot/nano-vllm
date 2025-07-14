import torch
from torch import nn

# 采样器，用于从模型输出的概率分布中采样
class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 将logits转换为float类型，以确保后续计算的精度
        logits = logits.to(torch.float)

        # 计算贪婪采样，选择概率最高的token
        greedy_tokens = logits.argmax(dim=-1)

        # 对logits进行温度调整
        # temperatures.unsqueeze(dim=1)将温度张量扩展为与logits形状匹配
        logits.div_(temperatures.unsqueeze(dim=1))

        # 计算softmax，将logits转换为概率分布
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        

        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        # 避免除以零的小值
        epsilon = 1e-10  
        
        # 生成与probs形状相同的指数分布随机数，并加一个小值epsilon防止除零
        # 将probs除以这些随机数，得到一个重新调整后的概率分布
        # argmax选择概率最高的token作为采样结果
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)  
        
        # 根据温度是否为0选择采样方式
        # 如果温度为0，使用贪婪采样；否则使用基于温度调整的采样
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
