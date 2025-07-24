import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size # 特征维度大小
        self.eps = eps # 用于数值稳定的微小值，默认为 1×10^−6
        # 初始化为全 1 的张量
        self.weight = nn.Parameter(torch.ones(hidden_size))

    # 编译优化，提高运行效率
    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # 保存输入张量的原始数据类型 
        orig_dtype = x.dtype
        # 将输入张量转换为 float32 类型进行计算，确保数值计算的精度。
        x = x.to(torch.float32)

        # 输入张量的平方，沿着最后一个维度上求均值（即每行求平均），保持维度
        # dim 取平均值的维度
        # keepdim=True 保持维度
        var = x.pow(2).mean(dim=-1, keepdim=True)

        # torch.rsqrt: 获取 input 张量中每一个元素的平方根的倒数
        # 然后与输入张量逐元素相乘，完成归一化操作。
        x.mul_(torch.rsqrt(var + self.eps))

        # 将张量转换回原始数据类型，与权重进行逐元素相乘
        x = x.to(orig_dtype).mul_(self.weight)
        
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 保存输入张量的原始数据类型 
        orig_dtype = x.dtype

        # 将输入 x 和残差 residual 转换为 float32 类型后相加
        x = x.to(torch.float32).add_(residual.to(torch.float32))

        residual = x.to(orig_dtype)
        
        var = x.pow(2).mean(dim=-1, keepdim=True)
        
        x.mul_(torch.rsqrt(var + self.eps))

        x = x.to(orig_dtype).mul_(self.weight)
        
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
