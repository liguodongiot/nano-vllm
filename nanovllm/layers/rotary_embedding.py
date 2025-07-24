from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    
    # 在倒数第二个维度上增加一个维度，方便后续操作
    cos = cos.unsqueeze(-2) 
    sin = sin.unsqueeze(-2)

    # 将张量 x 按最后一个维度分成两个部分 x1 和 x2
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1) 

    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int, # HEAD 大小
        rotary_dim: int, # 旋转维度
        max_position_embeddings: int, 
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size

        # 计算逆频率
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

        # 创建一个从 0 到 max_position_embeddings 的张量
        t = torch.arange(max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)

        # 计算余弦值
        cos = freqs.cos()
        # 计算正弦值
        sin = freqs.sin()

        # 将余弦和正弦张量拼接起来
        cache = torch.cat((cos, sin), dim=-1)

        # 注册一个缓冲区，用于缓存余弦和正弦张量
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor, # 位置张量
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 获取位置张量的第 0 维大小，即 token 数量
        num_tokens = positions.size(0)

        # 从缓存中获取与位置对应的部分余弦和正弦张量
        cos_sin = self.cos_sin_cache[positions]
        # 将余弦和正弦张量分开为 cos 和 sin
        cos, sin = cos_sin.chunk(2, dim=-1)

        # 保存Q张量的形状
        query_shape = query.shape
        # 调整Q张量的形状
        query = query.view(num_tokens, -1, self.head_size)
        # 应用旋转位置编码，并恢复查询张量的原始形状
        query = apply_rotary_emb(query, cos, sin).view(query_shape)
        
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        
        return query, key

# 使用 lru_cache 装饰器，缓存最近 1 次调用结果
# 如果不传则maxsize的默认值为128，typed的默认值为False。
# 其中，maxsize参数表示是的被装饰的方法最大可缓存结果数量， 
# 如果是默认值128，则表示被装饰方法最多可缓存128个返回结果；如果maxsize传入为None，则表示可以缓存无限个结果。
@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
