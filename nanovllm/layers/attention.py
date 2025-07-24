import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit # 表明这是一个Triton内核函数，将被编译为GPU可执行代码
def store_kvcache_kernel(
    key_ptr, # 指向当前批次K和V的指针
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr, # 指向缓存中K和V的指针
    v_cache_ptr,
    slot_mapping_ptr, # 每个KV对应该存储到缓存的哪个位置
    D: tl.constexpr, 
):
    # 获取当前程序ID，计算KV的偏移量，加载数据，并存储到指定的缓存位置
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

# 调用上面定义的Triton内核函数
# 接收PyTorch张量形式的K、V、KV缓存以及槽映射
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # 验证输入张量的形状和步长是否符合要求
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N

    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale, # 用于缩放注意力分数
        num_kv_heads, # KV对的注意力头数量
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 存储KV对的缓存，初始化为空张量
        # allocate_kv_cache 方法为 k_cache， v_cache 分配缓存
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # 获取上下文
        context = get_context()

        k_cache, v_cache = self.k_cache, self.v_cache

        # k_cache.numel() 返回张量中所有维度元素的乘积
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # 根据 预填充阶段 还是 解码阶段 ， 选择不同的 FlashAttention
        if context.is_prefill:
            
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache

            # 变长序列的Flash Attention计算
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, 
                                       cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, 
                                       cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, 
                                       causal=True, 
                                       block_table=context.block_tables)
        else:    # decode
            # 利用缓存进行高效解码
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
            
        # 将计算得到的输出张量重构为适合下游计算的形状
        o = o.view(-1, self.num_heads * self.head_dim)
        return o



