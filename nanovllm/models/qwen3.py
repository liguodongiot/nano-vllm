import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None, # 每个注意力头的维度，若未指定则自动计算
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False, # 是否使用QKV线性层的偏置
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        
        tp_size = dist.get_world_size()
        
        # 计算每个设备上的注意力头数量（总头数需能被设备数整除）
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        # 同样处理KV头的数量
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        
        # 计算每个注意力头的维度（若未指定则自动计算）
        self.head_dim = head_dim or hidden_size // self.total_num_heads

        # 计算Q向量和KV向量的总维度
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.scaling = self.head_dim**-0.5

        # QKV线性层
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )

        # 输出线性层
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )

        # 旋转位置嵌入层（用于捕捉序列中的位置信息）
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        # 注意力计算层（实现注意力机制的核心逻辑）
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # Q和K的RMS归一化层（稳定训练过程）
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)


    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        
        # 通过QKV线性层得到Q、K、V向量
        qkv = self.qkv_proj(hidden_states)

        # 将QKV张量分割为Q、K、V三部分
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # 重新组织Q向量形状并进行归一化
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        # 重新组织K向量形状并进行归一化
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k)

        o = self.attn(q, k, v)

        output = self.o_proj(o)
        
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        
        # 合并的列并行线性层（同时实现 gate 和 up）
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        # 行并行线性层（实现down）
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)

        x = self.act_fn(gate_up)
        
        x = self.down_proj(x)

        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()

        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

        # 输入归一化层（在注意力机制前）
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 注意力后归一化层（在前馈网络前）
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,  # 输入序列的位置信息
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None, # 残差连接输入（可能为None）
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # hidden_states 为使用RMSNorm归一化之后的值，residual为原始值。
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        
        # Embedding 层
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        # Decoder 层
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor, # 输入序列的位置信息
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        residual = None
        
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    
    # 在 vLLM 中，为了极致的推理性能，经常会对权重进行量化（例如，将 FP16 的权重转换为 INT8/INT4）。
    # 在量化过程中，多个原始的权重矩阵（如 Q、K、V 矩阵）可能会被合并（pack）成一个单一的、更大的量化后权重矩阵。
    # 这样做可以优化内存访问和计算效率。

    # packed_modules_mapping 的作用：这个字典就是用来描述这种“合并”关系的。
    # 键 (key): 通常是合并后（packed）的权重在 safetensors 文件中的名称的一部分。例如，可能是 "q_proj"。
    # 值 (value): 是一个元组 (v, shard_id)。
    # - v: 对应模型中原始、未合并的参数名称的一部分。例如，可能是 "query_key_value"。
    # - shard_id: 指示这个分片属于合并后大矩阵的哪个部分（例如，第0、1、2块分别对应Q、K、V）。
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        
        self.model = Qwen3Model(config)

        # 初始化一个 ParallelLMHead 实例，用于处理语言模型的输出层
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        # 是否应绑定模型的输入和输出词嵌入
        # 默认为true
        if config.tie_word_embeddings:
             # 将 lm_head 的权重数据设置为模型的 embed_tokens 权重数据，实现权重共享
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    # 模型推理 
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        
        hidden_states = self.model(input_ids, positions)

        return hidden_states

    # 执行完模型的forward方法之后，调用执行
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
