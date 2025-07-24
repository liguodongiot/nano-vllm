import torch
from torch import nn
import torch.nn.functional as F
# 导入分布式训练模块
import torch.distributed as dist

from nanovllm.utils.context import get_context

# Embedding 层
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int, # 嵌入层的词汇表大小
        embedding_dim: int, # 嵌入维度
    ):
        super().__init__()
        # 获取当前进程在分布式环境中的排名
        self.tp_rank = dist.get_rank()
        # 获取分布式环境的总进程数
        self.tp_size = dist.get_world_size()

        assert num_embeddings % self.tp_size == 0

        self.num_embeddings = num_embeddings

        # 计算每个进程负责的词汇表分区大小
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 计算当前进程负责的词汇表起始索引
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        # 计算当前进程负责的词汇表结束索引
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        # 初始化权重参数（每个进程只存储部分权重）
        # num_embeddings_per_partition行，embedding_dim列
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # 为权重参数添加加载方法
        self.weight.weight_loader = self.weight_loader

    # 定义权重加载方法
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # 获取参数数据
        param_data = param.data

        # 计算每个分片的大小，获取行数
        shard_size = param_data.size(0)

        # 计算当前进程的起始索引
        start_idx = self.tp_rank * shard_size

        # 从加载的权重中提取当前进程负责的部分
        # 按行切
        # dim: _int 切片维度
        # start: Tensor 开始的索引
        # length: Union[_int, SymInt] 切片长度
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        # 确保参数数据和加载的权重尺寸一致
        assert param_data.size() == loaded_weight.size()
        # 将加载的权重数据复制到参数中
        param_data.copy_(loaded_weight)


    def forward(self, x: torch.Tensor):
        
        if self.tp_size > 1:
            # 创建一个掩码，标记输入中属于当前进程词汇表分区的部分
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)

        # 执行嵌入操作
        # torch.Size([x.size(0), num_embeddings_per_partition, num_embeddings])
        y = F.embedding(x, self.weight)
        
        if self.tp_size > 1:
            # 使用掩码清除不属于当前进程词汇表分区的嵌入结果
            y = mask.unsqueeze(1) * y
            # 在所有进程间执行全归约操作，汇总所有进程的嵌入结果
            dist.all_reduce(y)

        return y

# 语言模型 Head 层
class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False, # 是否使用偏置
    ):
        super().__init__(num_embeddings, embedding_dim)

        if bias:
            # 初始化偏置参数（每个进程只存储部分偏置）
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)


    def forward(self, x: torch.Tensor):
        # 获取上下文
        context = get_context()

        # 如果当前处于预填充阶段
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()

        # 执行线性变换（计算 logits）
        logits = F.linear(x, self.weight, self.bias)
        
        if self.tp_size > 1:
            # 准备一个列表，用于收集所有进程的 logits（仅在 rank 0 进程中需要）
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            
            # 在所有进程间执行收集操作，将所有 logits 汇总到 rank 0 进程
            # gather_list:用于收集数据的适当大小的张量列表（默认为无，必须在目标rank指定）
            # dst: 目的地rank id
            dist.gather(logits, all_logits, 0)
            
            # 如果是 rank 0 进程，则将收集到的 logits 拼接起来
            # 拼接操作在最后一维（embedding 维度）进行
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        
        return logits



