import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

# 确保分子能被分母整除
def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator

# 基础线性层类
class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int, # 输入维度
        output_size: int, # 输出维度
        tp_dim: int | None = None, # 张量并行的维度，None 表示不使用张量并行，表示行并行还是列并行
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        # 0表示列并行，1表示行并行
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# 定义一个普通的线性层类，不涉及张量并行
class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size)

        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# 列并行线性层
class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, 0)

        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))

        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        
        # 该列并行线性层，该分区的列维度
        shard_size = param_data.size(self.tp_dim)

        start_idx = self.tp_rank * shard_size

        # 从加载的权重中提取当前分区的权重
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)

        # 将提取的权重复制到参数中
        param_data.copy_(loaded_weight)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)



# 合并的列并行
# gate_proj与up_proj 矩阵合并进行列并行
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        # 输出维度为 gate_proj 与 up_proj 矩阵输出维度的总和
        super().__init__(input_size, sum(output_sizes), bias=bias)


    # packed_modules_mapping
    # loaded_shard_id 为 0 , 1
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):

        param_data = param.data

        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size

        # 当前分区输出维度大小
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size

        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)

        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]

        param_data.copy_(loaded_weight)



# QKV并行线性层
# qkv_proj 合并矩阵进行并行
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads

        tp_size = dist.get_world_size()

        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)

        # 输入维度为隐藏层维度
        input_size = hidden_size
        # # 输出维度为 QKV 的总和
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        super().__init__(input_size, output_size, bias)


    # packed_modules_mapping
    # loaded_shard_id 为 "q", "k", "v"
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data

        assert loaded_shard_id in ["q", "k", "v"]
        
        # 计算QKV的开始索引和分片大小
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size

        # 获取到当前分区的权重部分
        # dim: _int 切片维度
        # start: Tensor 开始的索引
        # length: Union[_int, SymInt] 切片长度
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)

        # 从加载的权重中提取当前分区的权重部分
        # chunks：块数
        # dim：维度
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)



# 行并行线性层
# o_proj， down_proj 行并行
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, 1)

        # 每个线性层分区输入维度
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size

        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):

        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size

        # 从加载的权重中提取当前分区的权重
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bias 没有切分，因此，每个 rank 都有一份 bias，保留 rank 0 的 bias 即可。
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)

        if self.tp_size > 1:
            # 对所有分区的结果进行全归约操作
            dist.all_reduce(y)
            
        return y

