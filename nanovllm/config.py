import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:

    # 模型地址
    model: str
    
    # 单次迭代中要处理的最多 token 数
    max_num_batched_tokens: int = 16384
    
    # 单次迭代中要处理序列的最大数量
    max_num_seqs: int = 512
    
    # 模型上下文长度（提示和输出）。如果没有指定，会自动从模型配置中获取。
    # 当通过`--max-model-len`传递时，支持以可读格式输入k/m/g/K/M/G。例如：
    # - 1k -> 1000 （小写）
    # - 1K -> 1024 （大写）
    # - 25.6k -> 25,600 （小写）
    max_model_len: int = 4096
    
    gpu_memory_utilization: float = 0.9

    tensor_parallel_size: int = 1

    enforce_eager: bool = False
    
    hf_config: AutoConfig | None = None
    
    # eos token id
    eos: int = -1
    
    # kv cache 块大小
    kvcache_block_size: int = 256
    
    # kv cache 块数量
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8

        self.hf_config = AutoConfig.from_pretrained(self.model)

        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

        assert self.max_num_batched_tokens >= self.max_model_len
