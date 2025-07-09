from dataclasses import dataclass


@dataclass
class SamplingParams:
    # 温度
    temperature: float = 1.0

    # 每个输出序列生成令牌的最大数量
    max_tokens: int = 64
    
    # 是否忽略 eos token
    ignore_eos: bool = False
