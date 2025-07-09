from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams

# 序列状态，等待，运行，结束
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

# 存储序列的数据、状态和块信息
class Sequence:

    # 序列的块大小。应该和块管理器和缓存引擎使用的块大小一样。
    block_size = 256

    # 创建一个无限的迭代器，从指定的起始值开始，按指定的步长递增。
    # 比如：count(10, 2)  # 从10开始，每次加2
    # 默认从0开始，每次加1
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):

        # 序列ID
        self.seq_id = next(Sequence.counter)

        # 状态
        self.status = SequenceStatus.WAITING

        # token ids
        # 复制任意对象（如自定义类实例），浅拷贝
        self.token_ids = copy(token_ids)

        # 最近一个token
        self.last_token = token_ids[-1]

        # 总 token 数
        self.num_tokens = len(self.token_ids)

        # 提示token数
        self.num_prompt_tokens = len(token_ids)

        # 缓存token数
        self.num_cached_tokens = 0

        # 块表
        self.block_table: list[int] = []

        # 温度
        self.temperature = sampling_params.temperature

        # 每个输出序列生成令牌的最大数量
        self.max_tokens = sampling_params.max_tokens

        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    # 生成token的数量
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    # 提示token id
    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]
    
    # 生成token id
    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    # 缓存块数
    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    # 块数
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    # 最后一块token大小
    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 获取某一块token ids
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    # 增加 token id
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
