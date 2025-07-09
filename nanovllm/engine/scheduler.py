from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        # 单次迭代中要处理序列的最大数量
        self.max_num_seqs = config.max_num_seqs

        # # 单次迭代中要处理的最多 token 数
        self.max_num_batched_tokens = config.max_num_batched_tokens

        # eos token id
        self.eos = config.eos

        # 初始化块管理器
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # 等待队列
        self.waiting: deque[Sequence] = deque()
        
        # 运行队列
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # 调度，判读是执行预填充阶段还是解码阶段
    # 优先处理预填充
    def schedule(self) -> tuple[list[Sequence], bool]:

        # prefill
        
        # 调度序列列表
        scheduled_seqs = []

        # 批处理的序列数
        num_seqs = 0
        
        # 记录一个批次需处理的token数量
        num_batched_tokens = 0

        # 等待队列不为空，且序列数小于单次迭代中要处理序列的最大数量
        while self.waiting and num_seqs < self.max_num_seqs:
            # 从等待队列取出新序列
            seq = self.waiting[0]

            # 当前step已处理的token数量+新序列的token数量 > 单次迭代中要处理的最多 token 数
            # 块管理器当前的空闲块不够分配给新序列
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break

            # 序列数+1
            num_seqs += 1

            # 块管理器为序列分配块
            self.block_manager.allocate(seq)
            
            # 序列中已缓存的token部分，不参与预填充阶段的计算
            num_batched_tokens += len(seq) - seq.num_cached_tokens

            # 序列状态更改为运行
            seq.status = SequenceStatus.RUNNING

            # 从等待队列移除序列
            self.waiting.popleft()

            # 将序列加入到运行队列
            self.running.append(seq)

            scheduled_seqs.append(seq)
        
        # 如果调度序列列表不为空，表示进行预填充阶段
        if scheduled_seqs:
            return scheduled_seqs, True

        ###############################################################
        # decode

        # 运行队列不为空，且序列数小于单次迭代中要处理序列的最大数量
        while self.running and num_seqs < self.max_num_seqs:
            
            # 从运行队列中取出序列
            seq = self.running.popleft()

            # 判断能够追加新块
            while not self.block_manager.can_append(seq):
                # 如果不能追加新块
                
                # 运行队列不为空，将运行队列的序列重新放入等待队列
                # 然后再判断能够追加新块
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    # 如果运行队列已经为空，说明之前运行队列只有该序列，则直接将该序列下方到等待队列，并终止循环。且不会执行else部分。
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                # 根据序列长度，判断是否追加新块
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        # 断言scheduled_seqs不为空
        assert scheduled_seqs

        # 从运行队列中，弹出本批次待推理的序列
        self.running.extendleft(reversed(scheduled_seqs))

        return scheduled_seqs, False

    # 抢占，将序列放置到等待队列，释放序列占用的块
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)

        self.waiting.appendleft(seq)

    # 后处理
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            # 将生成的token追加到序列，并更新序列长度
            seq.append_token(token_id)

            # ignore_eos(忽略eos) 为 false，且token_id == eos
            # 已生成的token数 = 最大生成token数限制
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
