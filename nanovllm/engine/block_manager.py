from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


# 块
class Block:

    def __init__(self, block_id):

        # 块ID
        self.block_id = block_id

        # 该块的引用数
        self.ref_count = 0
        
        # 该块的哈希值
        self.hash = -1
        
        # 该块的 token ids
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids
    
    # 块重置
    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


# 块管理
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0

        # 块大小
        self.block_size = block_size

        # 根据块数量来初始化块
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]

        # 块哈希值 与 块ID 的映射
        self.hash_to_block_id: dict[int, int] = dict()

        # 空闲块ID队列（双端队列）
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        
        # 已使用块ID集合
        self.used_block_ids: set[int] = set()

    # 计算哈希值
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):

        # xxhash：这是一个高效的哈希算法库，xxh64 是其中的 64 位哈希算法
        h = xxhash.xxh64()
        if prefix != -1:
            # 将 prefix（一个整数）转换为字节序列
            # 8 表示字节长度，即占用 8 个字节。
            # "little" 表示字节序为小端模式（即低位字节在前）。
            h.update(prefix.to_bytes(8, "little"))
        
        # 将 NumPy 数组转换为字节序列
        # 将 token_ids 的字节序列更新到哈希对象中
        h.update(np.array(token_ids).tobytes())

        # 获取哈希对象的最终哈希值，并以整数形式返回。
        return h.intdigest()

    # 分配一个空块
    def _allocate_block(self, block_id: int) -> Block:

        # 根据块ID获取块
        block = self.blocks[block_id]

        # 断言块未被使用
        assert block.ref_count == 0

        # 块重置或初始化
        block.reset()
        
        # 将块ID从 空闲块ID队列 移除
        self.free_block_ids.remove(block_id)

        # 将块ID加入 已使用块ID集合
        self.used_block_ids.add(block_id)

        return self.blocks[block_id]

    # 释放块
    def _deallocate_block(self, block_id: int) -> Block:
        # 确保该块已经没有被引用
        assert self.blocks[block_id].ref_count == 0
        
        # 将块添加到空闲队列，并将其从使用队列移除。
        # 注意：块中相关的token_ids、hash等并没有释放，而是在新的序列申请该块时，重置。
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)




    
    def can_allocate(self, seq: Sequence) -> bool:
        # 空闲块数量大于某个序列所需要的块数
        return len(self.free_block_ids) >= seq.num_blocks

    # 为序列分配块
    def allocate(self, seq: Sequence):
        # 断言序列中的块表为空
        assert not seq.block_table
        h = -1
        cache_miss = False

        # 当前序列需要分割的块数
        for i in range(seq.num_blocks):
            # 获取当前序列中某块的token ids（分块操作）
            token_ids = seq.block(i)
            
            # 计算哈希值，当最新的一个块大小小于分块大小时，哈希值为-1
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

            # 根据哈希值获取块ID
            block_id = self.hash_to_block_id.get(h, -1)

            # 块管理器的 block_id 块的 token_ids 与 当前序列的第i块的token_ids 不等
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                # 块管理器中，缓存未命中，该序列如果第一块没命中，后续相当于都无法命中
                cache_miss = True
            
            if cache_miss: # 缓存未命中
                # 获取 空闲块ID队列 的 块ID
                block_id = self.free_block_ids[0]
                # 根据 块ID 分配一个空块
                block = self._allocate_block(block_id)
            else: # 缓存命中
                
                # 已缓存 token 数量
                seq.num_cached_tokens += self.block_size

                # 如果 块ID 在 已使用块ID集合 中
                if block_id in self.used_block_ids:
                    # 取出该块
                    block = self.blocks[block_id]
                    # 该块的引用数+1
                    block.ref_count += 1
                else:
                    # 根据 块ID 分配块
                    block = self._allocate_block(block_id)
            
            if h != -1:
                # 更新 块 的 哈希值 以及 token ids 
                block.update(h, token_ids)
                # 更新 块哈希值 与 块ID 的映射
                self.hash_to_block_id[h] = block_id
            
            # 序列的 块表 记录在块管理器中的 块ID （即分配的那一块）
            seq.block_table.append(block_id)

    # 释放序列
    def deallocate(self, seq: Sequence):

        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1

            if block.ref_count == 0:
                # 该块没有被引用，即可释放 BlockManager 管理的块
                self._deallocate_block(block_id)

        # 序列的缓存token数的记录清0
        seq.num_cached_tokens = 0
        # 清除序列的块表
        seq.block_table.clear()
        

    # 判断块管理器，是否有空闲块能够让序列追加新的块
    def can_append(self, seq: Sequence) -> bool:
        # 当序列长度与 % 块大小 == 1时，表示需要申请新块，因此需要空闲块大小大于1
        # 否则，不需要申请新块，空闲块大小始终>=0
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    # 根据实际序列长度追加新块
    def may_append(self, seq: Sequence):

        block_table = seq.block_table

        # 获取到该序列的最后一块
        last_block = self.blocks[block_table[-1]]

        if len(seq) % self.block_size == 1:
            # 前面推理时，序列长度能够整除块大小，因此最近一个块的哈希值不等于1
            # 断言最后一块的哈希值不等于1
            # 因此，需要重新分配新的块
            assert last_block.hash != -1

            # 申请新块
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

        elif len(seq) % self.block_size == 0:

            # 断言最后一块的哈希值等于1
            assert last_block.hash == -1
            
            # 取出最新块，重新计算哈希值
            token_ids = seq.block(seq.num_blocks-1)
            # 获取到倒数第二块的哈希作为前缀
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)

            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id

        else:

            assert last_block.hash == -1
