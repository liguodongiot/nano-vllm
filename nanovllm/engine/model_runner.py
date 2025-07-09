import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

# 模型运行器
class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):

        self.config = config
        hf_config = config.hf_config

        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager

        # 张量并行大小
        self.world_size = config.tensor_parallel_size

        self.rank = rank
        self.event = event

        # 初始化进程组
        # backend="nccl" 指定后端通信方式。nccl 是一种高性能的 GPU 通信库，专门用于 NVIDIA GPU 的分布式训练。

        # init_method="tcp://localhost:2333" 指定初始化方法，用于确定进程组中各个进程的通信方式和地址。
        # 在分布式训练中，init_method 通常是一个 URL，格式为 tcp://<master_node_ip>:<port>。

        # master_node_ip 是主节点的 IP 地址，所有进程都会通过这个地址进行通信初始化。
        # world_size 表示分布式训练中参与训练的进程总数。
        # rank 是当前进程的唯一标识符，范围从 0 到 world_size - 1。每个进程都有一个唯一的 rank，用于区分不同的进程。
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)

        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        # 初始化并加载Qwen3
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)

        # 初始化采样器
        self.sampler = Sampler()

        # 预热模型
        self.warmup_model()
        
        # 分配 KV 缓存
        self.allocate_kv_cache()

        if not self.enforce_eager:
            self.capture_cudagraph()
        
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                # 主进程，创建一个新的内存共享块
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                # 子进程，连接到现有空闲内存块
                self.shm = SharedMemory(name="nanovllm")
                # 循环执行
                self.loop()

    # ModelRunner 退出
    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        
        torch.cuda.synchronize()
        dist.destroy_process_group()


    # 子进程循环执行
    def loop(self):
        while True:
            # 从共享内存读取执行方法名（run、exit）及其参数
            method_name, args = self.read_shm()
            self.call(method_name, *args)

            if method_name == "exit":
                break


    def read_shm(self):
        # 确保当前环境是多进程环境，并且当前进程不是主进程
        assert self.world_size > 1 and self.rank

        # 当前进程会阻塞，直到 self.event 被设置（event.set()）
        self.event.wait()

        # 从共享内存的前 4 个字节读取数据长度 n，这里数据长度以小端字节序存储。
        n = int.from_bytes(self.shm.buf[0:4], "little")

        # 从共享内存中读取从第 4 个字节开始的 n 字节数据，并使用 pickle.loads 反序列化。
        # 反序列化后的数据是一个包含方法名和参数的列表，将其解包为 method_name 和 args。
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])

        # 清除事件状态，以便下一次写入操作可以重新设置事件。
        self.event.clear()

        return method_name, args

    def write_shm(self, method_name, *args):
        # 确保当前环境是多进程环境，并且当前进程不是主进程
        assert self.world_size > 1 and not self.rank
        
        # 将方法名和参数打包为一个列表，并使用 pickle.dumps 序列化为字节流。
        data = pickle.dumps([method_name, *args])

        n = len(data)
        # 将 n 转换为 4 个字节的小端字节序，并写入共享内存的前 4 个字节。
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        # 将序列化后的数据写入共享内存，从第 4 个字节开始。
        self.shm.buf[4:n+4] = data
        # 遍历所有事件对象，并将它们设置为已触发状态。这会通知其他进程，共享内存中的数据已经准备好。
        for event in self.event:
            event.set()


    def call(self, method_name, *args):
        # 当前环境是多进程环境，并且当前进程为主进程
        # 主进程 向共享内存中写入方法名及参数，比如：run、exit方法。通知子进程读取方法及参数执行
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)

        method = getattr(self, method_name, None)
        return method(*args)


    # 随机初始化一批序列 执行推理预热模型
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]

        # 执行预填充阶段的推理
        self.run(seqs, True)
        torch.cuda.empty_cache()

    # 分配 KV 缓存
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables


    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 设置上下文
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)

        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        # 设置上下文
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            # 获取上下文
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    # TODO
    # 运行模型推理
    # is_prefill 是否是预填充阶段
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:

        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # 模型推理
        logits = self.run_model(input_ids, positions, is_prefill)

        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        # 重置上下文
        reset_context()

        return token_ids

    """
    @torch.inference_mode() 作用：
    自动关闭梯度计算（torch.set_grad_enabled(False)）。这可以显著减少内存占用和计算时间。
    改变特定层的行为：
    - Dropout：在训练模式下，Dropout 会随机丢弃一部分神经元的输出；而在推理模式下，Dropout 层会被禁用，所有神经元的输出都会被保留。
    - Batch Normalization：在训练模式下，Batch Normalization 会根据当前批次的数据计算均值和方差；在推理模式下，它会使用训练阶段计算的全局均值和方差（移动平均值）。
    """
    @torch.inference_mode()
    def capture_cudagraph(self):

        config = self.config
        hf_config = config.hf_config
        
        # 最大批处理大小
        max_bs = min(self.config.max_num_seqs, 512)

        # 最大 KV cache 块数
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # 初始化了一系列张量，用于存储输入数据、位置信息、映射信息、上下文长度、块表以及模型输出。
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # 定义了捕捉 CUDA graph 的批量大小列表
        # [1, 2, 4, 8] + [16, 32, 48, 64, 80...]
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))

        # 存储不同批量大小对应的 CUDA 图
        self.graphs = {}

        # 图池
        # 1. 内存复用：提供了一个内存池，用于存储和复用这些分配的内存。当新的 CUDA 图需要内存时，它可以从 graph_pool 中获取已分配的内存，而不是每次都申请新的内存。这样可以减少内存碎片化，提高内存使用效率。
        # 2. 减少内存分配开销：可以减少内存分配的次数。当一个 CUDA 图完成其任务后，其占用的内存可以被释放回 graph_pool，供后续的 CUDA 图重复使用。这样可以显著减少内存分配和释放的开销，提高程序的整体性能。
        # 3. 优化资源管理：提供了一种资源管理机制，允许在多个 CUDA 图之间共享和复用内存。这样可以更好地利用有限的 GPU 内存资源，避免因内存不足而导致的程序崩溃或性能下降。
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            # 创建一个 CUDA 图实例
            graph = torch.cuda.CUDAGraph()

            # 设置当前批量大小相关的上下文信息
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            
            # 执行一次模型前向传播，确保模型加载到 GPU 上
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            
            # 捕捉图，该上下文管理器，记录模型前向传播的操作，这些操作会被保存到 CUDA 图中。
            # 如果 self.graph_pool 是 None，则表示不使用内存池，每次创建 CUDA 图时都会分配新的内存。
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            
            if self.graph_pool is None:
                # 在第一次创建 CUDA 图后，将 graph.pool() 的结果赋值给 self.graph_pool，从而初始化内存池。
                self.graph_pool = graph.pool()

            self.graphs[bs] = graph

            # 确保所有 GPU 操作完成
            torch.cuda.synchronize()
            # 重置上下文
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
