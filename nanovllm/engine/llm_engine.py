import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# PyTorch 提供的一个用于多进程编程的模块，
# 它是 Python 标准库 multiprocessing 的一个扩展，专门针对 PyTorch 的使用场景进行了优化。
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):

        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # 初始化Config配置类
        config = Config(model, **config_kwargs)

        # 进程列表
        self.ps = []

        # 事件对象列表
        self.events = []

        # 获取一个进程上下文（Context），并指定启动方式为 "spawn"
        # "spawn" 是一种进程启动方式，表示新进程会通过重新运行父进程的代码来启动。
        ctx = mp.get_context("spawn")

        for i in range(1, config.tensor_parallel_size):
            # 创建一个事件对象，一种用于进程间同步的机制，通常用于控制进程的执行顺序。
            # 在多进程程序中，事件可以用来通知其他进程某个操作已经完成，或者等待某个条件满足后再继续执行。
            event = ctx.Event()
            # 子进程，创建一个进程对象
            # 指定目标函数为 ModelRunner。这个函数将在新进程中被调用。
            # 同时指定传递给 ModelRunner 函数的参数。这些参数将在新进程中被传递给 ModelRunner 函数。
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            # 启动进程。
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # 主进程 创建 ModelRunner
        self.model_runner = ModelRunner(config, 0, self.events)

        # 创建 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

        # eos token id
        config.eos = self.tokenizer.eos_token_id

        # 创建调度器
        self.scheduler = Scheduler(config)

        # 将一个函数注册为程序退出时需要执行的函数。
        # 注册的函数会在程序退出时被调用，无论程序是正常退出还是因为异常退出。
        atexit.register(self.exit)

    # 退出执行函数
    def exit(self):
        # 退出模型运行器，同时主进程会向共享内存发送退出信号，子进程读取到共享内存的退出函数信息执行退出。
        self.model_runner.call("exit")
        del self.model_runner

        for p in self.ps:
            # 阻塞父进程，直到子进程完成
            # 确保子进程在父进程继续执行之前完成其任务，从而实现进程间的同步。
            p.join()

    # 调度器中，新增请求到等待队列
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    # 生成一个 token 
    def step(self):

        # 调度
        seqs, is_prefill = self.scheduler.schedule()
        # ([seq1, seq2], True)

        # 模型推理，返回每个序列生成的 token id
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # 后处理
        self.scheduler.postprocess(seqs, token_ids)

        # 过滤出结束的 序列id 以及 token ids
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        # 如果为预填充，预处理token的吞吐量为该step的所有序列总的 token 数
        # 如果为解码阶段，生成token的吞吐量则为该批次处理的序列数
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)

        return outputs, num_tokens

    # 调度器是否结束推理，即当等待队列与运行队列都为空时，结束
    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        
        if use_tqdm:
            # 创建进度条
            # total 参数指定了进度条的总进度值。
            # desc 参数用于设置进度条的描述信息。
            # dynamic_ncols 参数设置为 True，表示进度条的宽度会根据终端窗口的大小动态调整。
            # 这样可以确保进度条始终能够适应终端窗口的宽度，而不会因为窗口大小改变而显示不完整。
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # 将请求加入等待队列
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        outputs = {}
        prefill_throughput = decode_throughput = 0.

        # 判断是否结束推理
        while not self.is_finished():
            t = perf_counter()

            # 每步生成一个 token 
            output, num_tokens = self.step()

            if use_tqdm:
                # 统计预填充与解码阶段的吞吐量
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)

                # 通过 set_postfix() 方法，可以在进度条的末尾显示一些动态更新的额外信息，
                # 这些信息会随着进度条的更新而实时刷新。这在调试和监控任务执行过程中非常有用。
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })

            # 序列ID，生成的token ids
            for seq_id, token_ids in output:

                outputs[seq_id] = token_ids
                if use_tqdm:
                    # 更新进度条
                    pbar.update(1)

        # 将outputs根据序列ID排序
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        
        if use_tqdm:
            # 关闭进度条
            pbar.close()
        return outputs
