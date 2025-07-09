import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

# 定义默认的权重加载函数
def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    # 将加载的权重数据复制到模型参数中
    # param：模型中的参数对象
    # loaded_weight：权重张量
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    加载模型权重，支持从指定路径加载.safetensors文件。
    
    参数:
        model (nn.Module): 待加载权重的模型。
        path (str): 包含权重文件的目录路径。
    """
    # 获取模型的模块映射（如果有），用于处理打包模块
    # Qwen3ForCausalLM.packed_modules_mapping
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # 遍历指定路径下的所有.safetensors文件
    for file in glob(os.path.join(path, "*.safetensors")):
        # 使用safe_open打开文件，以PyTorch格式加载到CPU
        with safe_open(file, "pt", "cpu") as f:
            # 遍历文件中的所有权重名称
            for weight_name in f.keys():

                # 遍历打包模块映射
                # packed_modules_mapping字典示例："v_proj": ("qkv_proj", "v")
                for k in packed_modules_mapping:
                    # 如果当前权重名称包含映射中的键
                    if k in weight_name:
                        # 获取映射的目标名称和分片ID
                        v, shard_id = packed_modules_mapping[k]
                        
                        # 比如：将v_proj替换成qkv_proj
                        param_name = weight_name.replace(k, v)

                        # 获取到模型参数
                        param = model.get_parameter(param_name)

                        # 获取模型参数的权重加载函数（如果有）
                        weight_loader = getattr(param, "weight_loader")

                        # 调用权重加载函数，并传入模型参数、加载的权重和分片ID（通过分片ID告诉加载器这个张量应该被加载到目标大参数的哪一个分片）
                        # QKVParallelLinear
                        # MergedColumnParallelLinear
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 如果当前权重名称没有匹配到打包模块映射
                    # 直接获取模型中对应的参数
                    param = model.get_parameter(weight_name)

                    # 获取参数的权重加载函数，如果没有则使用默认加载函数
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    # 调用权重加载函数，传入参数和加载的权重
                    weight_loader(param, f.get_tensor(weight_name))



"""
- https://www.cnblogs.com/cavalier-chen/p/18935544
- https://www.cnblogs.com/cavalier-chen/p/18936769

什么是 "Packed Modules"? 

在 vLLM 中，为了极致的推理性能，经常会对权重进行量化（例如，将 FP16 的权重转换为 INT8/INT4）。

在量化过程中，多个原始的权重矩阵（如 Q、K、V 矩阵）可能会被合并（pack）成一个单一的、更大的量化后权重矩阵。

这样做可以优化内存访问和计算效率。

packed_modules_mapping 的作用：这个字典就是用来描述这种“合并”关系的。

键 (key): 通常是合并后（packed）的权重在 safetensors 文件中的名称的一部分。例如，可能是 "q_proj"。

值 (value): 是一个元组 (v, shard_id)。
- v: 对应模型中原始、未合并的参数名称的一部分。例如，可能是 "query_key_value"。
- shard_id: 指示这个分片属于合并后大矩阵的哪个部分（例如，第0、1、2块分别对应Q、K、V）。

举个例子：
假设在文件中有一个权重叫 layers.0.attention.q_proj.weight。
packed_modules_mapping 可能包含这样的条目：{"q_proj": ("query_key_value", 0)}。
q_proj 实际上是模型中 query_key_value 参数的一部分。
"""
