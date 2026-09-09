


silu(x)=x∗σ(x),where σ(x) is the logistic sigmoid.




RMSNorm:


rsqrt(x) = 1 / sqrt(x)


1. 计算均方根（RMS）
2. 归一化输入向量


```
# x: [batch_size, hidden_dim]
# weight: [hidden_dim]
def rms_norm(x, weight):
    return x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True)) * weight
```