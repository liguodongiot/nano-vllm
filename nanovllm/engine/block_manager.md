# 为序列分配块

```
block-size=4

序列1：[3,2,1,4, 4,4,3,3, 2,3,1]
序列2：[3,2,1,4, 4,4,3,2, 1]



```

```
num_batched_tokensnum_batched_tokens序列1：

# block 0:
h=h0 block_id=-1

# block 1:
h=h1 block_id=-1

# block 2
h=-1 block_id=-1


---------


hash_to_block_id:
h0: b0
h1: b1

seq.num_batched_tokens：0
seq.block_table = [0, 1, 2]
```

```
序列2：

# block 0:
h=h0 block_id=b0

# block 1:
h=h2 block_id=-1

# block 2
h=-1 block_id=-1

---------


hash_to_block_id:
h0: b0
h1: b3
h2: b4

seq.num_batched_tokens：4
seq.block_table = [0, 3, 4]


```
