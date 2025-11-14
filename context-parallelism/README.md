# context-parallelism

Comparing different context parallelism to continue pretraining on Malaysian multi-lingual corpus.

## Explanation

### Ring Attention

Quick intro about Blockwise Ring Attention, For simplicity sake, let say, we have 2 GPUs with seqlen of 4, partitioned to 2, 4 / 2 = 2, 2

- For GPU 0, should calculate q0k0v0 + q0k1v1
- For GPU 1, should calculate q1k0v0 + q1k1v1
- (+) denoted as blockwise attention.

so the attention is like,

```
      k0  | k1
q0    o x | x x
      o o | x x
      ---------
q1    o o | o x
      o o | o o
```

#### Forward

For GPU 0,

- q0k0v0 is causal.
- q0k1v1 not required to do anything.

```
cp_rank = 0
cp_world_size = 2
send = (cp_rank + 1) % cp_world_size = 1
receive = (cp_rank - 1) % cp_world_size = 1
q = q0
k = k0
v = v0

step == 0
    if step + 1 != comm.world_size: True
        send k, v to send
        receive k1, v1 from receive

    if not is_causal or step <= comm.rank: True
        calculate attention with is_causal and step == 0, so this causal # ✅ q0k0v0

    if step + 1 != comm.world_size: True
    k = k1
    v = v1

step == 1

    if step + 1 != comm.world_size: False
    if not is_causal or step <= comm.rank: False
    if step + 1 != comm.world_size: False
```

For GPU 1,

- q1k0v0 is full.
- q1k1v1 is causal.

```
cp_rank = 1
cp_world_size = 2
send = (cp_rank + 1) % cp_world_size = 0
receive = (cp_rank - 1) % cp_world_size = 0
q = q1
k = k1
v = v1

step == 0
    if step + 1 != comm.world_size: True
        send k, v to send
        receive k0, v0 from receive

    if not is_causal or step <= comm.rank: True
        calculate attention with is_causal and step == 0, so this causal, # ✅ q1k1v1

    if step + 1 != comm.world_size: True
    k = k0
    v = v0

step == 1
    if step + 1 != comm.world_size: False

    if not is_causal or step <= comm.rank: True
        calculate attention with is_causal and step == 1, full attention # ✅ q1k0v0

    if step + 1 != comm.world_size: False
```

Everything hit as it is.

#### Backward

Because for each GPU we have global out, global LSE, global ∂out, global ∂LSE, we should able to calculate ∂q, ∂k, ∂v,

- assumed backward function will overwrite ∂q, ∂k, ∂v when the parameter is ∂(global out, global LSE, global ∂out, q, k, v, ∂q, ∂k, ∂v)
- We know during forward,
    - For GPU 0, attention calculated as q0k0v0 + q0k1v1
    - For GPU 1, attention calculated as q1k0v0 + q1k1v1

- For GPU 0, attention calculated as q0k0v0 + q0k1v1,
    - to calculate ∂q0, you required to calculate ∂(global out, global LSE, global ∂out, q0, k0, v0, ∂q0, ∂k0, ∂v0) + ∂(global out, global LSE, global ∂out, q0, k1, v1, ∂q0, ∂k1, ∂v1)
    - to calculate ∂k0, you required to calculate ∂(global out, global LSE, global ∂out, q0, k0, v0, ∂q0, ∂k0, ∂v0) + ∂(global out, global LSE, global ∂out, q1, k0, v0, ∂q1, ∂k0, ∂v0)
    - to calculate ∂v0, you required to calculate ∂(global out, global LSE, global ∂out, q0, k0, v0, ∂q0, ∂k0, ∂v0) + ∂(global out, global LSE, global ∂out, q1, k0, v0, ∂q1, ∂k0, ∂v0)

- For GPU 1, attention calculated as q1k0v0 + q1k1v1,
    - to calculate ∂q1, you required to calculate ∂(global out, global LSE, global ∂out, q1, k0, v0, ∂q1, ∂k0, ∂v0) + ∂(global out, global LSE, global ∂out, q1, k1, v1, ∂q1, ∂k1, ∂v1)
    - to calculate ∂k1, you required to calculate ∂(global out, global LSE, global ∂out, q0, k1, v1, ∂q0, ∂k1, ∂v1) + ∂(global out, global LSE, global ∂out, q1, k1, v1, ∂q1, ∂k1, ∂v1)
    - to calculate ∂v1, you required to calculate ∂(global out, global LSE, global ∂out, q0, k1, v1, ∂q0, ∂k1, ∂v1) + ∂(global out, global LSE, global ∂out, q1, k1, v1, ∂q1, ∂k1, ∂v1)


#### Limitation

1. Some GPUs required less computation such as GPU 0, not required to compute q0k1v1, introduced unbalanced.