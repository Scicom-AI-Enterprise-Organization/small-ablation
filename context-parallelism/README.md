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

- assumed backward function like Flash Attention will overwrite ∂q, ∂k, ∂v when the parameter is ∂(global out, global LSE, global ∂out, q, k, v, ∂q, ∂k, ∂v), assumed 2 GPUs,
- We know during forward,
    - For GPU 0, attention calculated as q0k0v0 + q0k1v1
    - For GPU 1, attention calculated as q1k0v0 + q1k1v1

- For GPU 0, attention calculated as q0k0v0 + q0k1v1,
    - to calculate ∂q0, you required to calculate ∂(global out, global LSE, global ∂out, q0, k0, v0, ∂q0, ∂k0, ∂v0) + ∂(global out, global LSE, global ∂out, q0, k1, v1, ∂q0, ∂k1, ∂v1)
    - to calculate ∂k0, you required to calculate ∂(global out, global LSE, global ∂out, q0, k0, v0, ∂q0, ∂k0, ∂v0) + ∂(global out, global LSE, global ∂out, q1, k0, v0, ∂q1, ∂k0, ∂v0)
    - to calculate ∂v0, you required to calculate ∂(global out, global LSE, global ∂out, q0, k0, v0, ∂q0, ∂k0, ∂v0) + ∂(global out, global LSE, global ∂out, q1, k0, v0, ∂q1, ∂k0, ∂v0)

```
cp_rank = 0
cp_world_size = 2
send = (cp_rank + 1) % cp_world_size = 1
receive = (cp_rank - 1) % cp_world_size = 1
q = q0
k = k0
v = v0
dq, dk, dv = None, None, None
temp_dq, temp_dk, temp_dv = torch.empty(q.shape), torch.empty(q.shape), torch.empty(q.shape)
global_out = out0 # same shape q0
dglobal_out # same shape as global_out, received from autograd
global_lse # from forward, sync across devices, shape [B, H, L]
step == 0
    if step + 1 != comm.world_size: True
        send k, v to send
        receive k1, v1 from receive
    
    if not is_causal or step <= comm.rank: True
        calculate ∂ with is_causal and step == 0, so this causal
        ∂(global_out, global_lse, dglobal_out, q, k, v, temp_dq, temp_dk, temp_dv)
        # ✅ ∂(global out, global LSE, global ∂out, q0, k0, v0, ∂q0, ∂k0, ∂v0)
        # dk, dv from ∂(q0, k0, v0)

        if dq is None: True
            dq, dk, dv = temp_dq, temp_dk, temp_dv
        else: False # not required to update with other devices ∂
    
    elif step != 0: False
    if step + 1 != kv_comm.world_size: True
        k, v = k1, v1
    
    send dk, dv to send # dk, dv from ∂(q0, k0, v0)
    receive dk1, dv1 from receive # dk, dv from ∂(q1, k1, v1)

if step == 1
    if step + 1 != comm.world_size: False
    if not is_causal or step <= comm.rank: False
    elif step != 0: True
        dk, dv = dk1, dv1 # dk, dv from ∂(q1, k1, v1)
    if step + 1 != kv_comm.world_size: False

    send dk, dv to send # dk, dv from ∂(q1, k1, v1)
    receive dk1, dv1 from receive # dk, dv from ∂(q1, k0, v0) + ∂(q0, k0, v0)
    # dk1, dv1 is your final local dk, dv, ∂(q1, k0, v0) + ∂(q0, k0, v0) done in device 1
```

- For GPU 1, attention calculated as q1k0v0 + q1k1v1,
    - to calculate ∂q1, you required to calculate ∂(global out, global LSE, global ∂out, q1, k0, v0, ∂q1, ∂k0, ∂v0) + ∂(global out, global LSE, global ∂out, q1, k1, v1, ∂q1, ∂k1, ∂v1)
    - to calculate ∂k1, you required to calculate ∂(global out, global LSE, global ∂out, q0, k1, v1, ∂q0, ∂k1, ∂v1) + ∂(global out, global LSE, global ∂out, q1, k1, v1, ∂q1, ∂k1, ∂v1)
    - to calculate ∂v1, you required to calculate ∂(global out, global LSE, global ∂out, q0, k1, v1, ∂q0, ∂k1, ∂v1) + ∂(global out, global LSE, global ∂out, q1, k1, v1, ∂q1, ∂k1, ∂v1)

```
cp_rank = 1
cp_world_size = 2
send = (cp_rank + 1) % cp_world_size = 1
receive = (cp_rank - 1) % cp_world_size = 1
q = q1
k = k1
v = v1
dq, dk, dv = None, None, None
next_dk, next_dv = None, None, None
temp_dq, temp_dk, temp_dv = torch.empty(q.shape), torch.empty(q.shape), torch.empty(q.shape)
global_out = out1 # same shape q0
dglobal_out # same shape as global_out, received from autograd
global_lse # from forward, sync across devices, shape [B, H, L]
step == 0
    if step + 1 != comm.world_size: True
        send k, v to send
        receive k0, v0 from receive
    
    if not is_causal or step <= comm.rank: True
        calculate ∂ with is_causal and step == 0, so this causal
        ∂(global_out, global_lse, dglobal_out, q, k, v, temp_dq, temp_dk, temp_dv)
        # ✅ ∂(global out, global LSE, global ∂out, q1, k1, v1, ∂q1, ∂k1, ∂v1)
        # dk, dv from q1k1v1

        if dq is None: True
            dq, dk, dv = temp_dq, temp_dk, temp_dv
        else: False # not required to update with other devices ∂
    
    elif step != 0: False
    if step + 1 != kv_comm.world_size: True
        k, v = k0, v0
    
    send dk, dv to send # dk, dv from q1k1v1
    receive dk0, dv0 from receive # dk, dv from q0k0v0

if step == 1
    if step + 1 != comm.world_size: False
    if not is_causal or step <= comm.rank: True
        calculate ∂ with is_causal and step == 1, full attention
        ∂(global_out, global_lse, dglobal_out, q, k, v, temp_dq, temp_dk, temp_dv)
        # ✅ ∂(global out, global LSE, global ∂out, q1, k0, v0, ∂q1, ∂k0, ∂v0)
        # dk, dv from q1k0v0

        if dq is None: False
        else: True
            dq += temp_dq # ∂(q1, k1, v1) + ∂(q1, k0, v0)
            dk = temp_dk + dk0 # ∂(q1, k0, v0) + ∂(q0, k0, v0)
            dv = temp_dv + dv0 # ∂(q1, k0, v0) + ∂(q0, k0, v0)
    
    elif step != 0: False
    if step + 1 != kv_comm.world_size: False

    send dk, dv to send # dk, dv from ∂(q1, k0, v0) + ∂(q0, k0, v0)
    receive dk0, dv0 from receive # dk, dv from ∂(q1, k1, v1)
    # dk0, dv0 is your final local dk, dv, ∂(q1, k1, v1) done in device 0
    # we are not required to do ∂(q0, k1, v1) because empty masking
```

Voila!

#### Limitation

1. Some GPUs required less computation such as GPU 0, not required to compute q0k1v1, introduced unbalanced.