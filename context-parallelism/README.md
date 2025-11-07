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

For GPU 0,

```
    cp_rank = 0
    cp_world_size = 2
    send = (cp_rank + 1) % cp_world_size = 1
    receive = (cp_rank - 1) % cp_world_size = 1
    k = k0
    v = v0

    step == 0
        if step + 1 != comm.world_size: True
            send k, v to send
            receive k1, v1 from receive

        if not is_causal or step <= comm.rank: True
            calculate attention with is_causal and step == 0, so this causal

        if step + 1 != comm.world_size: True
        k = k1
        v = v1

    step == 1

        if step + 1 != comm.world_size: False
        if not is_causal or step <= comm.rank: False
        if step + 1 != comm.world_size: False
```

For GPU 1,

```
    cp_rank = 1
    cp_world_size = 2
    send = (cp_rank + 1) % cp_world_size = 0
    receive = (cp_rank - 1) % cp_world_size = 0
    k = k1
    v = v1

    step == 0
        if step + 1 != comm.world_size: True
            send k, v to send
            receive k0, v0 from receive

        if not is_causal or step <= comm.rank: True
            calculate attention with is_causal and step == 0, so this causal

        if step + 1 != comm.world_size: True
        k = k0
        v = v0

    step == 1
        if step + 1 != comm.world_size: False

        if not is_causal or step <= comm.rank: True
            calculate attention with is_causal and step == 1, full attention

        if step + 1 != comm.world_size: False
```

Everything hit as it is.
