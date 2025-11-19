import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from ring_fa3 import ring_flash_attn
import torch
import torch.distributed as dist
import time

batch_size = 1
nheads = 16
d = 128

def create(seqlen):
    world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    device = torch.device(f'cuda:{local_rank}')
    dtype = torch.bfloat16

    qkv = torch.randn(
        3, batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True,
    )
    dist.broadcast(qkv, src=0)

    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    local_qkv = qkv.chunk(world_size, dim=-3)[local_rank]
    local_dout = dout.chunk(world_size, dim=-3)[local_rank].clone().detach()

    del qkv, dout
    torch.cuda.empty_cache()

    return local_qkv, local_dout

def run(local_qkv, local_dout):

    local_q = local_qkv[0].clone().detach().requires_grad_(True)
    local_k = local_qkv[1].clone().detach().requires_grad_(True)
    local_v = local_qkv[2].clone().detach().requires_grad_(True)

    local_out, local_lse = ring_flash_attn(q=local_q, k=local_k, v=local_v, causal=True)
    local_out.backward(local_dout)

if __name__ == "__main__":
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])

    seqlens = [1024, 2048, 4096, 8192, 16384, 32768] + [10240 * i for i in range(4, 40, 1)]
    warmup = 3
    repeat = 5

    for seqlen in seqlens:
        local_qkv, local_dout = create(seqlen)
    
        for _ in range(warmup):
            run(local_qkv, local_dout)
            torch.cuda.synchronize()
            dist.barrier()
        
        torch.cuda.synchronize()
        dist.barrier()
        start = time.time()
        for _ in range(repeat):
            run(local_qkv, local_dout)
            torch.cuda.synchronize()
            dist.barrier()
    
        end = time.time()
        avg_time = (end - start) / repeat
        flops_per_step = 12 * batch_size * nheads * (seqlen ** 2) * d
        tflops = (flops_per_step / avg_time) / 1e12
        throughput = seqlen / avg_time
        if local_rank == 0:
            print(f"seqlen: {seqlen}")
            print(f"Average step time: {avg_time:.4f} sec")
            print(f"Throughput: {throughput:.2f} tokens/sec")
            print(f"TFLOPs/sec: {tflops:.2f}")
        
        del local_qkv, local_dout
        torch.cuda.empty_cache()
    
    dist.destroy_process_group()

"""
torchrun \
--nproc_per_node 4 \
benchmark/fa3_plot.py

seqlen: 1024
Average step time: 0.0048 sec
Throughput: 214939.66 tokens/sec
TFLOPs/sec: 5.41
seqlen: 2048
Average step time: 0.0059 sec
Throughput: 346743.04 tokens/sec
TFLOPs/sec: 17.45
seqlen: 4096
Average step time: 0.0061 sec
Throughput: 666604.68 tokens/sec
TFLOPs/sec: 67.10
seqlen: 8192
Average step time: 0.0070 sec
Throughput: 1177050.03 tokens/sec
TFLOPs/sec: 236.97
seqlen: 16384
Average step time: 0.0081 sec
Throughput: 2026896.00 tokens/sec
TFLOPs/sec: 816.14
seqlen: 32768
Average step time: 0.0158 sec
Throughput: 2069807.65 tokens/sec
TFLOPs/sec: 1666.83
seqlen: 40960
Average step time: 0.0224 sec
Throughput: 1829405.00 tokens/sec
TFLOPs/sec: 1841.54
seqlen: 51200
Average step time: 0.0333 sec
Throughput: 1539039.69 tokens/sec
TFLOPs/sec: 1936.56
seqlen: 61440
Average step time: 0.0465 sec
Throughput: 1322423.58 tokens/sec
TFLOPs/sec: 1996.79
seqlen: 71680
Average step time: 0.0627 sec
Throughput: 1144126.87 tokens/sec
TFLOPs/sec: 2015.50
seqlen: 81920
Average step time: 0.0797 sec
Throughput: 1028033.53 tokens/sec
TFLOPs/sec: 2069.70
seqlen: 92160
Average step time: 0.0981 sec
Throughput: 939406.67 tokens/sec
TFLOPs/sec: 2127.68
seqlen: 102400
Average step time: 0.1197 sec
Throughput: 855615.15 tokens/sec
TFLOPs/sec: 2153.23
seqlen: 112640
Average step time: 0.1482 sec
Throughput: 759962.60 tokens/sec
TFLOPs/sec: 2103.76
seqlen: 122880
Average step time: 0.1747 sec
Throughput: 703364.84 tokens/sec
TFLOPs/sec: 2124.09
seqlen: 133120
Average step time: 0.1998 sec
Throughput: 666248.89 tokens/sec
TFLOPs/sec: 2179.67
seqlen: 143360
Average step time: 0.2306 sec
Throughput: 621793.08 tokens/sec
TFLOPs/sec: 2190.71
seqlen: 153600
Average step time: 0.2598 sec
Throughput: 591198.40 tokens/sec
TFLOPs/sec: 2231.70
seqlen: 163840
Average step time: 0.2907 sec
Throughput: 563648.81 tokens/sec
TFLOPs/sec: 2269.55
seqlen: 174080
Average step time: 0.3363 sec
Throughput: 517603.58 tokens/sec
TFLOPs/sec: 2214.41
seqlen: 184320
Average step time: 0.3674 sec
Throughput: 501658.59 tokens/sec
TFLOPs/sec: 2272.44
seqlen: 194560
Average step time: 0.4141 sec
Throughput: 469783.10 tokens/sec
TFLOPs/sec: 2246.27
seqlen: 204800
Average step time: 0.4557 sec
Throughput: 449373.09 tokens/sec
TFLOPs/sec: 2261.77
seqlen: 215040
Average step time: 0.5113 sec
Throughput: 420562.97 tokens/sec
TFLOPs/sec: 2222.60
seqlen: 225280
Average step time: 0.5546 sec
Throughput: 406167.89 tokens/sec
TFLOPs/sec: 2248.74
seqlen: 235520
Average step time: 0.6116 sec
Throughput: 385107.74 tokens/sec
TFLOPs/sec: 2229.06
seqlen: 245760
Average step time: 0.6676 sec
Throughput: 368103.66 tokens/sec
TFLOPs/sec: 2223.27
seqlen: 256000
Average step time: 0.7128 sec
Throughput: 359145.74 tokens/sec
TFLOPs/sec: 2259.55
seqlen: 266240
Average step time: 0.7788 sec
Throughput: 341838.63 tokens/sec
TFLOPs/sec: 2236.69
seqlen: 276480
Average step time: 0.8278 sec
Throughput: 334007.48 tokens/sec
TFLOPs/sec: 2269.50
seqlen: 286720
Average step time: 0.8992 sec
Throughput: 318865.42 tokens/sec
TFLOPs/sec: 2246.86
seqlen: 296960
Average step time: 0.9565 sec
Throughput: 310452.65 tokens/sec
TFLOPs/sec: 2265.71
seqlen: 307200
Average step time: 1.0190 sec
Throughput: 301474.36 tokens/sec
TFLOPs/sec: 2276.06
seqlen: 317440
Average step time: 1.1090 sec
Throughput: 286250.73 tokens/sec
TFLOPs/sec: 2233.16
seqlen: 327680
Average step time: 1.1587 sec
Throughput: 282788.22 tokens/sec
TFLOPs/sec: 2277.31
seqlen: 337920
Average step time: 1.2533 sec
Throughput: 269629.53 tokens/sec
TFLOPs/sec: 2239.20
seqlen: 348160
Average step time: 1.3346 sec
Throughput: 260873.76 tokens/sec
TFLOPs/sec: 2232.14
seqlen: 358400
Average step time: 1.4231 sec
Throughput: 251841.79 tokens/sec
TFLOPs/sec: 2218.23
seqlen: 368640
Average step time: 1.4903 sec
Throughput: 247366.27 tokens/sec
TFLOPs/sec: 2241.06
seqlen: 378880
Average step time: 1.5447 sec
Throughput: 245270.69 tokens/sec
TFLOPs/sec: 2283.80
seqlen: 389120
Average step time: 1.6238 sec
Throughput: 239634.25 tokens/sec
TFLOPs/sec: 2291.63
seqlen: 399360
Average step time: 1.7559 sec
Throughput: 227438.05 tokens/sec
TFLOPs/sec: 2232.23
"""

"""
root@ff9250fe4fe3:/# nvidia-smi
Sun Nov  9 07:30:12 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.05              Driver Version: 560.35.05      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:06:00.0 Off |                    0 |
| N/A   26C    P0             70W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA H100 80GB HBM3          On  |   00000000:07:00.0 Off |                    0 |
| N/A   26C    P0             70W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA H100 80GB HBM3          On  |   00000000:08:00.0 Off |                    0 |
| N/A   32C    P0             73W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA H100 80GB HBM3          On  |   00000000:09:00.0 Off |                    0 |
| N/A   38C    P0             72W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

root@ff9250fe4fe3:/# nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    NIC0    NIC1    NIC2    NIC3    NIC4    NIC5    NIC6    NIC7    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV18    NV18    NV18    PHB     PHB     PHB     PHB     PHB     PHB     PHB     PHB     0-79    0               N/A
GPU1    NV18     X      NV18    NV18    PHB     PHB     PHB     PHB     PHB     PHB     PHB     PHB     0-79    0               N/A
GPU2    NV18    NV18     X      NV18    PHB     PHB     PHB     PHB     PHB     PHB     PHB     PHB     0-79    0               N/A
GPU3    NV18    NV18    NV18     X      PHB     PHB     PHB     PHB     PHB     PHB     PHB     PHB     0-79    0               N/A
NIC0    PHB     PHB     PHB     PHB      X      PHB     PHB     PHB     PHB     PHB     PHB     PHB
NIC1    PHB     PHB     PHB     PHB     PHB      X      PHB     PHB     PHB     PHB     PHB     PHB
NIC2    PHB     PHB     PHB     PHB     PHB     PHB      X      PHB     PHB     PHB     PHB     PHB
NIC3    PHB     PHB     PHB     PHB     PHB     PHB     PHB      X      PHB     PHB     PHB     PHB
NIC4    PHB     PHB     PHB     PHB     PHB     PHB     PHB     PHB      X      PHB     PHB     PHB
NIC5    PHB     PHB     PHB     PHB     PHB     PHB     PHB     PHB     PHB      X      PHB     PHB
NIC6    PHB     PHB     PHB     PHB     PHB     PHB     PHB     PHB     PHB     PHB      X      PHB
NIC7    PHB     PHB     PHB     PHB     PHB     PHB     PHB     PHB     PHB     PHB     PHB      X 

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2
  NIC3: mlx5_3
  NIC4: mlx5_4
  NIC5: mlx5_5
  NIC6: mlx5_6
  NIC7: mlx5_7
"""