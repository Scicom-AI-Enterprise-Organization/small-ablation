import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from ring_fa2 import ring_flash_attn
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
benchmark/fa2_plot.py

seqlen: 1024
Average step time: 0.0051 sec
Throughput: 201305.20 tokens/sec
TFLOPs/sec: 5.07
seqlen: 2048
Average step time: 0.0075 sec
Throughput: 273955.66 tokens/sec
TFLOPs/sec: 13.79
seqlen: 4096
Average step time: 0.0097 sec
Throughput: 420706.08 tokens/sec
TFLOPs/sec: 42.35
seqlen: 8192
Average step time: 0.0138 sec
Throughput: 595212.94 tokens/sec
TFLOPs/sec: 119.83
seqlen: 16384
Average step time: 0.0125 sec
Throughput: 1308070.37 tokens/sec
TFLOPs/sec: 526.70
seqlen: 32768
Average step time: 0.0251 sec
Throughput: 1304980.25 tokens/sec
TFLOPs/sec: 1050.91
seqlen: 40960
Average step time: 0.0370 sec
Throughput: 1106305.94 tokens/sec
TFLOPs/sec: 1113.64
seqlen: 51200
Average step time: 0.0578 sec
Throughput: 886527.59 tokens/sec
TFLOPs/sec: 1115.51
seqlen: 61440
Average step time: 0.0783 sec
Throughput: 784774.70 tokens/sec
TFLOPs/sec: 1184.97
seqlen: 71680
Average step time: 0.1059 sec
Throughput: 676896.35 tokens/sec
TFLOPs/sec: 1192.43
seqlen: 81920
Average step time: 0.1376 sec
Throughput: 595153.56 tokens/sec
TFLOPs/sec: 1198.20
seqlen: 92160
Average step time: 0.1707 sec
Throughput: 539900.50 tokens/sec
TFLOPs/sec: 1222.83
seqlen: 102400
Average step time: 0.2100 sec
Throughput: 487646.97 tokens/sec
TFLOPs/sec: 1227.20
seqlen: 112640
Average step time: 0.2503 sec
Throughput: 450009.21 tokens/sec
TFLOPs/sec: 1245.73
seqlen: 122880
Average step time: 0.3011 sec
Throughput: 408037.49 tokens/sec
TFLOPs/sec: 1232.23
seqlen: 133120
Average step time: 0.3450 sec
Throughput: 385902.19 tokens/sec
TFLOPs/sec: 1262.50
seqlen: 143360
Average step time: 0.4030 sec
Throughput: 355767.60 tokens/sec
TFLOPs/sec: 1253.45
seqlen: 153600
Average step time: 0.4604 sec
Throughput: 333628.80 tokens/sec
TFLOPs/sec: 1259.41
seqlen: 163840
Average step time: 0.5262 sec
Throughput: 311389.61 tokens/sec
TFLOPs/sec: 1253.82
seqlen: 174080
Average step time: 0.5891 sec
Throughput: 295511.14 tokens/sec
TFLOPs/sec: 1264.25
seqlen: 184320
Average step time: 0.6540 sec
Throughput: 281846.02 tokens/sec
TFLOPs/sec: 1276.72
seqlen: 194560
Average step time: 0.7326 sec
Throughput: 265566.72 tokens/sec
TFLOPs/sec: 1269.81
seqlen: 204800
Average step time: 0.8091 sec
Throughput: 253110.22 tokens/sec
TFLOPs/sec: 1273.95
seqlen: 215040
Average step time: 0.8902 sec
Throughput: 241556.02 tokens/sec
TFLOPs/sec: 1276.58
seqlen: 225280
Average step time: 0.9753 sec
Throughput: 230991.22 tokens/sec
TFLOPs/sec: 1278.88
seqlen: 235520
Average step time: 1.0584 sec
Throughput: 222517.52 tokens/sec
TFLOPs/sec: 1287.96
seqlen: 245760
Average step time: 1.1591 sec
Throughput: 212026.67 tokens/sec
TFLOPs/sec: 1280.60
seqlen: 256000
Average step time: 1.2536 sec
Throughput: 204208.36 tokens/sec
TFLOPs/sec: 1284.77
seqlen: 266240
Average step time: 1.3596 sec
Throughput: 195820.18 tokens/sec
TFLOPs/sec: 1281.27
seqlen: 276480
Average step time: 1.4645 sec
Throughput: 188782.90 tokens/sec
TFLOPs/sec: 1282.74
seqlen: 286720
Average step time: 1.5823 sec
Throughput: 181202.34 tokens/sec
TFLOPs/sec: 1276.83
seqlen: 296960
Average step time: 1.6944 sec
Throughput: 175264.25 tokens/sec
TFLOPs/sec: 1279.09
seqlen: 307200
Average step time: 1.8004 sec
Throughput: 170629.85 tokens/sec
TFLOPs/sec: 1288.21
seqlen: 317440
Average step time: 1.9298 sec
Throughput: 164490.84 tokens/sec
TFLOPs/sec: 1283.26
seqlen: 327680
Average step time: 2.0479 sec
Throughput: 160009.04 tokens/sec
TFLOPs/sec: 1288.56
seqlen: 337920
Average step time: 2.1815 sec
Throughput: 154904.64 tokens/sec
TFLOPs/sec: 1286.44
seqlen: 348160
Average step time: 2.3101 sec
Throughput: 150709.82 tokens/sec
TFLOPs/sec: 1289.53
seqlen: 358400
Average step time: 2.4576 sec
Throughput: 145830.38 tokens/sec
TFLOPs/sec: 1284.48
seqlen: 368640
Average step time: 2.5878 sec
Throughput: 142451.87 tokens/sec
TFLOPs/sec: 1290.57
seqlen: 378880
Average step time: 2.7278 sec
Throughput: 138898.22 tokens/sec
TFLOPs/sec: 1293.33
seqlen: 389120
Average step time: 2.8730 sec
Throughput: 135442.27 tokens/sec
TFLOPs/sec: 1295.24
seqlen: 399360
Average step time: 3.0239 sec
Throughput: 132067.54 tokens/sec
TFLOPs/sec: 1296.20
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