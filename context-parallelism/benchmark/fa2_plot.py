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

def run(seqlen):
    world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    device = torch.device(f'cuda:{local_rank}')
    dtype = torch.bfloat16

    assert seqlen % world_size == 0
    assert d % 8 == 0

    qkv = torch.randn(
        3, batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True,
    )
    dist.broadcast(qkv, src=0)

    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    local_qkv = qkv.chunk(world_size, dim=-3)[local_rank]
    local_dout = dout.chunk(world_size, dim=-3)[local_rank].clone().detach()

    local_q = local_qkv[0].clone().detach().requires_grad_(True)
    local_k = local_qkv[1].clone().detach().requires_grad_(True)
    local_v = local_qkv[2].clone().detach().requires_grad_(True)

    local_out, local_lse = ring_flash_attn(q=local_q, k=local_k, v=local_v, causal=True)
    local_out.backward(local_dout)

if __name__ == "__main__":
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])

    seqlens = [1024, 2048, 4096, 8192, 16384, 32768] + [10240 * i for i in range(4, 20, 1)]

    for seqlen in seqlens:
    
        warmup = 3
        repeat = 10
        for _ in range(warmup):
            run(seqlen)
            torch.cuda.synchronize()
            dist.barrier()
        
        torch.cuda.synchronize()
        dist.barrier()
        start = time.time()
        for _ in range(repeat):
            run(seqlen)
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
    
    dist.destroy_process_group()

"""
torchrun \
--nproc_per_node 4 \
benchmark/fa2_plot.py

seqlen: 1024
Average step time: 0.0025 sec
Throughput: 417526.25 tokens/sec
TFLOPs/sec: 10.51
seqlen: 2048
Average step time: 0.0032 sec
Throughput: 648096.41 tokens/sec
TFLOPs/sec: 32.62
seqlen: 4096
Average step time: 0.0033 sec
Throughput: 1228818.75 tokens/sec
TFLOPs/sec: 123.70
seqlen: 8192
Average step time: 0.0044 sec
Throughput: 1872506.12 tokens/sec
TFLOPs/sec: 376.99
seqlen: 16384
Average step time: 0.0091 sec
Throughput: 1809602.57 tokens/sec
TFLOPs/sec: 728.64
seqlen: 32768
Average step time: 0.0267 sec
Throughput: 1226286.06 tokens/sec
TFLOPs/sec: 987.54
seqlen: 40960
Average step time: 0.0389 sec
Throughput: 1053103.63 tokens/sec
TFLOPs/sec: 1060.09
seqlen: 51200
Average step time: 0.0600 sec
Throughput: 853371.45 tokens/sec
TFLOPs/sec: 1073.79
seqlen: 61440
Average step time: 0.0812 sec
Throughput: 756870.22 tokens/sec
TFLOPs/sec: 1142.84
seqlen: 71680
Average step time: 0.1095 sec
Throughput: 654716.20 tokens/sec
TFLOPs/sec: 1153.35
seqlen: 81920
Average step time: 0.1394 sec
Throughput: 587458.33 tokens/sec
TFLOPs/sec: 1182.71
seqlen: 92160
Average step time: 0.1733 sec
Throughput: 531915.60 tokens/sec
TFLOPs/sec: 1204.75
seqlen: 102400
Average step time: 0.2138 sec
Throughput: 479045.72 tokens/sec
TFLOPs/sec: 1205.56
seqlen: 112640
Average step time: 0.2542 sec
Throughput: 443086.27 tokens/sec
TFLOPs/sec: 1226.57
seqlen: 122880
Average step time: 0.3043 sec
Throughput: 403871.74 tokens/sec
TFLOPs/sec: 1219.65
seqlen: 133120
Average step time: 0.3500 sec
Throughput: 380303.25 tokens/sec
TFLOPs/sec: 1244.18
seqlen: 143360
Average step time: 0.4055 sec
Throughput: 353572.30 tokens/sec
TFLOPs/sec: 1245.71
seqlen: 153600
Average step time: 0.4645 sec
Throughput: 330654.56 tokens/sec
TFLOPs/sec: 1248.18
seqlen: 163840
Average step time: 0.5297 sec
Throughput: 309280.68 tokens/sec
TFLOPs/sec: 1245.33
seqlen: 174080
Average step time: 0.5915 sec
Throughput: 294325.33 tokens/sec
TFLOPs/sec: 1259.18
seqlen: 184320
Average step time: 0.6594 sec
Throughput: 279512.22 tokens/sec
TFLOPs/sec: 1266.15
seqlen: 194560
Average step time: 0.7369 sec
Throughput: 264010.73 tokens/sec
TFLOPs/sec: 1262.37
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