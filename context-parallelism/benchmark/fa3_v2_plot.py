import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from ring_fa3_v2 import ring_flash_attn
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
benchmark/fa3_v2_plot.py

seqlen: 1024
Average step time: 0.0031 sec
Throughput: 331039.08 tokens/sec
TFLOPs/sec: 8.33
seqlen: 2048
Average step time: 0.0036 sec
Throughput: 567554.32 tokens/sec
TFLOPs/sec: 28.57
seqlen: 4096
Average step time: 0.0044 sec
Throughput: 936049.01 tokens/sec
TFLOPs/sec: 94.23
seqlen: 8192
Average step time: 0.0044 sec
Throughput: 1855457.79 tokens/sec
TFLOPs/sec: 373.55
seqlen: 16384
Average step time: 0.0063 sec
Throughput: 2614697.39 tokens/sec
TFLOPs/sec: 1052.82
seqlen: 32768
Average step time: 0.0156 sec
Throughput: 2102329.55 tokens/sec
TFLOPs/sec: 1693.02
seqlen: 40960
Average step time: 0.0222 sec
Throughput: 1841207.92 tokens/sec
TFLOPs/sec: 1853.42
seqlen: 51200
Average step time: 0.0338 sec
Throughput: 1514791.57 tokens/sec
TFLOPs/sec: 1906.05
seqlen: 61440
Average step time: 0.0468 sec
Throughput: 1311715.42 tokens/sec
TFLOPs/sec: 1980.62
seqlen: 71680
Average step time: 0.0620 sec
Throughput: 1155816.34 tokens/sec
TFLOPs/sec: 2036.09
seqlen: 81920
Average step time: 0.0790 sec
Throughput: 1036923.27 tokens/sec
TFLOPs/sec: 2087.60
seqlen: 92160
Average step time: 0.0991 sec
Throughput: 929683.72 tokens/sec
TFLOPs/sec: 2105.66
seqlen: 102400
Average step time: 0.1206 sec
Throughput: 848963.80 tokens/sec
TFLOPs/sec: 2136.49
seqlen: 112640
Average step time: 0.1437 sec
Throughput: 783753.42 tokens/sec
TFLOPs/sec: 2169.62
seqlen: 122880
Average step time: 0.1742 sec
Throughput: 705332.15 tokens/sec
TFLOPs/sec: 2130.03
seqlen: 133120
Average step time: 0.1935 sec
Throughput: 688046.34 tokens/sec
TFLOPs/sec: 2250.98
seqlen: 143360
Average step time: 0.2312 sec
Throughput: 620010.69 tokens/sec
TFLOPs/sec: 2184.43
seqlen: 153600
Average step time: 0.2605 sec
Throughput: 589559.99 tokens/sec
TFLOPs/sec: 2225.51
seqlen: 163840
Average step time: 0.2925 sec
Throughput: 560138.09 tokens/sec
TFLOPs/sec: 2255.41
seqlen: 174080
Average step time: 0.3317 sec
Throughput: 524775.30 tokens/sec
TFLOPs/sec: 2245.09
seqlen: 184320
Average step time: 0.3652 sec
Throughput: 504749.10 tokens/sec
TFLOPs/sec: 2286.44
seqlen: 194560
Average step time: 0.4079 sec
Throughput: 477027.05 tokens/sec
TFLOPs/sec: 2280.91
seqlen: 204800
Average step time: 0.4543 sec
Throughput: 450758.80 tokens/sec
TFLOPs/sec: 2268.74
seqlen: 215040
Average step time: 0.5056 sec
Throughput: 425354.02 tokens/sec
TFLOPs/sec: 2247.92
seqlen: 225280
Average step time: 0.5469 sec
Throughput: 411931.70 tokens/sec
TFLOPs/sec: 2280.65
seqlen: 235520
Average step time: 0.6061 sec
Throughput: 388596.63 tokens/sec
TFLOPs/sec: 2249.25
seqlen: 245760
Average step time: 0.6546 sec
Throughput: 375444.71 tokens/sec
TFLOPs/sec: 2267.61
seqlen: 256000
Average step time: 0.7111 sec
Throughput: 359997.75 tokens/sec
TFLOPs/sec: 2264.91
seqlen: 266240
Average step time: 0.7634 sec
Throughput: 348754.49 tokens/sec
TFLOPs/sec: 2281.94
seqlen: 276480
Average step time: 0.8354 sec
Throughput: 330969.23 tokens/sec
TFLOPs/sec: 2248.86
seqlen: 286720
Average step time: 0.8748 sec
Throughput: 327763.57 tokens/sec
TFLOPs/sec: 2309.56
seqlen: 296960
Average step time: 0.9512 sec
Throughput: 312202.37 tokens/sec
TFLOPs/sec: 2278.48
seqlen: 307200
Average step time: 1.0302 sec
Throughput: 298192.84 tokens/sec
TFLOPs/sec: 2251.28
seqlen: 317440
Average step time: 1.0902 sec
Throughput: 291182.05 tokens/sec
TFLOPs/sec: 2271.63
seqlen: 327680
Average step time: 1.1415 sec
Throughput: 287059.83 tokens/sec
TFLOPs/sec: 2311.71
seqlen: 337920
Average step time: 1.2478 sec
Throughput: 270803.22 tokens/sec
TFLOPs/sec: 2248.95
seqlen: 348160
Average step time: 1.3093 sec
Throughput: 265922.03 tokens/sec
TFLOPs/sec: 2275.33
seqlen: 358400
Average step time: 1.3768 sec
Throughput: 260313.65 tokens/sec
TFLOPs/sec: 2292.85
seqlen: 368640
Average step time: 1.4688 sec
Throughput: 250976.62 tokens/sec
TFLOPs/sec: 2273.77
seqlen: 378880
Average step time: 1.5359 sec
Throughput: 246688.77 tokens/sec
TFLOPs/sec: 2297.01
seqlen: 389120
Average step time: 1.6217 sec
Throughput: 239952.90 tokens/sec
TFLOPs/sec: 2294.67
seqlen: 399360
Average step time: 1.6966 sec
Throughput: 235383.38 tokens/sec
TFLOPs/sec: 2310.21
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