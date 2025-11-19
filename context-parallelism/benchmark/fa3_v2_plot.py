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
Average step time: 0.0036 sec
Throughput: 281054.82 tokens/sec
TFLOPs/sec: 7.07
seqlen: 2048
Average step time: 0.0036 sec
Throughput: 572219.79 tokens/sec
TFLOPs/sec: 28.80
seqlen: 4096
Average step time: 0.0047 sec
Throughput: 880477.10 tokens/sec
TFLOPs/sec: 88.63
seqlen: 8192
Average step time: 0.0039 sec
Throughput: 2115382.72 tokens/sec
TFLOPs/sec: 425.88
seqlen: 16384
Average step time: 0.0059 sec
Throughput: 2786022.62 tokens/sec
TFLOPs/sec: 1121.80
seqlen: 32768
Average step time: 0.0156 sec
Throughput: 2104202.83 tokens/sec
TFLOPs/sec: 1694.53
seqlen: 40960
Average step time: 0.0230 sec
Throughput: 1780769.74 tokens/sec
TFLOPs/sec: 1792.58
seqlen: 51200
Average step time: 0.0344 sec
Throughput: 1488960.91 tokens/sec
TFLOPs/sec: 1873.55
seqlen: 61440
Average step time: 0.0489 sec
Throughput: 1256899.28 tokens/sec
TFLOPs/sec: 1897.85
seqlen: 71680
Average step time: 0.0632 sec
Throughput: 1133979.63 tokens/sec
TFLOPs/sec: 1997.63
seqlen: 81920
Average step time: 0.0802 sec
Throughput: 1021790.79 tokens/sec
TFLOPs/sec: 2057.14
seqlen: 92160
Average step time: 0.0997 sec
Throughput: 924379.11 tokens/sec
TFLOPs/sec: 2093.65
seqlen: 102400
Average step time: 0.1232 sec
Throughput: 831377.75 tokens/sec
TFLOPs/sec: 2092.23
seqlen: 112640
Average step time: 0.1467 sec
Throughput: 768000.73 tokens/sec
TFLOPs/sec: 2126.01
seqlen: 122880
Average step time: 0.1763 sec
Throughput: 697094.44 tokens/sec
TFLOPs/sec: 2105.15
seqlen: 133120
Average step time: 0.2006 sec
Throughput: 663578.24 tokens/sec
TFLOPs/sec: 2170.93
seqlen: 143360
Average step time: 0.2303 sec
Throughput: 622610.23 tokens/sec
TFLOPs/sec: 2193.59
seqlen: 153600
Average step time: 0.2653 sec
Throughput: 578945.38 tokens/sec
TFLOPs/sec: 2185.45
seqlen: 163840
Average step time: 0.2976 sec
Throughput: 550625.54 tokens/sec
TFLOPs/sec: 2217.11
seqlen: 174080
Average step time: 0.3380 sec
Throughput: 515023.83 tokens/sec
TFLOPs/sec: 2203.37
seqlen: 184320
Average step time: 0.3750 sec
Throughput: 491505.94 tokens/sec
TFLOPs/sec: 2226.45
seqlen: 194560
Average step time: 0.4171 sec
Throughput: 466409.03 tokens/sec
TFLOPs/sec: 2230.14
seqlen: 204800
Average step time: 0.4583 sec
Throughput: 446825.28 tokens/sec
TFLOPs/sec: 2248.95
seqlen: 215040
Average step time: 0.5218 sec
Throughput: 412143.68 tokens/sec
TFLOPs/sec: 2178.11
seqlen: 225280
Average step time: 0.5573 sec
Throughput: 404267.01 tokens/sec
TFLOPs/sec: 2238.22
seqlen: 235520
Average step time: 0.6009 sec
Throughput: 391936.13 tokens/sec
TFLOPs/sec: 2268.58
seqlen: 245760
Average step time: 0.6680 sec
Throughput: 367902.86 tokens/sec
TFLOPs/sec: 2222.06
seqlen: 256000
Average step time: 0.7183 sec
Throughput: 356397.80 tokens/sec
TFLOPs/sec: 2242.26
seqlen: 266240
Average step time: 0.7646 sec
Throughput: 348210.80 tokens/sec
TFLOPs/sec: 2278.38
seqlen: 276480
Average step time: 0.8353 sec
Throughput: 331006.22 tokens/sec
TFLOPs/sec: 2249.11
seqlen: 286720
Average step time: 0.9002 sec
Throughput: 318520.67 tokens/sec
TFLOPs/sec: 2244.43
seqlen: 296960
Average step time: 0.9803 sec
Throughput: 302922.28 tokens/sec
TFLOPs/sec: 2210.75
seqlen: 307200
Average step time: 1.0040 sec
Throughput: 305961.87 tokens/sec
TFLOPs/sec: 2309.93
seqlen: 317440
Average step time: 1.1043 sec
Throughput: 287450.16 tokens/sec
TFLOPs/sec: 2242.52
seqlen: 327680
Average step time: 1.1824 sec
Throughput: 277122.51 tokens/sec
TFLOPs/sec: 2231.69
seqlen: 337920
Average step time: 1.2326 sec
Throughput: 274148.56 tokens/sec
TFLOPs/sec: 2276.73
seqlen: 348160
Average step time: 1.3257 sec
Throughput: 262626.78 tokens/sec
TFLOPs/sec: 2247.13
seqlen: 358400
Average step time: 1.3857 sec
Throughput: 258632.63 tokens/sec
TFLOPs/sec: 2278.05
seqlen: 368640
Average step time: 1.4896 sec
Throughput: 247469.07 tokens/sec
TFLOPs/sec: 2241.99
seqlen: 378880
Average step time: 1.5489 sec
Throughput: 244612.82 tokens/sec
TFLOPs/sec: 2277.68
seqlen: 389120
Average step time: 1.6637 sec
Throughput: 233882.91 tokens/sec
TFLOPs/sec: 2236.63
seqlen: 399360
Average step time: 1.7257 sec
Throughput: 231425.48 tokens/sec
TFLOPs/sec: 2271.37
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