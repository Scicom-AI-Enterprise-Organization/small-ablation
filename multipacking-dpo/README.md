# Multipacking for DPO

We want to calculate sum inputs logprob and sum reference logprob without padding and lastly calculate DPO.

## Memory study

### [Liger kernel](memory-profiling-liger.ipynb)

<img src="liger-memory.png" width="50%">

Problem with liger kernel DPO, the chunk happened on first dimension, which is the first dimension, if you are flash attention maxxing, we know flash attention designed for ragged tensor, 0 padding, so concatenated dimension happened on sequence dimension.

Average memory requirement for 10k sequence dimension is 44.6GB

### [Naive Pytorch chunking](memory-profiling-chunk.ipynb)

<img src="chunk-pytorch-memory.png" width="50%">

We chunk the tensor on sequence dimenstion, for each chunk do matmul with weight after calculate chunk sum logprob and add in global sum logprob.

Average memory requirement for 10k sequence dimension is 31.9GB

### [Proper Pytorch chunking](memory-profiling-chunk-v2.ipynb)

<img src="chunk-pytorch-memory-v2-fix.png" width="50%">

Same as Naive Pytorch chunking, we properly wrapped `torch.func.grad_and_value` properly inside custom autograd.

Average memory requirement for 10k sequence dimension is 22GB