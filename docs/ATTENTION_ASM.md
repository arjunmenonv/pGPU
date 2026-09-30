# pGPU-attention: Implement Attention Kernel in Pseudo-Assembly on a GPU SM Simulator

`pGPU` (pseudo-GPU) is a simulator for a minimal GPU-like device, entirely in Python, with its own mini-ISA, DRAM, Scratchpad-Memory, Register-File and SM compute core models. The simulator comes with its own assembly language `pPTX`<sup><a href="#fn1">1</a></sup> (pseudo-PTX) that is used to program its ISA. The objective of this project is to implement an attention map computation kernel in this language and devise techniques to optimize the runtime of the kernel (latency cycles reported by the simulator) as close to the theoretical minimum. 

The attention map $`A`$, of _[Attention Is All You Need](https://arxiv.org/abs/1706.03762)_<sup><a href="#fn1">2</a></sup> fame, is computed on a query $`Q`$ and a key $`K`$ matrix as follows:

$$\text{A[b, h, :, :]} = \text{softmax}\left(\frac{Q[b, h, :, :] \times K[b, :, :]^\top}{\sqrt{d}}\right)$$

where, 

&emsp;$`Q`$ is a matrix of shape ($\text{Batch}, \text{heads}, \text{n}, \text{d}$)

&emsp;$`K`$ is a matrix of shape ($\text{Batch}, \text{n}, \text{d}$), $K^\top$ (transpose along the ($\text{n}, \text{d}$) dims) is of shape ($\text{Batch}, \text{d}, \text{n}$)

&emsp;$`A`$ is a matrix of shape ($\text{Batch}, \text{heads}, \text{n}, \text{n}$)

and the _softmax_ function is defined as:

$$\text{softmax}(\mathbf{z})_i = \frac{e^{\mathbf{z}_i - \max(\mathbf{z})}}{\sum_{j} e^{\mathbf{z}_j - \max(\mathbf{z})}}$$

The subtraction by $\max(\mathbf{z})$ in the above formula is performed to keep the computation numerically stable.

Your kernel implementation must cater to the following problem shape: $(\text{Batch} = 16, \text{heads} = 64, \text{n} = 8, \text{d} = 32)$<sup><a href="#fn1">3</a></sup>

## The `pGPU` Simulator

`pGPU` is a python-based single-SM simulator modelling various aspects of a GPU's SM. 

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                   DRAM                                      │
│                            (Shared across SMs)                              │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SM (Streaming Multiprocessor)                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         SRAM Scratch Memory                           │  │
│  │                   (Private to SM, shared between SMSPs)               │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    SMSP (SM Sub-Partition)                            │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                      Warp Scheduler                             │  │  │
│  │  │                  (4 warps, time-shared)                         │  │  │
│  │  └──────────────────────────┬──────────────────────────────────────┘  │  │
│  │                             │                                         │  │
│  │              ┌──────────────┴──────────────┐                          │  │
│  │              ▼                             ▼                          │  │
│  │  ┌───────────────────────┐   ┌─────────────────────────────────────┐  │  │
│  │  │     Register File     │   │          Execution Units            │  │  │
│  │  │ ┌───────────────────┐ │   │ ┌───────┐ ┌─────────┐ ┌───────────┐ │  │  │
│  │  │ │  256 Vector Regs  │ │   │ │ VALU  │ │  TRANS  │ │ GEMM Core │ │  │  │
│  │  │ │ (64 per warp max) │ │   │ │32-wide│ │  Core   │ │   8x16    │ │  │  │
│  │  │ └───────────────────┘ │   │ │ SIMD  │ │(exp,cos,│ │   tiles   │ │  │  │
│  │  │ ┌───────────────────┐ │   │ │       │ │tanh,..) │ │           │ │  │  │
│  │  │ │  32 Predicate Regs│ │   │ └───────┘ └─────────┘ └───────────┘ │  │  │
│  │  │ │ (8 per warp max)  │ │   │                                     │  │  │
│  │  │ └───────────────────┘ │   │                                     │  │  │
│  │  └───────────────────────┘   └─────────────────────────────────────┘  │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Notes:**
- The baseline simulator only supports 1 SMSP per SM 
- Each SMSP's 256 vector registers and 32 predicate registers are statically partitioned between 4 warps
- Per-warp register limits: 64 vector registers, 8 predicate registers

The simulator uses a naive round-robin scheduling algorithm to switch between warps in an SMSP when a warp is stalled on a long latency instruction. 

### Software Ecosystem and Tooling

- `pGPU` exposes an assembly like interface for programming the simulated ISA. Check out the `assemble_instruction` function in [pgpu/sw/kernel.py](https://github.com/arjunmenonv/pGPU/tree/main/pgpu/sw/kernel.py) for the asm syntax and the `WarpKernel.asm()` function in the same file for inserting inline assembly into your kernel. An example vector-add kernel is provided in [pgpu/examples/vector_add.py](https://github.com/arjunmenonv/pGPU/tree/main/pgpu/examples/vector_add.py).

- It is recommended to use and develop intrinsic functions (or rather pre-defined inline-asm blocks) to wrap commonly used `pPTX` asm routines in your kernel. This will avoid clutter in your kernel and will simplify your optimization workflow. You can find some in-built intrinsic functions in [pgpu/sw/intrinsics.py](https://github.com/arjunmenonv/pGPU/tree/main/pgpu/sw/intrinsics.py).

- The simulator comes with an `objdump` tool and a perfetto-compatible `KernelTrace` tool. 
    
    - Use `objdump` to dump out the `pPTX` assembly that your kernel corresponds to — this will come in handy when you have a bunch of intrinsic calls in your kernel. `objdump` also allows you to print the set of live registers for each `pPTX` asm instruction in your kernel (mirroring an interface similar to `nvdisasm -lrm`).
        
    - `KernelTrace` traces the simulation of your kernel and generates a json file that you can open with [perfetto](https://ui.perfetto.dev/) to see an `nsight-systems` timeline-like view of your kernel execution

- Finally, the entire source-code of the simulator is made available to you. You are encouraged to tinker with the [pgpu/sw](https://github.com/arjunmenonv/pGPU/tree/main/pgpu/sw) and [pgpu/devtools](https://github.com/arjunmenonv/pGPU/tree/main/pgpu/devtools) folders to add tools to simplify your workflow. 

### Pre-empted FAQs

#### 1. Where it differs from NVIDIA's PTX model

1. The architectural models and the latencies of the hardware resources in `pGPU` may be different from those on a real GPU. This means the trade-offs you prioritize when writing a kernel in `pPTX` may be different from the wisdom you have accumulated for an NVIDIA GPU in the GPU Programming Course.

2. `pGPU` is much simpler than a real GPU — given by the fact that the simulator only has a single SM Sub-Partition and that it uses a naive warp scheduling algorithm. This will simplify the programming experience when writing `pPTX` while also planting inefficiencies in your kernel that may be difficult to circumvent.

3. You don't get tinker close-to-metal on a real GPU everyday — you usually need to get past a very deep compiler and firmware infrastructure, all designed with the intention of extracting as much performance out of the hardware, to really see how things look like under the hood. `pGPU` lacks most of the virtualization layers you would see in a GPU allowing you to play with the low-level hardware model (albeit simulated) to a better extent.

4. `pGPU` ISA is different from `PTX` ISA with the differences being substantial at some places. Some of these differences are detrimental to performance, while some may be to your advantage. 

#### 2. Where it is NVIDIA-like

1. Your key objectives do not change: prioritize DRAM memory coalescing, avoid shared memory bank conflicts, improve instruction-level parallelism within a warp and avoid redundant data movement when writing performant kernels.

2. The `pGPU` hardware model is built to mimic the actual hardware model of an NVIDIA GPU. While it is not an exact replica of an NVIDIA GPU's SM, it is a good enough approximation around which you can build a mental model for performance optimization.<sup><a href="#fn1">4</a></sup>

3. Finally, the programming interface is made to look somewhat like an actual CUDA / PTX kernel. This is just there to help with the writer's block one may run into when writing kernels in assembly. 


#### 3. How should I approach this problem?

1. Read the [docs]() (TBD) and go through the files in the [pgpu/sw](https://github.com/arjunmenonv/pGPU/tree/main/pgpu/sw) folder to get a hold of the simulator features and the software APIs around it.

2. Read the [ISA](https://github.com/arjunmenonv/pGPU/tree/main/pgpu/arch/isa.py)! You might find unconventional instructions here that might come in handy for your kernel.

3. Repeat the traditional kernel performance optimization loop: Build Hypotheses for Performance Bottlenecks $\rightarrow$ Implement Optimization $\rightarrow$  Trace and Inspect ASM $\rightarrow$ Validate Hypothesis. The simulator workflow here allows to iterate over this faster than in the real-world.

4. Show up for your regular meetings with your mentor with ideas and enthusiasm!


#### 4. Why program in pPTX when I can program in PTX on real GPUs? _(aka Why should I pick this problem: an attempt at persuasion from the author)_

1. Courtesy points [1.3](./pGPU_AttentionAsm.md#1-where-it-differs-from-nvidias-ptx-model) and [2.2](./pGPU_AttentionAsm.md#2-where-it-is-nvidia-like), this problem gives you visibility to some close-to-metal traits of an actual GPU that are usually buried behind the multitude of GPU software, driver and virtual-memory abstractions that come in your way. 

2. This problem is devised with the intention of leading you towards developing a better understanding of a real GPU with some gold-nuggets planted along the way. We believe these nuggets will help you discover the nuances in a GPU and appreciate it even more than you currently do.

3. _So you think you can write a speed-of-light attention kernel in PTX? Do it on a simpler (simulated) GPU first!_

#### 5. Using LLMs to Code in pPTX<sup><a href="#fn1">5</a></sup>

The following points are gated by the official AI usage and honor code for the CS6023 course. If LLM tools are strictly forbidden in the course, you needn't even read the following points.

1. Please follow honor code and don't let LLMs rob you of the joy of performance optimization and discovering the gold nuggets in this problem!

2. Since the simulator is fully open-source, please avoid getting it indexed by a coding agent. 

3. If some amount of LLM usage is allowed in the course, you may use LLM chatbots to clarify doubts, iterate over ideas and explain code snippets. But please refrain from running a prompt like `/goal Modify the following kernel until it meets the theoretical minimum runtime for the simulator model.` 

## Relevant Links

`pGPU` Simulator: [arjunmenonv/pGPU](https://github.com/arjunmenonv/pGPU/tree/main/)

docs: [docs](https://github.com/arjunmenonv/pGPU/tree/main/docs)

example kernels: [vector-add](https://github.com/arjunmenonv/pGPU/tree/main/pgpu/examples/vector_add.py), [intrinsic-demo](https://github.com/arjunmenonv/pGPU/tree/main/pgpu/examples/intrinsic_demo.py)


<hr style="opacity: 0.6;">

<sup id="fn1">1</sup> Just like the popular commercial software with the same name, this language can also hang indefinitely if you aren't careful with parallel writes. But at least things are under your control here.

<sup id="fn1">2</sup> Sneaking in a reference to Vaswani et. al. here for engagement, good luck and prosperity.

<sup id="fn1">3</sup> Not happy seeing that the attention map is only operating over 8 tokens? We need your help to scale the kernel to support a larger context length!

<sup id="fn1">4</sup> One could say it is built atop the mental model I have built for myself over a few years of performance optimization on GPUs, and is likely to overlook many details and have many simplifying assumptions. 


<sup id="fn1">5</sup> The intention behind this problem was to cut the middle ~~man~~ LLM out from GPU kernel optimization. These tools have become quite effective over the past few months at writing CUDA and have robbed many of us of the creative joy of implementing and validating kernel optimizations. Surely no frontier LLM was trained on writing `pPTX` assembly, after all it was something I made up! But given that LLMs now are capable of solving Millenium Problems and that I am no Navier, Stokes or Reimann, I stand no chance against these tools.
