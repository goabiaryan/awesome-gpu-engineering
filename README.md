# Awesome GPU Engineering [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of resources for mastering GPU engineering from architecture and kernel programming to large-scale distributed systems and AI acceleration.

---

## 📘 Foundational Books

- **Programming Massively Parallel Processors: A Hands-on Approach** — *David B. Kirk & Wen-mei W. Hwu* 
  The canonical introduction to CUDA, memory hierarchies, and parallel patterns. *[Amazon](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0323912311)* , notes: [*Abi's Concise Notes*](https://github.com/goabiaryan/awesome-gpu-engineering/blob/main/notes/Abi's%20PMPP%20Notes.pdf)
- **CUDA by Example** — *Jason Sanders & Edward Kandrot*  
  A practical introduction to CUDA for beginners. *[Amazon](https://www.amazon.com/CUDA-Example-Introduction-General-Purpose-Programming/dp/0131387685)*
- **The Ultra-Scale Playbook: Training LLMs on GPU Clusters** - Hugging Face *[Web Version](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview)*


## 💻 GPU Programming Frameworks

- **[CUDA](https://developer.nvidia.com/cuda-toolkit)** — NVIDIA’s proprietary GPU programming platform.
  - Libraries: [cuBLAS](https://developer.nvidia.com/cublas), [cuDNN](https://developer.nvidia.com/cudnn)
- **[ROCm](https://github.com/RadeonOpenCompute/ROCm)** — AMD’s open compute stack.
- **[OpenCL](https://www.khronos.org/opencl/)** — Cross-platform parallel computing standard.
- **[SYCL / oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)** — Intel’s C++ abstraction for heterogeneous compute.
- **[Vulkan Compute](https://www.khronos.org/vulkan/)** — Low-level GPU compute API.
- **[Kompute](github.com/komputeproject/kompute)** — Higher level general purpose GPU compute framework built on Vulkan.
- **[Metal Performance Shaders](https://developer.apple.com/metal/)** — Apple’s GPU framework.
- **[Mojo🔥](https://mojolang.org)** - Write like Python, run like C++.


## 🧩 Optimization and Performance

- **[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)** — System-wide GPU profiler.
- **[Nsight Compute](https://developer.nvidia.com/nsight-compute)** — Kernel-level performance analysis.
- **Occupancy Calculator** — NVIDIA spreadsheet for kernel configuration.
- **[CUTLASS](https://github.com/NVIDIA/cutlass)** — CUDA templates for linear algebra subroutines.
- **[TensorRT](https://developer.nvidia.com/tensorrt)** — High-performance deep learning inference.
- **[OpenAI Triton](https://triton-lang.org/)** — Python DSL for writing high-performance GPU kernels.
- **[Helion](https://helionlang.com)** - A Python-embedded DSL that makes it easy to write fast, scalable ML kernels with minimal boilerplate.
- **[Roofline Model](https://jax-ml.github.io/scaling-book/)** — Analytical model to reason about compute/memory bottlenecks.


## 🧠 Architecture and Low-Level Design

- **[NVIDIA Ampere Whitepaper](https://developer.nvidia.com/ampere-architecture)**
- **[AMD RDNA & CDNA Architectures](https://gpuopen.com/learn/)**
- SIMT execution and warp scheduling
- Memory hierarchy and coalescing
- Shared memory and cache optimization
- Warp divergence and thread occupancy


## ⚙️ Systems and Multi-GPU Engineering

- **[NCCL](https://developer.nvidia.com/nccl)** — Multi-GPU communication primitives.
- **[vLLM](https://github.com/vllm-project/vllm)** - Inference and serving engine for LLMs
- **[Hugging Face Accelerate](https://github.com/huggingface/accelerate)** - Simplify abstractions for distributed training
- **[SGLang](https://github.com/sgl-project/sglang)**
- **[Prime Intellect](https://github.com/PrimeIntellect-ai/prime-cli)**
- **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)**
- **[TGI by Hugging Face](https://huggingface.co/docs/text-generation-inference/en/index)**
- **[Horovod](https://github.com/horovod/horovod)** — Distributed deep learning across GPUs.
- **NVLink & PCIe Topology** — GPU interconnects and bandwidth optimization.
- **[GPUDirect RDMA](https://developer.nvidia.com/gpudirect)** — Zero-copy GPU networking.
- **[Ray Train](https://docs.ray.io/en/latest/train/index.html)**, **[DeepSpeed](https://github.com/microsoft/DeepSpeed)**, **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** — Large-scale GPU orchestration frameworks.
- **[Iris by AMD](https://github.com/ROCm/iris)** - open-source multi-GPU programming framework built for compiler-visible performance and optimized multi-GPU execution.
- **[Vivacious Cloud](https://github.com/Viavcious-cloud/vivacious-cli)** - Zero-code multi-cloud GPU orchestrator with preflight VRAM guards and spot arbitrage.


## 🧪 Tutorials and Courses

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Triton Tutorials (OpenAI)](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [CUDA in 12 hours by FreeCodeCamp](https://www.youtube.com/watch?v=86FAWCzIe_4)  and [Video Repo](https://github.com/infatoshi/cuda-course)
- [Stanford CS149, Fall 2025 Parallel Computing Course Fall 2025](https://gfxcourses.stanford.edu/cs149/fall25/)
- [CMU 15-418/618: Parallel Computer Architecture & Programming](https://www.cs.cmu.edu/~418/)
- [MIT 6.5940: TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940)
- [GPU MODE video lecture series](https://www.youtube.com/@GPUMODE/videos)
- [Red Hat vLLM Office Hours video series](https://www.youtube.com/playlist?list=PLbMP1JcGBmSHxp4-lubU5WYmJ9YgAQcf3)
- [The courses of the Programming Massively Parallel Processors book's authors](https://www.youtube.com/@pmpp-book/playlists)



## 📄 Research Papers and Articles

- *[Optimization techniques for GPU programming](https://dl.acm.org/doi/pdf/10.1145/3570638)* - Hijma, Pieter, et al.
- *[Efficient Multi-GPU Programming in Python: Reducing Synchronization and Access Overheads](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11186485)* - Oden, Lena, and Klaus Nölp
- *[Evolving GPU Architecture](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9623445&casa_token=Zknb-Go77Y4AAAAA:03tRVI5oLoyDZMx-UZZiWp9h7JRTc-UHNmiHykq2MZWBKNFBwjxEUpuddkX54Z246I6gjDUpdw&tag=1)* — Kirk & Hwu
- *[Deep Learning Workload Scheduling in GPU Datacenters: Taxonomy, Challenges and Vision](https://arxiv.org/abs/2205.11913)* - Wei Gao et al
- *[Optimizing Machine Learning Models with CUDA: A Comprehensive Performance Analysis](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11064558)*  - Niteesh, L., and M. B. Ampareeshan
- NVIDIA Research Papers on *[Model Parallelism](https://dl.acm.org/doi/pdf/10.1145/3458817.3476209?casa_token=p3epEa_Z4xEAAAAA:fZgVzYD2uMH5NcafdBN9g7EgIbESqB7WsHjL0X6LU2zdm6EdgQkMyIFk0yZAfWGl1o3PeUSB4xhg)* and *[Megatron-LM](https://arxiv.org/pdf/1909.08053)*
- *[GPU Virtualization and Multi-Tenant Scheduling](https://dl.acm.org/doi/pdf/10.1145/3068281?casa_token=bbU9Dvrt3vsAAAAA:jxP-NNGr8GEmjOng-EFlb1Rd6wVSQAXg65GTK1jDPlGIkGjNIirMWkDZcjnTw0xDZmLGZ489LwHX)*
- *[A Survey of Multi-Tenant Deep Learning Inference on GPU](https://arxiv.org/abs/2203.09040)*
- *[Efficient Performance-Aware GPU Sharing with Compatibility and Isolation through Kernel Space Interception](https://www.youtube.com/watch?v=e54BVwcdJ4Y)*


## 🧰 Tools and Utilities

- **nvprof**, **nvvp**, **Nsight Systems / Compute** — NVIDIA profiling tools.
- **cuda-memcheck**, **compute-sanitizer** — Memory and correctness tools.
- **[GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim)**, **[Accel-Sim](https://accel-sim.github.io/)** — GPU simulation frameworks.
- **[Ingero](https://github.com/ingero-io/ingero)** — eBPF-based GPU causal observability agent. Traces CUDA Runtime/Driver APIs and host kernel events to build causal chains explaining GPU latency. <2% overhead, production-safe.
- **Perfetto**, **Nsight UI** — Visual profilers for tracing GPU workloads.

### Learning Tools

- **[Tensara](https://gpuengineering.com)**
- **[LeetGPU](https://leetgpu.com/)**
- **[GPU MODE Discord](https://discord.gg/FnjEVAhW)**
- **[GPU Glossary](https://modal.com/gpu-glossary)** - A dictionary of terms related to programming GPUs
- **[Mojo🔥 GPU Puzzles](https://puzzles.modular.com)**


## 🧑‍🔬 GPU for AI & ML

- **PyTorch CUDA Extensions** — Custom kernels for PyTorch.
- **JAX + XLA** — Compiler-based GPU vectorization.
- **TensorFlow XLA Compiler** — Ahead-of-time GPU graph compilation.
- **FlashAttention**, **FlashConv** — Kernel optimization techniques for transformers.
- **DeepSpeed**, **FSDP**, **Megatron-LM** — Distributed training systems.

## 🧱 GPU Systems Design Topics For Interview Prep

- FlashAttention and PagedAttention
- Matmul Operations
- GPU scheduling algorithms and runtime systems.
- Memory oversubscription and unified memory models.
- Resource allocation in GPU clusters.
- GPU virtualization
- Kernel fusion and graph execution
- Dataflow optimization
- Persistent threads model

---

## 🧑‍💻 Contributors

Contributions welcome!
Please read the [contribution guidelines](CONTRIBUTING.md) before submitting a pull request.

## 🧾 License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — feel free to share and adapt with attribution.

## ⭐ Acknowledgements

Inspired by:
- [Awesome HPC](https://github.com/trevor-vincent/awesome-high-performance-computing)
- [Awesome Computer Architecture](https://github.com/aalhour/awesome-computer-architecture)
- [Awesome CUDA](https://github.com/coderonion/awesome-cuda-and-hpc)

---

> “GPU engineering is not just about writing kernels. It’s about understanding how systems work.”  — [Model Craft](https://modelcraft.substack.com/p/fundamentals-of-gpu-engineering)

