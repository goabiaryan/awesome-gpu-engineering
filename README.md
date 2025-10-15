# 🧠 Awesome GPU Engineering [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of resources for mastering GPU engineering from architecture and kernel programming to large-scale distributed systems and AI acceleration.

---

## 📘 Foundational Books

- **Programming Massively Parallel Processors: A Hands-on Approach** — *David B. Kirk & Wen-mei W. Hwu*  
  The canonical introduction to CUDA, memory hierarchies, and parallel patterns.
- **GPU Pro / GPU Zen Series** — *Wolfgang Engel*  
  Real-world graphics and compute programming techniques.
- **CUDA by Example** — *Jason Sanders & Edward Kandrot*  
  A practical introduction to CUDA for beginners.
- **Heterogeneous Computing with OpenCL 2.0** — *Benedict Gaster et al.*  
  Cross-platform perspective on GPU compute.
- **Parallel Programming and Optimization with GPUs** — *Udacity + NVIDIA*  
  Covers GPU architecture and performance optimization concepts.

---

## 💻 GPU Programming Frameworks

- **[CUDA](https://developer.nvidia.com/cuda-toolkit)** — NVIDIA’s proprietary GPU programming platform.  
  - Libraries: [cuBLAS](https://developer.nvidia.com/cublas), [cuDNN](https://developer.nvidia.com/cudnn)
- **[ROCm](https://github.com/RadeonOpenCompute/ROCm)** — AMD’s open compute stack.  
- **[OpenCL](https://www.khronos.org/opencl/)** — Cross-platform parallel computing standard.  
- **[SYCL / oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)** — Intel’s C++ abstraction for heterogeneous compute.  
- **[Vulkan Compute](https://www.khronos.org/vulkan/)** — Low-level GPU compute API.  
- **[Metal Performance Shaders](https://developer.apple.com/metal/)** — Apple’s GPU framework.

---

## 🧩 Optimization and Performance

- **[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)** — System-wide GPU profiler.  
- **[Nsight Compute](https://developer.nvidia.com/nsight-compute)** — Kernel-level performance analysis.  
- **Occupancy Calculator** — NVIDIA spreadsheet for kernel configuration.  
- **[CUTLASS](https://github.com/NVIDIA/cutlass)** — CUDA templates for linear algebra subroutines.  
- **[TensorRT](https://developer.nvidia.com/tensorrt)** — High-performance deep learning inference.  
- **[OpenAI Triton](https://triton-lang.org/)** — Python DSL for writing high-performance GPU kernels.  
- **Roofline Model** — Analytical model to reason about compute/memory bottlenecks.

---

## 🧠 Architecture and Low-Level Design

- **[NVIDIA Ampere Whitepaper](https://developer.nvidia.com/ampere-architecture)**  
- **[AMD RDNA & CDNA Architectures](https://gpuopen.com/learn/)**  
- Topics:
  - SIMT execution and warp scheduling  
  - Memory hierarchy and coalescing  
  - Shared memory and cache optimization  
  - Warp divergence and thread occupancy  

---

## ⚙️ Systems and Multi-GPU Engineering

- **[NCCL](https://developer.nvidia.com/nccl)** — Multi-GPU communication primitives.  
- **[Horovod](https://github.com/horovod/horovod)** — Distributed deep learning across GPUs.  
- **NVLink & PCIe Topology** — GPU interconnects and bandwidth optimization.  
- **[GPUDirect RDMA](https://developer.nvidia.com/gpudirect)** — Zero-copy GPU networking.  
- **[Ray Train](https://docs.ray.io/en/latest/train/index.html)**, **[DeepSpeed](https://github.com/microsoft/DeepSpeed)**, **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** — Large-scale GPU orchestration frameworks.

---

## 🧪 Tutorials and Courses

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)  
- [Triton Tutorials (OpenAI)](https://triton-lang.org/main/getting-started/tutorials/index.html)  
- [Udacity: Parallel Programming with CUDA](https://www.udacity.com/course/intro-to-parallel-programming--cs344)  
- [MIT 6.889: GPU Programming and Architecture](https://ocw.mit.edu/)  
- [CMU 15-418/618: Parallel Computer Architecture & Programming](http://15418.courses.cs.cmu.edu/)  

---

## 📄 Research Papers and Articles

- *Evolving GPU Architecture* — Kirk & Hwu  
- *Understanding and Mitigating Control Divergence on GPUs*  
- *Memory Coalescing Techniques for Modern GPU Architectures*  
- NVIDIA Research Papers on *Model Parallelism* and *Megatron-LM*  
- *GPU Virtualization and Multi-Tenant Scheduling*  

---

## 🧰 Tools and Utilities

- **nvprof**, **nvvp**, **Nsight Systems / Compute** — NVIDIA profiling tools.  
- **cuda-memcheck**, **compute-sanitizer** — Memory and correctness tools.  
- **[GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim)**, **[Accel-Sim](https://accel-sim.github.io/)** — GPU simulation frameworks.  
- **Perfetto**, **Nsight UI** — Visual profilers for tracing GPU workloads.

---

## 🧑‍🔬 GPU for AI & ML

- **PyTorch CUDA Extensions** — Custom kernels for PyTorch.  
- **JAX + XLA** — Compiler-based GPU vectorization.  
- **TensorFlow XLA Compiler** — Ahead-of-time GPU graph compilation.  
- **FlashAttention**, **FlashConv** — Kernel optimization techniques for transformers.  
- **DeepSpeed**, **FSDP**, **Megatron-LM** — Distributed training systems.  

---

## 🧱 GPU Systems Design

- GPU scheduling algorithms and runtime systems.  
- Memory oversubscription and unified memory models.  
- Resource allocation in GPU clusters.  
- Topics:
  - GPU virtualization  
  - Kernel fusion and graph execution  
  - Dataflow optimization  
  - Persistent threads model  

---

## 🧑‍💻 Contributors

Contributions welcome!  
Please read the [contribution guidelines](CONTRIBUTING.md) before submitting a pull request.

---

## 🧾 License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — feel free to share and adapt with attribution.

---

## ⭐ Acknowledgements

Inspired by:  
- [Awesome HPC](https://github.com/awesome-hpc/awesome-hpc)  
- [Awesome Computer Architecture](https://github.com/aalhour/awesome-computer-architecture)  
- [Awesome CUDA](https://github.com/Erkaman/awesome-cuda)

---

> “GPU engineering is not about writing kernels. It’s about understanding how systems work.”  — [Model Craft Newsletter](https://modelcraft.substack.com/p/fundamentals-of-gpu-engineering)*