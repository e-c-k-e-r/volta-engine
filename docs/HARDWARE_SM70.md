# HARDWARE_SM70.md — Physical Architecture & Invariants

This document establishes the immutable physical constants and constraints of the target hardware cluster: **4× NVIDIA Tesla V100-SXM2-32GB**. Every kernel, memory layout, and scheduling decision in `volta-engine` must adhere to these specifications.

---

## 1. System Topology & Host Interconnect

- **Host Processor**: Dual-socket Intel Xeon system.
- **NUMA Affinity**: 
  - The SXM2 baseboard is physically attached to the PCIe root complex of **NUMA Node 0**.
  - All engine threads must bind to **CPU cores 0–9** (`taskset -c 0-9`).
  - Accessing GPUs from NUMA Node 1 incurs an inter-socket UPI bridge penalty, degrading host-device DMA bandwidth and IPC latency by **20–25%**.
- **Host Memory**: 64+ GB DDR4 RAM.

---

## 2. GPU Silicon (NVIDIA GV100, Compute Capability 7.0)

Per GPU specifications:
- **Streaming Multiprocessors (SMs)**: **80 SMs**.
- **CUDA Cores**: 5,120 FP32 cores (64 per SM) + 2,560 FP64 cores.
- **Tensor Cores**: **640 Volta Tensor Cores** (8 per SM).
- **Core Clock**: Unlocked boost clock of **1,530 MHz**.
- **Register File**: 64 KB (65,536 32-bit registers) per SM. Max 255 registers per thread.
- **Warp Schedulers**: 4 warp schedulers per SM (each dispatching up to 1 instruction per cycle).

### On-Chip SRAM Hierarchy:
- **L1 Data Cache / Shared Memory**: 128 KB combined SRAM per SM.
- **Default Shared Memory Limit**: **48 KB** per threadblock.
- **Maximum Shared Memory Opt-In**: **96 KB** per threadblock.
  - Kernels requiring $>48\text{ KB}$ (such as Trellis Shape 4 GEMM) **must** invoke host opt-in before launch:
    ```cpp
    cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    ```
- **L2 Cache**: 6,144 KB (6 MB) shared across all 80 SMs.

---

## 3. High Bandwidth Memory (HBM2)

- **Capacity**: **32,768 MiB (32 GB)** per GPU (131,072 MiB total across 4 GPUs).
- **Memory Technology**: 4 stacks of Samsung 8-Hi HBM2.
- **Bus Width**: **4,096 bits** (1,024 bits per stack).
- **Memory Clock**: 877 MHz (1.755 Gbps effective).
- **Theoretical Peak Bandwidth**: **900.0 GB/s per GPU** (3,600 GB/s aggregate across the 4 GPUs).
- **Line Rate Cycle Math (@ 1.53 GHz)**:
  $$\frac{900\text{ GB/s}}{80\text{ SMs} \times 1.53\text{ GHz}} = \mathbf{7.35\text{ bytes per SM cycle}}$$
  A 32-byte weight slice must be processed in **$\le 4.3$ cycles** to sustain 100% bus saturation.

---

## 4. Inter-GPU Interconnect (NVLink 2.0)

- **Total Links**: 6 NVLink 2.0 links per GPU.
- **Raw Signaling Rate**: 25.78125 Gbps per differential pair (sub-link).
- **Bandwidth per Link**: 25 GB/s unidirectional / 50 GB/s bidirectional.
- **Total GPU Bandwidth**: **300 GB/s bidirectional** per GPU.
- **SXM2 4-Way Fully Meshed Crossbar**:
  - Every GPU is directly connected to every other GPU via **2 dedicated links**.
  - Direct P2P Bandwidth: **100 GB/s bidirectional per GPU pair**.
  - Zero hop contention: GPU 0 can stream to GPU 1, 2, and 3 simultaneously at full line rate without routing through intermediate switches or PCIe.

---

## 5. Critical Hardware Invariants & Architectural Landmines

### 5.1 No Hardware Asynchronous Copy (`cp.async`)
- `cp.async` was introduced in Ampere (SM80).
- On Volta (SM70), global-to-shared memory transfers **must route through general-purpose registers**:
  $$\text{Global HBM2} \xrightarrow{\text{LD.GLOBAL}} \text{Registers} \xrightarrow{\text{ST.SHARED}} \text{Shared Memory}$$
- Software double-buffering must account for register pressure. A warp cannot issue a background DMA without consuming thread issue slots.

### 5.2 No Hardware FP4
- Native FP4 tensor cores were introduced in Blackwell (SM100).
- Volta Tensor Cores only execute **FP16 / FP32 multiply-accumulate**:
  $$\text{mma.sync.aligned.m8n8k4.row.col}$$
- All NVFP4 inference on Volta uses **software nibble unpacking to FP16 registers** before issuing `m8n8k4`.

### 5.3 Independent Thread Scheduling & Warp Convergence
- Volta features Independent Thread Scheduling with per-thread program counters and call stacks.
- Threads within a warp do not execute in implicit lockstep.
- **Rule**: Never spin-poll on shared memory or NVLink IPC memory flags without `__syncwarp()` or explicit memory fences (`ld.acquire.sys`, `st.release.sys`).

### 5.4 Cooperative Grid Sync (`cg::this_grid().sync()`) Ceiling
- Cooperative groups require **all blocks in the grid to be active simultaneously**.
- Volta GV100 has **80 SMs**.
- If a kernel's shared memory / register allocation allows only 1 resident block per SM:
  $$\text{Max Cooperative Blocks} = 80 \times 1 = \mathbf{80\text{ blocks}}$$
- Launching $\ge 81$ blocks will cause the first 80 blocks to wait at `grid.sync()` for the 81st block, which cannot be scheduled until one of the first 80 exits $\rightarrow$ **Permanent hardware deadlock**.
- **Rule**: Never use cooperative grid sync for layers where grid size exceeds resident capacity (e.g. wide MLP layers where $N / 128 > 80$).
