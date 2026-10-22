# flashattn-scratch

FlashAttention-2 forward kernels written from scratch in **CUDA C++ with inline PTX**, tuned for Ampere (developed on an **RTX 3050 6 GB Laptop**, sm_86).

Benchmarks compare kernel latency and throughput against **PyTorch SDPA** across head dimensions, head counts and sequence lengths up to 32K. This is a learning and profiling project, not a replacement for PyTorch SDPA or the official FlashAttention libraries.

## Implementation

- Tensor Core `mma.sync` (`m16n8k16`) for QK^T and PV, with FP16 operands and FP32 accumulation.
- Online softmax in the base-2 domain (`softmax_scale · log2(e)` folded into the scores, `exp2f`) and output accumulation in registers.
- Four warps per block; K and V tiles streamed through shared memory with `cp.async`, each load overlapped with the other operand's compute (V_j loads during QK^T, K_{j+1} during softmax and PV).
- Q is staged through shared memory once, moved to registers with `ldmatrix`, and its rows are then reused for K/V, so the block needs only `max(64, 2·Bc)` tile rows of shared memory.

The forward kernel supports **dense and causal attention, GQA/MQA and unequal query/KV lengths**, including lengths that are not multiples of the tile size. A baseline backward pass is included.

[Design notes](#design-notes) | [Profiling](#profiling-and-diagnostics) | [Project layout](#project-layout) | [Build options](#build-options)

## Kernel Performance

**RTX 3050 6 GB Laptop (GA107, sm_86), PyTorch 2.7.1+cu118, CUDA 11.8.** Batch size 1, eight Q/KV heads, head dimension 64. SDPA backend `auto`. FP16 inputs, FP32 accumulation. Each reported latency is the median across three full sweeps; each sweep uses the median of timed calls with an L2 flush between calls.

TFLOP/s is `4·B·H·N²·D` FLOPs (halved for causal) divided by the median latency.

![Kernel throughput](docs/figures/kernel-throughput.png)

Execution time (ms), lower is better:

| Mask | Sequence length | Custom | SDPA |
|---|---:|---:|---:|
| Dense | 1024 | 0.156 | 0.178 |
| Dense | 2048 | 0.482 | 0.525 |
| Dense | 4096 | 1.892 | 1.910 |
| Dense | 8192 | 7.726 | 7.656 |
| Dense | 16384 | 31.151 | 30.790 |
| Dense | 32768 | 124.405 | 124.677 |
| Causal | 1024 | 0.095 | 0.115 |
| Causal | 2048 | 0.294 | 0.329 |
| Causal | 4096 | 1.054 | 1.128 |
| Causal | 8192 | 4.152 | 4.261 |
| Causal | 16384 | 17.371 | 16.881 |
| Causal | 32768 | 72.574 | 67.523 |

The full sweep covers **108 cases**: head dimension 64/128/256, 8/16/32 heads, sequence lengths 1024 to 32768, dense and causal. Custom throughput spans **13.80–18.16 TFLOP/s for dense** and **11.28–16.55 TFLOP/s for causal** attention, using the per-case median latency across three full sweeps.

[All cases](benchmarks/results/fwd.csv)

### Reproduce

```bash
# full sweep, batch size 1 for every N
python -m benchmarks.bench_fwd --tokens 1 --csv benchmarks/results/fwd.csv

# chart + markdown table for one (D, H) slice
python -m benchmarks.plot_fwd --csv benchmarks/results/fwd.csv --dim 64 --heads 8
```

The sweep takes a while at N = 32768 (seconds per call); `--budget-ms` caps the per-config time by shrinking warmup and iteration counts.

## Quick start

### Requirements

- Linux, NVIDIA GPU with compute capability 8.0 or newer.
- CUDA Toolkit with `nvcc` on `PATH`, matching the CUDA version used by your PyTorch install.
- Python 3.9+ and a CUDA-enabled PyTorch install.
- Tested: Python 3.10, PyTorch 2.7.1+cu118, CUDA 11.8, RTX 3050 6GB Laptop.

### Install

```bash
git clone https://github.com/akshay919191/flashattn-V1.git
cd flashattn-V1
pip install -e . --no-build-isolation
```

### Smoke test

```python
import torch
from flash_acc_reg import flash_attn

q = torch.randn(2, 16, 1024, 64, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

out = flash_attn(q, k, v, causal=True)
print(out.shape)   # torch.Size([2, 16, 1024, 64])
```

## API

### High-level autograd wrapper

```python
from flash_acc_reg import flash_attn
O = flash_attn(Q, K, V, causal=False)
```

### Low-level kernel interface

```python
from flash_acc_reg import _C
O, L = _C.flash_fwd(Q, K, V, causal)
```

`L` is the per-row log-sum-exp in natural log (`m + log(l)`, FP32), needed for gradient computation.

### Tensor shapes

```
Q, O:  [B, H,    Sq,  D]
K, V:  [B, H_kv, Skv, D]
L:     [B, H,    Sq]
```

All tensors must be contiguous. `H_kv` must divide `H`: `H_kv == H` is standard MHA, `H_kv == 1` is MQA.

## Supported inputs

| Property | Status |
|---|---|
| Device | CUDA, compute capability 8.0+ |
| Dtype | FP16 |
| Accumulation | FP32 (MMA accumulators and softmax statistics) |
| Masking | None, or causal (bottom-right aligned) |
| GQA / MQA | Supported |
| Cross-attention | Supported (`Sq ≠ Skv`) |
| Sequence lengths | Any positive `Sq`, `Skv` (ragged tiles handled) |
| Head dimension | `{32, 64, 80, 96, 128, 160, 192, 224, 256}` |
| BF16, dropout, variable-length sequences | Not implemented |

Causal masking is aligned to the bottom-right corner: query row `i` attends to keys `0 … i + (Skv − Sq)`, matching FlashAttention-2. For `Sq == Skv` this is the usual lower-triangular mask. When `Sq > Skv`, the leading query rows have no visible key and their output is NaN.

Unsupported shapes (wrong head dimension, `H` not divisible by `H_kv`, non-contiguous tensors) raise an error at launch.

## Design notes

**Shared memory.** Tiles use a padded row stride of `D + 8` elements to avoid `ldmatrix` bank conflicts. Q is loaded into the same region that later holds K and V: it is read into registers, a barrier is taken, and then K/V overwrite it. The allocation is `max(64, 2·Bc)` rows × `(D + 8)` × 2 bytes.

| `D` | `Bc` | Tile rows | Shared memory per block |
|---:|---:|---:|---:|
|  32 | 64 | 128 | 10 KiB |
|  64 | 64 | 128 | 18 KiB |
|  80 | 64 | 128 | 22 KiB |
|  96 | 64 | 128 | 26 KiB |
| 128 | 64 | 128 | 34 KiB |
| 160 | 32 |  64 | 21 KiB |
| 192 | 32 |  64 | 25 KiB |
| 224 | 32 |  64 | 29 KiB |
| 256 | 32 |  64 | 33 KiB |

**Occupancy.** GA107 has 100 KiB of shared memory per SM, and CUDA reserves 1 KiB per resident block. Two blocks per SM therefore need at most about 49 KiB each; every configuration above fits, and the launcher requests the maximum shared-memory carveout so the driver actually provides it.

**Large head dimensions.** At `D = 256` each thread holds 128 FP32 output accumulators plus 64 registers of Q fragments, so the register file, not shared memory, is the limiting resource. Check `-Xptxas -v` for spills if you change `Bc` there.

**Causal.** KV tiles beyond a block's last visible key are skipped, and the mask is applied only to tiles that cross the diagonal (checked per warp).

**Ragged lengths.** The launcher selects the full-tile kernel when `Sq % 64 == 0` and `Skv % Bc == 0`. Otherwise it uses a variant with zero-filling `cp.async` loads, key-bound masking and guarded stores.

## Profiling and diagnostics

These commands only measure the kernel; none of them change it. They were used to check register pressure and to find where the remaining gap to SDPA comes from.

### Register and spill counts

Compiling `csrc/flashattn.cu` directly does not instantiate `flashattn_fwd` for a given `Bc`, so `-Xptxas -v` prints nothing for it. Force the instantiations with a probe file instead (run from the repo root, and match the template parameter list of your kernel):

```bash
cat > /tmp/probe.cu << 'EOF'
#include "csrc/fwd_kernel.cuh"
// <Br, Bc, D_PAD, masked, FULL_TILES>
template __global__ void flashattn_fwd<64, 64, 128, true, true>(
    const __half*, const __half*, const __half*,
    __half*, float*, int, int, int, int);
template __global__ void flashattn_fwd<64, 32, 128, true, true>(
    const __half*, const __half*, const __half*,
    __half*, float*, int, int, int, int);
EOF

nvcc -Xptxas=-v -arch=sm_86 -I. --expt-relaxed-constexpr \
     /tmp/probe.cu -o /dev/null 2>&1 | grep -E "registers|spill"
```

Measured for `D = 128`, causal:

| `Bc` | Registers per thread | Spills |
|---:|---:|---:|
| 64 | 157 | 0 |
| 32 | 167 | 0 |

The smaller tile uses more registers and spills nothing either way, so it buys nothing here; `Bc = 64` is used for `D = 128`.

To see whether a tighter register budget (more resident blocks) would force spills, add `-maxrregcount=128` to the same command. This has not been run yet.

### Where the remaining time goes (Nsight Compute)

Profile only the forward kernel on one small, representative shape, not the whole sweep:

```bash
ncu -k regex:flashattn_fwd -c 3 \
  --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__t_sector_hit_rate.pct,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct \
  python -m benchmarks.bench_fwd --dims 128 --heads 8 --seqlens 4096 \
         --causal true --tokens 1 --warmup 1 --iters 2
```

| Metric | What it tells you |
|---|---|
| `sm__warps_active` | Achieved occupancy: whether two blocks per SM are actually resident |
| `sm__pipe_tensor_op_hmma_cycles_active` | How busy the tensor cores are; low means they are waiting on something else |
| `..._stalled_barrier` | Cost of the `__syncthreads()` calls (two per KV tile) |
| `..._stalled_long_scoreboard` | Waiting on global or local memory; also shows register spills |
| `..._stalled_mio_throttle` | Pressure on the shared-memory / `ldmatrix` instruction queue |
| `l1tex__t_sector_hit_rate` | L1 behaviour; `cp.async.cg` bypasses L1, so a low value is expected for K/V |

If `ncu` rejects a metric name, list the valid ones for your GPU with `ncu --query-metrics`. If it fails with `ERR_NVGPUCTRPERM`, run it with `sudo` or enable profiling for non-admin users in the NVIDIA driver settings.

## Project layout

```
flashattn-V1/
├── csrc/
│   ├── fwd_kernel.cuh     # forward kernel (MMA, online softmax, masking)
│   ├── fwd_launch.cuh     # Bc dispatch, smem sizing, launch config
│   ├── bwd_kernel.cuh     # backward kernels (baseline)
│   ├── bwd_launch.cuh     # backward dispatch
│   ├── mma_helpers.cuh    # ldmatrix, tile loaders, pack helpers
│   ├── flashattn.cu       # entry points
│   └── bindings.cpp       # pybind11 module
├── flash_acc_reg/
│   ├── __init__.py        # compiled extension or JIT fallback
│   ├── _jit.py
│   └── ops.py             # autograd wrapper and input validation
├── tests/
├── docs/figures/          # charts generated by benchmarks/plot_fwd.py
└── benchmarks/
    ├── bench_fwd.py       # forward sweep vs SDPA
    └── plot_fwd.py        # chart + markdown table from the sweep CSV
```

## Build options

| Variable | Default | Effect |
|---|---|---|
| `TORCH_CUDA_ARCH_LIST` | Detected from GPU | Target arch(s), e.g. `"8.0;8.6"` |
| `FLASH_DEBUG` | `0` | `1` → `-O0 -G` for `compute-sanitizer` / `cuda-gdb` |
| `FLASH_FAST_MATH` | `1` | `0` → disable `--use_fast_math` |

## Clean rebuild

```bash
pip uninstall -y flash-acc-reg
rm -rf build *.egg-info
find . -name '*.so' -delete
rm -rf ~/.cache/torch_extensions/*/flash_acc_reg_jit
pip install -e . --no-build-isolation
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `nvcc not found` | Add CUDA Toolkit `bin/` to `PATH` |
| PyTorch/CUDA version mismatch | Compare `nvcc --version` with `torch.version.cuda` |
| `unsupported D_PAD` / head-dim error | `D` must be in the table under [Supported inputs](#supported-inputs) |
| NaN in the first rows of `O` | Causal with `Sq > Skv`: those rows have no visible key |
| Old behavior after editing CUDA | Clean build and JIT cache, then reinstall |

## References

- Dao et al., *FlashAttention*, NeurIPS 2022.
- Dao, *FlashAttention-2*, ICLR 2024.
- NVIDIA, [Ampere Tuning Guide — Occupancy](https://docs.nvidia.com/cuda/archive/12.5.1/ampere-tuning-guide/index.html#occupancy).
