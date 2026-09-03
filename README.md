# sovereign-kimi

Sovereign GPU kernel stack for Kimi K3 — fused GEMM + online softmax attention (FlashAttention-style block tiling) for Ampere+ (sm_80+).

## Architecture

```
┌─────────────────────────────────────────────────┐
│              sovereign-kimi                      │
├─────────────────────────────────────────────────┤
│  kernels/                                        │
│    gemm_online_softmax.cu    PTX/CUDA (sm_80+)  │
│    gemm_online_softmax.mojo  Mojo/MAX           │
│  src/                                            │
│    (expansion: batching, causal masks, etc.)    │
└─────────────────────────────────────────────────┘
```

## Kernel Features

- **FP16 inputs / FP32 accumulation**
- **128×64 tiles** (Q rows × K columns), K-inner = 64
- **128-thread blocks** (4 warps)
- **Online softmax** with running max + denominator (no full S or P matrix)
- **Tensor Core** `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
- **cp.async** double buffering, shared-memory padding for bank conflicts
- **Warp `__shfl_xor_sync` reductions**
- Q/K/V stay in global memory; only tiles + compact row state live on-chip

## Build

### CUDA/PTX

```bash
nvcc -O3 -arch=sm_80 -std=c++17 kernels/gemm_online_softmax.cu -o gemm_online_softmax
```

For sm_86 (RTX 3080):

```bash
nvcc -O3 -arch=sm_86 -std=c++17 kernels/gemm_online_softmax.cu -o gemm_online_softmax
```

### Mojo

```bash
mojo build --target-accelerator=nvidia:sm_80 kernels/gemm_online_softmax.mojo
```

## PTX Fragments

```ptx
// Tensor Core MMA (Ampere)
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    { %f0, %f1, %f2, %f3 },  // C (FP32)
    { %r0, %r1, %r2, %r3 },  // A (FP16)
    { %r4, %r5 },             // B (FP16)
    { %f0, %f1, %f2, %f3 };  // C in

// Async copy (double-buffer)
cp.async.cg.shared.global [smem_addr], [gmem_addr], 16;
cp.async.commit_group;
cp.async.wait_group 0;
```

## Online Softmax

```cuda
float warp_max = val;
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1)
    warp_max = fmaxf(warp_max, __shfl_xor_sync(0xffffffff, warp_max, offset));
```

## Next Steps

- Correctness validation against cuBLAS/cuDNN reference (tolerance ~1e-3)
- Nsight Compute / Nsight Systems profiling
- Register pressure tuning, shared-memory optimization (~97 KB target)
- Multi-head batching, causal masks, variable sequence lengths
- Hopper/Blackwell extensions (TMA + WGMMA)

## License

Sovereign Source License v1.0 + BSL-1.1 + AGPL-3.0 (tri-license)

## Author

Ahmad Ali Parr · Bel Esprit D'Accord Irrevocable Trust · EIN 42-697643
Contact: Ahmad <ahmedparr93@gmail.com>
