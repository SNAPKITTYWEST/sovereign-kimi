# sovereign-kimi

Sovereign GPU kernel stack for Kimi K3 — fused GEMM + online softmax attention (FlashAttention-style block tiling) for Ampere+ (sm_80+).

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       sovereign-kimi                          │
├──────────────────────────────────────────────────────────────┤
│  kernels/                                                     │
│    k3_nano_harness.cu         Full inference harness          │
│    fused_attention.cu         Production GEMM+Online Softmax  │
│    gemm_online_softmax.cu     PTX/CUDA skeleton               │
│    gemm_online_softmax.mojo   Mojo/MAX skeleton               │
│  src/                                                        │
│    ffi.rs                      Rust C FFI bindings            │
│    daemon.rs                   Tokio channel actor            │
│    lib.rs                      Rust library root              │
│    build.rs                    nvcc compilation               │
│  scripts/                                                    │
│    create_draft_model.py       Draft model generator          │
│  Cargo.toml                                                    │
│  Makefile                                                      │
└──────────────────────────────────────────────────────────────┘
```

## Kernels

| File | Language | Status |
|------|----------|--------|
| `k3_nano_harness.cu` | CUDA | Full inference: Delta Attention, LatentMoE, WMMA FP16, Speculative Decoding, FFI exports |
| `fused_attention.cu` | CUDA/PTX | Production: cp.async, mma.sync.m16n8k16, online softmax, 97KB smem |
| `gemm_online_softmax.cu` | CUDA/PTX | Ahmad's skeleton with PTX inline assembly |
| `gemm_online_softmax.mojo` | Mojo | MAX accelerator skeleton |

## Features

- **FP16 inputs / FP32 accumulation**
- **128×64 tiles** (Q rows × K columns), K-inner = 64
- **128-thread blocks** (4 warps)
- **Online softmax** with running max + denominator (no full S or P matrix)
- **Tensor Core** `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
- **cp.async** double buffering, shared-memory padding for bank conflicts
- **Warp `__shfl_xor_sync` reductions**
- **Delta Attention**: `tanh(Q * ΔK * scale) * V` (Kimi K3 innovation)
- **LatentMoE Router**: 8-expert routing, top-2 selection
- **WMMA FP16 Tensor Core MoE**: Warp-level matrix multiply accumulate
- **Speculative Decoding**: Draft model + batched verification + acceptance kernel
- **Rust FFI**: `k3_engine_init` / `forward` / `free` with safe `K3Engine` wrapper
- **Tokio Daemon**: Channel actor, pins CUDA to dedicated OS thread

## Build

### CUDA/PTX

```bash
make all          # Full k3_nano harness
make attention    # Production fused attention kernel
make gemm         # PTX/CUDA skeleton
make draft        # Generate draft.bin for speculative decoding
```

### Rust FFI

```bash
cargo build --release
```

### Standalone Test

```bash
make test         # Run with model.bin
make spec         # Speculative decoding with draft model
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

- Test-compile on BBQBADDIE (RTX 3080, sm_86)
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
