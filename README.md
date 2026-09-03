<div align="center">

# sovereign-kimi

**Kimi K3 Nano CUDA Attention Kernels · GEMM + Online Softmax · WMMA FP16 · Speculative Decoding**

[![License: Sovereign](https://img.shields.io/badge/License-Sovereign%20v1.0-blue.svg)](LICENSE)
[![License: BSL-1.1](https://img.shields.io/badge/License-BSL--1.1-green.svg)](LICENSE)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-red.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/Language-CUDA%20C++-ff6f00.svg)](k3_nano/kernels/)
[![Mojo](https://img.shields.io/badge/Language-Mojo-ff2d55.svg)](k3_nano/kernels/)
[![Rust](https://img.shields.io/badge/Language-Rust-ce4225.svg)](k3_nano/src/)
[![FP16](https://img.shields.io/badge/Compute-FP16%20WMMA-9cf.svg)](k3_nano/kernels/)
[![cuDNN](https://img.shields.io/badge/Backend-cuDNN%209-76b900.svg)](k3_nano/)

Sovereign GPU attention engine: fused GEMM + online softmax, WMMA tensor core kernels, speculative decoding, Rust FFI, and a production inference harness.

</div>

---

## Kernel Pipeline

```mermaid
graph LR
    subgraph Input
        Q[Query]
        K[Key]
        V[Value]
    end
    subgraph GEMM["Fused GEMM + Softmax"]
        G1[GEMM: Q × Kᵀ] --> G2[Online Softmax<br/>running max + denom]
        G2 --> G3[Scale by √d]
        G3 --> G4[Attention Weights]
    end
    subgraph Output
        G4 --> G5[GEMM: P × V]
        G5 --> OUT[Output]
    end
    Q --> G1
    K --> G1
    V --> G5
```

## Harness Architecture

```mermaid
graph TB
    subgraph "k3_nano_harness.cu"
        H1[Main Loop] --> H2[Load Weights<br/>from GGUF]
        H2 --> H3[Token Embed]
        H3 --> H4[Transformer Layers<br/>× N]
        H4 --> H5[LM Head]
        H5 --> H6[Token Out]
    end
    subgraph "kernels/"
        K1[fused_attention.cu<br/>GEMM+Online Softmax]
        K2[gemm_online_softmax.cu<br/>PTX/CUDA skeleton]
        K3[gemm_online_softmax.mojo<br/>Mojo/MAX skeleton]
    end
    subgraph "src/"
        S1[daemon.rs<br/>Tokio Channel Actor]
        S2[ffi.rs<br/>Rust C FFI Bindings]
    end
    H4 --> K1
    H4 --> K2
    S1 --> H1
    S2 --> H1
```

## Kernel Files

| File | Type | Description |
|------|------|-------------|
| `kernels/fused_attention.cu` | CUDA C++ | Production GEMM + online softmax kernel |
| `kernels/gemm_online_softmax.cu` | PTX/CUDA | PTX-level skeleton with inline assembly |
| `kernels/gemm_online_softmax.mojo` | Mojo/MAX | GPU kernel skeleton for MAX runtime |

## Fused Attention Flow

```mermaid
sequenceDiagram
    participant Host
    participant GPU
    participant SRAM

    Host->>GPU: Launch kernel (Q, K, V, d, N)
    loop For each tile
        GPU->>SRAM: Load Q tile (128×64)
        GPU->>SRAM: Load K tile (64×64)
        SRAM->>GPU: GEMM: S = Q × Kᵀ
        Note over GPU: Online Softmax<br/>m_new = max(m, tile_max)<br/>d = d × exp(m−m_new) + Σexp(s−m_new)
        GPU->>SRAM: Load V tile (64×64)
        SRAM->>GPU: GEMM: O = P × V
    end
    GPU->>Host: Return output
```

## Rust FFI + Daemon

```mermaid
graph LR
    PY[Python Caller] -->|C ABI| FFI[ffi.rs<br/>extern C]
    FFI --> DAEMON[daemon.rs<br/>Tokio Actor]
    DAEMON -->|channel| HGPU[k3_nano_harness.cu]
    HGPU -->|result| DAEMON
    DAEMON -->|channel| FFI
    FFI -->|result| PY
```

## Build

```bash
cd k3_nano

# Full build (all kernels)
make all

# Individual targets
make attention      # fused_attention only
make gemm           # gemm_online_softmax only
make draft          # speculative drafting
make test           # compile + run test
make spec           # load spec from config/g6_architecture.json
```

## Speculative Decoding

```mermaid
graph TD
    DRAFT[Draft Model<br/>K3 Nano 0.3B] -->|k candidates| VERIFY[Verify Step<br/>full model]
    VERIFY -->|accepted| TOKEN[Token Output]
    VERIFY -->|rejected| RETRY[Retry with<br/>new candidates]
    RETRY --> DRAFT
```

## G6 Architecture

| Parameter | Value |
|-----------|-------|
| Draft model | K3 Nano 0.3B |
| Target model | G6 6.15B |
| Speculation depth | k=4 |
| Acceptance rate target | ≥70% |
| Latency reduction | 2–3× |

---

## Topics

`CUDA` `attention` `GEMM` `online-softmax` `WMMA` `FP16` `tensor-cores` `speculative-decoding` `Kimi` `K3-Nano` `Rust-FFI` `Mojo` `Mojo-MAX` `GPU-kernels` `sovereign`

---

**Sovereign Source License v1.0 + BSL-1.1 + AGPL-3.0 (tri-license)**

Ahmad Ali Parr · Bel Esprit D'Accord Irrevocable Trust · EIN 42-697643
