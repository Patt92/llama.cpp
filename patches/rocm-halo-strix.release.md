# Patt92 ROCm / Strix Halo - upstream 0df017d6d refresh

Cumulative ROCm/HIP optimization branch for AMD Strix Halo (`gfx1151` / RDNA3.5),
rebased onto upstream llama.cpp `0df017d6dd246edb1d06577976c759cf7a3c50d6`.

The release carries only `rocm-halo-strix.patch`. Apply it to a clean checkout of that
exact upstream commit:

```sh
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 0df017d6dd246edb1d06577976c759cf7a3c50d6
git apply --check /path/to/rocm-halo-strix.patch
git apply /path/to/rocm-halo-strix.patch
```

SHA-256: `c77b2a28c1baef7265d4467f7756dabb417ac012df5184358d250daea75241a0`

## What changed

- Merges current upstream through `0df017d6d` and adapts the fork-owned GLM5Next
  architecture to upstream's per-layer expert-width API.
- Restores current upstream CUDA/HIP fusion dispatch in full. The branch no longer
  dispatches its own Q8.1, MoE down-scale, paired-MMVQ, shared-expert or SSM fusion
  patterns, and no longer restricts HIP `MUL_MAT_ID` fusion to one token.
- Retains separate gfx1151 kernel/routing work: hipCUB RPC top-k/argsort, Strix Halo
  MMQ/MMVQ/Flash-Attention tuning, rank-1 LoRA MMVF safety, DeepSeek sparse HIP Flash
  Attention, FastMTP, GLM5Next, and narrow ROCmFPX Q8 support.

## Verification

- `git diff --check` is clean for the source/documentation patch.
- `git apply --check` was run against a fresh detached worktree at the exact base above.
- `llama-server` built and reported version `0.3.0-dev (build 10880, commit db102d054)`
  on Apple M4 Pro with CPU/RPC backends.

The build host has no ROCm runtime. HIP/gfx1151 compilation and inference remain the
required hardware validation before deployment. Controller and RPC worker must use the
same build.
