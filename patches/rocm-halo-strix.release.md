# patt92 ROCm / Strix Halo - upstream f027c4f1b refresh

Cumulative ROCm/HIP optimization branch for AMD Strix Halo (`gfx1151` / RDNA3.5),
rebased onto upstream llama.cpp `f027c4f1b025e05d6a2fc3b741047bda07b85ef7`.

The release carries **only** `rocm-halo-strix.patch`. Apply it to a clean checkout of
that exact upstream commit:

```sh
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout f027c4f1b025e05d6a2fc3b741047bda07b85ef7
git apply --check /path/to/rocm-halo-strix.patch
git apply /path/to/rocm-halo-strix.patch
```

## What changed this round

This refresh is the upstream merge that the previous release (b10844) deliberately
deferred. It turns on a single upstream commit.

Upstream `8e93a9773` ("CUDA + ggml: add sparse-fa for DSV4/GLM") implements the same
feature this branch already carries, but guards it with

```
GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA");
```

under `#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)`. Taking it wholesale would
have deleted sparse attention on `gfx1151`. Both paths now stand side by side behind
those guards:

- **`ggml.h` / `llama-graph.cpp`** keep both APIs. Upstream derives `n_kv_max` from the
  mask (`ggml_flash_attn_ext_set_n_kv_max`, op-param slot 4); this branch passes explicit
  column indices (`ggml_flash_attn_ext_add_top_k`). They are independent.
- **`fattn-common.cuh`** appends `use_sparse` to `launch_fattn` rather than taking
  upstream's parameter 5. `int` converts to `bool` silently, so a positional slip in that
  list would compile and only surface as wrong output.
- **`fattn-mma-f16.cuh`** carries both trailing template flags. Upstream's `use_sparse`
  claimed the same slot this branch uses for `stream_k_strided`; the two
  kernel-selection branches are concatenated behind their mutually exclusive HIP guards,
  so no translation unit instantiates both.
- **`deepseek4.cpp`** computes `n_kv_max` as
  `min(raw_mask->ne[0], n_swa) + top_k->ne[0]`.
- **`test-backend-ops.cpp`** carries `n_kv_raw`, `n_top_k` and `n_kv_max`. Upstream's ten
  sparse cases passed `n_kv_max` as the 17th positional argument, which is `n_kv_raw`
  here; they now name the skipped defaults explicitly.

Upstream's `need_f16_K` / `need_f16_V = true` change in `fattn-tile.cuh` was
**deliberately not applied** to this branch's HIP-only call sites, because it could not
be verified for them on the build host.

The other 13 upstream commits merged cleanly and are carried unmodified. Three are worth
naming:

- `cff184438` **Update ROCm to 10.0.0 release (#27803)** - upstream's CI and release
  builds move from ROCm 7.14.0 to 10.0.0, and the wheel index changes from
  `repo.amd.com/rocm/whl-multi-arch/` to `stable.repo.amd.com/rocm/whl-next/`. This
  touches only `.github/`; it changes no source and imposes nothing on a local build.
  It does mean upstream's published Linux and Windows ROCm binaries are now built
  against ROCm 10.
- `3d3d7c818` (ggml-cuda: remove unused vars) drops 16 lines from `mmq-vec-dot.cuh` and
  `mmq.cuh` - both files this branch modifies. Merged without conflict.
- `9400c8946` and `7798007a2` extend DeepSeek-V4 vision input and mtmd support.

Carried unchanged from b10844 and still unexercised on hardware:
`[TAG_TOPK_RADIX_OVER_CUB_SORT]` and `[TAG_KPOOL_POOL_TOPK]`.

## Verification

Verified on Apple M4 Pro, 2026-09-02:

- `git apply --check` clean at `f027c4f1b`; applying it reproduces the branch tip's tree
  bit for bit, with a single exception: `patches/rocm-halo-strix.patch` itself, which the
  patch necessarily cannot contain its own final bytes for. Every source file, test and
  document matches exactly.
- `git diff --check` clean outside the stored `patches/*.patch` artifacts.
- No Vulkan, SYCL, OpenCL or WebGPU file touched.
- `test-backend-ops`: 3974/3974 across three backends.
- `test-backend-ops -o FLASH_ATTN_EXT`: 11993/11993 on Metal.
- `test-llama-archs`: OK for `glm5next`, `deepseek4` and `qwen4exp`.
- `ctest`: 61/62. The one failure, `test-tokenizers-ggml-vocabs`, is an unfetched
  git-lfs vocab pointer on the build host and is unrelated to any code here.
- Removed-upstream-line audit: `ggml-cuda.cu` unchanged at 19; tree-wide 476 -> 491, with
  all 15 new removals inside the three sparse-FA files.

## Known limits

**The build host compiles no HIP and runs no ROCm.** Every result above covers the CPU
and Metal paths only. `fattn-mma-f16.cuh`, `fattn-common.cuh` and `fattn-tile.cuh` reach
you unbuilt for HIP, which makes this the highest-risk refresh of the series. Confirm it
on hardware before stacking anything on top of it.

Nodes must be rebuilt in lockstep: the RPC protocol version has stayed at 6.0 since
`a7cc83bba`, so a mixed client/server state can fail without a useful client-side error.

If an RPC worker aborts in `mmvq.cu`, set `GGML_CUDA_DISABLE_FUSION=1` in
`/etc/llama-cluster/rpc-worker.env`. See the README section "RPC worker aborts and
backend fusion".
