# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Patt92 ROCm Halo Strix additions

Based on upstream llama.cpp commit [`465e49b9cea78a68b9c244ffb48d0ee24a82873d`](https://github.com/ggml-org/llama.cpp/commit/465e49b9cea78a68b9c244ffb48d0ee24a82873d).

This branch tracks the current upstream `llama.cpp` master and intentionally keeps upstream ROCm fusion, Qwen, DeepSeek, and Ornith graph semantics intact. Its backend delta is limited to tested gfx1151 MMQ layouts, scoped hipCUB argsort support, the AMD `MUL_MAT_ID` guard, and bounded multi-backend scheduler splits; `TOP_K` remains on the upstream HIP implementation.

- Adds isolated `glm5next` / GLM-5.3-Flash text inference, including its hybrid KDA/MLA memory layout and NextN/MTP draft context.
- Keeps completed GLM indexer pool keys in a persistent cache instead of rebuilding the entire context on every pass.
- Ranks GLM indexer pools before expanding the selected pools to cells, removing several context-by-ubatch intermediates from the graph.
- Keeps both mathematically equivalent GLM indexer scorers and defaults to the fused one. The CUDA Lightning Indexer has no WMMA kernel on HIP and falls back to its vector kernel, which is why the fused path was once assumed to lose on gfx1151. Measured, it wins by a wide margin: **150 t/s peak prefill unfused against 200 t/s fused**. The unfused chain has to materialize the per-head score, `[n_pool, n_head_idx, n_tokens]` F32 - 1.9 GB per layer at 29k context with a 2048-token ubatch - and then walk it again for the ReLU, a permuted copy, the weighting and the row sum, roughly 169 GB of traffic per ubatch across the full-attention layers. `LLAMA_GLM5NEXT_FUSED_LID=0` restores the unfused chain.
- Adds GLM-5.3-Flash MMProj support, including its vision-specific clamped SwiGLU tower.
- The GLM-specific hyper-connection fused nodes are retained because they keep GLM graph reservation tractable; no global fusion policy or backend dispatcher is replaced.
- Adds external NextN/MTP draft-head support for Qwen3.8-Flash-Next (`qwen4exp`). The target exports its four-stream hyper-connection state, while a self-contained MTP sidecar loads only the trailing NextN block and its own HC output mixer. The streams remain separate through `eh_proj`; averaging them first destroys draft acceptance. The draft block uses the correctness-first dense-attention path and does not enable the experimental sparse-FA transplant.
- Adds `ggml_flash_attn_ext_add_top_k`, an explicit-index counterpart to upstream's mask-derived `ggml_flash_attn_ext_set_n_kv_max`. Upstream's sparse selection is CUDA-only - `ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse` returns false on HIP and MUSA - so on those backends attention reads every KV column even where the caller already knows which few matter. The new call takes the indices directly, on `src[5]` and op_params slot 5, both previously unused; the two APIs are independent and one graph may carry both. No backend reads it yet, so this is API and graph plumbing only and changes nothing on its own.
- Gates the AMD `MUL_MAT_ID` float path on `ggml_cuda_should_use_mmvf`. That branch called `mul_mat_vec_f` unconditionally for any non-quantized `src0`, while the kernel asserts `ncols % 2 == 0` and needs its strides aligned to `2*type_size`. Every other caller consults the predicate; this one did not, so a model with float expert or dense weights aborted at load with `mmvf.cu:426`. Verified on gfx1151 with DeepSeek-V4-Flash UD-Q8_K_XL, whose dense stack is BF16 rather than quantized.
- On ROCm with rocPRIM 4.4 or newer, enables hipCUB for RPC argsort without changing upstream HIP top-k selection.
- Restores the measured gfx1151 MMQ warp distribution and the Q8_0/Q5_K/Q6_K RDNA3.5 tile choices without replacing upstream's MMQ implementation.
- Keeps multi-backend graph split boundaries fixed across requests to avoid growing RPC compute-buffer peaks under pipeline parallelism.
- Widens the `gated_delta_net` warp grid on gfx1151: eight warps at 32 heads, sixteen plus a shared-memory input cache at 64 or more, for prefill batches of 2048 tokens and up. GDN carries the whole Qwen3.8-Flash-Next prefill and the upstream kernel launches a fixed four warps regardless of batch size. Decode, KDA and state-keeping runs are untouched.
- Adds an AMD WMMA kernel for the lightning indexer, used when the indexer K cache is f16 and the batch is at least 16 rows. The indexer is the `n_kv`-proportional term of DeepSeek-V4 and GLM prefill and previously ran the scalar float4 kernel on HIP, one warp per KV row with every head re-reading global memory. Decode and quantized indexer caches keep the old path.
- Adds a coalesced dim-0 `concat` for a transposed `src1`, the shape DeepSeek-V4 builds when it joins its SWA-bounded `raw_k` to the narrowed `csa_k`. The generic kernel reads one element per row stride; the new one stages a 32x32 tile through shared memory.
- Gives RDNA3.5 its own MMVQ parameter table instead of aliasing it onto RDNA2, which always resolved to one warp per block. Token generation now takes two warps for MXFP4, Q4_K, Q5_K, Q6_K and Q8_0, and Q8_0 gets a vec-dot ratio of 4 so its K loop retires in half the trips.
- Shares the mm-ids helper and the q8_1 quantization of the activation across the gate/up `MUL_MAT_ID` pair of a MoE FFN. The two matmuls stay separate MMQ launches with unchanged arguments, so no GLU, bias or scale is folded in and this is not the class of fusion behind upstream #28113.
- Chooses the `MUL_MAT_ID` J tile on gfx1151 from the type and the average columns per expert rather than from the generic minimum-tile-count scan. `GGML_CUDA_MMQ_ID_J=<J>` forces a value, `=auto` restores the default.
- Caches the q8_1 activation for the duration of one graph evaluation on gfx1151, so MoE decode quantizes each row once instead of once per expert matmul. Off while a HIP graph is captured or replayed; `GGML_CUDA_DISABLE_MMVQ_Q8_1_CACHE=1` disables it.

The seven items above are ported from [myhacsint/llama.cpp `production/strix-halo-qwen4exp-b10685`](https://github.com/myhacsint/llama.cpp/tree/production/strix-halo-qwen4exp-b10685). That branch is Vulkan-first and states that its ROCm paths are not claimed to be validated there, so each was re-verified here: the full `test-backend-ops` suite passes 14747/14747 on gfx1151, and coverage was added for the two shapes upstream does not exercise -- `concat` with a transposed `src1`, and `gated_delta_net` at 2048 tokens. The `MUL_MAT_ID` pair and the q8_1 cache only engage in a real MoE graph, which single-op tests cannot reach.

The model-specific ports are architecture-gated: they do not alter the Qwen3.5/Ornith or DeepSeek graph implementations.

### Qwen3.8-Flash-Next MTP

Use a Qwen3.8-Flash-Next target together with its matching self-contained MTP sidecar. Start with two draft tokens; only raise it after checking the server's reported acceptance rate. The MTP head should stay on the controller GPU when the target is split over RPC.

```sh
--spec-type draft-mtp \
--spec-draft-model /opt/models/Qwen3.8-Flash-Next-Uncensored/Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf \
--spec-draft-device ROCm0 \
--spec-draft-ngl all \
--spec-draft-n-max 2 \
--spec-draft-type-k f16 \
--spec-draft-type-v f16
```

The implementation comes from [`ggml-org/llama.cpp#27836`](https://github.com/ggml-org/llama.cpp/pull/27836) by [@rmonsurate](https://github.com/rmonsurate). Draft-only sidecar loading follows [`unslothai/llama.cpp#144`](https://github.com/unslothai/llama.cpp/pull/144) by [@danielhanchen](https://github.com/danielhanchen). The loader guard and the CPU/ROCm regression coverage are maintained here by [@Patt92](https://github.com/Patt92).

### Measured on gfx1151

All figures from AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), ROCm 7.15, 124 GB unified memory
per node. Prefill and generation are quoted with the KV depth they were taken at, because both
fall with context and a number without one is meaningless.

**Ornith-1.5-35B-A3B Q8_0, one node, `ctx 32768`, `ubatch 2048`:**

| context | prefill | tg |
|---|---|---|
| 1 | 33.8 t/s | 46.6 t/s |
| 4096 | 1356 t/s | 45.9 t/s |
| 16384 | 1166 t/s | 43.6 t/s |

**The same model split across two nodes over RPC costs 20% of tg and 23% of prefill** (36.5 t/s
and 1050 t/s at 4096). Its weights are 37.8 GB and a single node has 124 GB, so routing it through
RPC buys nothing. Only models that genuinely exceed one node - GLM-5.3-Flash at ~154 GB - should
be split.

**GLM-5.3-Flash Q3_K_M, two nodes over RPC, `ctx 131072`:** prefill decomposes into a fixed cost
per ubatch and a term proportional to KV depth. At `ubatch 2048` that is roughly 9.2 s fixed per
ubatch plus 0.5 ms per KV unit, so 4k context is 82% fixed cost while 20k is roughly half and
half. Raising the ubatch to 4096 does **not** help - measured 117 t/s against 125 t/s at 2048
around 32k - because the fixed part is already amortized at 2048.

The dominant remaining cost at depth is that flash attention reads every KV column although the
DSA indexer selected 2052 of them: 9.7x the necessary work at 20k and 14.3x at 29k, plus two
full-width masks that make a larger ubatch expensive in VRAM. That is what
`ggml_flash_attn_ext_add_top_k` exists for; no backend consumes it yet.

### Applying the standalone patch

```sh
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 465e49b9cea78a68b9c244ffb48d0ee24a82873d
git apply --check /path/to/rocm-halo-strix.patch
git apply /path/to/rocm-halo-strix.patch
```

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
