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

Based on upstream llama.cpp commit [`52d42686560a9e8f441f9b9780c8890c37d2802d`](https://github.com/ggml-org/llama.cpp/commit/52d42686560a9e8f441f9b9780c8890c37d2802d).

This branch tracks the current upstream `llama.cpp` master and intentionally keeps upstream ROCm fusion, Qwen, DeepSeek, and Ornith graph semantics intact. Its backend delta is limited to tested gfx1151 MMQ layouts, scoped hipCUB argsort support, the AMD `MUL_MAT_ID` guard, and bounded multi-backend scheduler splits; `TOP_K` remains on the upstream HIP implementation.

- Adds isolated `glm5next` / GLM-5.3-Flash text inference, including its hybrid KDA/MLA memory layout and NextN/MTP draft context.
- Keeps completed GLM indexer pool keys in a persistent cache instead of rebuilding the entire context on every pass.
- Ranks GLM indexer pools before expanding the selected pools to cells, removing several context-by-ubatch intermediates from the graph.
- Keeps both mathematically equivalent GLM indexer scorers and defaults to the fused one. The CUDA Lightning Indexer has no WMMA kernel on HIP and falls back to its vector kernel, which is why the fused path was once assumed to lose on gfx1151. Measured, it wins by a wide margin: **150 t/s peak prefill unfused against 200 t/s fused**. The unfused chain has to materialize the per-head score, `[n_pool, n_head_idx, n_tokens]` F32 - 1.9 GB per layer at 29k context with a 2048-token ubatch - and then walk it again for the ReLU, a permuted copy, the weighting and the row sum, roughly 169 GB of traffic per ubatch across the full-attention layers. `LLAMA_GLM5NEXT_FUSED_LID=0` restores the unfused chain.
- Adds GLM-5.3-Flash MMProj support, including its vision-specific clamped SwiGLU tower.
- Holds the GLM indexer's pooled key cache in F16 rather than F32. `ggml_cuda_lightning_indexer` only takes the AMD WMMA kernel when K is F16, and GLM's indexer is exactly that kernel's shape (`hsk=128, nh=32`), so at F32 gfx1151 fell back to the scalar float4 kernel -- one warp per KV row, every head re-reading global memory. Measured on gfx1151: 5472 -> 1570 us at ubatch 2048, 3.48x on the term that scales with `n_kv`. This is precision the inputs never had, since a pool key is a weighted mean of indexer keys that are themselves cached F16, and the score built from it only ranks pools for a top-k. Decode does not reach the kernel -- it needs a batch of 16 -- and pays about 4% more on the scalar path, which is 0.04% of a token. The cache halves as a side effect.
- Compacts masked-out KV rows inside the flash-attention tile kernel. GLM's indexer selects roughly 2048 of `n_kv` cells and expresses that as a full-width mask; the attention read every column anyway, because absorbed MLA makes Q 512 wide and so lands on the tile kernel, where upstream's sparse path -- wired only into `fattn-mma` -- never reaches it. The kernel now ballots which rows of each warp-sized group survive the mask and remaps lanes onto them, so the KQ and VKQ work scales with the selection. Row granularity rather than block skipping is what this model needs: its selection is 512 contiguous runs of four cells scattered over the context. Measured on gfx1151 at `hsk=512, gqa 8, kv=32768, ubatch 2048, 2048 selected`: 6.88 -> 16.65 TFLOPS, 2.42x. Prefill only; decode picks a different tile shape and is unaffected. No cache-type change is needed -- a q8_0 cache is converted to F16 scratch before the kernel as before.
- The GLM-specific hyper-connection fused nodes are retained because they keep GLM graph reservation tractable; no global fusion policy or backend dispatcher is replaced.
- Adds external NextN/MTP draft-head support for Qwen3.8-Flash-Next (`qwen4exp`). The target exports its four-stream hyper-connection state, while a self-contained MTP sidecar loads only the trailing NextN block and its own HC output mixer. The streams remain separate through `eh_proj`; averaging them first destroys draft acceptance. The draft block uses the correctness-first dense-attention path and does not enable the experimental sparse-FA transplant.
- Adds `ggml_flash_attn_ext_add_top_k`, an explicit-index counterpart to upstream's mask-derived `ggml_flash_attn_ext_set_n_kv_max`. Upstream's sparse selection is CUDA-only - `ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse` returns false on HIP and MUSA - so on those backends attention reads every KV column even where the caller already knows which few matter. The new call takes the indices directly, on `src[5]` and op_params slot 5, both previously unused; the two APIs are independent and one graph may carry both. No backend reads it yet, so this is API and graph plumbing only and changes nothing on its own.
- Gates the AMD `MUL_MAT_ID` float path on `ggml_cuda_should_use_mmvf`. That branch called `mul_mat_vec_f` unconditionally for any non-quantized `src0`, while the kernel asserts `ncols % 2 == 0` and needs its strides aligned to `2*type_size`. Every other caller consults the predicate; this one did not, so a model with float expert or dense weights aborted at load with `mmvf.cu:426`. Verified on gfx1151 with DeepSeek-V4-Flash UD-Q8_K_XL, whose dense stack is BF16 rather than quantized.
- On ROCm with rocPRIM 4.4 or newer, enables hipCUB for RPC argsort without changing upstream HIP top-k selection.
- Zeroes the MTP hidden-state graph input on token-only batches. `llm_graph_input_embd_h::set_input` writes `h` only when the ubatch carries embeddings, so a token-only ubatch left the `DECODER_MTP` graph reading whatever the compute buffer held. Upstream master has the same gap.
- Restores `prop.integrated` on RDNA3.5. Upstream reverted it for all HIP builds over corrupted output in #15034, but gfx1151 has no VRAM carveout worth the name -- `mem_info_vram_total` reports 0.5 GB and the model lives in GTT -- so with the flag off, `ggml_backend_cuda_device_supports_buft` refuses host buffers and the scheduler keeps a device copy of memory the GPU could address in place. On a 124 GB node with two models resident that was the difference between 82 GB used and 119 GB used with 8 GB of swap, and prefill segments falling from 325 t/s to 12.6 t/s as ubatches hit swapped pages. Restored for RDNA3.5 only; every other architecture keeps upstream's `false`.
- Restores the measured gfx1151 MMQ warp distribution and the Q8_0/Q5_K/Q6_K RDNA3.5 tile choices without replacing upstream's MMQ implementation.
- Widens the `gated_delta_net` warp grid on gfx1151: eight warps at 32 heads, sixteen plus a shared-memory input cache at 64 or more, for prefill batches of 2048 tokens and up. GDN carries the whole Qwen3.8-Flash-Next prefill and the upstream kernel launches a fixed four warps regardless of batch size. Decode, KDA and state-keeping runs are untouched.
- Adds an AMD WMMA kernel for the lightning indexer, used when the indexer K cache is f16 and the batch is at least 16 rows. The indexer is the `n_kv`-proportional term of DeepSeek-V4 and GLM prefill and previously ran the scalar float4 kernel on HIP, one warp per KV row with every head re-reading global memory. Decode and quantized indexer caches keep the old path.
- Adds a coalesced dim-0 `concat` for a transposed `src1`, the shape DeepSeek-V4 builds when it joins its SWA-bounded `raw_k` to the narrowed `csa_k`. The generic kernel reads one element per row stride; the new one stages a 32x32 tile through shared memory.
- Gives RDNA3.5 its own MMVQ parameter table instead of aliasing it onto RDNA2, which always resolved to one warp per block. Token generation now takes two warps for MXFP4, Q4_K, Q5_K, Q6_K and Q8_0, and Q8_0 gets a vec-dot ratio of 4 so its K loop retires in half the trips.
- Shares the mm-ids helper and the q8_1 quantization of the activation across the gate/up `MUL_MAT_ID` pair of a MoE FFN. The two matmuls stay separate MMQ launches with unchanged arguments, so no GLU, bias or scale is folded in and this is not the class of fusion behind upstream #28113.
- Chooses the `MUL_MAT_ID` J tile on gfx1151 from the type and the average columns per expert rather than from the generic minimum-tile-count scan. `GGML_CUDA_MMQ_ID_J=<J>` forces a value, `=auto` restores the default.
- Caches the q8_1 activation for the duration of one graph evaluation on gfx1151, so MoE decode quantizes each row once instead of once per expert matmul. Off while a HIP graph is captured or replayed; `GGML_CUDA_DISABLE_MMVQ_Q8_1_CACHE=1` disables it.

The seven items above are ported from [myhacsint/llama.cpp `production/strix-halo-qwen4exp-b10685`](https://github.com/myhacsint/llama.cpp/tree/production/strix-halo-qwen4exp-b10685). That branch is Vulkan-first and states that its ROCm paths are not claimed to be validated there, so each was re-verified here: the full `test-backend-ops` suite passes 14747/14747 on gfx1151, and coverage was added for the two shapes upstream does not exercise -- `concat` with a transposed `src1`, and `gated_delta_net` at 2048 tokens. The `MUL_MAT_ID` pair and the q8_1 cache only engage in a real MoE graph, which single-op tests cannot reach.

#### When these engage

The gates carry the original author's measured thresholds and were not widened. Read them against
the model's own metadata before expecting a change, because several are narrower than they look:

| tune | engages when | Qwen3.8-Flash-Next Q5_K_M |
| --- | --- | --- |
| `gated_delta_net` warp grid | `ssm.time_step_rank` is 32, or 64 and above | **no** -- the model has 48 |
| AMD WMMA lightning indexer | `attention.indexer.head_count` is 32 or 64, indexer K is f16, batch >= 16 | **no** -- the model has 4 |
| transposed-`src1` `concat` | dim-0 concat of a 2-D tensor with a transposed operand | no -- DeepSeek-V4 shape |
| RDNA3.5 MMVQ table | MXFP4/Q4_K/Q5_K/Q6_K/Q8_0 at one output column | yes, but token generation on this model is bandwidth-bound, so the warp count is not what limits it |
| `MUL_MAT_ID` pair | two adjacent `MUL_MAT_ID` over one activation, prefill | yes |
| `MUL_MAT_ID` auto J | RDNA3.5; Q5_K/Q6_K with 256 or more experts, or Q8_0 | yes -- 512 experts, so J is forced to 64 |
| q8_1 activation cache | RDNA3.5 MoE decode outside a HIP graph | yes |

Widening the first two is mechanical -- 48 heads divide evenly into the sixteen-warp path, and the
indexer kernel is templated on head count -- but the thresholds above are where the original author
measured, so anything wider needs its own measurement rather than an assumption.

Upstream reached the same conclusion about MoE tile sizing independently in
[`#24546`](https://github.com/ggml-org/llama.cpp/pull/24546), which sized routed-MoE MMQ N-tiles
from typical expert width, and then reverted it in
[`#28551`](https://github.com/ggml-org/llama.cpp/pull/28551). The revert was about where the logic
lives, not about whether it works: the objection was that it changed kernel configurations when the
choice belongs entirely on the host side, and the suggested shape is an `ncols_opt` field on
`mmq_args` decided in `ggml_cuda_mul_mat_q`. The version carried here is already host-side -- it
only overrides the J that `mul_mat_q_switch_J` would have picked -- and it is gated to RDNA3.5,
which that PR explicitly excluded. Its measurements are still the best evidence available for the
idea on this hardware: at a typical expert width of 16 on gfx1151, `+21.7%` for Q4_K and `+7.5%`
for Q5_K at the operator level, with width-64 negative controls flat.

The model-specific ports are architecture-gated: they do not alter the Qwen3.5/Ornith or DeepSeek graph implementations.

### Qwen3.8-Flash-Next MTP

Use a Qwen3.8-Flash-Next target together with its matching self-contained MTP sidecar. Two or three
draft tokens is the working range; measured on a 125 GB Q5_K_M target this branch reaches
`draft acceptance = 0.70-0.90, mean len = 3.1-3.7` at `--spec-draft-n-max 3`.

**A high acceptance rate is not a reason to draft deeper.** These sidecars carry a single NextN
block that is applied recursively, so from the second draft token onward the head conditions on its
own guess instead of on a verified token. Raising `--spec-draft-n-max` from 3 to 6 on this model
took acceptance from 0.899 to 0.286 and halved token generation; accepted tokens per round fell
from 2.70 to 1.70 while the drafted count doubled. Any acceptance figure is an average over the
depths that were actually drafted and does not extrapolate past them.

**Know where the draft model lands when the target is split over RPC.** Without
`--spec-draft-device` the draft's device list is empty, so it inherits the target's `--split-mode`
and `--tensor-split` and is spread across the same devices, remote one included. Pinning it to the
controller with `--spec-draft-device ROCm0 --spec-draft-ngl all` is worth testing but is not a
guaranteed win: with a layer split the target's last layers are the remote ones and the MTP head
consumes their hidden state, so pinning the head locally can add a round trip per draft step rather
than remove one. Measure it as a single change.

```sh
--spec-type draft-mtp \
--spec-draft-model /opt/models/Qwen3.8-Flash-Next-Uncensored/Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf \
--spec-draft-device ROCm0 \
--spec-draft-ngl all \
--spec-draft-n-max 2 \
--spec-draft-type-k f16 \
--spec-draft-type-v f16
```

#### Worked example: 125 GB Q5_K_M target across two Strix Halo nodes

The target does not fit in one node's 124 GB, so the layer split is mandatory rather than a choice.
A second model (Ornith Q8_0) is resident alongside it, which is what constrains the split ratio.

```sh
llama-server \
  --model /opt/models/Qwen3.8-Flash-Next-Uncensored/Q5_K_M/Qwen3.8-Flash-Next-Uncensored-Q5_K_M-00001-of-00003.gguf \
  --alias qwen3.8-flash-next --host 127.0.0.1 --port 5807 \
  --gpu-layers all --fit off --load-mode none \
  --ctx-size 262144 --parallel 1 \
  --rpc 10.44.0.2:50053 --split-mode layer --tensor-split 30,100 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --batch-size 8192 --ubatch-size 2048 --cont-batching \
  --jinja --reasoning on --reasoning-format deepseek --reasoning-effort medium \
  --temp 0.7 --top-p 0.95 --top-k 20 --repeat-penalty 1.05 \
  --mmproj /opt/models/Qwen3.8-Flash-Next-Uncensored/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf \
  --spec-type draft-mtp \
  --model-draft /opt/models/Qwen3.8-Flash-Next-Uncensored/Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf \
  --spec-draft-device ROCm0 --spec-draft-ngl all \
  --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-p-min 0.2 \
  --spec-draft-type-k f16 --spec-draft-type-v f16
```

Why each non-obvious value is what it is:

- **`--spec-draft-n-max 3`, and do not raise it on an acceptance rate alone.** A measured 0.899
  acceptance at `n-max 3` looks like room for a deeper draft, and it is not: raising it to 6 on
  this model collapsed acceptance to 0.286 and cut token generation in half, from 30.9 to 15.5 t/s.
  Accepted tokens per round went *down*, from 2.70 to 1.70, while the draft work doubled. The
  reason is that a sidecar with a single NextN block is applied recursively, so from step two
  onward the head conditions on its own guess rather than on a verified token and the error
  compounds. An acceptance figure measured at depth 3 is an average over depths 1-3 and says
  nothing about depth 6. Move this value one step at a time and read `draft acceptance` and
  `mean len` after each.
- **`--spec-draft-device ROCm0 --spec-draft-ngl all` is a placement question worth testing, not a
  settled win.** Without it the draft's device list is empty, so it inherits the target's
  `--tensor-split` and lands partly on the remote node. Against that: with a layer split the
  target's *last* layers are the remote ones, and the MTP head consumes their hidden state, so
  pinning the head locally can add a round trip per draft step instead of removing one. Which way
  it falls depends on the split ratio. Measure it on its own.
- **`--temp 0.7 --top-k 20`** is primarily an output-quality choice, close to the vendor's
  thinking-mode recommendation. Its effect on speculative decoding is not one-directional: a lower
  temperature sharpens the target distribution and helps agreement, but `--top-k` truncates that
  distribution, and every draft token outside the surviving set is rejected outright. If token
  generation matters more than sampling behaviour here, test `--top-k` on its own.
- **`--batch-size 8192`** with an unchanged `--ubatch-size 2048` puts four ubatches in flight
  instead of two, which is what gives the two-node layer split something to pipeline.
- **`--parallel 1`** is deliberate: two slots cost roughly a third of token generation on this
  hardware, because `n_stream` is computed for both even when only one is active.
- **`--ctx-size 262144` costs almost nothing.** With `full_attention_interval` 4, only 12 of 48
  layers hold a KV cache; at `head_count_kv` 2 and a key/value length of 256 in q8_0 that is
  13.1 KB per token, so the full 256k context is about 3.4 GB. Shortening the context to save
  memory is the obvious move here and it is the wrong one.
- **`--tensor-split 30,100`** gives the controller 23% of the layers. That is set by what the
  co-resident model leaves free, not by what is best for throughput -- a more even split lets the
  prefill pipeline overlap better, so move toward `50,100` if memory allows and confirm with
  `free -g` on both nodes after the load.

Baseline to compare against, same host, 52k of context: prompt processing 231-253 t/s, generation
30.7-30.9 t/s, `draft acceptance = 0.899, mean len = 3.70`. The server prints all of these, so every
change in this block is a before/after with no separate benchmark run.

Two cautions when reading them. The `eval time` rate is cumulative and includes the ramp: on one
run here it read 18.7 t/s at 100 generated tokens and 31.4 t/s at 739, so a short generation is not
comparable to a long one -- use `tg_3s`, or compare runs of similar length. And token generation
falls with context depth, so a figure without the KV depth it was taken at means nothing. Change
one flag at a time; `draft acceptance` is the leading indicator for anything speculative, with the
generation rate only its consequence.

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
git checkout 52d42686560a9e8f441f9b9780c8890c37d2802d
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
