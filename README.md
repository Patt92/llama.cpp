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

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Patt92 ROCm / Strix Halo fork

This branch is based on upstream llama.cpp commit
[`b81c99b479d4c24e5eeca10de99032ebd343ef8f`](https://github.com/ggml-org/llama.cpp/commit/b81c99b479d4c24e5eeca10de99032ebd343ef8f).
It is a self-contained cumulative ROCm/HIP optimization branch for AMD Strix Halo
(`gfx1151` / RDNA3.5): it does not depend on the continued existence of any earlier
optimization branch. It retains normal upstream functionality, but is not intended to
replace the portable default upstream build.

### Included changes relative to upstream

#### HIP, RPC and long-context routing

- Enables hipCUB for HIP top-k and argsort, including the distributed RPC execution
  path, so these operations remain GPU-capable on ROCm.
- Adds a gfx1151-local Lightning Indexer top-k specialization for 512, 1024 and 2048
  candidates up to 8192 score entries, avoiding the expensive generic device-wide sort
  during DeepSeek-V4 long-context prefill.
- `[TAG_TOPK_RADIX_OVER_CUB_SORT]` Beyond that 8192-column limit, prefers upstream's
  radix select over the argsort fallback on HIP. Upstream compiles the radix select only
  when hipCUB is absent, because with hipCUB present `TOP_K` is expected to reach
  `DeviceTopK`. On ROCm it never does: that path requires `CCCL_MAJOR_VERSION >= 3.2`,
  a macro hipCUB does not define, so enabling hipCUB silently routed `TOP_K` into a full
  sort of every column to keep `k` of them. With a Lightning Indexer configured for
  `top_k = 2048` against a long KV cache that is a device-wide sort of tens of thousands
  of entries per row per layer, once per ubatch. The radix select is O(ncols) per row and
  batches all rows into one launch; hipCUB remains in use for `ARGSORT`, which is the
  reason this fork enables it.
- Adds indexed sparse HIP Flash Attention for the DeepSeek-V4 CSA F16 layout. Large
  prefills read only the raw attention window plus the 512 rows selected by the
  Lightning Indexer instead of scanning the full compressed KV cache. The path is
  enabled only when the dense cache is at least three times larger than the active
  set; decode, small verification batches and short contexts retain the existing path.
- Keeps the Lightning Indexer key cache in F16 when a quantized KV cache is selected,
  where required by the DeepSeek path.
- Resolves fused operations per layer and device instead of disabling an entire graph
  on a single ROCm/RPC backend mismatch; the default fallback remains unfused GPU work,
  not CPU work.
- Restores a fixed scheduler split-input cut-off after a dynamically grown split buffer.
  This prevents cross-backend input copies from remaining live across ever-larger graph
  regions in controller-plus-RPC layouts.
- Validates HIP MMVF dispatch for rank-1 MoE LoRA-B `MUL_MAT_ID` tensors, as used by
  DeepSeek-V4 expert adapters. The even-K MMVF kernel is never selected for this
  layout; it uses the tested general GPU fallback instead. This avoids an RPC-worker
  GPU fault at the cost of graph reuse for the affected LoRA operation.

#### Strix Halo compute tuning

- Overrides four entries of the RDNA3.5 MMQ configuration table for wide Q8_0,
  Q5_K and Q6_K tiles, and adds the expert-MMQ tile selection that picks the `MUL_MAT_ID`
  J dimension from the average rows per expert (`GGML_CUDA_MMQ_ID_J` to force or disable
  it). The overrides replace existing entries rather than removing them, so no shape loses
  its MMQ configuration.
- Adds RDNA3.5/gfx1151 launch tuning for MMVQ and Q8 MoE.
- Adds optional ROCTX ranges to `llama-bench` (`-DLLAMA_BENCH_ROCTX=ON`, requires the ROCm
  `rocprofiler-sdk-roctx` headers), with profiling scoped to the prompt runs, so
  `rocprofv3` output can be attributed to prefill instead of the whole process.
- Caches MMVQ Q8_1 activations and partitions MMQ waves across rows and columns.
- Adds AMD WMMA Lightning Indexer support and tuned tiled Flash Attention, including a
  fix for NaNs in the compacted-tile mask path.
- Tunes Gated Delta Net and adds quantized-KV Flash Attention support for Strix Halo.
- Extends the gfx1151 MMVQ selection to the relevant Q4/Q5/Q6/Q8 and MXFP4 decode
  types, including the Q6_K VDR=2 decode kernel.
- Keeps HIP integrated-GPU host-buffer handling safe.

#### GLM-5.3-Flash (`glm5next`) — experimental

- Carries the still-open upstream draft PR #27752 (`eauchs`), which adds the `glm5next`
  architecture: converter, text graph, KDA linear attention, MLA with a DSA indexer and
  k-pool compression, gated-residual hyper-connections and the 288-expert MoE. Without it,
  a GLM-5.3-Flash GGUF fails at load with `unknown model architecture: 'glm5next'`.
- The DSA path uses its own `llama_memory_hybrid_kpool` container, renamed from
  `llama_memory_hybrid_idx` when upstream introduced a different class of that name for
  Qwen3.8-Flash-Next (`qwen4exp`, PR #27742). The two are unrelated: upstream's derives from
  `llama_memory_hybrid` and has no k-pool; this one is standalone and carries the k-pool
  metadata GLM-5.3 needs. Both coexist. `llama_kv_cache`,
  `llama_memory_hybrid` and this fork's DeepSeek-V4 caches are not modified, which is why
  this PR was chosen over the competing draft #27754 — the latter also rewrites the shared
  KV cache.
- Includes the vision tower, the 2026-08-26 image preprocessor and the bicubic resampler
  from the competing draft #27754, ported onto this text graph. Only its `tools/mtmd`
  changes and the two `gguf-py` writer helpers were taken; its text and KV-cache rewrite
  was not. `test-mtmd-impl` passes 216 assertions including the glm5next budget and resize
  cases.
- Reads the k-pool size from either `attention.indexer.block_size` (#27752) or
  `attention.indexer.kpool` (#27754), because the two converters disagree on the key name.
  Without this, a GGUF from the other converter aborts at
  `GLM5NEXT requires index_kpool`.
- Adds the NextN/MTP draft head. It follows the
  GLM-5.2 head in `glm-dsa.cpp`: `enorm(embed) + hnorm(prev_hidden) -> concat -> eh_proj ->
  one dense DSA decoder block -> shared_head_norm -> shared LM head`. It reuses the trunk's
  own attention and FFN builders rather than duplicating them, runs without the mHC mixer
  (the NextN block has no `hc_*` tensors) and without the DSA indexer, and an MTP context
  allocates a KV cache holding only the NextN layers. Enable with `--spec-type draft-mtp`.
  **Never executed.** There is no MTP harness in `test-llama-archs` for any architecture,
  so this is compile-verified only; measure the acceptance rate before relying on it.
  Ordinary loading is unchanged: the NextN tensors keep `TENSOR_SKIP` unless MTP is asked
  for, so a run without `--spec-type draft-mtp` behaves exactly as before.
- The converter already emits the NextN block: `Glm5NextModel` inherits `block_count`,
  `filter_tensors` and `nextn_predict_layers` from `GlmMoeDsaModel`, so a GGUF converted
  with this tree carries the MTP head. Published GLM-5.3-Flash GGUFs generally do not —
  if `n_layer_nextn` is 0 in the startup log, that file has no NextN block and
  `--spec-type draft-mtp` will refuse the context with a clear error rather than
  misbehaving.
- The chat template shipped inside the published GLM-5.3-Flash GGUFs does not render under
  this project's Jinja engine. It indexes list elements as `m.content.0.type`, which Jinja2
  accepts but which fails here with `Static member property must be an identifier`. That
  breaks the object-arguments capability probe, so tool-call arguments stay a JSON string,
  and the template then calls `.items()` on that string. The visible symptom is
  `Unable to generate parser for this template ... Callee is not a function: got Undefined
  (hint: 'items')` on the first request with tools. A corrected template that changes only
  those four subscripts is published at a stable URL:
  [`glm-5.3-flash-chat-template.jinja`](https://github.com/Patt92/llama.cpp/releases/download/model-assets/glm-5.3-flash-chat-template.jinja).
  Pass it with `--chat-template-file`. No rebuild and no re-quantisation are needed.
- Treat it as experimental. Neither draft PR has been validated against the real 328 GB
  checkpoint by its author, and the 288-expert stacking, the FP8 `weight_scale_inv`
  dequantisation and the NextN block have only been exercised on a synthetic model.
  GGUFs produced by a different converter may use tensor names this loader does not expect;
  that surfaces as a missing-tensor error at load, not as silent corruption.

#### ROCmFPX Q8 GGUF support

- Supports published `Q8_0_ROCMFPX` / `Q8_0_ROCMFPX_AGENT` GGUFs (on-disk type 103,
  file types 111 and 115), including CPU and HIP UE4M3-scale dequantization and HIP
  `GET_ROWS`.
- Uses a dedicated gfx1151 MMQ tile loader for the 33-byte ROCmFPX-Q8 block layout.
  It materializes the current Q8 shared-memory representation, then uses the current
  Q8 WMMA/MMQ dot-product and RDNA3.5 configuration. Ordinary `Q8_0` remains on its
  own unchanged path.
- Treats ROCmFPX Q8 as experimental: validate model quality and throughput locally
  before deploying it as a production default.

#### DeepSeek-V4 and speculative decoding

- Makes the grouped output-projection input contiguous for small multi-token
  DeepSeek-V4 speculative batches.
- Uses matching MMVQ and Flash-Attention kernel configurations for decode and small
  speculative/MTP verification batches, preventing numerical divergence between the
  logits used for acceptance checks.
- Resets deferred MTP hidden state when a reused slot starts a genuinely new sequence
  and serializes that state with context checkpoints. This addresses the inter-request
  state leak tracked by upstream issue #26425 without discarding valid cached-prefix
  state.
- Keeps fused DeepSeek-V4 HC and Gated Delta Net operations on devices that support
  them while safely falling back only for mismatching layers.

#### Qwen3.5 / Qwen3.8 and hybrid SSM models

- Fuses SSM gate/beta projections, the SSM convolution-output L2 norm, and the SSM
  pre-scan chain (convolution, L2 norm, gate/beta).
- Folds SSM convolution-input concatenation into QKV MMVQ and fuses paired MMVQ
  matmuls that share an activation.
- Caches graph-local Q8_1 matmul inputs, folds Q8_1 quantization into RMS norm and
  gating-MUL, and fuses a matmul-plus-add-through-view sequence.
- Folds MoE top-k weights into the down projection and fuses the shared-expert output
  chain.
- Fuses IMRoPE and set-rows for BF16 KV cache use, and aligns the attention-gate
  tensor-parallel split with attention-Q.
- Builds the recurrent Qwen3.5-MoE attention result into the graph before the
  Gated Delta Net state update, preserving the required execution order.
- Capability-gates DFlash2 activation against the graph that is actually built: an
  unsupported DFlash2 selector on a DeepSeek-V4 backbone is rejected at load, because
  that backbone's graph does not build the selector lattice.
- Adds FastMTP compact-draft-vocabulary support for Qwen3.8 sidecars carrying an I64
  `d2t` map. The draft LM head evaluates only its compact vocabulary and scatters the
  logits back into the full target vocabulary before verification. Shape, I32/I64 index type,
  vocabulary range and required output-head invariants are validated at load/build time.

### DeepSeek-V4 long-context prefill

The indexed sparse HIP Flash Attention path is automatic and has no new command-line
flag. It requires the existing DeepSeek-V4 CSA graph, `--flash-attn on`, and F16 target
K/V cache types. A representative server configuration includes:

```sh
--flash-attn on \
--cache-type-k f16 \
--cache-type-v f16 \
--batch-size 2048 \
--ubatch-size 512
```

The optimization starts only for prefill batches of at least 64 tokens and when the
full compressed cache is at least three times larger than raw-window plus Top-K. This
keeps short-context and token-generation behavior unchanged. It reduces the
context-linear CSA Flash Attention work, but other model layers can still limit total
prompt throughput; measure identical prompts and cache state before and after the
build. Set `GGML_CUDA_DISABLE_DSV4_SPARSE_FA=1` in the `llama-server` environment to
force the previous dense path for an A/B comparison without rebuilding.

### glm5next indexer: pool-level selection

The Lightning Indexer scores pools of `kpool` cells and then picks `index_topk` cells for
the sparse attention to read. Upstream's shape for this expands every pool score to its
`kpool` member cells before ranking them, which means the ranking sorts `kpool` copies of
each number and the cut lands on a pool boundary regardless.

`[TAG_KPOOL_POOL_TOPK]` ranks the pools directly and expands only the winners through a
pool-to-cells table. The selection is the same set - the old width of `index_topk + kpool
- 1` cells was `index_topk/kpool` complete pools plus the incomplete tail - but the
`[n_kv, n_tokens]` score array, its two permute/cont copies and the `[n_kv, n_tokens]`
bias all leave the graph. At a 262144 context and a 2048-token ubatch each of those was
2.1 GB of compute buffer per indexer layer.

This matters twice over on a memory-tight split. The compute buffer is reserved for the
worst case `n_kv`, so it scales as `ctx-size` times `ubatch-size` whatever the actual
context; that product, not the KV cache, is usually what caps `--ubatch-size`. And a
larger ubatch is the main lever on MoE prefill throughput, because every ubatch reads
essentially the whole expert set once regardless of how many tokens it carries.

The tail is the one behavioural difference. It belongs to no complete pool and so cannot
be ranked; it is concatenated onto the winners instead of being forced to the top with a
`+1e9` bias. Over-selection stays harmless because `build_attn_mask_top_k` only clears
mask entries and adds the real KQ mask back afterwards.

### RPC worker aborts and backend fusion

An RPC worker running this branch can abort inside the CUDA/HIP backend with

```
ggml/src/ggml-cuda/mmvq.cu: GGML_ASSERT(ids || dst->ne[1] == 1) failed
  ggml_cuda_mul_mat_vec_q
  ggml_backend_graph_compute
  rpc_server::graph_compute
```

systemd reports `code=dumped, status=6/ABRT` and restarts the worker, so the symptom on
the client is `recv failed (bytes_recv=0)` followed by `Remote RPC server crashed or
returned malformed response`. The host itself keeps running, which makes this look like
a network fault rather than a compute fault.

That assert sits **inside** the `if (fusion)` precondition block, so it is unreachable
when backend fusion is off. Setting

```sh
GGML_CUDA_DISABLE_FUSION=1
```

in the worker's environment (for example an `EnvironmentFile` referenced by the systemd
unit) removes the abort by construction. The remote node then computes the same graph
without fused epilogues, which costs some throughput on that node only - the host keeps
its own fusion. Because the switch is decisive rather than cosmetic, it doubles as the
A/B: if the aborts stop, a fusion group is at fault; if they continue, they are not.

This is **not** the same defect as the multi-token `MUL_MAT_ID` fusion tracked in
upstream issue #28113 and held back here by `[TAG_MMID_FUSION_ONE_TOKEN_HIP]`. The two
asserts are adjacent but cover different operations:

```c
GGML_ASSERT( !ids || dst->ne[2] <= get_mmvq_mmid_max_batch(...));  // #28113, MUL_MAT_ID
GGML_ASSERT(  ids || dst->ne[1] == 1);                             // this abort, plain MUL_MAT
```

`#28113` is the multi-token expert path and produces garbage tokens rather than an
abort; the hold-back in this fork only gates `GGML_OP_MUL_MAT_ID` and therefore does
nothing for the case above. Both are failures of *which nodes get fused* rather than of
the kernels that compute them, so they are likely related, but they are two defects.

### Recommended `llama-server` profiles for Qwen3.8-27B

FastMTP improves token generation only when the compact draft predicts target tokens
with useful acceptance. It does not accelerate target-model prompt processing. Measure
`draft acceptance`, accepted mean length and final `eval time`, not draft speed alone.

#### Correctness baseline: no speculative decoder

```sh
--spec-type none \
--parallel 1 \
--batch-size 2048 \
--ubatch-size 512 \
--flash-attn on
```

Run the same prompt, seed and sampler more than once before enabling a draft. This is
the reference for both output stability and true target-model generation speed.

#### Embedded Qwen MTP head

```sh
--spec-type draft-mtp \
--spec-draft-n-max 2 \
--spec-draft-device ROCm0 \
--spec-draft-ngl all \
--spec-draft-type-k f16 \
--spec-draft-type-v f16 \
--parallel 1 \
--batch-size 2048 \
--ubatch-size 512 \
--flash-attn on
```

Do not pass `--spec-draft-model` for the embedded head. Test `--spec-draft-n-max 1`,
`2` and `3`; keep the value with the best end-to-end generation rate rather than the
largest value. Do not add `--spec-default`, because that also enables an independent
ngram drafter.

#### HauhauCS compact FastMTP sidecar

```sh
--spec-type draft-mtp \
--spec-draft-model /opt/models/Qwen3.8-27B/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf \
--spec-draft-device ROCm0 \
--spec-draft-ngl all \
--spec-draft-n-max 3 \
--spec-draft-p-min 0 \
--spec-draft-type-k f16 \
--spec-draft-type-v f16 \
--parallel 1 \
--ctx-size 204800 \
--batch-size 2048 \
--ubatch-size 512 \
--flash-attn on \
--load-mode none \
--temp 1.0 \
--top-k 20 \
--top-p 0.95 \
--min-p 0.0 \
--repeat-penalty 1.0 \
--presence-penalty 0.0
```

The sidecar is model-variant-specific: use the matching target/checkpoint family and
do not assume that another Qwen3.8 quant or fine-tune has an identical token map.
Depth 3 is the published starting point, not a universal optimum. The source model's
reported speedups were measured on NVIDIA Blackwell, not ROCm; this fork supplies the
backend-neutral graph plus HIP `FILL`/`SET_ROWS` execution, but gfx1151 throughput must
be established locally. Start with F16 draft K/V; try Q8 only after repeated requests
remain stable.

For maximum single-request speed use `--parallel 1`. Higher parallelism can improve
aggregate throughput but usually lowers per-request generation speed. The shown
`--batch-size 2048 --ubatch-size 512` is already sufficient for small MTP verification
batches; making either value much larger is primarily a memory/performance experiment,
not a correctness fix.

### Qwen3.8 speculative-decoding correctness notes

- Do not combine `--spec-default` with `--spec-type draft-mtp` while isolating output
  corruption. `--spec-default` adds an independent modified-ngram drafter; it is not a
  set of harmless MTP defaults. Use one speculative method at a time.
- Establish a deterministic no-draft baseline first, then test plain
  `--spec-type draft-mtp` with `--spec-draft-n-max 1` and `2`. Draft K/V should start
  at F16 for correctness isolation; quantize it only after the same prompt and seed
  remain stable.
- This fork contains a source-level MTP sequence/reset and checkpoint-state fix for the
  inter-request degradation reported in upstream issue #26425. During the first ROCm
  soak test, `--ctx-checkpoints 0` remains a useful conservative comparison rather
  than a permanent requirement.
- The fork keeps decode and small speculative-verification MMVQ/Flash-Attention launch
  shapes consistent on gfx1151. If output is correct locally but corrupt over RPC, set
  `GGML_CUDA_DISABLE_FUSION=1` in the RPC worker environment for an A/B run. RPC 5.1 can
  execute backend fusions remotely that older RPC builds never reached. The same switch
  addresses worker aborts - see "RPC worker aborts and backend fusion" above.
- `--batch-size` and `--ubatch-size` primarily affect throughput and memory pressure;
  they are not correctness switches for committed speculative tokens. Change them
  only after the no-draft and single-drafter comparisons are clean.
### Sources and provenance

- [@ggml-org](https://github.com/ggml-org/llama.cpp) provides the upstream base. This
  branch is rebased on the exact upstream commit stated above.
- [@Patt92](https://github.com/Patt92/llama.cpp) maintains this integration branch and
  supplies the HIP RPC top-k/argsort patch, including the current HIP-only rank-1 MoE
  LoRA MMVF dispatch fix.
- [@Geramy](https://github.com/Geramy/llama.cpp) is the provenance for the original
  hipCUB/CUB-on-HIP top-k and argsort approach. Its useful HIP content was rebased
  into the Patt92 RPC patch; no older branch is required at build time.
- [@Nathanw1014](https://github.com/Nathanw1014/llama.cpp) supplies the selected
  Strix Halo/RDNA3.5 HIP tuning, DeepSeek-V4, Flash-Attention, MMQ/MMVQ and
  backend-neutral fused-op work, including the speculative graph-capability guard. This
  branch also ports the indexed sparse DeepSeek-V4 attention design from Nathan's Vulkan
  research to a separate HIP tile implementation; Vulkan shader code is not copied.
- [@antirez](https://github.com/antirez/ds4) is the algorithmic source for the
  gfx1151-local Lightning Indexer top-k adaptation. Only the compatible HIP
  specialization is used; ds4 runtime, model and storage code is not included.
- [@stew675](https://github.com/stew675/llama.cpp) supplies the selected
  ROCm-compatible Qwen3.5/Qwen3.8 and hybrid-SSM fusions, MMVQ tuning and
  speculative decode/verification consistency fixes.
- [@HauhauCS](https://huggingface.co/HauhauCS) supplies the
  [compact-vocabulary FastMTP format and original Qwen3.8 runtime patch](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF).
  This branch retains that algorithm while adding explicit loader and graph validation.

### Deliberate exclusions

- Native-BF16 Flash-Attention tile changes from an external ROCm research series are
  not included. They overlap the tested gfx1151 tiled-FA path in this branch and need
  their own ROCm correctness and performance benchmark before replacement.
- The external adaptive-MTP depth experiment is not included. It is backend-neutral,
  but needs a separate correctness and throughput test with the RPC memory layout
  before it can be considered for this branch.
- No unreviewed change is used to reinterpret or skip target-model verification for
  MTP. DFlash2's additional verifier is used only by DFlash2 checkpoints.
- Vulkan shaders, Vulkan resource management and Vulkan-only environment flags are not
  copied into this HIP branch. Algorithmic ideas from them may be ported separately,
  with a dedicated HIP implementation and benchmark.

### Applying the standalone patch

Every published state of this branch also ships as a single standalone patch,
`rocm-halo-strix.patch`, attached to the matching release. The patch applies to a clean
checkout of the exact upstream commit named above and to no other commit: it is generated
with full context from that base, so `git apply` refuses it anywhere else instead of
producing a silently mismatched tree.

```sh
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout b81c99b479d4c24e5eeca10de99032ebd343ef8f
git apply --check /path/to/rocm-halo-strix.patch
git apply /path/to/rocm-halo-strix.patch
```

`git apply --check` performs a dry run and reports conflicts without touching the working
tree; run it first. Use `git apply --stat` to list the affected files. The patch contains
binary hunks, so `patch -p1` cannot be used. Nothing is committed by `git apply` — commit
the result yourself if you want it in history.

Build it exactly like upstream llama.cpp. A ROCm build for Strix Halo is:

```sh
HIPCXX=/opt/rocm/lib/llvm/bin/clang HIP_PATH=/opt/rocm ROCM_PATH=/opt/rocm \
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGGML_RPC=ON \
  -DGPU_TARGETS=gfx1151
cmake --build build --parallel "$(nproc)"
```

Adjust `GPU_TARGETS` for a different AMD GPU. The optimizations in this branch are gated on
RDNA3.5/gfx1151 at runtime, so other targets fall back to ordinary upstream behavior.

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
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
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
