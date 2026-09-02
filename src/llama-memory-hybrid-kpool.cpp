#include "llama-memory-hybrid-kpool.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

//
// llama_memory_hybrid_kpool
//

llama_memory_hybrid_kpool::llama_memory_hybrid_kpool(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* indexer */
                 uint32_t   idx_row_size,
                 uint32_t   idx_kpool,
                     bool   idx_select_tail,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    hparams(model.hparams),
    hparams_idx(model.hparams),
    kpool(idx_kpool),
    select_tail(idx_select_tail),
    mem_attn(new llama_kv_cache(
        model,
        model.hparams,
        type_k,
        type_v,
        v_trans,
        offload,
        unified,
        kv_size,
        n_seq_max,
        n_pad,
        n_swa,
        swa_type,
        nullptr,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn,
        nullptr,
        nullptr
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr
    )),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        GGML_ASSERT(idx_row_size > 0 && idx_kpool > 0);

        // everything the indexer caches goes into K - llama_kv_cache allocates no V tensor
        // for an MLA model, which is what keeps this cache down to one row per token
        GGML_ASSERT(hparams_idx.is_mla() && "the indexer cache stores everything in K");

        // one key head of idx_row_size per layer, holding whatever the graph packs into it
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = idx_row_size;
        hparams_idx.n_embd_head_k_swa  = idx_row_size;

        // the indexer keys are read raw out of the cache, never through build_attn, so they
        // must not be Hadamard-rotated - keep them F16 rather than forwarding type_k
        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells, row = %u, kpool = %u\n",
                __func__, kv_size, idx_row_size, idx_kpool);

        return new llama_kv_cache(
            model,
            hparams_idx,
            GGML_TYPE_F16,
            GGML_TYPE_F16,
            v_trans,
            offload,
            unified,
            kv_size,
            n_seq_max,
            n_pad,
            n_swa,
            swa_type,
            nullptr,
            filter_idx,
            nullptr,
            nullptr);
    }()) {
    if (!mem_idx) {
        return;
    }

    // [TAG_KPOOL_KEY_CACHE] the indexer packs key | gate into one row, so a key is half of it
    GGML_ASSERT(idx_row_size % 2 == 0);

    idx_head   = idx_row_size/2;
    n_pool_max = kv_size/kpool;
    n_stream   = mem_attn->get_n_stream();

    GGML_ASSERT(n_pool_max > 0);

    pool_valid.assign((size_t) n_pool_max*n_stream, 0);

    const int32_t n_layer = (int32_t) hparams.n_layer();

    pool_k_l.resize(n_layer, nullptr);

    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    for (int32_t il = 0; il < n_layer; ++il) {
        if (!filter_idx(il)) {
            continue;
        }

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        if (offload) {
            buft = ggml_backend_dev_buffer_type(model.dev_layer(il));
        }

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for the k-pool key cache");
        }

        // + 1 row per stream: the scratch slot that absorbs padded refresh writes
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, idx_head, (int64_t)(n_pool_max + 1)*n_stream);
        ggml_format_name(t, "cache_kpool_l%d", il);
        pool_k_l[il] = t;
    }

    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for the k-pool key cache");
        }
        ggml_backend_buffer_clear(buf, 0);
        LLAMA_LOG_INFO("%s: %10s k-pool key cache size = %8.2f MiB (%u pools, head %u)\n", __func__,
                ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0,
                n_pool_max, idx_head);
        ctxs_bufs_pool.emplace_back(std::move(ctx), buf);
    }
}

void llama_memory_hybrid_kpool::invalidate_pool_keys() {
    std::fill(pool_valid.begin(), pool_valid.end(), 0);
}

uint32_t llama_memory_hybrid_kpool::n_pool_pending() const {
    if (!mem_idx || n_pool_max == 0) {
        return 0;
    }

    const uint32_t r = kpool;

    uint32_t n_max = 0;

    // a pool needs its key when it is complete but not yet cached. Completeness is decided the
    // same way set_input_kpool decides it, so the two always agree on the pool set.
    for (uint32_t s = 0; s < n_stream; ++s) {
        // a unified cache keeps every sequence in one stream; otherwise stream s is sequence s
        const llama_seq_id seq_of_stream = n_stream == 1 ? 0 : (llama_seq_id) s;

        const auto & cells = mem_attn->get_cells(seq_of_stream);

        std::vector<uint32_t> filled(n_pool_max, 0);

        for (uint32_t j = 0; j < cells.size(); ++j) {
            if (cells.is_empty(j)) {
                continue;
            }

            const uint32_t b = (uint32_t) (cells.pos_get(j)/(llama_pos) r);
            if (b < n_pool_max) {
                filled[b]++;
            }
        }

        uint32_t n = 0;
        for (uint32_t b = 0; b < n_pool_max; ++b) {
            if (filled[b] >= r && !pool_valid[s*n_pool_max + b]) {
                n++;
            }
        }

        n_max = std::max(n_max, n);
    }

    return n_max;
}

llama_memory_context_ptr llama_memory_hybrid_kpool::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                const bool unified = (mem_attn->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                const uint32_t n_rs_seq = mem_recr->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!mem_recr->prepare(ubatches)) {
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_kpool_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = mem_attn->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_kpool_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // note: mem_idx->prepare() is deliberately not called - the indexer cache reuses these
        //       very slots, see [TAG_IDX_CACHE_SHARE_SLOTS]
        return std::make_unique<llama_memory_hybrid_kpool_context>(
                this, std::move(heads_attn), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_kpool_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_kpool::init_full() {
    return std::make_unique<llama_memory_hybrid_kpool_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_kpool::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_kpool_context>(this, lctx, optimize);
}

bool llama_memory_hybrid_kpool::get_can_shift() const {
    // shifting is trivially supported for recurrent, and the indexer keys carry no positional
    // information at all (the arch using this container is NoPE), so only the cells move
    return mem_attn->get_can_shift() && (!mem_idx || mem_attn->get_size() == mem_idx->get_size());
}

void llama_memory_hybrid_kpool::clear(bool data) {
    invalidate_pool_keys();

    mem_attn->clear(data);
    mem_recr->clear(data);
    if (mem_idx) {
        mem_idx->clear(data);
    }
}

bool llama_memory_hybrid_kpool::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // try removing from the recurrent cache first since it may fail. If it does fail,
    // the cache will not have been mutated.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    // both attention caches always get the same call, so their cells stay in lockstep
    invalidate_pool_keys();

    const bool res = mem_attn->seq_rm(seq_id, p0, p1);
    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }
    return res;
}

void llama_memory_hybrid_kpool::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    invalidate_pool_keys();

    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_memory_hybrid_kpool::seq_keep(llama_seq_id seq_id) {
    invalidate_pool_keys();

    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_kpool::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    invalidate_pool_keys();

    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid_kpool::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    invalidate_pool_keys();

    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }
}

llama_pos llama_memory_hybrid_kpool::seq_pos_min(llama_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid_kpool::seq_pos_max(llama_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_kpool::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }
    return mb;
}

void llama_memory_hybrid_kpool::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_write(io, seq_id, flags);
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }
    mem_recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid_kpool::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    invalidate_pool_keys();

    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_read(io, seq_id, flags);
        if (mem_idx) {
            mem_idx->state_read(io, seq_id, flags);
        }
    }
    mem_recr->state_read(io, seq_id, flags);
}

//
// llama_memory_hybrid_kpool_context
//

llama_memory_hybrid_kpool_context::llama_memory_hybrid_kpool_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_kpool_context::llama_memory_hybrid_kpool_context(llama_memory_hybrid_kpool * mem) :
    mem(mem),
    kpool(mem->get_kpool()),
    select_tail(mem->get_select_tail()),
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    ctx_idx (mem->get_mem_idx() ? mem->get_mem_idx()->init_full() : nullptr),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_kpool_context::llama_memory_hybrid_kpool_context(
        llama_memory_hybrid_kpool * mem,
                  llama_context * lctx,
                           bool   optimize) :
    mem(mem),
    kpool(mem->get_kpool()),
    select_tail(mem->get_select_tail()),
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    ctx_idx (mem->get_mem_idx() ? mem->get_mem_idx()->init_update(lctx, optimize) : nullptr),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_kpool_context::llama_memory_hybrid_kpool_context(
          llama_memory_hybrid_kpool * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches) :
    mem(mem),
    kpool(mem->get_kpool()),
    select_tail(mem->get_select_tail()),
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_attn(new llama_kv_cache_context(mem->get_mem_attn(), sinfos_attn, this->ubatches)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    // [TAG_IDX_CACHE_SHARE_SLOTS] the indexer cache is a side buffer addressed by the
    // attention cache's cells, so it takes that slot layout instead of finding its own
    ctx_idx (mem->get_mem_idx()
            ? new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_attn), this->ubatches)
            : nullptr),
    // note: ctx_idx is built from the very same slot infos, so its status always mirrors ctx_attn's
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

bool llama_memory_hybrid_kpool_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();
    if (ctx_idx) {
        ctx_idx->next();
    }

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_kpool_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();
    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    return res;
}

llama_memory_status llama_memory_hybrid_kpool_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_kpool_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_context * llama_memory_hybrid_kpool_context::get_attn() const {
    return static_cast<const llama_kv_cache_context *>(ctx_attn.get());
}

const llama_memory_recurrent_context * llama_memory_hybrid_kpool_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}

const llama_kv_cache_context * llama_memory_hybrid_kpool_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

void llama_memory_hybrid_kpool_context::set_input_kpool(
        ggml_tensor * pool_cells,
        ggml_tensor * bias,
        ggml_tensor * tail_cells,
        ggml_tensor * rec_pool_idx,
        ggml_tensor * rec_pool_cells,
        const llama_ubatch * ubatch) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(bias->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(tail_cells->buffer));
    GGML_ASSERT(pool_cells->type == GGML_TYPE_I32);
    GGML_ASSERT(bias->type == GGML_TYPE_F32);
    GGML_ASSERT(tail_cells->type == GGML_TYPE_I32);

    // [TAG_KPOOL_KEY_CACHE] both refresh inputs exist together or not at all
    GGML_ASSERT((rec_pool_idx == nullptr) == (rec_pool_cells == nullptr));

    const int64_t n_rec = rec_pool_idx ? rec_pool_idx->ne[0] : 0;

    if (n_rec > 0) {
        GGML_ASSERT(ggml_backend_buffer_is_host(rec_pool_idx->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(rec_pool_cells->buffer));
        GGML_ASSERT(rec_pool_idx->type == GGML_TYPE_I32 && rec_pool_cells->type == GGML_TYPE_I32);
        GGML_ASSERT(rec_pool_cells->ne[0] == n_rec*(int64_t) kpool);
    }

    const int64_t r      = kpool;
    const int64_t n_pool = bias->ne[0];
    const int64_t n_ns   = pool_cells->ne[1];  // streams in this ubatch
    const int64_t n_kv   = (int64_t) get_attn()->get_n_kv();

    GGML_ASSERT(pool_cells->ne[0] == r*n_pool);
    GGML_ASSERT(tail_cells->ne[0] == r);

    GGML_ASSERT(r > 0 && n_pool > 0);
    GGML_ASSERT(ubatch->n_tokens % n_ns == 0);

    const int64_t n_tps = ubatch->n_tokens/n_ns;

    int32_t * dst_pool_cells = (int32_t *) pool_cells->data;
    float   * dst_bias       = (float   *) bias->data;
    int32_t * dst_tail       = (int32_t *) tail_cells->data;

    // [TAG_KPOOL_POOL_TOPK] the pool -> cells map is a graph input again: the selection now
    // names pools, and the winners are expanded through this table on the device.

    // one pass per stream: cell j is a different token in each
    std::vector<int32_t> pool_of(n_kv);
    std::vector<int32_t> filled(n_pool);

    // [TAG_KPOOL_POOL_TOPK] members that also carry this stream's sequence. A pool is usable
    // only when all kpool of them do, which is the per-pool form of the per-cell seq_has test
    // the old cell bias applied.
    std::vector<int32_t> filled_seq(n_pool);

    // position -> cell, over the narrow window the tails of this ubatch can reach. Built in the
    // same O(n_kv) pass, so collecting a token's tail costs kpool lookups instead of a scan.
    std::vector<int32_t> cell_of_pos;

    for (int64_t s = 0; s < n_ns; ++s) {
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];

        const auto & cells = mem->get_mem_attn()->get_cells(seq_of_stream);

        int32_t * cur_pool_cells = dst_pool_cells + s*r*n_pool;

        // pool b covers token positions [b*r, (b+1)*r). -1 = the cell has no usable pool.
        std::fill(pool_of.begin(),    pool_of.end(),    -1);
        std::fill(filled.begin(),     filled.end(),      0);
        std::fill(filled_seq.begin(), filled_seq.end(),  0);
        std::fill(cur_pool_cells, cur_pool_cells + r*n_pool, 0);

        // the tail of a token at position q spans [q - r + 1, q] at most
        llama_pos q_min = ubatch->pos[s*n_tps];
        llama_pos q_max = q_min;
        for (int64_t ii = 1; ii < n_tps; ++ii) {
            q_min = std::min(q_min, ubatch->pos[s*n_tps + ii]);
            q_max = std::max(q_max, ubatch->pos[s*n_tps + ii]);
        }

        const llama_pos pos_base = std::max<llama_pos>(0, q_min - (llama_pos) r + 1);
        const int64_t   pos_span = (int64_t) (q_max - pos_base) + 1;

        cell_of_pos.assign((size_t) pos_span, -1);

        for (int64_t j = 0; j < n_kv; ++j) {
            if (cells.is_empty(j)) {
                continue;
            }

            const llama_pos p      = cells.pos_get(j);
            const bool      in_seq = cells.seq_has(j, seq_of_stream);

            if (in_seq && p >= pos_base && p - pos_base < (llama_pos) pos_span) {
                cell_of_pos[p - pos_base] = (int32_t) j;
            }

            const int64_t b = p/r;

            if (b >= n_pool) {
                continue;
            }

            pool_of[j] = (int32_t) b;
            cur_pool_cells[b*r + (p%r)] = (int32_t) j;
            filled[b]++;
            filled_seq[b] += in_seq ? 1 : 0;
        }

        // an incomplete pool cannot be pooled: its cells are reachable only through the tail
        // bias below, and cell 0 in pool_cells only keeps the gather in range
        for (int64_t b = 0; b < n_pool; ++b) {
            if (filled[b] < (int32_t) r) {
                std::fill(cur_pool_cells + b*r, cur_pool_cells + (b + 1)*r, 0);
            }
        }

        // [TAG_KPOOL_KEY_CACHE] name the complete pools whose key is not cached yet - the graph
        // refreshes exactly these, and every other pool is read straight out of the cache.
        // Marking them valid here is safe: the graph that consumes these inputs is the one that
        // writes the keys, and any cache edit in between clears the whole map anyway.
        if (n_rec > 0) {
            int32_t * cur_rec_idx   = (int32_t *) rec_pool_idx->data   + s*n_rec;
            int32_t * cur_rec_cells = (int32_t *) rec_pool_cells->data + s*n_rec*r;

            int64_t n = 0;
            for (int64_t b = 0; b < n_pool && n < n_rec; ++b) {
                if (filled[b] < (int32_t) r || mem->is_pool_valid((uint32_t) s, (uint32_t) b)) {
                    continue;
                }

                cur_rec_idx[n] = (int32_t) b;
                std::copy(cur_pool_cells + b*r, cur_pool_cells + (b + 1)*r, cur_rec_cells + n*r);

                mem->mark_pool_valid((uint32_t) s, (uint32_t) b);
                n++;
            }

            // park the unused refresh slots on the scratch row, which is never read back
            for (int64_t i = n; i < n_rec; ++i) {
                cur_rec_idx[i] = (int32_t) mem->get_n_pool_max();
                std::fill(cur_rec_cells + i*r, cur_rec_cells + (i + 1)*r, 0);
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];
            const llama_pos    q      = ubatch->pos[i];

            // the rest is an incomplete pool that is always attended to, which is what lands
            // the selection on pool boundaries like the reference. Without select_tail the
            // trailing cells simply stay invisible.
            const llama_pos tail_start = select_tail ? (q + 1)/(llama_pos) r*(llama_pos) r : q + 1;

            GGML_ASSERT(seq_id == seq_of_stream);

            // [TAG_KPOOL_POOL_TOPK] pool b covers positions [b*r, (b+1)*r). It is usable by this
            // token exactly when it is complete, entirely this token's sequence, and lies wholly
            // before the tail - then every member satisfies pos <= q and the per-cell test the
            // old bias ran n_kv times collapses to one test per pool.
            float * cur_bias = dst_bias + i*n_pool;

            for (int64_t b = 0; b < n_pool; ++b) {
                const bool ok = filled[b]     >= (int32_t) r &&
                                filled_seq[b] >= (int32_t) r &&
                                (llama_pos) ((b + 1)*r) <= tail_start;

                cur_bias[b] = ok ? 0.0f : -INFINITY;
            }

            // the tail belongs to no complete pool, so it cannot be ranked - it is handed to the
            // graph separately and concatenated onto the winners. Padding repeats cell 0, which
            // the KQ mask masks again if it is not actually visible.
            int32_t * cur_tail = dst_tail + i*r;
            int64_t   n_tail   = 0;

            for (llama_pos p = tail_start; p <= q && n_tail < r; ++p) {
                const int64_t idx = (int64_t) (p - pos_base);

                if (idx >= 0 && idx < (int64_t) cell_of_pos.size() && cell_of_pos[idx] >= 0) {
                    cur_tail[n_tail++] = cell_of_pos[idx];
                }
            }

            for (int64_t t = n_tail; t < r; ++t) {
                cur_tail[t] = 0;
            }
        }
    }
}
