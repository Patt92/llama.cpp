#include "llama-memory-hybrid-kpool.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cmath>

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
    }()) {}

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
    const bool res = mem_attn->seq_rm(seq_id, p0, p1);
    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }
    return res;
}

void llama_memory_hybrid_kpool::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_memory_hybrid_kpool::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_kpool::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid_kpool::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
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
        ggml_tensor * cell_pool,
        ggml_tensor * pool_cells,
        ggml_tensor * bias,
        const llama_ubatch * ubatch) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(cell_pool->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(bias->buffer));
    GGML_ASSERT(cell_pool->type == GGML_TYPE_I32 && pool_cells->type == GGML_TYPE_I32);
    GGML_ASSERT(bias->type == GGML_TYPE_F32);

    const int64_t r      = kpool;
    const int64_t n_kv   = cell_pool->ne[0];
    const int64_t n_ns   = cell_pool->ne[1];   // streams in this ubatch
    const int64_t n_pool = pool_cells->ne[0]/r;

    GGML_ASSERT(r > 0 && n_pool > 0);
    GGML_ASSERT(ubatch->n_tokens % n_ns == 0);

    const int64_t n_tps = ubatch->n_tokens/n_ns;

    int32_t * dst_cell_pool  = (int32_t *) cell_pool->data;
    int32_t * dst_pool_cells = (int32_t *) pool_cells->data;
    float   * dst_bias       = (float   *) bias->data;

    // one pass per stream: cell j is a different token in each
    std::vector<int32_t> pool_of(n_kv);
    std::vector<int32_t> filled(n_pool);

    for (int64_t s = 0; s < n_ns; ++s) {
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];

        const auto & cells = mem->get_mem_attn()->get_cells(seq_of_stream);

        int32_t * cur_cell_pool  = dst_cell_pool  + s*n_kv;
        int32_t * cur_pool_cells = dst_pool_cells + s*(r*n_pool);

        // pool b covers token positions [b*r, (b+1)*r). -1 = the cell has no usable pool.
        std::fill(pool_of.begin(), pool_of.end(), -1);
        std::fill(filled.begin(),  filled.end(),   0);
        std::fill(cur_pool_cells, cur_pool_cells + r*n_pool, 0);

        for (int64_t j = 0; j < n_kv; ++j) {
            if (cells.is_empty(j)) {
                continue;
            }

            const llama_pos p = cells.pos_get(j);
            const int64_t   b = p/r;

            if (b >= n_pool) {
                continue;
            }

            pool_of[j] = (int32_t) b;
            cur_pool_cells[b*r + (p%r)] = (int32_t) j;
            filled[b]++;
        }

        // an incomplete pool cannot be pooled: its cells are reachable only through the tail
        // bias below, and cell 0 in pool_cells only keeps the gather in range
        for (int64_t b = 0; b < n_pool; ++b) {
            if (filled[b] < (int32_t) r) {
                std::fill(cur_pool_cells + b*r, cur_pool_cells + (b + 1)*r, 0);
            }
        }

        for (int64_t j = 0; j < n_kv; ++j) {
            if (pool_of[j] >= 0 && filled[pool_of[j]] < (int32_t) r) {
                pool_of[j] = -1;
            }
            cur_cell_pool[j] = pool_of[j] < 0 ? 0 : pool_of[j];
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];
            const llama_pos    q      = ubatch->pos[i];

            // the rest is an incomplete pool that is always attended to, which is what lands
            // the selection on pool boundaries like the reference. Without select_tail the
            // trailing cells simply stay invisible.
            const llama_pos tail_start = select_tail ? (q + 1)/(llama_pos) r*(llama_pos) r : q + 1;

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id) && cells.pos_get(j) <= q) {
                    // finite, so it can never meet a -inf and produce a nan
                    v = cells.pos_get(j) >= tail_start ? 1e9f : (pool_of[j] < 0 ? -INFINITY : 0.0f);
                }

                cur_bias[j] = v;
            }
        }
    }
}
