#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-memory.h"
#include "llama-memory-recurrent.h"

#include <memory>
#include <vector>

//
// llama_memory_hybrid_kpool
//

// utilizes three memories for architectures that are hybrid recurrent/attention AND run
//   sparse attention driven by a lightning indexer (GLM-5.3-Flash):
//     - llama_memory_recurrent : the linear-attention (KDA) layer states
//     - llama_kv_cache         : the (MLA) K cache of the full-attention layers
//     - llama_kv_cache         : the indexer cache of those same layers, one row per token
//
// [TAG_IDX_CACHE_SHARE_SLOTS] the indexer cache never runs find_slot: it is handed the
//   attention cache's slot infos, so cell j always means the same token in both. Letting it
//   pick its own slots would silently drift after a context rewrite (the two caches have
//   independent ring-buffer heads), and the top-k indices would then address the wrong cells.
//
// the context also exposes the host-side k-pool metadata, built from the public
//   llama_kv_cache::get_cells(), the same way llama_kv_cache_msa exposes its position maps.
//   nothing arch-specific is added to llama_kv_cache itself.

class llama_memory_hybrid_kpool : public llama_memory_i {
public:
    llama_memory_hybrid_kpool(
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
                 uint32_t   idx_row_size,   // floats cached per token (the graph packs its own layout)
                 uint32_t   idx_kpool,      // tokens per k-pool
                     bool   idx_select_tail,// the incomplete trailing pool is always visible
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn = nullptr,
    const layer_filter_cb & filter_recr = nullptr,
    const layer_filter_cb & filter_idx  = nullptr);

    ~llama_memory_hybrid_kpool() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_kpool specific API
    //

    llama_kv_cache         * get_mem_attn() const { return mem_attn.get(); }
    llama_memory_recurrent * get_mem_recr() const { return mem_recr.get(); }

    // null when the model has no indexer - the graph then falls back to dense attention
    llama_kv_cache         * get_mem_idx () const { return mem_idx.get();  }

    uint32_t get_kpool()       const { return kpool;       }
    bool     get_select_tail() const { return select_tail; }

private:
    const llama_hparams & hparams;

    // llama_kv_cache keeps only a reference to its hparams, so the tuned copy lives here
    llama_hparams hparams_idx;

    const uint32_t kpool       = 1;
    const bool     select_tail = false;

    const std::unique_ptr<llama_kv_cache>         mem_attn;
    const std::unique_ptr<llama_memory_recurrent> mem_recr;
    const std::unique_ptr<llama_kv_cache>         mem_idx;
};

class llama_memory_hybrid_kpool_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // init failure
    explicit llama_memory_hybrid_kpool_context(llama_memory_status status);

    // init full
    explicit llama_memory_hybrid_kpool_context(llama_memory_hybrid_kpool * mem);

    // init update
    explicit llama_memory_hybrid_kpool_context(
        llama_memory_hybrid_kpool * mem,
                  llama_context * lctx,
                           bool   optimize);

    // init success - sinfos_attn drives BOTH the attention and the indexer cache
    llama_memory_hybrid_kpool_context(
          llama_memory_hybrid_kpool * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_kpool_context() = default;

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_hybrid_kpool_context specific API
    //

    const llama_kv_cache_context         * get_attn() const;
    const llama_memory_recurrent_context * get_recr() const;

    // null when the model has no indexer
    const llama_kv_cache_context         * get_idx () const;

    uint32_t get_kpool() const { return kpool; }

    // host-side k-pool metadata for one ubatch, built from the attention cache cells:
    //   cell_pool  I32 [n_kv, ns]            the pool a cell belongs to (0 when it has none)
    //   pool_cells I32 [kpool*n_pools, ns]   the cells making up each complete pool
    //   bias       F32 [n_kv, n_tps, ns]     -INF invisible, +1e9 always-visible tail, 0 otherwise
    // a pool is a run of `kpool` consecutive token *positions*, so nothing here assumes the
    // cache is laid out contiguously. Cells of an incomplete pool cannot be pooled and are
    // reachable only through the tail bias. Pooling by position, not by cache order, differs
    // from the reference after a seq_rm or a context shift.
    //
    // limitation: pools are built once per stream, so a unified cache holding several
    // sequences pools their cells together and degrades the selection. The attention mask
    // itself stays per token, so no sequence ever sees another one's tokens.
    void set_input_kpool(
            ggml_tensor * cell_pool,
            ggml_tensor * pool_cells,
            ggml_tensor * bias,
            const llama_ubatch * ubatch) const;

private:
    llama_memory_hybrid_kpool * mem = nullptr;

    const uint32_t kpool       = 1;
    const bool     select_tail = false;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_attn;
    const llama_memory_context_ptr ctx_recr;
    const llama_memory_context_ptr ctx_idx; // null when the model has no indexer

    const llama_memory_status status;
};
