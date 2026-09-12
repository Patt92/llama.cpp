// [TAG_HC_FUSED_OPS] fused hyper-connection tails for Qwen3.8-Flash-Next (qwen4exp).
//
// A decode step of this model is launch-bound: every layer carries two hyper-connection mixes
// and two scatters, each a chain of small elementwise kernels over [n_embd, hc, n_tokens]. The
// two ops below replace those chains with one kernel each, same arithmetic in the same order
// per element (sigmoid, multiply, sum over the streams, scale).

#include "hc-fused.cuh"

// result[i, t] = (1/hc) * sum_h x[i, h, t] * sigmoid(gate[i, h, t])
static __global__ void hc_gate_mix_f32(
        const float * x,
        const float * gate,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sg0, int64_t sg1, int64_t sg2,
        int64_t sd0, int64_t sd1) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x   [i0*sx0 + ih*sx1 + it*sx2];
        const float gv = gate[i0*sg0 + ih*sg1 + it*sg2];
        sum += xv * (1.0f / (1.0f + expf(-gv)));
    }

    dst[i0*sd0 + it*sd1] = sum * (1.0f / (float) hc);
}

// result[i, h, t] = residual[i, h, t] + x[i, t] * 2 * sigmoid(inject[h, t] * scale)
static __global__ void hc_combine_f32(
        const float * residual,
        const float * x,
        const float * inject,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        float scale,
        int64_t sr0, int64_t sr1, int64_t sr2,
        int64_t sx0, int64_t sx1,
        int64_t si0, int64_t si1,
        int64_t sd0, int64_t sd1, int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t ih = (ir / n_embd) % hc;
    const int64_t it = ir / (n_embd * hc);

    const float rv = residual[i0*sr0 + ih*sr1 + it*sr2];
    const float xv = x       [i0*sx0 + it*sx1];
    const float iv = inject  [ih*si0 + it*si1];

    const float w = 2.0f / (1.0f + expf(-iv * scale));

    dst[i0*sd0 + ih*sd1 + it*sd2] = rv + xv * w;
}

void ggml_cuda_op_hc_gate_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x    = dst->src[0];
    const ggml_tensor * gate = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(gate->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,    nb);
    GGML_TENSOR_LOCALS(size_t, nbg, gate, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,  nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(hc_gate_mix_f32, launch_params,
            (const float *) x->data, (const float *) gate->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbg0 / sizeof(float), nbg1 / sizeof(float), nbg2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float));
}

void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * residual = dst->src[0];
    const ggml_tensor * x        = dst->src[1];
    const ggml_tensor * inject   = dst->src[2];

    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(inject->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float scale = 1.0f;
    memcpy(&scale, dst->op_params, sizeof(scale));

    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbi, inject,   nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const int64_t n_embd   = residual->ne[0];
    const int64_t hc       = residual->ne[1];
    const int64_t n_tokens = residual->ne[2];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(hc_combine_f32, launch_params,
            (const float *) residual->data, (const float *) x->data, (const float *) inject->data, (float *) dst->data,
            n_embd, hc, n_tokens, scale,
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbi0 / sizeof(float), nbi1 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
}
