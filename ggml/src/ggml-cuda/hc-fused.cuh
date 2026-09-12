#include "common.cuh"
#include "ggml.h"

// [TAG_HC_FUSED_OPS] qwen4exp hyper-connection tails
void ggml_cuda_op_hc_gate_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
