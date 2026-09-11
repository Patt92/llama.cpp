// [TAG_RPC_CACHE_ATOMIC] exercise: a compute-buffer tensor above the hash threshold must not be
// cached; a weights tensor is; a truncated cache file must not be served.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <filesystem>
#include <fstream>
int main(int argc, char ** argv) {
    const char * endpoint = argv[1];
    const char * cache = argv[2];
    ggml_backend_t be = ggml_backend_rpc_init(endpoint, 0);
    if (!be) { printf("no backend\n"); return 1; }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(be);
    const int64_t n = 4*1024*1024; // 16 MB f32 > 10 MB threshold
    ggml_init_params ip = { 2*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t_act = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * t_w   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    std::vector<float> a(n), b(n);
    for (int64_t i = 0; i < n; ++i) { a[i] = (float) (i % 977) * 0.5f; b[i] = -a[i]; }
    auto count = [&]() { size_t c = 0; for (auto & e : std::filesystem::directory_iterator(cache)) { (void) e; c++; } return c; };
    // 1. compute usage: nothing may be cached
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    size_t c0 = count();
    std::vector<float> r(n);
    ggml_backend_tensor_set(t_act, a.data(), 0, n*4);
    ggml_backend_tensor_get(t_act, r.data(), 0, n*4);
    size_t c1 = count();
    bool ok1 = memcmp(r.data(), a.data(), n*4) == 0 && c1 == c0;
    printf("compute tensor: data ok=%d cache files before=%zu after=%zu -> %s\n", memcmp(r.data(), a.data(), n*4) == 0, c0, c1, ok1 ? "OK" : "FAIL");
    // 2. weights usage: cached once, second send hits the cache and data stays right
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(t_w, b.data(), 0, n*4);
    ggml_backend_tensor_get(t_w, r.data(), 0, n*4); // SET_TENSOR has no reply: the round trip orders the count after it
    size_t c2 = count();
    ggml_backend_tensor_set(t_w, a.data(), 0, n*4); // different data, new entry
    ggml_backend_tensor_set(t_w, b.data(), 0, n*4); // back to b: served from cache
    ggml_backend_tensor_get(t_w, r.data(), 0, n*4);
    bool ok2 = memcmp(r.data(), b.data(), n*4) == 0 && c2 == c0 + 1;
    printf("weights tensor: data ok=%d cache files=%zu (expect %zu) -> %s\n", memcmp(r.data(), b.data(), n*4) == 0, c2, c0 + 1, ok2 ? "OK" : "FAIL");
    // 3. truncate every cache file to 1 MB: a hash hit must be refused and the data re-sent
    for (auto & e : std::filesystem::directory_iterator(cache)) { std::filesystem::resize_file(e.path(), 1024*1024); }
    ggml_backend_tensor_set(t_w, a.data(), 0, n*4);
    ggml_backend_tensor_set(t_w, b.data(), 0, n*4);
    ggml_backend_tensor_get(t_w, r.data(), 0, n*4);
    bool ok3 = memcmp(r.data(), b.data(), n*4) == 0;
    printf("truncated cache: data ok=%d -> %s\n", ok3, ok3 ? "OK" : "FAIL");
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    return (ok1 && ok2 && ok3) ? 0 : 1;
}
