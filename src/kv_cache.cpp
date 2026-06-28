#include "llmengine/kv_cache.hpp"
#include <stdexcept>

namespace llmengine {

KVCache::KVCache(int64_t max_seq_len, int64_t n_layers, int64_t n_heads, int64_t head_dim)
    : max_seq_len_(max_seq_len), n_layers_(n_layers), n_heads_(n_heads), head_dim_(head_dim) {
    const size_t element_size = ggml_type_size(GGML_TYPE_F32);
    const size_t total_elements = static_cast<size_t>(head_dim_ * n_heads_ * max_seq_len_ * n_layers_);
    const size_t cache_bytes = total_elements * element_size;
    const size_t meta_overhead = 2 * ggml_tensor_overhead() + 4096;

    buffer_.resize(cache_bytes * 2 + meta_overhead);

    ggml_init_params params = {
        .mem_size = buffer_.size(),
        .mem_buffer = buffer_.data(),
        .no_alloc = false,
    };

    ctx_ = ggml_init(params);
    if (!ctx_) {
        throw std::runtime_error("KVCache: Native structural ggml_init context creation failure.");
    }

    k_cache_ = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, head_dim_, n_heads_, max_seq_len_, n_layers_);
    v_cache_ = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, head_dim_, n_heads_, max_seq_len_, n_layers_);

    if (!k_cache_ || !v_cache_) {
        throw std::runtime_error("KVCache: Internal multi-dimensional tensor allocation failed.");
    }

    std::fill_n(reinterpret_cast<float*>(k_cache_->data), total_elements, 0.0f);
    std::fill_n(reinterpret_cast<float*>(v_cache_->data), total_elements, 0.0f);
}

KVCache::~KVCache() {
    if (ctx_) {
        ggml_free(ctx_);
    }
}

} // namespace llmengine