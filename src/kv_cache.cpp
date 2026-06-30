#include "llmengine/kv_cache.hpp"

#include <cassert>
#include <limits>
#include <stdexcept>

namespace llmengine {

constexpr size_t ContextOverhead = 4096;

template<typename T, typename U> size_t checked_mul(T a, U b) {
    const size_t lhs = static_cast<size_t>(a);
    const size_t rhs = static_cast<size_t>(b);

    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::overflow_error("Integer multiplication overflow.");
    }

    return lhs * rhs;
}

KVCache::KVCache(int64_t max_seq_len, int64_t n_layers, int64_t n_heads, int64_t head_dim, ggml_type elem_type)
    : max_seq_len_(max_seq_len), n_layers_(n_layers), n_heads_(n_heads), head_dim_(head_dim), elem_type_(elem_type) {
    if (max_seq_len <= 0)
        throw std::invalid_argument("max_seq_len is equal or less than zero");
    if (n_layers <= 0)
        throw std::invalid_argument("n_layers is equal or less than zero");
    if (n_heads <= 0)
        throw std::invalid_argument("n_heads is equal or less than zero");
    if (head_dim <= 0)
        throw std::invalid_argument("head_dim is equal or less than zero");

    size_t total_elements = head_dim_;
    total_elements = checked_mul(total_elements, n_heads_);
    total_elements = checked_mul(total_elements, max_seq_len_);
    total_elements = checked_mul(total_elements, n_layers_);

    const size_t element_size = ggml_type_size(elem_type_);
    const size_t cache_bytes = total_elements * element_size;
    const size_t meta_overhead = 2 * ggml_tensor_overhead() + ContextOverhead;

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

    k_cache_ = ggml_new_tensor_4d(ctx_, elem_type_, head_dim_, n_heads_, max_seq_len_, n_layers_);
    v_cache_ = ggml_new_tensor_4d(ctx_, elem_type_, head_dim_, n_heads_, max_seq_len_, n_layers_);

    if (!k_cache_ || !v_cache_) {
        throw std::runtime_error("KVCache: Internal multi-dimensional tensor allocation failed.");
    }

    std::fill_n(reinterpret_cast<float*>(k_cache_->data), total_elements, 0.0f);
    std::fill_n(reinterpret_cast<float*>(v_cache_->data), total_elements, 0.0f);
}

void KVCache::increment_sequence() {
    if (seq_len_ == max_seq_len_) {
        throw std::out_of_range("Cannot append token: KV cache capacity exceeded.");
    }

    ++seq_len_;
}

KVCache::~KVCache() {
    if (ctx_) {
        ggml_free(ctx_);
    }
}

} // namespace llmengine