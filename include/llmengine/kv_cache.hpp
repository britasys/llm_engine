// kv_cache.hpp
#pragma once

#include <cstdint>
#include <ggml.h>
#include <vector>

namespace llmengine {

class KVCache {
public:
    KVCache(int64_t max_seq_len, int64_t n_layers, int64_t n_heads, int64_t head_dim, ggml_type ggml_type = GGML_TYPE_F32);
    ~KVCache();

    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;

    void clear() noexcept { seq_len_ = 0; }

    [[nodiscard]] ggml_tensor* k_cache() const noexcept { return k_cache_; }
    [[nodiscard]] ggml_tensor* v_cache() const noexcept { return v_cache_; }

    [[nodiscard]] int64_t size() const noexcept { return seq_len_; }
    [[nodiscard]] int64_t capacity() const noexcept { return max_seq_len_; }
    [[nodiscard]] bool empty() const noexcept { return seq_len_ == 0; }
    [[nodiscard]] ggml_type get_elem_type() const noexcept { return elem_type_; }

    void increment_sequence();

private:
    int64_t max_seq_len_;
    int64_t n_layers_;
    int64_t n_heads_;
    int64_t head_dim_;
    int64_t seq_len_ = 0;
    ggml_type elem_type_;

    ggml_context* ctx_ = nullptr;
    std::vector<uint8_t> buffer_;

    ggml_tensor* k_cache_ = nullptr;
    ggml_tensor* v_cache_ = nullptr;
};

} // namespace llmengine