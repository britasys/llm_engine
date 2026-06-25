#pragma once

#include <cstdint>
#include <vector>

#include "tensor.hpp"

namespace llmengine {

class KVCache {
public:
    KVCache() = default;

    KVCache(int64_t max_seq_len, int64_t n_layers, int64_t n_heads, int64_t head_dim);

    void clear();

    void append(int64_t layer, const Tensor& key, const Tensor& value);

    [[nodiscard]] Tensor keys(int64_t layer) const;

    [[nodiscard]] Tensor values(int64_t layer) const;

    [[nodiscard]] int64_t size() const noexcept;

    [[nodiscard]] int64_t capacity() const noexcept;

    [[nodiscard]] bool empty() const noexcept;

private:
    int64_t max_seq_len_ = 0;
    int64_t n_layers_ = 0;
    int64_t n_heads_ = 0;
    int64_t head_dim_ = 0;

    int64_t seq_len_ = 0;

    std::vector<Tensor> keys_;
    std::vector<Tensor> values_;
};

} // namespace llmengine