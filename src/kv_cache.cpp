#include "llmengine/kv_cache.hpp"

#include <stdexcept>

namespace llmengine {

KVCache::KVCache(int64_t max_seq_len, int64_t n_layers, int64_t n_heads, int64_t head_dim)
    : max_seq_len_(max_seq_len), n_layers_(n_layers), n_heads_(n_heads), head_dim_(head_dim) {

    keys_.reserve(n_layers_);
    values_.reserve(n_layers_);

    for (int64_t i = 0; i < n_layers_; ++i) {
        keys_.push_back(Tensor::zeros({max_seq_len_, n_heads_, head_dim_}));

        values_.push_back(Tensor::zeros({max_seq_len_, n_heads_, head_dim_}));
    }
}

void KVCache::clear() {
    seq_len_ = 0;
}

void KVCache::append(int64_t layer, const Tensor& key, const Tensor& value) {
    if (layer < 0 || layer >= n_layers_) {
        throw std::runtime_error("invalid layer");
    }

    if (seq_len_ >= max_seq_len_) {
        throw std::runtime_error("kv cache full");
    }

    if (key.shape() != value.shape()) {
        throw std::runtime_error("shape mismatch");
    }

    if (key.ndim() != 2) {
        throw std::runtime_error("expected [heads, head_dim]");
    }

    auto src_k = key.as_f32();
    auto src_v = value.as_f32();

    auto dst_k = keys_[layer].as_f32();
    auto dst_v = values_[layer].as_f32();

    const int64_t offset = seq_len_ * n_heads_ * head_dim_;

    for (int64_t i = 0; i < key.numel(); ++i) {
        dst_k[offset + i] = src_k[i];
        dst_v[offset + i] = src_v[i];
    }

    if (layer == n_layers_ - 1) {
        ++seq_len_;
    }
}

Tensor KVCache::keys(int64_t layer) const {
    if (layer < 0 || layer >= n_layers_) {
        throw std::runtime_error("invalid layer");
    }

    return keys_[layer].clone();
}

Tensor KVCache::values(int64_t layer) const {
    if (layer < 0 || layer >= n_layers_) {
        throw std::runtime_error("invalid layer");
    }

    return values_[layer].clone();
}

int64_t KVCache::size() const noexcept {
    return seq_len_;
}

int64_t KVCache::capacity() const noexcept {
    return max_seq_len_;
}

bool KVCache::empty() const noexcept {
    return seq_len_ == 0;
}

} // namespace llmengine