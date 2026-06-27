#include "llmengine/rope_cache.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace llmengine {

RopeCache::RopeCache(int64_t max_seq_len, int64_t head_dim, float theta_base) {
    initialize(max_seq_len, head_dim, theta_base);
}

void RopeCache::initialize(int64_t max_seq_len, int64_t head_dim, float theta_base) {
    if (max_seq_len <= 0)
        throw std::invalid_argument("RoPE sequence length must be positive");

    if (head_dim <= 0)
        throw std::invalid_argument("RoPE head dimension must be positive");

    if ((head_dim & 1) != 0)
        throw std::invalid_argument("RoPE head dimension must be even");

    if (!std::isfinite(theta_base) || theta_base <= 0.0f) {
        throw std::invalid_argument("RoPE theta must be positive and finite");
    }

    max_seq_len_ = max_seq_len;
    head_dim_ = head_dim;
    half_dim_ = head_dim / 2;

    const auto seq = static_cast<std::size_t>(max_seq_len_);

    const auto half = static_cast<std::size_t>(half_dim_);

    if (half != 0 && seq > std::numeric_limits<std::size_t>::max() / half) {
        throw std::length_error("RoPE cache size overflow");
    }

    table_.resize(seq * half);

    for (int64_t pos = 0; pos < max_seq_len_; ++pos) {
        RopePair* row_ptr = table_.data() + static_cast<std::size_t>(pos) * static_cast<std::size_t>(half_dim_);

        for (int64_t i = 0; i < half_dim_; ++i) {
            const float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim_);

            const float inv_freq = std::pow(theta_base, -exponent);

            const float angle = static_cast<float>(pos) * inv_freq;

            row_ptr[i].cos = std::cos(angle);

            row_ptr[i].sin = std::sin(angle);
        }
    }
}

const RopePair* RopeCache::row(int64_t position) const {
    if (position < 0 || position >= max_seq_len_) {
        throw std::out_of_range("RoPE position out of range");
    }

    return table_.data() + static_cast<std::size_t>(position) * static_cast<std::size_t>(half_dim_);
}

void apply_rope(std::span<float> v, int64_t position, const RopeCache& cache) {
    if (cache.head_dim() <= 0)
        throw std::logic_error("RoPE cache is not initialized");

    if (v.size() != static_cast<std::size_t>(cache.head_dim())) {
        throw std::invalid_argument("RoPE vector size mismatch");
    }

    const RopePair* row = cache.row(position);

    const int64_t half = cache.head_dim() / 2;

    for (int64_t i = 0; i < half; ++i) {
        const int64_t index = 2 * i;

        const float x = v[index];

        const float y = v[index + 1];

        const float c = row[i].cos;

        const float s = row[i].sin;

        v[index] = x * c - y * s;

        v[index + 1] = x * s + y * c;
    }
}

} // namespace llmengine