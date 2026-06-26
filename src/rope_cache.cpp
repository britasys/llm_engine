#include "llmengine/rope_cache.hpp"

#include <cmath>
#include <stdexcept>

namespace llmengine {

RopeCache::RopeCache(int64_t max_seq_len, int64_t head_dim, float theta_base) {
    initialize(max_seq_len, head_dim, theta_base);
}

void RopeCache::initialize(int64_t max_seq_len, int64_t head_dim, float theta_base) {
    if (head_dim % 2 != 0)
        throw std::runtime_error("RoPE head dimension must be even.");

    max_seq_len_ = max_seq_len;
    head_dim_ = head_dim;
    half_dim_ = head_dim / 2;

    table_.resize(static_cast<size_t>(max_seq_len_) * static_cast<size_t>(half_dim_));

    for (int64_t pos = 0; pos < max_seq_len_; ++pos) {

        RopePair* row = table_.data() + pos * half_dim_;

        for (int64_t i = 0; i < half_dim_; ++i) {

            float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim_);

            float inv_freq = std::pow(theta_base, -exponent);

            float angle = static_cast<float>(pos) * inv_freq;

            row[i].cos = std::cos(angle);
            row[i].sin = std::sin(angle);
        }
    }
}

void apply_rope(std::span<float> v, int64_t position, const RopeCache& cache) {
    const RopePair* row = cache.row(position);

    const int64_t half = cache.head_dim() / 2;

    for (int64_t i = 0; i < half; ++i) {

        float x0 = v[2 * i];
        float x1 = v[2 * i + 1];

        float c = row[i].cos;
        float s = row[i].sin;

        v[2 * i] = x0 * c - x1 * s;
        v[2 * i + 1] = x0 * s + x1 * c;
    }
}

} // namespace llmengine