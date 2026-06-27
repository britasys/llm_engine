#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace llmengine {

struct RopePair {
    float cos;
    float sin;
};

class RopeCache {
public:
    RopeCache() = default;

    RopeCache(int64_t max_seq_len, int64_t head_dim, float theta_base);

    void initialize(int64_t max_seq_len, int64_t head_dim, float theta_base);

    [[nodiscard]]
    int64_t max_seq_len() const noexcept {
        return max_seq_len_;
    }

    [[nodiscard]]
    int64_t head_dim() const noexcept {
        return head_dim_;
    }

    [[nodiscard]]
    const RopePair* row(int64_t position) const;

private:
    int64_t max_seq_len_ = 0;
    int64_t head_dim_ = 0;
    int64_t half_dim_ = 0;

    std::vector<RopePair> table_;
};

void apply_rope(std::span<float> v, int64_t position, const RopeCache& cache);

} // namespace llmengine