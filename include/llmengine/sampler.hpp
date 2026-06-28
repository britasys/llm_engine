// sampler.hpp
#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace llmengine {

class Sampler {
public:
    explicit Sampler(uint32_t seed = std::random_device{}());

    [[nodiscard]] int32_t argmax(const float* logits, std::size_t n) const;
    [[nodiscard]] int32_t sample(const float* logits, std::size_t n, float temperature = 1.0f);
    [[nodiscard]] int32_t sample_top_k(const float* logits, std::size_t n, int32_t k,
                                       float temperature = 1.0f);
    [[nodiscard]] int32_t sample_top_p(const float* logits, std::size_t n, float p,
                                       float temperature = 1.0f);

private:
    mutable std::mt19937 rng_;

    [[nodiscard]] static std::vector<float> softmax(const float* logits, std::size_t n,
                                                    float temperature);
};

} // namespace llmengine