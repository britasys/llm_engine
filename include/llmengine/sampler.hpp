#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "tensor.hpp"

namespace llmengine {

class Sampler {
public:
    explicit Sampler(uint32_t seed = std::random_device{}());

    [[nodiscard]] int32_t argmax(const Tensor& logits) const;

    [[nodiscard]] int32_t sample(const Tensor& logits, float temperature = 1.0f);

    [[nodiscard]] int32_t sample_top_k(const Tensor& logits, int32_t k, float temperature = 1.0f);

    [[nodiscard]] int32_t sample_top_p(const Tensor& logits, float p, float temperature = 1.0f);

private:
    std::mt19937 rng_;

    [[nodiscard]] static std::vector<float> softmax(const float* logits, std::size_t n,
                                                    float temperature);
};

} // namespace llmengine