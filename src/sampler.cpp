#include "llmengine/sampler.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace llmengine {

Sampler::Sampler(uint32_t seed) : rng_(seed) {}

std::vector<float> Sampler::softmax(const float* logits, std::size_t n, float temperature) {
    if (n == 0)
        return {};

    std::vector<float> probs(n);
    float max_logit = logits[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (logits[i] > max_logit)
            max_logit = logits[i];
    }

    float sum = 0.0f;
    float inv_temp = 1.0f / (temperature > 0.0f ? temperature : 1.0f);

    for (std::size_t i = 0; i < n; ++i) {
        probs[i] = std::exp((logits[i] - max_logit) * inv_temp);
        sum += probs[i];
    }

    for (std::size_t i = 0; i < n; ++i) {
        probs[i] /= sum;
    }

    return probs;
}

int32_t Sampler::argmax(const float* logits, std::size_t n) const {
    if (n == 0)
        return -1;
    return static_cast<int32_t>(std::distance(logits, std::max_element(logits, logits + n)));
}

int32_t Sampler::sample(const float* logits, std::size_t n, float temperature) {
    if (temperature <= 0.0f) {
        return argmax(logits, n);
    }

    auto probs = softmax(logits, n, temperature);
    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    return dist(rng_);
}

int32_t Sampler::sample_top_k(const float* logits, std::size_t n, int32_t k, float temperature) {
    if (k <= 1 || temperature <= 0.0f) {
        return argmax(logits, n);
    }

    auto probs = softmax(logits, n, temperature);
    std::vector<std::pair<float, int32_t>> indexed_probs(n);
    for (std::size_t i = 0; i < n; ++i) {
        indexed_probs[i] = {probs[i], static_cast<int32_t>(i)};
    }

    std::size_t top_k_boundary = std::min(static_cast<std::size_t>(k), n);
    std::nth_element(indexed_probs.begin(), indexed_probs.begin() + top_k_boundary,
                     indexed_probs.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<float> k_probs(top_k_boundary);
    float sum = 0.0f;
    for (std::size_t i = 0; i < top_k_boundary; ++i) {
        k_probs[i] = indexed_probs[i].first;
        sum += k_probs[i];
    }

    for (float& p : k_probs)
        p /= sum;

    std::discrete_distribution<int32_t> dist(k_probs.begin(), k_probs.end());
    return indexed_probs[static_cast<std::size_t>(dist(rng_))].second;
}

int32_t Sampler::sample_top_p(const float* logits, std::size_t n, float p, float temperature) {
    if (p <= 0.0f || p >= 1.0f || temperature <= 0.0f) {
        return argmax(logits, n);
    }

    auto probs = softmax(logits, n, temperature);
    std::vector<std::pair<float, int32_t>> indexed_probs(n);
    for (std::size_t i = 0; i < n; ++i) {
        indexed_probs[i] = {probs[i], static_cast<int32_t>(i)};
    }

    std::sort(indexed_probs.begin(), indexed_probs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    float cumulative_prob = 0.0f;
    std::size_t cutoff = 0;
    for (; cutoff < n; ++cutoff) {
        cumulative_prob += indexed_probs[cutoff].first;
        if (cumulative_prob >= p) {
            cutoff++;
            break;
        }
    }
    if (cutoff == 0)
        cutoff = 1;

    std::vector<float> p_probs(cutoff);
    float sum = 0.0f;
    for (std::size_t i = 0; i < cutoff; ++i) {
        p_probs[i] = indexed_probs[i].first;
        sum += p_probs[i];
    }

    for (float& prob : p_probs)
        prob /= sum;

    std::discrete_distribution<int32_t> dist(p_probs.begin(), p_probs.end());
    return indexed_probs[static_cast<std::size_t>(dist(rng_))].second;
}

} // namespace llmengine