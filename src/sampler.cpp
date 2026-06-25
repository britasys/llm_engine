#include "llmengine/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace llmengine {

Sampler::Sampler(uint32_t seed) : rng_(seed) {}

std::vector<float> Sampler::softmax(const float* logits, std::size_t n, float temperature) {
    if (temperature <= 0.f) {
        throw std::runtime_error("temperature must be positive");
    }

    std::vector<float> probs(n);

    float max_logit = logits[0];

    for (std::size_t i = 1; i < n; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }

    float sum = 0.f;

    for (std::size_t i = 0; i < n; ++i) {
        probs[i] = std::exp((logits[i] - max_logit) / temperature);

        sum += probs[i];
    }

    for (float& p : probs) {
        p /= sum;
    }

    return probs;
}

int32_t Sampler::argmax(const Tensor& logits) const {
    auto values = logits.as_f32();

    return static_cast<int32_t>(
        std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

int32_t Sampler::sample(const Tensor& logits, float temperature) {
    auto values = logits.as_f32();

    auto probs = softmax(values.data(), values.size(), temperature);

    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());

    return dist(rng_);
}

int32_t Sampler::sample_top_k(const Tensor& logits, int32_t k, float temperature) {
    auto values = logits.as_f32();

    if (k <= 0) {
        throw std::runtime_error("invalid k");
    }

    struct Candidate {
        int32_t token;
        float logit;
    };

    std::vector<Candidate> candidates;

    candidates.reserve(values.size());

    for (std::size_t i = 0; i < values.size(); ++i) {
        candidates.push_back({static_cast<int32_t>(i), values[i]});
    }

    std::partial_sort(
        candidates.begin(), candidates.begin() + std::min<std::size_t>(k, candidates.size()),
        candidates.end(), [](const auto& a, const auto& b) { return a.logit > b.logit; });

    k = std::min<int32_t>(k, static_cast<int32_t>(candidates.size()));

    std::vector<float> top_logits(k);

    for (int32_t i = 0; i < k; ++i) {
        top_logits[i] = candidates[i].logit;
    }

    auto probs = softmax(top_logits.data(), top_logits.size(), temperature);

    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());

    return candidates[dist(rng_)].token;
}

int32_t Sampler::sample_top_p(const Tensor& logits, float p, float temperature) {
    auto values = logits.as_f32();

    if (p <= 0.f || p > 1.f) {
        throw std::runtime_error("invalid p");
    }

    auto probs = softmax(values.data(), values.size(), temperature);

    struct Candidate {
        int32_t token;
        float prob;
    };

    std::vector<Candidate> candidates;

    for (std::size_t i = 0; i < probs.size(); ++i) {
        candidates.push_back({static_cast<int32_t>(i), probs[i]});
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.prob > b.prob; });

    float cumulative = 0.f;

    std::vector<Candidate> nucleus;

    for (const auto& c : candidates) {
        nucleus.push_back(c);

        cumulative += c.prob;

        if (cumulative >= p) {
            break;
        }
    }

    std::vector<double> weights;

    for (const auto& c : nucleus) {
        weights.push_back(c.prob);
    }

    std::discrete_distribution<int32_t> dist(weights.begin(), weights.end());

    return nucleus[dist(rng_)].token;
}

} // namespace llmengine