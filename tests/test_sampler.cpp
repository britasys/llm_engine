// test_sampler.cpp
#include "llmengine/sampler.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <map>
#include <vector>

using llmengine::Sampler;

TEST(SamplerTest, ArgmaxPicksHighestLogit) {
    Sampler s(42);
    std::vector<float> logits = {0.1f, 0.5f, 3.2f, -1.0f, 2.9f};
    EXPECT_EQ(s.argmax(logits.data(), logits.size()), 2);
}

TEST(SamplerTest, ArgmaxSingleElement) {
    Sampler s(42);
    std::vector<float> logits = {5.0f};
    EXPECT_EQ(s.argmax(logits.data(), logits.size()), 0);
}

TEST(SamplerTest, ArgmaxFirstOnTie) {
    Sampler s(42);
    std::vector<float> logits = {2.0f, 2.0f, 2.0f};
    EXPECT_EQ(s.argmax(logits.data(), logits.size()), 0);
}

TEST(SamplerTest, ArgmaxNegativeValues) {
    Sampler s(42);
    std::vector<float> logits = {-5.0f, -1.0f, -3.0f};
    EXPECT_EQ(s.argmax(logits.data(), logits.size()), 1);
}

TEST(SamplerTest, SampleReturnsValidIndex) {
    Sampler s(42);
    std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < 100; ++i) {
        int32_t idx = s.sample(logits.data(), logits.size());
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, static_cast<int32_t>(logits.size()));
    }
}

TEST(SamplerTest, SampleDeterministicWithSameSeed) {
    Sampler s1(123);
    Sampler s2(123);
    std::vector<float> logits = {1.0f, 2.0f, 0.5f, 3.0f, -1.0f};
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(s1.sample(logits.data(), logits.size()), s2.sample(logits.data(), logits.size()));
    }
}

TEST(SamplerTest, SampleDistributionFavorsHigherLogit) {
    Sampler s(7);
    std::vector<float> logits = {0.0f, 10.0f};
    int count_high = 0;
    const int trials = 1000;
    for (int i = 0; i < trials; ++i) {
        if (s.sample(logits.data(), logits.size()) == 1) {
            ++count_high;
        }
    }
    EXPECT_GT(count_high, trials * 0.95);
}

TEST(SamplerTest, SampleZeroTemperatureBehavesLikeArgmax) {
    Sampler s(42);
    std::vector<float> logits = {0.1f, 0.5f, 3.2f, -1.0f, 2.9f};
    int32_t result = s.sample(logits.data(), logits.size(), 0.0f);
    EXPECT_EQ(result, s.argmax(logits.data(), logits.size()));
}

TEST(SamplerTest, SampleHighTemperatureNearUniform) {
    Sampler s(99);
    std::vector<float> logits = {1.0f, 1.0f, 1.0f, 1.0f};
    std::map<int32_t, int> counts;
    const int trials = 4000;
    for (int i = 0; i < trials; ++i) {
        counts[s.sample(logits.data(), logits.size(), 1.0f)]++;
    }
    EXPECT_EQ(counts.size(), 4u);
    for (auto& [k, v] : counts) {
        EXPECT_GT(v, trials / 4 * 0.7);
        EXPECT_LT(v, trials / 4 * 1.3);
    }
}

TEST(SamplerTest, TopKReturnsValidIndex) {
    Sampler s(42);
    std::vector<float> logits = {1.0f, 5.0f, 2.0f, 4.0f, 0.5f};
    for (int i = 0; i < 100; ++i) {
        int32_t idx = s.sample_top_k(logits.data(), logits.size(), 2);
        EXPECT_TRUE(idx == 1 || idx == 3);
    }
}

TEST(SamplerTest, TopKEqualToOneIsArgmax) {
    Sampler s(42);
    std::vector<float> logits = {1.0f, 5.0f, 2.0f, 4.0f, 0.5f};
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(s.sample_top_k(logits.data(), logits.size(), 1), 1);
    }
}

TEST(SamplerTest, TopKGreaterThanNBehavesLikeFullSample) {
    Sampler s(42);
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    for (int i = 0; i < 50; ++i) {
        int32_t idx = s.sample_top_k(logits.data(), logits.size(), 100);
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, static_cast<int32_t>(logits.size()));
    }
}

TEST(SamplerTest, TopKNeverPicksExcludedLowIndices) {
    Sampler s(13);
    std::vector<float> logits = {10.0f, -10.0f, 9.0f, -9.0f, 8.0f};
    for (int i = 0; i < 200; ++i) {
        int32_t idx = s.sample_top_k(logits.data(), logits.size(), 3);
        EXPECT_TRUE(idx == 0 || idx == 2 || idx == 4);
    }
}

TEST(SamplerTest, TopPReturnsValidIndex) {
    Sampler s(55);
    std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    for (int i = 0; i < 100; ++i) {
        int32_t idx = s.sample_top_p(logits.data(), logits.size(), 0.9f);
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, static_cast<int32_t>(logits.size()));
    }
}

TEST(SamplerTest, TopPVerySmallActsLikeArgmax) {
    Sampler s(42);
    std::vector<float> logits = {0.1f, 0.5f, 8.0f, -1.0f, 0.2f};
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(s.sample_top_p(logits.data(), logits.size(), 0.01f), 2);
    }
}

TEST(SamplerTest, TopPLargeBehavesLikeFullSample) {
    Sampler s(42);
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    for (int i = 0; i < 50; ++i) {
        int32_t idx = s.sample_top_p(logits.data(), logits.size(), 1.0f);
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, static_cast<int32_t>(logits.size()));
    }
}

TEST(SamplerTest, TopPExcludesLowProbabilityTail) {
    Sampler s(21);
    std::vector<float> logits = {10.0f, 10.0f, -20.0f, -20.0f, -20.0f};
    for (int i = 0; i < 200; ++i) {
        int32_t idx = s.sample_top_p(logits.data(), logits.size(), 0.5f);
        EXPECT_TRUE(idx == 0 || idx == 1);
    }
}

TEST(SamplerTest, DifferentSeedsCanProduceDifferentSequences) {
    Sampler s1(1);
    Sampler s2(2);
    std::vector<float> logits = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    bool any_diff = false;
    for (int i = 0; i < 50; ++i) {
        if (s1.sample(logits.data(), logits.size()) != s2.sample(logits.data(), logits.size())) {
            any_diff = true;
            break;
        }
    }
    EXPECT_TRUE(any_diff);
}

TEST(SamplerTest, SingleElementAlwaysReturnsZero) {
    Sampler s(42);
    std::vector<float> logits = {3.14f};
    EXPECT_EQ(s.sample(logits.data(), logits.size()), 0);
    EXPECT_EQ(s.sample_top_k(logits.data(), logits.size(), 5), 0);
    EXPECT_EQ(s.sample_top_p(logits.data(), logits.size(), 0.9f), 0);
}
