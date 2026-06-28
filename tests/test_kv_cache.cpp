#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>
#include <vector>

#include "llmengine/kv_cache.hpp"

namespace llmengine {

class KVCacheTest : public ::testing::Test {
protected:
    static constexpr int64_t SEQ = 32;
    static constexpr int64_t LAYERS = 12;
    static constexpr int64_t HEADS = 16;
    static constexpr int64_t DIM = 64;

    KVCache make_cache() { return KVCache(SEQ, LAYERS, HEADS, DIM); }
};

//
// Construction
//

TEST_F(KVCacheTest, ConstructionCreatesObject) {
    KVCache cache(SEQ, LAYERS, HEADS, DIM);

    SUCCEED();
}

TEST_F(KVCacheTest, StartsWithEmptyState) {
    auto cache = make_cache();

    EXPECT_EQ(cache.size(), 0);
    EXPECT_TRUE(cache.empty());
}

TEST_F(KVCacheTest, CapacityIsPreserved) {
    auto cache = make_cache();

    EXPECT_EQ(cache.capacity(), SEQ);
}

//
// Tensor existence
//

TEST_F(KVCacheTest, CreatesKeyTensor) {
    auto cache = make_cache();

    ASSERT_NE(cache.k_cache(), nullptr);
}

TEST_F(KVCacheTest, CreatesValueTensor) {
    auto cache = make_cache();

    ASSERT_NE(cache.v_cache(), nullptr);
}

TEST_F(KVCacheTest, KeyAndValueAreDifferentBuffers) {
    auto cache = make_cache();

    EXPECT_NE(cache.k_cache()->data, cache.v_cache()->data);
}

//
// Tensor metadata
//

TEST_F(KVCacheTest, TensorTypeIsFloat32) {
    auto cache = make_cache();

    EXPECT_EQ(cache.k_cache()->type, GGML_TYPE_F32);

    EXPECT_EQ(cache.v_cache()->type, GGML_TYPE_F32);
}

TEST_F(KVCacheTest, TensorDimensionCount) {
    auto cache = make_cache();

    ggml_tensor* k = cache.k_cache();
    ggml_tensor* v = cache.v_cache();

    EXPECT_GT(k->ne[0], 0);
    EXPECT_GT(k->ne[1], 0);
    EXPECT_GT(k->ne[2], 0);
    EXPECT_GT(k->ne[3], 0);

    EXPECT_GT(v->ne[0], 0);
    EXPECT_GT(v->ne[1], 0);
    EXPECT_GT(v->ne[2], 0);
    EXPECT_GT(v->ne[3], 0);

    // ggml stores dimensions until ne[i] == 1
    EXPECT_EQ(ggml_n_dims(k), 4);

    EXPECT_EQ(ggml_n_dims(v), 4);
}

TEST_F(KVCacheTest, TensorShapeMatchesConstructor) {
    auto cache = make_cache();

    ggml_tensor* k = cache.k_cache();

    EXPECT_EQ(k->ne[0], DIM);
    EXPECT_EQ(k->ne[1], HEADS);
    EXPECT_EQ(k->ne[2], SEQ);
    EXPECT_EQ(k->ne[3], LAYERS);
}

TEST_F(KVCacheTest, TensorElementCountCorrect) {
    auto cache = make_cache();

    ggml_tensor* k = cache.k_cache();

    int64_t expected = DIM * HEADS * SEQ * LAYERS;

    EXPECT_EQ(ggml_nelements(k), expected);
}

TEST_F(KVCacheTest, TensorSizeCorrect) {
    auto cache = make_cache();

    size_t expected = DIM * HEADS * SEQ * LAYERS * sizeof(float);

    EXPECT_EQ(ggml_nbytes(cache.k_cache()), expected);
}

//
// Sequence tracking
//

TEST_F(KVCacheTest, IncrementWorks) {
    auto cache = make_cache();

    for (int i = 1; i <= 10; i++) {
        cache.increment_sequence();

        EXPECT_EQ(cache.size(), i);
    }
}

TEST_F(KVCacheTest, ReachingMaximumSequence) {
    KVCache cache(5, LAYERS, HEADS, DIM);

    for (int i = 0; i < 5; i++)
        cache.increment_sequence();

    EXPECT_EQ(cache.size(), 5);
}

TEST_F(KVCacheTest, CannotOverflowSequence) {
    KVCache cache(5, LAYERS, HEADS, DIM);

    for (int i = 0; i < 100; i++)
        cache.increment_sequence();

    EXPECT_EQ(cache.size(), 5);
}

TEST_F(KVCacheTest, ClearAfterUsage) {
    auto cache = make_cache();

    for (int i = 0; i < 10; i++)
        cache.increment_sequence();

    cache.clear();

    EXPECT_EQ(cache.size(), 0);

    EXPECT_TRUE(cache.empty());
}

TEST_F(KVCacheTest, ClearDoesNotDestroyTensor) {
    auto cache = make_cache();

    auto* k = cache.k_cache();

    cache.increment_sequence();

    cache.clear();

    EXPECT_EQ(cache.k_cache(), k);
}

//
// Memory integrity
//

TEST_F(KVCacheTest, TensorMemoryIsWritable) {
    auto cache = make_cache();

    float* data = static_cast<float*>(cache.k_cache()->data);

    ASSERT_NE(data, nullptr);

    data[0] = 123.0f;

    EXPECT_FLOAT_EQ(data[0], 123.0f);
}

TEST_F(KVCacheTest, KVMemoryIndependent) {
    auto cache = make_cache();

    float* k = static_cast<float*>(cache.k_cache()->data);

    float* v = static_cast<float*>(cache.v_cache()->data);

    k[0] = 99.0f;

    EXPECT_NE(k[0], v[0]);
}

//
// Lifetime
//

TEST_F(KVCacheTest, CanCreateAndDestroyManyTimes) {
    for (int i = 0; i < 1000; i++) {
        KVCache cache(SEQ, LAYERS, HEADS, DIM);
    }

    SUCCEED();
}

TEST_F(KVCacheTest, DifferentConfigurationsWork) {
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> configs = {{1, 1, 1, 1}, {128, 2, 4, 32}, {512, 32, 32, 128}};

    for (auto [seq, layers, heads, dim] : configs) {
        KVCache cache(seq, layers, heads, dim);

        EXPECT_EQ(cache.capacity(), seq);

        EXPECT_NE(cache.k_cache(), nullptr);
    }
}

//
// Compile-time API checks
//

TEST_F(KVCacheTest, TypeTraits) {
    static_assert(!std::is_copy_constructible_v<KVCache>);

    static_assert(!std::is_copy_assignable_v<KVCache>);

    static_assert(!std::is_move_constructible_v<KVCache>);

    static_assert(!std::is_move_assignable_v<KVCache>);

    static_assert(noexcept(std::declval<KVCache&>().clear()));

    static_assert(noexcept(std::declval<KVCache&>().increment_sequence()));

    SUCCEED();
}

} // namespace llmengine