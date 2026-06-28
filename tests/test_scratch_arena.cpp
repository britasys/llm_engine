#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>
#include <vector>

#include "llmengine/scratch_arena.hpp"

namespace llmengine {

class ScratchArenaTest : public ::testing::Test {
protected:
    static constexpr std::size_t SIZE = 1024 * 1024;
};

//
// Construction
//

TEST_F(ScratchArenaTest, ConstructionCreatesContext) {
    ScratchArena arena(SIZE);

    EXPECT_NE(arena.ctx(), nullptr);
}

TEST_F(ScratchArenaTest, ContextIsValid) {
    ScratchArena arena(SIZE);

    ggml_context* ctx = arena.ctx();

    ASSERT_NE(ctx, nullptr);

    EXPECT_EQ(ggml_get_no_alloc(ctx), false);
}

//
// Allocation
//

TEST_F(ScratchArenaTest, CanAllocateTensor) {
    ScratchArena arena(SIZE);

    ggml_context* ctx = arena.ctx();

    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);

    ASSERT_NE(t, nullptr);

    EXPECT_EQ(t->type, GGML_TYPE_F32);

    EXPECT_EQ(t->ne[0], 128);
}

TEST_F(ScratchArenaTest, CanAllocateMultipleTensors) {
    ScratchArena arena(SIZE);

    ggml_context* ctx = arena.ctx();

    std::vector<ggml_tensor*> tensors;

    for (int i = 1; i <= 100; i++) {
        auto* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, i);

        ASSERT_NE(t, nullptr);

        tensors.push_back(t);
    }

    EXPECT_EQ(tensors.size(), 100);
}

TEST_F(ScratchArenaTest, TensorMemoryIsUsable) {
    ScratchArena arena(SIZE);

    ggml_tensor* t = ggml_new_tensor_1d(arena.ctx(), GGML_TYPE_F32, 16);

    ASSERT_NE(t, nullptr);

    float* data = static_cast<float*>(t->data);

    ASSERT_NE(data, nullptr);

    data[0] = 42.0f;

    EXPECT_FLOAT_EQ(data[0], 42.0f);
}

//
// Reset behavior
//

TEST_F(ScratchArenaTest, ResetDoesNotDestroyContext) {
    ScratchArena arena(SIZE);

    auto* ctx_before = arena.ctx();

    arena.reset();

    EXPECT_EQ(arena.ctx(), ctx_before);
}

TEST_F(ScratchArenaTest, ResetAllowsReuse) {
    ScratchArena arena(SIZE);

    ggml_tensor* first = ggml_new_tensor_1d(arena.ctx(), GGML_TYPE_F32, 256);

    ASSERT_NE(first, nullptr);

    arena.reset();

    ggml_tensor* second = ggml_new_tensor_1d(arena.ctx(), GGML_TYPE_F32, 256);

    EXPECT_NE(second, nullptr);
}

TEST_F(ScratchArenaTest, ResetClearsOldAllocations) {
    ScratchArena arena(SIZE);

    ggml_tensor* before = ggml_new_tensor_1d(arena.ctx(), GGML_TYPE_F32, 16);

    ASSERT_NE(before, nullptr);

    arena.reset();

    ggml_tensor* after = ggml_new_tensor_1d(arena.ctx(), GGML_TYPE_F32, 16);

    ASSERT_NE(after, nullptr);

    EXPECT_EQ(after->ne[0], 16);
}

//
// Edge cases
//

TEST_F(ScratchArenaTest, SmallArenaWorks) {
    ScratchArena arena(1024);

    EXPECT_NE(arena.ctx(), nullptr);
}

TEST_F(ScratchArenaTest, LargeArenaWorks) {
    ScratchArena arena(128 * 1024 * 1024);

    EXPECT_NE(arena.ctx(), nullptr);
}

TEST_F(ScratchArenaTest, ManyResets) {
    ScratchArena arena(SIZE);

    for (int i = 0; i < 1000; i++) {
        ggml_new_tensor_1d(arena.ctx(), GGML_TYPE_F32, 32);

        arena.reset();
    }

    SUCCEED();
}

//
// Alignment / layout
//

TEST_F(ScratchArenaTest, TensorAlignmentIsValid) {
    ScratchArena arena(SIZE);

    ggml_tensor* t = ggml_new_tensor_1d(arena.ctx(), GGML_TYPE_F32, 32);

    ASSERT_NE(t, nullptr);

    auto address = reinterpret_cast<std::uintptr_t>(t->data);

    EXPECT_EQ(address % GGML_MEM_ALIGN, 0);
}

TEST_F(ScratchArenaTest, TensorSizeIsCorrect) {
    ScratchArena arena(SIZE);

    ggml_tensor* t = ggml_new_tensor_1d(arena.ctx(), GGML_TYPE_F32, 100);

    EXPECT_EQ(ggml_nbytes(t), 100 * sizeof(float));
}

//
// Type traits
//

TEST_F(ScratchArenaTest, IsNonCopyable) {
    static_assert(!std::is_copy_constructible_v<ScratchArena>);

    static_assert(!std::is_copy_assignable_v<ScratchArena>);
}

//
// Lifetime stress
//

TEST_F(ScratchArenaTest, CanCreateAndDestroyRepeatedly) {
    for (int i = 0; i < 1000; i++) {
        ScratchArena arena(1024 * 1024);

        EXPECT_NE(arena.ctx(), nullptr);
    }
}

} // namespace llmengine