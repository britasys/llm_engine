// test_backend.cpp
#include "llmengine/backend.hpp"

#include <gtest/gtest.h>

#include <utility>

using llmengine::Backend;


TEST(BackendTest, ConstructsSuccessfully) {
    EXPECT_NO_THROW({
        Backend backend;
        EXPECT_NE(backend.get(), nullptr);
    });
}

TEST(BackendTest, MultipleInstancesHaveDistinctHandles) {
    Backend b1;
    Backend b2;
    EXPECT_NE(b1.get(), nullptr);
    EXPECT_NE(b2.get(), nullptr);
    EXPECT_NE(b1.get(), b2.get());
}

TEST(BackendTest, MoveConstructionTransfersHandle) {
    Backend original;
    ggml_backend_t original_handle = original.get();
    ASSERT_NE(original_handle, nullptr);

    Backend moved(std::move(original));

    EXPECT_EQ(moved.get(), original_handle);
    EXPECT_EQ(original.get(), nullptr);
}

TEST(BackendTest, MoveAssignmentTransfersHandle) {
    Backend a;
    Backend b;

    ggml_backend_t b_handle = b.get();
    ASSERT_NE(b_handle, nullptr);

    a = std::move(b);

    EXPECT_EQ(a.get(), b_handle);
    EXPECT_EQ(b.get(), nullptr);
}

TEST(BackendTest, SelfMoveAssignmentIsSafe) {
    Backend a;
    ggml_backend_t a_handle = a.get();
    ASSERT_NE(a_handle, nullptr);
    
    Backend& a_ref = a;
    a = std::move(a_ref);

    EXPECT_EQ(a.get(), a_handle);
}

TEST(BackendTest, MovedFromBackendHasNullHandle) {
    Backend original;
    Backend moved(std::move(original));

    EXPECT_EQ(original.get(), nullptr);
    EXPECT_EQ(original.get(), nullptr);
}

TEST(BackendTest, DestroyingMovedFromBackendIsSafe) {
    EXPECT_NO_THROW({
        Backend original;
        {
            Backend moved(std::move(original));
        }
    });
}

TEST(BackendTest, MoveAssignmentFreesExistingHandleBeforeTakingNew) {
    EXPECT_NO_THROW({
        Backend a;
        Backend b;
        a = std::move(b);
    });
}

TEST(BackendTest, ChainedMovesPropagateHandleCorrectly) {
    Backend b1;
    ggml_backend_t handle = b1.get();
    ASSERT_NE(handle, nullptr);

    Backend b2(std::move(b1));
    EXPECT_EQ(b2.get(), handle);
    EXPECT_EQ(b1.get(), nullptr);

    Backend b3(std::move(b2));
    EXPECT_EQ(b3.get(), handle);
    EXPECT_EQ(b2.get(), nullptr);
    EXPECT_EQ(b1.get(), nullptr);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}