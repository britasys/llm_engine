include(FetchContent)

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)
FetchContent_MakeAvailable(googletest)

add_executable(llmengine_tests
    test_backend.cpp
    test_scratch_arena.cpp
    test_kv_cache.cpp
    test_sampler.cpp
    test_gguf_loader.cpp
    test_model.cpp
    test_tokenizer.cpp
)

target_link_libraries(llmengine_tests PRIVATE llmengine GTest::gtest_main GTest::gmock)
target_compile_options(llmengine_tests PRIVATE -Wall -Wextra)

include(GoogleTest)
gtest_discover_tests(llmengine_tests)
