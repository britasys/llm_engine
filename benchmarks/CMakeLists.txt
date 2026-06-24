include(FetchContent)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.1
)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(bench_tensor
    bench_tensor.cpp
)

add_executable(bench_ops
    bench_ops.cpp
)

target_link_libraries(bench_tensor PRIVATE llmengine benchmark::benchmark)
target_link_libraries(bench_ops PRIVATE llmengine benchmark::benchmark)

target_compile_options(bench_tensor PRIVATE -O3)
target_compile_options(bench_ops PRIVATE -O3)
