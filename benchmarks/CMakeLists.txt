include(FetchContent)

# google/benchmark for honest, statistically-sound microbenchmarks. This is
# how we back up "high-performance" claims with numbers instead of vibes.
#
# We disable benchmark's own test suite and its bundled gtest dependency
# (BENCHMARK_ENABLE_GTEST_TESTS) since we already fetch our own GoogleTest
# in tests/CMakeLists.txt -- pulling in a second copy would create duplicate
# CMake targets and slow down configure for no benefit.
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

# Benchmarks should reflect realistic optimized performance, not debug-build
# numbers -- force release-level optimization for this target specifically
# regardless of the project-wide CMAKE_BUILD_TYPE, so `cmake --build .` in
# Debug mode (handy for the test suite + sanitizers) doesn't silently give
# you meaningless benchmark numbers.
target_compile_options(bench_tensor PRIVATE -O3)
target_compile_options(bench_ops PRIVATE -O3)
