cmake_minimum_required(VERSION 3.21)
project(llmengine LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Core language settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# ---------------------------------------------------------------------------
# Project-wide options
# ---------------------------------------------------------------------------
option(LLMENGINE_BUILD_TESTS "Build unit tests" ON)
option(LLMENGINE_BUILD_BENCHMARKS "Build google/benchmark performance suite" ON)
option(LLMENGINE_BUILD_EXAMPLES "Build example binaries (CLI chat, etc.)" ON)
option(LLMENGINE_ENABLE_SANITIZERS "Build with ASan+UBSan (debug aid, not for release)" OFF)
option(LLMENGINE_NATIVE_ARCH "Compile with -march=native for max local performance" OFF)

# ---------------------------------------------------------------------------
# The core library
# ---------------------------------------------------------------------------
add_library(llmengine
    src/tensor.cpp
    src/gguf_loader.cpp
    src/tokenizer.cpp
    src/ops.cpp
    src/kv_cache.cpp
    src/sampler.cpp
    src/model.cpp
    src/model_helper.cpp
    src/engine.cpp
    src/rope_cache.cpp
    src/scratch_arena.cpp
)

target_include_directories(llmengine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_options(llmengine PRIVATE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion
)

if(LLMENGINE_NATIVE_ARCH)
    target_compile_options(llmengine PUBLIC -march=native)
endif()

if(LLMENGINE_ENABLE_SANITIZERS)
    target_compile_options(llmengine PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(llmengine PUBLIC -fsanitize=address,undefined)
endif()

# Threads for future parallel matmul work (std::thread / OpenMP-style fan-out)
find_package(Threads REQUIRED)
target_link_libraries(llmengine PUBLIC Threads::Threads)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# if(LLMENGINE_BUILD_TESTS)
#     enable_testing()
#     add_subdirectory(tests)
# endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
# if(LLMENGINE_BUILD_BENCHMARKS)
#     add_subdirectory(benchmarks)
# endif()

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(LLMENGINE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
