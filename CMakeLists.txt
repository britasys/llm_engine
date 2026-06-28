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

include(FetchContent)

# ---------------------------------------------------------------------------
# ggml
# ---------------------------------------------------------------------------

include(FetchContent)

FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggml-org/ggml.git
    GIT_TAG eced84c
    UPDATE_DISCONNECTED ON
)
FetchContent_MakeAvailable(ggml)

# ---------------------------------------------------------------------------
# The core library
# ---------------------------------------------------------------------------
add_library(llmengine
    src/backend.cpp
    src/context.cpp
    src/engine.cpp
    src/gguf_loader.cpp
    src/graph_executor.cpp
    src/kv_cache.cpp
    src/model_loader.cpp
    src/model.cpp
    src/sampler.cpp
    src/scratch_arena.cpp
    src/tensor_loader.cpp
    src/tokenizer.cpp
    src/transformer_block.cpp
    src/platform/memory_mapped_file_win.cpp
    src/platform/memory_mapped_file_posix.cpp
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
target_link_libraries(llmengine PUBLIC Threads::Threads ggml)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(LLMENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

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
