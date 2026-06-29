
# llm-engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Language: C++23](https://img.shields.io/badge/C++-23-blue.svg)](https://en.cppreference.com/w/cpp/23)

A from-scratch, highly modular C++23 LLM inference engine. This project serves as a comprehensive, educational guided tour through the core architectural concepts powering production engines like `llama.cpp`. 

Rather than presenting a monolithic codebase, **llm-engine** breaks down GGUF model loading, tokenization, transformer forward passes, KV-caching, and sampling into independent, 1:1 tested modules packed with extensive teaching comments.

> 💡 **Core Philosophy:** Every header file explains **why** a specific design decision was made, not just *what* the code does.

---

## 🗺️ System Architecture

For a high-level module map, check out [`docs/architecture.md`](docs/architecture.md). Deep dives into specific LLM mechanics can be found in [`docs/concepts/`](docs/concepts/).

## Building

Requires CMake ≥ 3.21 and a C++23 compiler (GCC ≥ 13 or Clang ≥ 17 tested).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Useful options (`-D<OPTION>=ON/OFF` at configure time):

| Option | Default | Purpose |
|---|---|---|
| `LLMENGINE_BUILD_TESTS` | ON | Build the GoogleTest unit test suite |
| `LLMENGINE_BUILD_BENCHMARKS` | ON | Build the google/benchmark performance suite |
| `LLMENGINE_BUILD_EXAMPLES` | ON | Build `examples/cli_chat` |
| `LLMENGINE_ENABLE_SANITIZERS` | OFF | ASan + UBSan, for catching memory bugs while developing |
| `LLMENGINE_NATIVE_ARCH` | OFF | `-march=native`, for local performance testing only |


## Project layout

```
include/llmengine/   public headers, one per module, heavily commented
src/                  implementations
tests/                GoogleTest unit tests, mirrored 1:1 with modules
examples/             cli_chat: the end-user-facing REPL
```

## Why build this instead of just reading llama.cpp?

Reading mature production C code teaches you *what* works; rebuilding the
core ideas from an empty repo, in idiomatic modern C++, teaches you *why*
each piece exists and what breaks if you skip it. The structure here
intentionally separates concerns that are more entangled in llama.cpp
itself (tensor math, file format, tokenizer, sampling, orchestration) so
each one can be understood and tested in isolation before being wired
together.

## License

MIT — see [`LICENSE`](LICENSE).
