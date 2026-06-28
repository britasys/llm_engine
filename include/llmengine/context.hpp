// context.hpp
#pragma once

#include <cstddef>
#include <ggml.h>

namespace llmengine {

class Context {
public:
    explicit Context(std::size_t mem_size, bool no_alloc = false);
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&& other) noexcept;
    Context& operator=(Context&& other) noexcept;
    ~Context();

    [[nodiscard]] ggml_context* get() const noexcept;

private:
    ggml_context* ctx_ = nullptr;
};

} // namespace llmengine