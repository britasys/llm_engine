// context.cpp
#include "llmengine/context.hpp"

#include <stdexcept>
#include <utility>

namespace llmengine {

Context::Context(std::size_t mem_size, bool no_alloc) {
    ggml_init_params params{};
    params.mem_size = mem_size;
    params.mem_buffer = nullptr;
    params.no_alloc = no_alloc;

    ctx_ = ggml_init(params);

    if (!ctx_) {
        throw std::runtime_error("Failed to initialize ggml context.");
    }
}

Context::Context(Context&& other) noexcept : ctx_(std::exchange(other.ctx_, nullptr)) {}

Context& Context::operator=(Context&& other) noexcept {
    if (this != &other) {
        if (ctx_) {
            ggml_free(ctx_);
        }
        ctx_ = std::exchange(other.ctx_, nullptr);
    }
    return *this;
}

Context::~Context() {
    if (ctx_) {
        ggml_free(ctx_);
    }
}

ggml_context* Context::get() const noexcept {
    return ctx_;
}

} // namespace llmengine