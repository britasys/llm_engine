#include "llmengine/scratch_arena.hpp"
#include <stdexcept>

namespace llmengine {

ScratchArena::ScratchArena(std::size_t bytes) : buffer_(bytes) {
    ctx_.reset(ggml_init({
        .mem_size = buffer_.size(),
        .mem_buffer = buffer_.data(),
        .no_alloc = false,
    }));

    if (!ctx_) {
        throw std::runtime_error("ScratchArena: ggml_init failed");
    }
}

void ScratchArena::reset() noexcept {
    ggml_reset(ctx_.get());
}

} // namespace llmengine