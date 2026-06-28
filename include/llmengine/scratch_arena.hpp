// scratch_arena.hpp
#pragma once

#include <cstddef>
#include <ggml.h>
#include <memory>
#include <vector>

namespace llmengine {

class ScratchArena {
public:
    explicit ScratchArena(std::size_t bytes);
    ~ScratchArena() = default;

    ScratchArena(const ScratchArena&) = delete;
    ScratchArena& operator=(const ScratchArena&) = delete;

    ScratchArena(ScratchArena&&) noexcept = default;
    ScratchArena& operator=(ScratchArena&&) noexcept = default;

    [[nodiscard]] ggml_context* ctx() const noexcept { return ctx_.get(); }

    void reset() noexcept;

private:
    struct CtxDeleter {
        void operator()(ggml_context* c) const noexcept {
            if (c)
                ggml_free(c);
        }
    };

    std::vector<std::byte> buffer_;
    std::unique_ptr<ggml_context, CtxDeleter> ctx_;
};

} // namespace llmengine