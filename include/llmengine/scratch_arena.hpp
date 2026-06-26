#pragma once

#include "llmengine/tensor.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace llmengine {

class ScratchArena {
public:
    explicit ScratchArena(std::size_t bytes);

    void reset() noexcept;

    void alloc(Tensor& tensor, std::span<const int64_t> shape, DType dtype = DType::F32);

    [[nodiscard]]
    std::size_t mark() const noexcept;

    void rewind(std::size_t mark) noexcept;

    [[nodiscard]]
    std::size_t capacity() const noexcept {
        return buffer_.size();
    }

    [[nodiscard]]
    std::size_t used() const noexcept {
        return offset_;
    }

private:
    static constexpr std::size_t kAlignment = 64;

    static std::size_t align_up(std::size_t value) noexcept;

    std::vector<std::byte> buffer_;
    std::size_t offset_ = 0;
};

class ScratchScope {
public:
    explicit ScratchScope(ScratchArena& arena) noexcept : arena_(arena), mark_(arena.mark()) {}

    ScratchScope(const ScratchScope&) = delete;
    ScratchScope& operator=(const ScratchScope&) = delete;

    ~ScratchScope() { arena_.rewind(mark_); }

private:
    ScratchArena& arena_;
    std::size_t mark_;
};

} // namespace llmengine