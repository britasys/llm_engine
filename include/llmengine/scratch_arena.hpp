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

    void alloc(Tensor& tensor, std::span<const int64_t> shape, DType dtype);

    [[nodiscard]]
    std::size_t capacity() const noexcept {
        return buffer_.size();
    }

    [[nodiscard]]
    std::size_t used() const noexcept {
        return offset_;
    }

    [[nodiscard]]
    std::size_t available() const noexcept {
        return buffer_.size() - offset_;
    }

private:
    static constexpr std::size_t kAlignment = 64;

    static std::size_t align_up(std::size_t value) noexcept;

    std::vector<std::byte> buffer_;
    std::size_t offset_ = 0;
};

} // namespace llmengine