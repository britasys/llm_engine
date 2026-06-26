#include "llmengine/scratch_arena.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace llmengine {

namespace {

std::size_t get_dtype_size(DType dtype) {
    switch (dtype) {
    case DType::F32:
        return sizeof(float);
    default:
        throw std::runtime_error("ScratchArena: unsupported dtype.");
    }
}

std::size_t bytes_required(std::span<const int64_t> shape, DType dtype) {
    std::size_t numel = 1;

    for (int64_t d : shape) {
        if (d <= 0) {
            throw std::runtime_error("ScratchArena: invalid tensor shape.");
        }

        numel *= static_cast<std::size_t>(d);
    }

    return numel * get_dtype_size(dtype);
}

} // namespace

ScratchArena::ScratchArena(std::size_t bytes) : buffer_(align_up(bytes)) {}

std::size_t ScratchArena::align_up(std::size_t value) noexcept {
    return (value + kAlignment - 1) & ~(kAlignment - 1);
}

void ScratchArena::reset() noexcept {
    offset_ = 0;
}

std::size_t ScratchArena::mark() const noexcept {
    return offset_;
}

void ScratchArena::rewind(std::size_t mark) noexcept {
    offset_ = mark;
}

void ScratchArena::alloc(Tensor& tensor, std::span<const int64_t> shape, DType dtype) {
    const std::size_t raw_bytes = bytes_required(shape, dtype);

    std::uintptr_t base = reinterpret_cast<std::uintptr_t>(buffer_.data());
    std::uintptr_t ptr = base + offset_;

    std::uintptr_t aligned =
        (ptr + kAlignment - 1) & ~(static_cast<std::uintptr_t>(kAlignment - 1));

    const std::size_t padding = aligned - ptr;
    const std::size_t total = padding + raw_bytes;

    if (offset_ + total > buffer_.size()) {
        throw std::runtime_error("ScratchArena: out of memory.");
    }

    std::vector<int64_t> shape_vec(shape.begin(), shape.end());

    tensor = Tensor::view(reinterpret_cast<void*>(aligned), shape_vec, dtype);

    offset_ += total;
}

} // namespace llmengine