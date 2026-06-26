#include "llmengine/scratch_arena.hpp"

#include <stdexcept>
#include <vector>

namespace llmengine {

namespace {

// Inline helper since dtype_size isn't exposed globally
std::size_t get_dtype_size(DType dtype) {
    switch (dtype) {
    case DType::F32:
        return 4;
    // If you have other types used in the arena, add them here
    default:
        return 4;
    }
}

std::size_t bytes_required(std::span<const int64_t> shape, DType dtype) {
    std::size_t numel = 1;

    for (int64_t d : shape) {
        if (d <= 0)
            throw std::runtime_error("ScratchArena: invalid tensor shape.");

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

void ScratchArena::alloc(Tensor& tensor, std::span<const int64_t> shape, DType dtype) {
    const std::size_t raw_bytes = bytes_required(shape, dtype);

    std::uintptr_t base_ptr = reinterpret_cast<std::uintptr_t>(buffer_.data());
    std::uintptr_t current_ptr = base_ptr + offset_;

    std::uintptr_t aligned_ptr = (current_ptr + kAlignment - 1) & ~(kAlignment - 1);

    std::size_t alignment_padding = aligned_ptr - current_ptr;
    std::size_t total_allocation_bytes = alignment_padding + raw_bytes;

    if (offset_ + total_allocation_bytes > buffer_.size())
        throw std::runtime_error("ScratchArena: out of memory.");

    // FIX: Convert std::span to std::vector<int64_t> to match your exact Tensor::view signature
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());

    // FIX: Match signature taking a void* (no need to keep it std::byte*)
    tensor = Tensor::view(reinterpret_cast<void*>(aligned_ptr), shape_vec, dtype);

    offset_ += total_allocation_bytes;
}

} // namespace llmengine