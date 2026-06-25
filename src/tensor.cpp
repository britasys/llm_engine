#include "llmengine/tensor.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llmengine {

DTypeInfo dtype_info(DType dt) {
    switch (dt) {
    case DType::F32:
        return {4, 1};
    case DType::F16:
        return {2, 1};
    case DType::Q8_0:
        return {32 + 4, 32};
    }
    throw std::runtime_error("unknown DType");
}

const char* dtype_name(DType dt) {
    switch (dt) {
    case DType::F32:
        return "f32";
    case DType::F16:
        return "f16";
    case DType::Q8_0:
        return "q8_0";
    }
    return "unknown";
}

namespace {
int64_t compute_numel(const std::vector<int64_t>& shape) {
    if (shape.empty())
        throw std::invalid_argument("Tensor shape must have at least 1 dimension");
    if (shape.size() > kMaxDims)
        throw std::invalid_argument("Tensor supports at most " + std::to_string(kMaxDims) +
                                    " dimensions");
    int64_t n = 1;
    for (auto d : shape) {
        if (d <= 0)
            throw std::invalid_argument("Tensor dimensions must be positive");
        n *= d;
    }
    return n;
}

std::size_t compute_size_bytes(int64_t numel, DType dt) {
    auto info = dtype_info(dt);
    if (numel % static_cast<int64_t>(info.block_elems) != 0)
        throw std::invalid_argument("element count is not a multiple of the dtype's block size");
    return static_cast<std::size_t>(numel) / info.block_elems * info.block_bytes;
}
} // namespace

Tensor Tensor::zeros(std::vector<int64_t> shape, DType dt) {
    Tensor t;
    t.shape_ = std::move(shape);
    t.numel_ = compute_numel(t.shape_);
    t.dtype_ = dt;
    t.size_bytes_ = compute_size_bytes(t.numel_, dt);
    t.data_ = static_cast<std::byte*>(::operator new(t.size_bytes_));
    std::memset(t.data_, 0, t.size_bytes_);
    t.owns_ = true;
    return t;
}

Tensor Tensor::view(void* data, std::vector<int64_t> shape, DType dt) {
    Tensor t;
    t.shape_ = std::move(shape);
    t.numel_ = compute_numel(t.shape_);
    t.dtype_ = dt;
    t.size_bytes_ = compute_size_bytes(t.numel_, dt);
    t.data_ = static_cast<std::byte*>(data);
    t.owns_ = false;
    return t;
}

void Tensor::free_if_owned() noexcept {
    if (owns_ && data_) {
        ::operator delete(data_);
    }
    data_ = nullptr;
}

Tensor::~Tensor() {
    free_if_owned();
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), numel_(other.numel_), dtype_(other.dtype_),
      data_(other.data_), size_bytes_(other.size_bytes_), owns_(other.owns_) {
    other.data_ = nullptr;
    other.owns_ = false;
    other.numel_ = 0;
    other.size_bytes_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        free_if_owned();
        shape_ = std::move(other.shape_);
        numel_ = other.numel_;
        dtype_ = other.dtype_;
        data_ = other.data_;
        size_bytes_ = other.size_bytes_;
        owns_ = other.owns_;
        other.data_ = nullptr;
        other.owns_ = false;
        other.numel_ = 0;
        other.size_bytes_ = 0;
    }
    return *this;
}

Tensor Tensor::clone() const {
    Tensor t = Tensor::zeros(shape_, dtype_);
    std::memcpy(t.data_, data_, size_bytes_);
    return t;
}

std::span<float> Tensor::as_f32() {
    if (dtype_ != DType::F32)
        throw std::runtime_error("as_f32() called on non-F32 tensor (" +
                                 std::string(dtype_name(dtype_)) + ")");
    return {reinterpret_cast<float*>(data_), static_cast<std::size_t>(numel_)};
}
std::span<const float> Tensor::as_f32() const {
    if (dtype_ != DType::F32)
        throw std::runtime_error("as_f32() called on non-F32 tensor (" +
                                 std::string(dtype_name(dtype_)) + ")");
    return {reinterpret_cast<const float*>(data_), static_cast<std::size_t>(numel_)};
}

std::span<float> Tensor::row(int64_t r) {
    if (ndim() != 2)
        throw std::runtime_error("row() requires a 2D tensor, got " + shape_string());
    int64_t cols = shape_[1];
    if (r < 0 || r >= shape_[0])
        throw std::out_of_range("row index out of range");
    auto full = as_f32();
    return full.subspan(static_cast<std::size_t>(r * cols), static_cast<std::size_t>(cols));
}
std::span<const float> Tensor::row(int64_t r) const {
    if (ndim() != 2)
        throw std::runtime_error("row() requires a 2D tensor, got " + shape_string());
    int64_t cols = shape_[1];
    if (r < 0 || r >= shape_[0])
        throw std::out_of_range("row index out of range");
    auto full = as_f32();
    return full.subspan(static_cast<std::size_t>(r * cols), static_cast<std::size_t>(cols));
}

std::string Tensor::shape_string() const {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        oss << shape_[i];
        if (i + 1 < shape_.size())
            oss << ", ";
    }
    oss << "] (" << dtype_name(dtype_) << ")";
    return oss.str();
}

} // namespace llmengine
