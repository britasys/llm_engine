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

    case DType::Q4_0:
        // ggml_type_q4_0
        // 2-byte fp16 scale + 16 packed bytes
        return {18, 32};
    case DType::Q4_1:
        // fp16 scale + fp16 min + 16 packed bytes
        return {20, 32};

    case DType::Q5_0:
        // fp16 scale + 4-byte high bits + 16 packed bytes
        return {22, 32};
    case DType::Q5_1:
        // fp16 scale + fp16 min + 4-byte high bits + 16 packed bytes
        return {24, 32};

    case DType::Q8_0:
        // fp32 scale + 32 int8 values
        return {36, 32};
    case DType::Q8_1:
        // fp32 scale + fp32 sum + 32 int8 values
        return {40, 32};

    case DType::Q2_K:
        return {84, 256};        
    case DType::Q3_K:
        return {110, 256};
    case DType::Q4_K:
        return {144, 256};
    case DType::Q5_K:
        return {176, 256};
    case DType::Q6_K:
        return {210, 256};
    case DType::Q8_K:
        return {292, 256};

    case DType::IQ2_XXS:
        return {66, 256};
    case DType::IQ2_XS:
        return {74, 256};
    case DType::IQ3_XXS:
        return {98, 256};
    case DType::IQ1_S:
        return {50, 256};
    case DType::IQ4_NL:
        return {144, 256};
    case DType::IQ3_S:
        return {110, 256};
    case DType::IQ2_S:
        return {82, 256};
    case DType::IQ4_XS:
        return {136, 256};
    case DType::IQ1_M:
        return {56, 256};
    }

    throw std::runtime_error("unknown DType: " + std::to_string(static_cast<int>(dt)));
}

const char* dtype_name(DType dt) {
    switch (dt) {
    case DType::F32:
        return "F32";
    case DType::F16:
        return "F16";

    case DType::Q4_0:
        return "Q4_0";
    case DType::Q4_1:
        return "Q4_1";

    case DType::Q5_0:
        return "Q5_0";
    case DType::Q5_1:
        return "Q5_1";

    case DType::Q8_0:
        return "Q8_0";
    case DType::Q8_1:
        return "Q8_1";

    case DType::Q2_K:
        return "Q2_K";
    case DType::Q3_K:
        return "Q3_K";
    case DType::Q4_K:
        return "Q4_K";
    case DType::Q5_K:
        return "Q5_K";
    case DType::Q6_K:
        return "Q6_K";
    case DType::Q8_K:
        return "Q8_K";

    case DType::IQ2_XXS:
        return "IQ2_XXS";
    case DType::IQ2_XS:
        return "IQ2_XS";
    case DType::IQ3_XXS:
        return "IQ3_XXS";
    case DType::IQ1_S:
        return "IQ1_S";
    case DType::IQ4_NL:
        return "IQ4_NL";
    case DType::IQ3_S:
        return "IQ3_S";
    case DType::IQ2_S:
        return "IQ2_S";
    case DType::IQ4_XS:
        return "IQ4_XS";
    case DType::IQ1_M:
        return "IQ1_M";
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
