#include "llmengine/ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llmengine::ops {

void matmul(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.dtype() != DType::F32 || b.dtype() != DType::F32 || out.dtype() != DType::F32)
        throw std::runtime_error("matmul requires F32");

    if (a.ndim() != 2 || b.ndim() != 2 || out.ndim() != 2)
        throw std::runtime_error("matmul requires 2D tensors");

    const int64_t m = a.dim(0);
    const int64_t k = a.dim(1);

    if (k != b.dim(0))
        throw std::runtime_error("matmul shape mismatch");

    const int64_t n = b.dim(1);

    if (out.dim(0) != m || out.dim(1) != n)
        throw std::runtime_error("matmul output shape mismatch");

    auto A = a.as_f32();
    auto B = b.as_f32();
    auto C = out.as_f32();

    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {

            float sum = 0.f;

            for (int64_t p = 0; p < k; ++p)
                sum += A[i * k + p] * B[p * n + j];

            C[i * n + j] = sum;
        }
    }
}

void add(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.shape() != b.shape() || out.shape() != a.shape())
        throw std::runtime_error("add shape mismatch");

    auto A = a.as_f32();
    auto B = b.as_f32();
    auto O = out.as_f32();

    for (int64_t i = 0; i < a.numel(); ++i)
        O[i] = A[i] + B[i];
}

void add_inplace(Tensor& x, const Tensor& y) {
    if (x.shape() != y.shape())
        throw std::runtime_error("add shape mismatch");

    auto X = x.as_f32();
    auto Y = y.as_f32();

    for (int64_t i = 0; i < x.numel(); ++i)
        X[i] += Y[i];
}

void rms_norm(const Tensor& x, const Tensor& weight, Tensor& out, float eps) {
    if (x.ndim() != 2)
        throw std::runtime_error("rms_norm expects 2D tensor");

    const int64_t rows = x.dim(0);
    const int64_t cols = x.dim(1);

    if (weight.numel() != cols)
        throw std::runtime_error("rms_norm weight mismatch");

    if (out.shape() != x.shape())
        throw std::runtime_error("rms_norm output shape mismatch");

    auto X = x.as_f32();
    auto W = weight.as_f32();
    auto O = out.as_f32();

    for (int64_t r = 0; r < rows; ++r) {

        float ss = 0.f;

        for (int64_t c = 0; c < cols; ++c) {
            float v = X[r * cols + c];
            ss += v * v;
        }

        float scale = 1.f / std::sqrt(ss / static_cast<float>(cols) + eps);

        for (int64_t c = 0; c < cols; ++c)
            O[r * cols + c] = X[r * cols + c] * scale * W[c];
    }
}

void silu_inplace(Tensor& x) {
    auto X = x.as_f32();

    for (int64_t i = 0; i < x.numel(); ++i)
        X[i] = X[i] / (1.f + std::exp(-X[i]));
}

void silu(const Tensor& x, Tensor& out) {    
    auto src = x.as_f32();
    auto dst = out.as_f32();

    for (int64_t i = 0; i < x.numel(); ++i) {
        float val = src[i];
        dst[i] = val / (1.0f + std::exp(-val));
    }
}

void softmax(const Tensor& x, Tensor& out) {
    if (x.ndim() != 2)
        throw std::runtime_error("softmax expects 2D tensor");

    if (out.shape() != x.shape())
        throw std::runtime_error("softmax output shape mismatch");

    auto X = x.as_f32();
    auto O = out.as_f32();

    const int64_t rows = x.dim(0);
    const int64_t cols = x.dim(1);

    for (int64_t r = 0; r < rows; ++r) {

        float max_v = X[r * cols];

        for (int64_t c = 1; c < cols; ++c)
            max_v = std::max(max_v, X[r * cols + c]);

        float sum = 0.f;

        for (int64_t c = 0; c < cols; ++c) {
            float e = std::exp(X[r * cols + c] - max_v);
            O[r * cols + c] = e;
            sum += e;
        }

        float inv = 1.f / sum;

        for (int64_t c = 0; c < cols; ++c)
            O[r * cols + c] *= inv;
    }
}

void transpose(const Tensor& x, Tensor& out) {
    if (x.ndim() != 2)
        throw std::runtime_error("transpose expects 2D tensor");

    const int64_t rows = x.dim(0);
    const int64_t cols = x.dim(1);

    if (out.dim(0) != cols || out.dim(1) != rows)
        throw std::runtime_error("transpose output shape mismatch");

    auto X = x.as_f32();
    auto O = out.as_f32();

    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c)
            O[c * rows + r] = X[r * cols + c];
}

} // namespace llmengine::ops