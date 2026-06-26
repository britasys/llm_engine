#pragma once

#include "tensor.hpp"

namespace llmengine::ops {

void matmul(const Tensor& a, const Tensor& b, Tensor& out);

void add(const Tensor& a, const Tensor& b, Tensor& out);

void add_inplace(Tensor x, const Tensor& y, Tensor& out);

void rms_norm(const Tensor& x, const Tensor& weight, float eps = 1e-5f, Tensor& out);

void silu(const Tensor& x, Tensor& out);

void softmax(const Tensor& x, Tensor& out);

void transpose(const Tensor& x, Tensor& out);

} // namespace llmengine::ops