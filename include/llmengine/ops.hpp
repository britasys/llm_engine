#pragma once

#include "tensor.hpp"

namespace llmengine::ops {

Tensor matmul(const Tensor& a, const Tensor& b);

Tensor add(const Tensor& a, const Tensor& b);

Tensor add_inplace(Tensor x, const Tensor& y);

Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps = 1e-5f);

Tensor silu(const Tensor& x);

Tensor softmax(const Tensor& x);

Tensor transpose(const Tensor& x);

} // namespace llmengine::ops