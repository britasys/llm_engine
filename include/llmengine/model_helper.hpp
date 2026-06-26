# pragma once

#include "tensor.hpp"
#include "gguf_loader.hpp"

#include <unordered_map>

namespace llmengine {

float f16_to_f32(uint16_t h);

Tensor dequantize_q4_0_to_f32(const Tensor& t);

Tensor dequantize_q4_1_to_f32(const Tensor& t);

Tensor dequantize_q5_1_to_f32(const Tensor& t);

Tensor dequantize_q2_k_to_f32(const Tensor& t);

Tensor dequantize_q3_k_to_f32(const Tensor& t);

void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m);

Tensor dequantize_q4_k_to_f32(const Tensor& t);

Tensor dequantize_q5_k_to_f32(const Tensor& t);

Tensor dequantize_q8_k_to_f32(const Tensor& t);

Tensor dequantize_q6_k_to_f32(const Tensor& t);

Tensor dequantize_q5_0_to_f32(const Tensor& t);

Tensor dequantize_q8_0_to_f32(const Tensor& t);

Tensor load_f32(GGUFLoader& loader, const std::string& name);

int64_t meta_int(const std::unordered_map<std::string, std::string>& meta, const std::string& key, int64_t fallback);

float meta_float(const std::unordered_map<std::string, std::string>& meta, const std::string& key,
                 float fallback);

} // namespace llmengine