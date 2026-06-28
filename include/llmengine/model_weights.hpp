// model_weights.hpp
#pragma once

#include <vector>

#include <ggml.h>

namespace llmengine {

struct LayerWeights {
    ggml_tensor* attn_norm = nullptr;

    ggml_tensor* query_weight = nullptr;
    ggml_tensor* query_bias = nullptr;

    ggml_tensor* key_weight = nullptr;
    ggml_tensor* key_bias = nullptr;

    ggml_tensor* value_weight = nullptr;
    ggml_tensor* value_bias = nullptr;

    ggml_tensor* output_weight = nullptr;

    ggml_tensor* ffn_norm = nullptr;

    ggml_tensor* gate_weight = nullptr;
    ggml_tensor* up_weight = nullptr;
    ggml_tensor* down_weight = nullptr;
};

struct ModelWeights {
    ggml_tensor* token_embedding = nullptr;
    ggml_tensor* output_norm = nullptr;
    ggml_tensor* output_projection = nullptr;

    std::vector<LayerWeights> layers;
};

} // namespace llmengine