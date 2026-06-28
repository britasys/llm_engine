// transformer_block.hpp
#pragma once

#include <cstdint>

#include <ggml.h>

#include "backend.hpp"
#include "kv_cache.hpp"
#include "model_weights.hpp"
#include "model_config.hpp"

namespace llmengine {

class TransformerBlock {
public:
    TransformerBlock(const ModelConfig& config, const LayerWeights& weights);

    [[nodiscard]] ggml_tensor* forward(ggml_context* ctx, ggml_cgraph* graph, ggml_tensor* x, int64_t pos, ggml_tensor* k_cache, ggml_tensor* v_cache,
                                       int64_t layer_index) const;

private:
    const ModelConfig& config_;
    const LayerWeights& weights_;
};

} // namespace llmengine