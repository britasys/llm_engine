#include "llmengine/model.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llmengine {

Model::Model(ModelConfig config, Backend backend, Context weight_context, ScratchArena scratch_arena, ModelWeights weights)
    : config_(std::move(config)), backend_(std::move(backend)), executor_(backend_), weight_context_(std::move(weight_context)),
      scratch_arena_(std::move(scratch_arena)), weights_(std::move(weights)) {
    blocks_.reserve(weights_.layers.size());
    for (const auto& layer : weights_.layers) {
        blocks_.emplace_back(config_, layer);
    }
}

const ModelConfig& Model::config() const noexcept {
    return config_;
}

ggml_tensor* Model::embed(ggml_context* ctx, TokenId token) const {

    ggml_tensor* token_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    static_cast<int32_t*>(token_tensor->data)[0] = token;

    return ggml_get_rows(ctx, weights_.token_embedding, token_tensor);
}

ggml_tensor* Model::output(ggml_context* ctx, ggml_tensor* hidden) const {
    hidden = ggml_rms_norm(ctx, hidden, config_.rms_eps);
    hidden = ggml_mul(ctx, hidden, weights_.output_norm);
    return ggml_mul_mat(ctx, weights_.output_projection, hidden);
}

ggml_tensor* Model::forward(TokenId token, int64_t position, ggml_tensor* key_cache, ggml_tensor* value_cache) {
    scratch_arena_.reset();

    ggml_context* ctx = scratch_arena_.ctx();
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_tensor* hidden = embed(ctx, token);

    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        hidden = blocks_[i].forward(ctx, graph, hidden, position, key_cache, value_cache, static_cast<int64_t>(i));
    }

    ggml_tensor* logits = output(ctx, hidden);
    ggml_build_forward_expand(graph, logits);

    executor_.execute(graph);

    return logits;
}

} // namespace llmengine