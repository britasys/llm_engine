#include "llmengine/model.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llmengine {

namespace {
ggml_tensor* load_tensor_native(ggml_context* ctx, GGUFLoader& loader, const std::string& name) {
    if (!loader.has_tensor(name))
        throw std::runtime_error("Missing tensor: " + name);
    const auto& info = loader.tensor_info(name);

    ggml_tensor* t = nullptr;
    if (info.shape.size() == 1)
        t = ggml_new_tensor_1d(ctx, info.dtype, info.shape[0]);
    else if (info.shape.size() == 2)
        t = ggml_new_tensor_2d(ctx, info.dtype, info.shape[0], info.shape[1]);
    else if (info.shape.size() == 3)
        t = ggml_new_tensor_3d(ctx, info.dtype, info.shape[0], info.shape[1], info.shape[2]);
    else if (info.shape.size() == 4)
        t = ggml_new_tensor_4d(ctx, info.dtype, info.shape[0], info.shape[1], info.shape[2], info.shape[3]);

    if (!t)
        throw std::runtime_error("Failed to allocate weight layout: " + name);
    t->data = const_cast<void*>(loader.tensor_data(name));
    return t;
}

std::size_t calculate_scratch_size(const ModelConfig& config) {
    (void)config;
    return 128ULL * 1024ULL * 1024ULL;
}

void inline_assign_i32(ggml_tensor* t, size_t index, int32_t val) {
    if (t && t->data) {
        static_cast<int32_t*>(t->data)[index] = val;
    }
}
} // namespace

Model::Model(GGUFLoader& loader)
    : config_(ModelConfig::from_metadata(loader)), executor_(backend_),
      weight_context_(loader.tensors().size() * ggml_tensor_overhead() + 4096, true), scratch_arena_(calculate_scratch_size(config_)) {
    load_weights(loader);

    blocks_.reserve(config_.n_layers);

    for (const auto& layer : weights_.layers) {
        blocks_.emplace_back(config_, layer);
    }
}

const ModelConfig& Model::config() const noexcept {
    return config_;
}

void Model::load_weights(GGUFLoader& loader) {
    TensorLoader tensor_loader(weight_context_, loader);

    weights_.token_embedding = tensor_loader.load("token_embd.weight");

    weights_.output_norm = tensor_loader.load("output_norm.weight");

    if (tensor_loader.contains("output.weight")) {
        weights_.output_projection = tensor_loader.load("output.weight");
    } else {
        weights_.output_projection = weights_.token_embedding;
    }

    weights_.layers.reserve(config_.n_layers);

    for (int64_t i = 0; i < config_.n_layers; ++i) {
        const std::string prefix = "blk." + std::to_string(i) + ".";

        LayerWeights layer;

        layer.attn_norm = tensor_loader.load(prefix + "attn_norm.weight");

        layer.query_weight = tensor_loader.load(prefix + "attn_q.weight");

        layer.query_bias = tensor_loader.load(prefix + "attn_q.bias");

        layer.key_weight = tensor_loader.load(prefix + "attn_k.weight");

        layer.key_bias = tensor_loader.load(prefix + "attn_k.bias");

        layer.value_weight = tensor_loader.load(prefix + "attn_v.weight");

        layer.value_bias = tensor_loader.load(prefix + "attn_v.bias");

        layer.output_weight = tensor_loader.load(prefix + "attn_output.weight");

        layer.ffn_norm = tensor_loader.load(prefix + "ffn_norm.weight");

        layer.gate_weight = tensor_loader.load(prefix + "ffn_gate.weight");

        layer.up_weight = tensor_loader.load(prefix + "ffn_up.weight");

        layer.down_weight = tensor_loader.load(prefix + "ffn_down.weight");

        weights_.layers.push_back(layer);
    }
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