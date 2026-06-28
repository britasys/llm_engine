#include "llmengine/model_loader.hpp"

namespace llmengine {

Model ModelLoader::load(GGUFLoader& loader) {
    ModelConfig config = ModelConfig::from_metadata(loader);

    Backend backend;
    Context weight_context(loader.tensors().size() * ggml_tensor_overhead() + 4096, /*no_alloc=*/true);
    ScratchArena scratch_arena(calculate_scratch_size(config));

    ModelWeights weights = load_weights(weight_context, loader, config);

    return Model(std::move(config), std::move(backend), std::move(weight_context), std::move(scratch_arena), std::move(weights));
}

ModelWeights ModelLoader::load_weights(Context& weight_context, GGUFLoader& loader, const ModelConfig& config) {
    TensorLoader tensor_loader(weight_context, loader);

    ModelWeights weights;

    weights.token_embedding = tensor_loader.load("token_embd.weight");
    weights.output_norm = tensor_loader.load("output_norm.weight");

    if (tensor_loader.contains("output.weight")) {
        weights.output_projection = tensor_loader.load("output.weight");
    } else {
        weights.output_projection = weights.token_embedding;
    }

    weights.layers.reserve(static_cast<std::size_t>(config.n_layers));

    for (int64_t i = 0; i < config.n_layers; ++i) {
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

        weights.layers.push_back(layer);
    }

    return weights;
}

std::size_t ModelLoader::calculate_scratch_size(const ModelConfig& config) {
    (void)config;
    return 128ULL * 1024ULL * 1024ULL;
}

} // namespace llmengine