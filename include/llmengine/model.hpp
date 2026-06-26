// include/llmengine/model.hpp
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf_loader.hpp"
#include "kv_cache.hpp"
#include "rope_cache.hpp"
#include "scratch_arena.hpp"
#include "tensor.hpp"
#include "tokenizer.hpp"

namespace llmengine {

struct ModelConfig {
    int64_t vocab_size = 0;
    int64_t n_embd = 0;
    int64_t n_layers = 0;
    int64_t n_heads = 0;
    int64_t n_kv_heads = 0;
    int64_t n_ff = 0;
    int64_t max_seq_len = 0;
    float rope_theta = 10000.0f;
    float rms_eps = 1e-5f;

    [[nodiscard]] int64_t head_dim() const noexcept { return n_heads > 0 ? n_embd / n_heads : 0; }

    [[nodiscard]] static ModelConfig
    from_metadata(const std::unordered_map<std::string, std::string>& meta);
};

struct LayerWeights {
    Tensor attn_norm;
    Tensor wq, wk, wv;
    Tensor wo;

    Tensor ffn_norm;
    Tensor w_gate;
    Tensor w_up;
    Tensor w_down;
};

class Model {
public:
    explicit Model(GGUFLoader& loader);

    [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }

    [[nodiscard]] Tensor forward(TokenId token, int64_t pos, KVCache& kv_cache) const;

private:
    ModelConfig config_;

    RopeCache rope_cache_;

    mutable ScratchArena scratch_arena_;

    Tensor token_embd_;
    Tensor output_norm_;
    Tensor output_weight_;

    std::vector<LayerWeights> layers_;

    void attention(int64_t layer_idx, const Tensor& x_norm, int64_t pos, KVCache& kv_cache,
                   Tensor& out) const;
    void forward_layer(int64_t layer_idx, const Tensor& x, int64_t pos, KVCache& kv_cache,
                       Tensor& out) const;
    void feed_forward(int64_t layer_idx, const Tensor& x_norm, Tensor& out) const;
};

} // namespace llmengine