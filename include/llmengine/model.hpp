#pragma once

#include <cstdint>
#include <ggml-backend.h>
#include <ggml.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf_loader.hpp"
#include "scratch_arena.hpp"

namespace llmengine {

using TokenId = int32_t;

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
    [[nodiscard]] static ModelConfig from_metadata(const GGUFLoader& loader);
};

struct LayerWeights {
    ggml_tensor* attn_norm = nullptr;
    ggml_tensor* wq = nullptr;
    ggml_tensor* wk = nullptr;
    ggml_tensor* wv = nullptr;
    ggml_tensor* wo = nullptr;
    ggml_tensor* bq = nullptr;
    ggml_tensor* bk = nullptr;
    ggml_tensor* bv = nullptr;
    ggml_tensor* ffn_norm = nullptr;
    ggml_tensor* w_gate = nullptr;
    ggml_tensor* w_up = nullptr;
    ggml_tensor* w_down = nullptr;
};

class Model {
public:
    explicit Model(GGUFLoader& loader);
    ~Model();

    [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }
    
    [[nodiscard]] ggml_tensor* forward(TokenId token, int64_t pos, ggml_tensor* k_cache, ggml_tensor* v_cache) const;

private:
    ModelConfig config_;
    ggml_context* weights_ctx_ = nullptr;
    mutable ScratchArena scratch_arena_;

    ggml_backend_t backend_;

    ggml_tensor* token_embd_ = nullptr;
    ggml_tensor* output_norm_ = nullptr;
    ggml_tensor* output_weight_ = nullptr;

    std::vector<LayerWeights> layers_;
};

} // namespace llmengine