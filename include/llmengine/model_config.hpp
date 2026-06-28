// model_config.hpp
#pragma once

#include <cstdint>

#include "gguf_loader.hpp"

namespace llmengine {

using TokenId = int32_t;

namespace {

int64_t count_tokenizer_vocab(const GGUFLoader& loader) {
    try {
        const auto& tokens = loader.get_meta_array("tokenizer.ggml.tokens");

        return static_cast<int64_t>(tokens.values.size());
    } catch (const std::exception&) {
        return 0;
    }
}

} // namespace

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

    [[nodiscard]] static ModelConfig from_metadata(const GGUFLoader& loader) {
        ModelConfig c;

        std::string arch = loader.get_meta_string("general.architecture");

        try {
            c.vocab_size = loader.get_meta_int(arch + ".vocab_size");
        } catch (...) {
            c.vocab_size = count_tokenizer_vocab(loader);
        }

        c.n_embd = loader.get_meta_int(arch + ".embedding_length");
        c.n_layers = loader.get_meta_int(arch + ".block_count");
        c.n_heads = loader.get_meta_int(arch + ".attention.head_count");
        c.n_kv_heads = loader.get_meta_int(arch + ".attention.head_count_kv");
        c.n_ff = loader.get_meta_int(arch + ".feed_forward_length");
        c.max_seq_len = loader.get_meta_int(arch + ".context_length");
        c.rope_theta = loader.get_meta_float(arch + ".rope.freq_base");
        c.rms_eps = loader.get_meta_float(arch + ".attention.layer_norm_rms_epsilon");

        if (c.vocab_size <= 0 || c.n_embd <= 0 || c.n_layers <= 0 || c.n_heads <= 0) {
            throw std::runtime_error("ModelConfig::from_metadata: architecture configuration missing.");
        }
        if (c.n_kv_heads <= 0 || c.n_heads % c.n_kv_heads != 0) {
            throw std::runtime_error("ModelConfig::from_metadata: invalid head_count_kv.");
        }
        if (c.n_embd % c.n_heads != 0) {
            throw std::runtime_error("ModelConfig::from_metadata: embedding_length not divisible by head_count.");
        }

        return c;
    }
};

} // namespace llmengine