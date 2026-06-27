#include "llmengine/engine.hpp"
#include <stdexcept>

namespace llmengine {

Engine::Engine(Model& model, Tokenizer& tokenizer)
    : model_(model), tokenizer_(tokenizer),
      kv_cache_(model.config().max_seq_len, model.config().n_layers, model.config().n_heads,
                model.config().head_dim()) {}

TokenId Engine::pick_next_token(ggml_tensor* logits, const GenerationConfig& config) {
    const float* data = static_cast<const float*>(logits->data);
    std::size_t vocab_size = static_cast<std::size_t>(logits->ne[0]);

    if (config.temperature <= 0.0f) {
        return sampler_.argmax(data, vocab_size);
    }
    if (config.top_p < 1.0f) {
        return sampler_.sample_top_p(data, vocab_size, config.top_p, config.temperature);
    }
    if (config.top_k > 0) {
        return sampler_.sample_top_k(data, vocab_size, config.top_k, config.temperature);
    }
    return sampler_.sample(data, vocab_size, config.temperature);
}

void Engine::generate_text(const std::string& prompt, const GenerationConfig& config,
                           const TextPieceCallback& callback) {
    auto prompt_tokens = tokenizer_.encode(prompt);
    if (prompt_tokens.empty())
        return;

    int64_t pos = static_cast<int64_t>(kv_cache_.size());
    ggml_tensor* logits = nullptr;

    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        logits = model_.forward(prompt_tokens[i], pos, kv_cache_.k_cache(), kv_cache_.v_cache());
        if (i < prompt_tokens.size() - 1) {
            kv_cache_.increment_sequence();
            pos++;
        }
    }

    TokenId next_token = pick_next_token(logits, config);
    if (callback)
        callback(next_token, tokenizer_.decode(next_token));

    int32_t tokens_produced = 1;
    while (kv_cache_.size() < kv_cache_.capacity() - 1 && tokens_produced < config.max_new_tokens) {
        kv_cache_.increment_sequence();
        pos++;

        logits = model_.forward(next_token, pos, kv_cache_.k_cache(), kv_cache_.v_cache());
        next_token = pick_next_token(logits, config);
        tokens_produced++;

        if (callback)
            callback(next_token, tokenizer_.decode(next_token));
            
        if (next_token == 2)
            break;
    }
}

} // namespace llmengine