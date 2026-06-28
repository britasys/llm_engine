// engine.cpp
#include "llmengine/engine.hpp"
#include <stdexcept>

namespace llmengine {

Engine::Engine(Model& model, Tokenizer& tokenizer)
    : model_(model), tokenizer_(tokenizer),
      kv_cache_(model.config().max_seq_len, model.config().n_layers, model.config().n_kv_heads, model.config().head_dim()) {}

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

void Engine::generate_text(const std::string& prompt, const GenerationConfig& config, const TextPieceCallback& callback) {

    std::string formatted_prompt = "<|im_start|>user\n" + prompt +
                                   "<|im_end|>\n"
                                   "<|im_start|>assistant\n";
    auto prompt_tokens = tokenizer_.encode(formatted_prompt);
    if (prompt_tokens.empty())
        return;

    ggml_tensor* logits = nullptr;

    //
    // Prefill
    //
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        int64_t pos = kv_cache_.size();
        logits = model_.forward(prompt_tokens[i], pos, kv_cache_.k_cache(), kv_cache_.v_cache());
        kv_cache_.increment_sequence();
    }

    //
    // First generated token
    //
    std::vector<TokenId> generated;
    generated.reserve(config.max_new_tokens);

    TokenId next_token = pick_next_token(logits, config);
    generated.push_back(next_token);

    if (callback) {
        callback(next_token, tokenizer_.decode(std::span<const TokenId>(generated.data(), generated.size())));
    }

    int32_t tokens_produced = 1;

    //
    // Decode loop
    //
    while (kv_cache_.size() < kv_cache_.capacity() - 1 && tokens_produced < config.max_new_tokens) {
        int64_t pos = kv_cache_.size();

        logits = model_.forward(next_token, pos, kv_cache_.k_cache(), kv_cache_.v_cache());

        kv_cache_.increment_sequence();

        next_token = pick_next_token(logits, config);
        generated.push_back(next_token);

        tokens_produced++;

        if (callback) {
            callback(next_token, tokenizer_.decode(std::span<const TokenId>(generated.data(), generated.size())));
        }

        if (next_token == tokenizer_.eos_token()) {
            break;
        }
    }
}

} // namespace llmengine