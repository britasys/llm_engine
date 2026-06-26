#include "llmengine/engine.hpp"

#include <stdexcept>

namespace llmengine {

Engine::Engine(Model& model, Tokenizer& tokenizer, uint32_t seed)
    : model_(model), tokenizer_(tokenizer), sampler_(seed),
      kv_cache_(model.config().max_seq_len, model.config().n_layers,
                model.config().n_kv_heads > 0 ? model.config().n_kv_heads : model.config().n_heads,
                model.config().head_dim()) {}

TokenId Engine::pick_next_token(const Tensor& logits, const GenerationConfig& config) {
    if (config.temperature <= 0.0f) {
        return sampler_.argmax(logits);
    }
    if (config.top_k > 0) {
        return sampler_.sample_top_k(logits, config.top_k, config.temperature);
    }
    if (config.top_p > 0.0f) {
        return sampler_.sample_top_p(logits, config.top_p, config.temperature);
    }
    return sampler_.sample(logits, config.temperature);
}

std::vector<TokenId> Engine::generate(const std::vector<TokenId>& prompt,
                                      const GenerationConfig& config) {
    if (prompt.empty()) {
        throw std::runtime_error("prompt cannot be empty");
    }

    reset();

    const int64_t capacity = kv_cache_.capacity();
    if (static_cast<int64_t>(prompt.size()) > capacity) {
        throw std::runtime_error("prompt length exceeds model's max sequence length");
    }

    Tensor logits;
    int64_t pos = 0;

    // Prefill: run every prompt token, keeping only the final logits.
    for (TokenId token : prompt) {
        logits = model_.forward(token, pos, kv_cache_);
        ++pos;
    }

    std::vector<TokenId> generated; // NOTE: only the new tokens, not the prompt
    generated.reserve(static_cast<std::size_t>(config.max_new_tokens));

    for (int32_t i = 0; i < config.max_new_tokens; ++i) {
        TokenId next_token = pick_next_token(logits, config);

        if (config.eos_token >= 0 && next_token == config.eos_token) break;

        generated.push_back(next_token);

        if (pos >= capacity) break; // out of context window: stop before forward() overruns kv_cache_

        logits = model_.forward(next_token, pos, kv_cache_);
        ++pos;
    }

    return generated;
}

std::string Engine::generate_text(const std::string& prompt, const GenerationConfig& config) {
    auto prompt_tokens = tokenizer_.encode(prompt);

    auto output_tokens = generate(prompt_tokens, config);

    return tokenizer_.decode(output_tokens);
}

void Engine::reset() {
    kv_cache_.clear();
}

} // namespace llmengine