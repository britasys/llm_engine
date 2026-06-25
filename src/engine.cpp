#include "llmengine/engine.hpp"

#include <stdexcept>

namespace llmengine {

Engine::Engine(Model& model, Tokenizer& tokenizer, uint32_t seed)
    : model_(model), tokenizer_(tokenizer), sampler_(seed),
      kv_cache_(model.config().max_seq_len, model.config().n_layers,
                model.config().n_kv_heads > 0 ? model.config().n_kv_heads : model.config().n_heads,
                model.config().head_dim()) {}

std::vector<TokenId> Engine::generate(const std::vector<TokenId>& prompt,
                                      const GenerationConfig& config) {
    if (prompt.empty()) {
        throw std::runtime_error("prompt cannot be empty");
    }

    reset();

    Tensor logits;

    int64_t pos = 0;

    for (TokenId token : prompt) {
        logits = model_.forward(token, pos, kv_cache_);

        ++pos;
    }

    std::vector<TokenId> output = prompt;

    for (int32_t i = 0; i < config.max_new_tokens; ++i) {

        TokenId next_token;

        if (config.top_k > 0) {
            next_token = sampler_.sample_top_k(logits, config.top_k, config.temperature);
        } else {
            next_token = sampler_.sample(logits, config.temperature);
        }

        output.push_back(next_token);

        logits = model_.forward(next_token, pos, kv_cache_);

        ++pos;
    }

    return output;
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