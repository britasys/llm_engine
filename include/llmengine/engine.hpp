#pragma once

#include "kv_cache.hpp"
#include "model.hpp"
#include "sampler.hpp"
#include "tokenizer.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace llmengine {

struct GenerationConfig {
    int32_t max_new_tokens = 128;
    float temperature = 0.8f;
    int32_t top_k = 40;
    float top_p = 0.95f;
    TokenId eos_token = -1; // -1 disables early stopping
};

class Engine {
public:
    Engine(Model& model, Tokenizer& tokenizer, uint32_t seed = 42);

    using TokenCallback = std::function<void(TokenId, const std::string&)>;
    std::vector<TokenId> generate(const std::vector<TokenId>& prompt,
                                  const GenerationConfig& config = {},
                                  const TokenCallback& on_token = nullptr);
    [[nodiscard]]
    std::string generate_text(const std::string& prompt, const GenerationConfig& config = {},
                              const TokenCallback& on_token = nullptr);

    void reset();

private:
    Model& model_;
    Tokenizer& tokenizer_;
    Sampler sampler_;
    KVCache kv_cache_;

    [[nodiscard]]
    TokenId pick_next_token(const Tensor& logits, const GenerationConfig& config);
};

} // namespace llmengine