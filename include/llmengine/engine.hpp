#pragma once

#include <functional>
#include <ggml.h>
#include <string>
#include <vector>

#include "kv_cache.hpp"
#include "model.hpp"
#include "sampler.hpp"
#include "tokenizer.hpp"

namespace llmengine {

struct GenerationConfig {
    float temperature = 1.0f;
    int32_t top_k = 40;
    float top_p = 0.95f;
    int32_t max_new_tokens = 128;
};

using TextPieceCallback = std::function<void(TokenId, const std::string&)>;

class Engine {
public:
    Engine(Model& model, Tokenizer& tokenizer);

    void reset() noexcept { kv_cache_.clear(); }

    TokenId pick_next_token(ggml_tensor* logits, const GenerationConfig& config);

    void generate_text(const std::string& prompt, const GenerationConfig& config,
                       const TextPieceCallback& callback);

private:
    Model& model_;
    Tokenizer& tokenizer_;
    KVCache kv_cache_;
    Sampler sampler_;
};

} // namespace llmengine