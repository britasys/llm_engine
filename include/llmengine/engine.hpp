#pragma once

#include <functional>
#include <ggml.h>
#include <string>
#include <vector>

#include "kv_cache.hpp"
#include "model.hpp"
#include "sampler.hpp"
#include "tokenizer.hpp" // Make sure your tokenizer file path matches this entry

namespace llmengine {

struct GenerationConfig {
    float temperature = 1.0f;
    int32_t top_k = 40;
    float top_p = 0.95f;
    int32_t max_new_tokens = 128; // FIX: Added missing property field
};

using TextPieceCallback = std::function<void(TokenId, const std::string&)>;

class Engine {
public:
    // FIX: Constructor signature accepts tokenizer reference directly
    Engine(Model& model, Tokenizer& tokenizer);

    void reset() noexcept { kv_cache_.clear(); } // FIX: Added missing clear method

    TokenId pick_next_token(ggml_tensor* logits, const GenerationConfig& config);

    void generate_text(const std::string& prompt, const GenerationConfig& config,
                       const TextPieceCallback& callback);

private:
    Model& model_;
    Tokenizer& tokenizer_;
    KVCache kv_cache_; // Managed directly inside engine context layout tracking
    Sampler sampler_;
};

} // namespace llmengine