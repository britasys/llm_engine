// model.hpp
#pragma once

#include <cstdint>
#include <vector>

#include <ggml.h>

#include "model_config.hpp"
#include "backend.hpp"
#include "context.hpp"
#include "gguf_loader.hpp"
#include "graph_executor.hpp"
#include "model_weights.hpp"
#include "scratch_arena.hpp"
#include "tensor_loader.hpp"
#include "transformer_block.hpp"

namespace llmengine {

using TokenId = int32_t;

class Model {
public:
    explicit Model(GGUFLoader& loader);

    [[nodiscard]] const ModelConfig& config() const noexcept;

    [[nodiscard]] ggml_tensor* forward(
        TokenId token,
        int64_t position,
        ggml_tensor* key_cache,
        ggml_tensor* value_cache);

private:
    void load_weights(GGUFLoader& loader);

    ggml_tensor* embed(
        ggml_context* ctx,
        TokenId token) const;

    ggml_tensor* output(
        ggml_context* ctx,
        ggml_tensor* hidden) const;

private:
    ModelConfig config_;

    Backend backend_;
    GraphExecutor executor_;

    Context weight_context_;
    ScratchArena scratch_arena_;

    ModelWeights weights_;

    std::vector<TransformerBlock> blocks_;
};

} // namespace llmengine