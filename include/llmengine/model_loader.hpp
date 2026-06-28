#pragma once

#include "gguf_loader.hpp"
#include "model.hpp"

namespace llmengine {

class ModelLoader {
public:
    // Loads config + weights from a GGUF file and returns a ready-to-use Model.
    [[nodiscard]] static Model load(GGUFLoader& loader);

private:
    [[nodiscard]] static ModelWeights load_weights(Context& weight_context, GGUFLoader& loader, const ModelConfig& config);

    [[nodiscard]] static std::size_t calculate_scratch_size(const ModelConfig& config);
};

} // namespace llmengine