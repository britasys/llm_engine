// model_loader.hpp
#pragma once

#include "gguf_loader.hpp"
#include "model.hpp"

namespace llmengine {

class ModelLoader {
public:
    [[nodiscard]] static Model load(GGUFLoader& loader);

private:
    [[nodiscard]] static ModelWeights load_weights(Context& weight_context, GGUFLoader& loader, const ModelConfig& config);
    [[nodiscard]] static std::size_t calculate_scratch_size(const ModelConfig& config);
};

} // namespace llmengine