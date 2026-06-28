// tensor_loader.hpp
#pragma once

#include <ggml.h>
#include <string>

namespace llmengine {

class GGUFLoader;
class Context;

class TensorLoader {
public:
    TensorLoader(Context& context, GGUFLoader& loader);

    [[nodiscard]] ggml_tensor* load(const std::string& name) const;
    [[nodiscard]] bool contains(const std::string& name) const;

private:
    Context& context_;
    GGUFLoader& loader_;
};

} // namespace llmengine