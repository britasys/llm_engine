// tensor_loader.cpp
#include "llmengine/tensor_loader.hpp"

#include <stdexcept>

#include "llmengine/context.hpp"
#include "llmengine/gguf_loader.hpp"

namespace llmengine {

TensorLoader::TensorLoader(Context& context, GGUFLoader& loader) : context_(context), loader_(loader) {}

bool TensorLoader::contains(const std::string& name) const {
    return loader_.has_tensor(name);
}

ggml_tensor* TensorLoader::load(const std::string& name) const {
    if (!loader_.has_tensor(name)) {
        throw std::runtime_error("Missing tensor: " + name);
    }

    const auto& info = loader_.tensor_info(name);

    ggml_tensor* tensor = nullptr;

    switch (info.shape.size()) {
    case 1:
        tensor = ggml_new_tensor_1d(context_.get(), info.dtype, info.shape[0]);
        break;

    case 2:
        tensor = ggml_new_tensor_2d(context_.get(), info.dtype, info.shape[0], info.shape[1]);
        break;

    case 3:
        tensor = ggml_new_tensor_3d(context_.get(), info.dtype, info.shape[0], info.shape[1], info.shape[2]);
        break;

    case 4:
        tensor = ggml_new_tensor_4d(context_.get(), info.dtype, info.shape[0], info.shape[1], info.shape[2], info.shape[3]);
        break;

    default:
        throw std::runtime_error("Unsupported tensor rank: " + name);
    }

    if (!tensor) {
        throw std::runtime_error("Failed to allocate tensor: " + name);
    }

    tensor->data = const_cast<void*>(loader_.tensor_data(name));

    return tensor;
}

} // namespace llmengine