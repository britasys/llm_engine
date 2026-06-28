// backend.cpp
#include "llmengine/backend.hpp"

#include <stdexcept>
#include <utility>

namespace llmengine {

Backend::Backend() {
    backend_ = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);

    if (!backend_) {
        throw std::runtime_error("Failed to initialize ggml backend.");
    }
}

Backend::Backend(Backend&& other) noexcept : backend_(std::exchange(other.backend_, nullptr)) {}

Backend& Backend::operator=(Backend&& other) noexcept {
    if (this != &other) {
        if (backend_) {
            ggml_backend_free(backend_);
        }
        backend_ = std::exchange(other.backend_, nullptr);
    }
    return *this;
}

Backend::~Backend() {
    if (backend_) {
        ggml_backend_free(backend_);
    }
}

ggml_backend_t Backend::get() const noexcept {
    return backend_;
}

} // namespace llmengine