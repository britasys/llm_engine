// backend.hpp
#pragma once

#include <ggml-backend.h>

namespace llmengine {

class Backend {
public:
    Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    Backend(Backend&& other) noexcept;
    Backend& operator=(Backend&& other) noexcept;
    ~Backend();

    [[nodiscard]] ggml_backend_t get() const noexcept;

private:
    ggml_backend_t backend_ = nullptr;
};

} // namespace llmengine