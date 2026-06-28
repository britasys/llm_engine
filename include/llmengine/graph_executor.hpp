// graph_executor.hpp
#pragma once

#include <ggml-backend.h>
#include <ggml.h>

#include "backend.hpp"

namespace llmengine {

class GraphExecutor {
public:
    explicit GraphExecutor(Backend& backend);

    GraphExecutor(const GraphExecutor&) = delete;
    GraphExecutor& operator=(const GraphExecutor&) = delete;

    void execute(ggml_cgraph* graph) const;

private:
    Backend& backend_;
};

} // namespace llmengine