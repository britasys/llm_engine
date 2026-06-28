// graph_executor.cpp
#include "llmengine/graph_executor.hpp"

#include <stdexcept>

namespace llmengine {

GraphExecutor::GraphExecutor(Backend& backend)
    : backend_(backend) {}

void GraphExecutor::execute(ggml_cgraph* graph) const {
    if (!graph) {
        throw std::runtime_error("Cannot execute a null graph.");
    }

    ggml_backend_graph_compute(backend_.get(), graph);
}

} // namespace llmengine