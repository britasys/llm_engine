// cli_chat.cpp
//
// Simple interactive chat client for llm-engine.
//
// Usage:
//
//   ./llmengine_chat model.gguf
//
// Commands:
//
//   /exit      quit
//   /reset     clear KV cache/history
//
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "llmengine/engine.hpp"
#include "llmengine/gguf_loader.hpp"
#include "llmengine/model.hpp"
#include "llmengine/tokenizer.hpp"

using namespace llmengine;

namespace {

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " <model.gguf>\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    try {
        GGUFLoader loader(argv[1]);

        loader.load();

        std::cout << "GGUF loaded successfully\n\n";
        std::cout << "version       : " << loader.version() << '\n';
        std::cout << "tensors       : " << loader.tensor_count() << '\n';
        std::cout << "metadata      : " << loader.metadata_count() << "\n\n";
        std::cout << "Metadata:\n";

        // for (const auto& [key, value] : loader.metadata()) {
        //     std::cout << "  " << key << '\n';
        // }

        // std::cout << "\nFirst tensors:\n";

        // int count = 0;

        // for (const auto& [name, info] : loader.tensors()) {
        //     std::cout << "  " << name << '\n';

        //     std::cout << "    shape: ";

        //     for (auto dim : info.shape) {
        //         std::cout << dim << " ";
        //     }

        //     std::cout << '\n';

        //     std::cout << "    dtype: " << static_cast<int>(info.dtype) << '\n';

        //     std::cout << "    offset: " << info.offset << '\n';
        //     std::cout << "    nbytes: " << info.nbytes << "\n\n";

        //     if (++count == 10)
        //         break;
        // }

        std::cout << "Loading model...\n";

        Model model(loader);

        Tokenizer tokenizer(loader);

        Engine engine(model, tokenizer);

        std::cout << "Model loaded.\n";
        std::cout << "vocab_size  = " << model.config().vocab_size << '\n';
        std::cout << "n_layers    = " << model.config().n_layers << '\n';
        std::cout << "n_embd      = " << model.config().n_embd << '\n';
        std::cout << "max_seq_len = " << model.config().max_seq_len << "\n\n";

        GenerationConfig gen_cfg;
        gen_cfg.max_new_tokens = 128;
        gen_cfg.temperature = 0.8f;
        gen_cfg.top_k = 40;
        gen_cfg.top_p = 0.95f;

        std::string line;

        while (true) {
            std::cout << "> ";

            if (!std::getline(std::cin, line)) {
                break;
            }

            if (line.empty()) {
                continue;
            }

            if (line == "/exit") {
                break;
            }

            if (line == "/reset") {
                engine.reset();

                std::cout << "[context cleared]\n";

                continue;
            }

            try {
                std::cout << std::flush; // ensure "> " prompt itself wasn't buffered either

                engine.generate_text(line, gen_cfg, [](TokenId id, const std::string& piece) {
                    std::cout << piece;
                    std::cout.flush();
                });

                std::cout << "\n\n";
            } catch (const std::exception& e) {
                std::cerr << "\ngeneration error: " << e.what() << '\n';
            }
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << '\n';

        return EXIT_FAILURE;
    }
}
