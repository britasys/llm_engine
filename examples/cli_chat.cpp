#include <cstdlib>
#include <iostream>
#include <string>

#include "llmengine/engine.hpp"
#include "llmengine/gguf_loader.hpp"
#include "llmengine/model_loader.hpp"
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

        std::cout << "Loading model...\n";

        Tokenizer tokenizer(loader);
        Model model = ModelLoader::load(loader);
        Engine engine(model, tokenizer);

        const auto& cfg = model.config();

        std::cout << "Model loaded.\n";
        std::cout << "vocab_size  : " << cfg.vocab_size << '\n';
        std::cout << "n_layers    : " << cfg.n_layers << '\n';
        std::cout << "n_embd      : " << cfg.n_embd << '\n';
        std::cout << "n_heads     : " << cfg.n_heads << '\n';
        std::cout << "max_seq_len : " << cfg.max_seq_len << "\n\n";

        GenerationConfig generation;
        generation.max_new_tokens = 128;
        generation.temperature = 0.8f;
        generation.top_k = 40;
        generation.top_p = 0.95f;

        std::string prompt;

        while (true) {
            std::cout << "> ";

            if (!std::getline(std::cin, prompt))
                break;

            if (prompt.empty())
                continue;

            if (prompt == "/exit")
                break;

            if (prompt == "/reset") {
                engine.reset();
                std::cout << "[context cleared]\n";
                continue;
            }

            try {
                std::string previous;

                engine.generate_text(prompt, generation, [&](TokenId id, const std::string& piece) {
                    if (id != tokenizer.eos_token()) {
                        std::cout << piece.substr(previous.size());
                        std::cout.flush();
                    }

                    previous = piece;
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
