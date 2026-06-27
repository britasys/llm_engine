#include "llmengine/gguf_loader.hpp"
#include "llmengine/tokenizer.hpp"

#include <stdexcept>
#include <string>

namespace llmengine {

namespace {

TokenId find_token(const GGUFLoader& loader, const std::string& key) {
    try {
        const auto& value = loader.get_meta_string(key);

        return static_cast<TokenId>(std::stoi(value));
    } catch (...) {
        return -1;
    }
}

} // namespace

void Tokenizer::load_special_tokens(const GGUFLoader& loader) {
    bos_token_ = find_token(loader, "tokenizer.ggml.bos_token_id");

    eos_token_ = find_token(loader, "tokenizer.ggml.eos_token_id");

    unk_token_ = find_token(loader, "tokenizer.ggml.unknown_token_id");

    pad_token_ = find_token(loader, "tokenizer.ggml.padding_token_id");
}

} // namespace llmengine