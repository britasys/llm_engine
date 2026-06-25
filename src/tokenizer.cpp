#include "llmengine/tokenizer.hpp"
#include "llmengine/gguf_loader.hpp"

#include <sstream>
#include <stdexcept>

namespace llmengine {

Tokenizer::Tokenizer(const GGUFLoader& loader) {
    auto it = loader.metadata().find("tokenizer.ggml.tokens");

    if (it == loader.metadata().end()) {
        throw std::runtime_error("GGUF missing tokenizer.ggml.tokens");
    }

    const std::string& tokens = it->second;

    std::size_t pos = 0;
    TokenId id = 0;

    while (pos < tokens.size()) {
        auto next = tokens.find(',', pos);

        std::string piece =
            tokens.substr(pos, next == std::string::npos ? std::string::npos : next - pos);

        add_token(id++, piece);

        if (next == std::string::npos)
            break;

        pos = next + 1;
    }
}

void Tokenizer::add_token(TokenId id, std::string piece) {
    piece_to_id_[piece] = id;
    id_to_piece_[id] = std::move(piece);
}

bool Tokenizer::contains(TokenId id) const {
    return id_to_piece_.contains(id);
}

std::string Tokenizer::decode(TokenId id) const {
    auto it = id_to_piece_.find(id);

    if (it == id_to_piece_.end()) {
        throw std::runtime_error("unknown token id");
    }

    return it->second;
}

std::string Tokenizer::decode(const std::vector<TokenId>& tokens) const {
    std::string result;

    for (auto id : tokens) {
        result += decode(id);
    }

    return result;
}

std::vector<TokenId> Tokenizer::encode(const std::string& text) const {
    std::vector<TokenId> output;

    for (char c : text) {
        std::string piece(1, c);

        auto it = piece_to_id_.find(piece);

        if (it != piece_to_id_.end()) {
            output.push_back(it->second);
        }
    }

    return output;
}

std::size_t Tokenizer::vocab_size() const noexcept {
    return id_to_piece_.size();
}

} // namespace llmengine