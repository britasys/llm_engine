#include "llmengine/tokenizer.hpp"

#include <sstream>
#include <stdexcept>

namespace llmengine {

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
    std::istringstream stream(text);

    std::vector<TokenId> output;

    std::string piece;

    while (stream >> piece) {
        auto it = piece_to_id_.find(piece);

        if (it == piece_to_id_.end()) {
            throw std::runtime_error("unknown token piece: " + piece);
        }

        output.push_back(it->second);
    }

    return output;
}

std::size_t Tokenizer::vocab_size() const noexcept {
    return id_to_piece_.size();
}

} // namespace llmengine