#include "llmengine/tokenizer.hpp"
#include "llmengine/gguf_loader.hpp"

#include <sstream>
#include <stdexcept>

namespace llmengine {

namespace {

std::string parse_next_element(const std::string& s, std::size_t& pos) {
    if (pos >= s.size())
        return "";

    if (s[pos] == '"') {
        // Quoted string element: scan to the matching closing quote.
        std::size_t start = pos + 1;
        std::size_t end = s.find('"', start);
        if (end == std::string::npos) {
            throw std::runtime_error("malformed tokenizer.ggml.tokens: unterminated quoted string");
        }
        std::string piece = s.substr(start, end - start);
        pos = end + 1; // skip closing quote
        // skip trailing ',' or ']'
        if (pos < s.size() && (s[pos] == ',' || s[pos] == ']'))
            ++pos;
        return piece;
    }

    // Unquoted (numeric/bool) element -- shouldn't occur for a token
    // array, but handled defensively rather than assuming.
    std::size_t end = s.find_first_of(",]", pos);
    if (end == std::string::npos)
        end = s.size();
    std::string piece = s.substr(pos, end - pos);
    pos = (end < s.size() && s[end] == ',') ? end + 1 : end + 1;
    return piece;
}

} // namespace

Tokenizer::Tokenizer(const GGUFLoader& loader) {
    auto it = loader.metadata().find("tokenizer.ggml.tokens");

    if (it == loader.metadata().end()) {
        throw std::runtime_error("GGUF missing tokenizer.ggml.tokens");
    }

    const std::string& raw = it->second;
    if (raw.size() < 2 || raw.front() != '[' || raw.back() != ']') {
        throw std::runtime_error("tokenizer.ggml.tokens is not in the expected array format");
    }

    std::size_t pos = 1; // skip leading '['
    TokenId id = 0;

    while (pos < raw.size() && raw[pos] != ']') {
        std::string piece = parse_next_element(raw, pos);
        add_token(id++, std::move(piece));
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