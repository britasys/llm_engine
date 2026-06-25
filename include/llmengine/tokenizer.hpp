#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmengine {

using TokenId = int32_t;

class Tokenizer {
public:
    Tokenizer() = default;

    void add_token(TokenId id, std::string piece);

    [[nodiscard]] bool contains(TokenId id) const;

    [[nodiscard]] std::string decode(TokenId id) const;

    [[nodiscard]] std::string decode(const std::vector<TokenId>& tokens) const;

    [[nodiscard]] std::vector<TokenId> encode(const std::string& text) const;

    [[nodiscard]] std::size_t vocab_size() const noexcept;

private:
    std::unordered_map<TokenId, std::string> id_to_piece_;
    std::unordered_map<std::string, TokenId> piece_to_id_;
};

} // namespace llmengine