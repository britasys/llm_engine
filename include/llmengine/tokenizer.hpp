#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace llmengine {

class GGUFLoader;

using TokenId = int32_t;

enum class TokenizerModel { Unknown, SentencePiece, BPE, Unigram };

class Tokenizer {
public:
    explicit Tokenizer(const GGUFLoader& loader);

    [[nodiscard]] TokenizerModel model() const noexcept;
    [[nodiscard]] bool contains(TokenId id) const noexcept;
    [[nodiscard]] std::size_t vocab_size() const noexcept;
    [[nodiscard]] const std::string& decode(TokenId id) const;
    [[nodiscard]] std::string decode(std::span<const TokenId> tokens) const;
    [[nodiscard]] std::vector<TokenId> encode(std::string_view text) const;

private:
    void load_vocab(const GGUFLoader& loader);
    void load_model(const GGUFLoader& loader);
    void load_special_tokens(const GGUFLoader& loader);

    std::vector<TokenId> encode_sentencepiece(std::string_view text) const;
    std::vector<TokenId> encode_bpe(std::string_view text) const;
    std::vector<TokenId> encode_unigram(std::string_view text) const;

    void add_token(TokenId id, std::string piece);

private:
    TokenizerModel model_ = TokenizerModel::Unknown;

    std::vector<std::string> id_to_piece_;
    std::unordered_map<std::string, TokenId> piece_to_id_;

    TokenId bos_token_ = -1;
    TokenId eos_token_ = -1;
    TokenId unk_token_ = -1;
    TokenId pad_token_ = -1;
};

} // namespace llmengine