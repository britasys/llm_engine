// llmengine/tokenizer.hpp
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llmengine {

class GGUFLoader;

using TokenId = int32_t;

enum class TokenizerModel { Unknown, SentencePiece, BPE, Unigram };

struct PairHash {
    std::size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
        return std::hash<std::string>{}(p.first) ^ (std::hash<std::string>{}(p.second) << 1);
    }
};

class Tokenizer {
public:
    explicit Tokenizer(const GGUFLoader& loader);

    [[nodiscard]] TokenizerModel model() const noexcept;
    [[nodiscard]] bool contains(TokenId id) const noexcept;
    [[nodiscard]] std::size_t vocab_size() const noexcept;
    [[nodiscard]] const std::string& decode(TokenId id) const;
    [[nodiscard]] std::string decode(std::span<const TokenId> tokens) const;
    [[nodiscard]] std::vector<TokenId> encode(std::string_view text) const;

    [[nodiscard]] TokenId eos_token() const noexcept { return eos_token_; }

private:
    void load_vocab(const GGUFLoader& loader);
    void load_model(const GGUFLoader& loader);
    void load_special_tokens(const GGUFLoader& loader);
    void load_merges(const GGUFLoader& loader);
    void init_byte_unicode_table();
    void load_special_token_strings();

    std::vector<TokenId> encode_sentencepiece(std::string_view text) const;
    std::vector<TokenId> encode_bpe(std::string_view text) const;
    std::vector<TokenId> encode_unigram(std::string_view text) const;

    void add_token(TokenId id, std::string piece);

private:
    TokenizerModel model_ = TokenizerModel::Unknown;

    std::vector<std::string> id_to_piece_;
    std::unordered_map<std::string, TokenId> piece_to_id_;

    std::array<char32_t, 256> byte_to_unicode_{};
    std::unordered_map<char32_t, uint8_t> unicode_to_byte_;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merge_ranks_;
    std::vector<std::pair<std::string, TokenId>> special_tokens_;

    TokenId bos_token_ = -1;
    TokenId eos_token_ = -1;
    TokenId unk_token_ = -1;
    TokenId pad_token_ = -1;
};

} // namespace llmengine