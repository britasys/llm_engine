#include "llmengine/tokenizer.hpp"
#include "llmengine/gguf_loader.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string_view>

namespace llmengine {

namespace {

void skip_whitespace(std::string_view s, std::size_t& pos) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
        ++pos;
    }
}

char decode_escape(char c) {
    switch (c) {
    case '"':
        return '"';
    case '\\':
        return '\\';
    case '/':
        return '/';
    case 'b':
        return '\b';
    case 'f':
        return '\f';
    case 'n':
        return '\n';
    case 'r':
        return '\r';
    case 't':
        return '\t';
    default:
        throw std::runtime_error("invalid escape sequence");
    }
}

std::string parse_string(std::string_view s, std::size_t& pos) {
    if (pos >= s.size() || s[pos] != '"')
        throw std::runtime_error("expected string");

    ++pos;

    std::string out;

    while (pos < s.size()) {
        char c = s[pos++];

        if (c == '"')
            return out;

        if (c == '\\') {
            if (pos >= s.size())
                throw std::runtime_error("unterminated escape sequence");

            char esc = s[pos++];

            if (esc == 'u')
                throw std::runtime_error("\\u escapes are not implemented");

            out.push_back(decode_escape(esc));
            continue;
        }

        out.push_back(c);
    }

    throw std::runtime_error("unterminated string");
}

std::vector<std::string> parse_string_array(std::string_view s) {
    std::vector<std::string> result;

    std::size_t pos = 0;

    skip_whitespace(s, pos);

    if (pos >= s.size() || s[pos] != '[')
        throw std::runtime_error("expected '['");

    ++pos;

    for (;;) {
        skip_whitespace(s, pos);

        if (pos >= s.size())
            throw std::runtime_error("unexpected end of array");

        if (s[pos] == ']') {
            ++pos;
            break;
        }

        result.emplace_back(parse_string(s, pos));

        skip_whitespace(s, pos);

        if (pos >= s.size())
            throw std::runtime_error("unexpected end of array");

        if (s[pos] == ',') {
            ++pos;
            continue;
        }

        if (s[pos] == ']') {
            ++pos;
            break;
        }

        throw std::runtime_error("expected ',' or ']'");
    }

    return result;
}

} // namespace

Tokenizer::Tokenizer(const GGUFLoader& loader) {
    load_model(loader);
    load_vocab(loader);
    load_special_tokens(loader);
}

void Tokenizer::load_vocab(const GGUFLoader& loader) {
    const auto& tokens = loader.get_meta_array("tokenizer.ggml.tokens");

    id_to_piece_.clear();
    piece_to_id_.clear();

    id_to_piece_.reserve(tokens.values.size());
    piece_to_id_.reserve(tokens.values.size());

    for (TokenId id = 0; id < static_cast<TokenId>(tokens.values.size()); ++id) {
        const auto& value = tokens.values[static_cast<size_t>(id)];

        if (!std::holds_alternative<std::string>(value))
            throw std::runtime_error("token is not a string");

        add_token(id, std::get<std::string>(value));
    }
}

void Tokenizer::load_model(const GGUFLoader& loader) {
    try {
        const auto& value = loader.get_meta_string("tokenizer.ggml.model");

        if (value == "llama")
            model_ = TokenizerModel::SentencePiece;
        else if (value == "gpt2")
            model_ = TokenizerModel::BPE;
        else
            model_ = TokenizerModel::Unknown;
    } catch (const std::exception&) {
        model_ = TokenizerModel::Unknown;
    }
}

TokenizerModel Tokenizer::model() const noexcept {
    return model_;
}

bool Tokenizer::contains(TokenId id) const noexcept {
    return id >= 0 && static_cast<std::size_t>(id) < id_to_piece_.size();
}

std::size_t Tokenizer::vocab_size() const noexcept {
    return id_to_piece_.size();
}

void Tokenizer::add_token(TokenId id, std::string piece) {
    if (static_cast<std::size_t>(id) != id_to_piece_.size())
        throw std::runtime_error("non-contiguous token ids");

    piece_to_id_.emplace(piece, id);
    id_to_piece_.emplace_back(std::move(piece));
}

const std::string& Tokenizer::decode(TokenId id) const {
    if (!contains(id))
        throw std::out_of_range("invalid token id");

    return id_to_piece_[static_cast<std::size_t>(id)];
}

std::vector<TokenId> Tokenizer::encode(std::string_view text) const {
    switch (model_) {
    case TokenizerModel::SentencePiece:
        return encode_sentencepiece(text);

    case TokenizerModel::BPE:
        return encode_bpe(text);

    case TokenizerModel::Unigram:
        return encode_unigram(text);

    default:
        throw std::runtime_error("unsupported tokenizer model");
    }
}

} // namespace llmengine
