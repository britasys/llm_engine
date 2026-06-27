// llmengine/tokenizer.cpp
#include "llmengine/tokenizer.hpp"
#include "llmengine/gguf_loader.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <iostream>
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

// ---- UTF-8 helpers ----

std::size_t utf8_decode(std::string_view s, std::size_t pos, char32_t& out) {
    unsigned char c0 = static_cast<unsigned char>(s[pos]);

    if (c0 < 0x80) {
        out = c0;
        return 1;
    }
    if ((c0 & 0xE0) == 0xC0 && pos + 1 < s.size()) {
        out = (static_cast<char32_t>(c0 & 0x1F) << 6) | (static_cast<unsigned char>(s[pos + 1]) & 0x3F);
        return 2;
    }
    if ((c0 & 0xF0) == 0xE0 && pos + 2 < s.size()) {
        out = (static_cast<char32_t>(c0 & 0x0F) << 12) | ((static_cast<unsigned char>(s[pos + 1]) & 0x3F) << 6) |
              (static_cast<unsigned char>(s[pos + 2]) & 0x3F);
        return 3;
    }
    if ((c0 & 0xF8) == 0xF0 && pos + 3 < s.size()) {
        out = (static_cast<char32_t>(c0 & 0x07) << 18) | ((static_cast<unsigned char>(s[pos + 1]) & 0x3F) << 12) |
              ((static_cast<unsigned char>(s[pos + 2]) & 0x3F) << 6) | (static_cast<unsigned char>(s[pos + 3]) & 0x3F);
        return 4;
    }

    out = c0;
    return 1;
}

void utf8_encode(char32_t cp, std::string& out) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// add to the anonymous namespace in tokenizer.cpp, near the other UTF-8 helpers

bool is_ascii_letter(char32_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

bool is_digit_cp(char32_t c) {
    return c >= '0' && c <= '9';
}

bool is_letter_cp(char32_t c) {
    // ASCII letters, plus treat any non-ASCII codepoint as "letter-like"
    // (covers CJK, Cyrillic, accented Latin, etc. well enough for splitting purposes)
    return is_ascii_letter(c) || c >= 0x80;
}

bool is_space_cp(char32_t c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == 0x0B || c == 0x0C;
}

// Splits text into GPT2/Qwen2-style pretokenizer chunks (on raw UTF-8 text,
// BEFORE byte-level remapping). Each returned chunk is later byte-remapped
// and BPE-merged independently.
std::vector<std::string> pretokenize_split(std::string_view text) {
    std::vector<std::string> chunks;

    std::vector<char32_t> cps;
    std::vector<std::size_t> byte_offsets; // byte offset of each codepoint
    {
        std::size_t i = 0;
        while (i < text.size()) {
            char32_t cp;
            std::size_t len = utf8_decode(text, i, cp);
            cps.push_back(cp);
            byte_offsets.push_back(i);
            i += len;
        }
        byte_offsets.push_back(text.size());
    }

    std::size_t n = cps.size();
    std::size_t i = 0;

    auto emit = [&](std::size_t start, std::size_t end) {
        if (start < end)
            chunks.emplace_back(text.substr(byte_offsets[start], byte_offsets[end] - byte_offsets[start]));
    };

    while (i < n) {
        // 1. Contractions: 's 't 're 've 'm 'll 'd (case-insensitive lead char check skipped for simplicity)
        if (cps[i] == '\'' && i + 1 < n) {
            auto starts_with = [&](std::string_view suf) -> std::size_t {
                if (i + suf.size() > n)
                    return 0;
                for (std::size_t k = 0; k < suf.size(); ++k) {
                    char32_t c = cps[i + 1 + k];
                    char lc = static_cast<char>((c >= 'A' && c <= 'Z') ? c + 32 : c);
                    if (lc != suf[k + 1])
                        return 0;
                }
                return suf.size() + 1;
            };
            static const char* conts[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
            bool matched = false;
            for (const char* c : conts) {
                std::size_t len = starts_with(c);
                if (len > 0) {
                    emit(i, i + len);
                    i += len;
                    matched = true;
                    break;
                }
            }
            if (matched)
                continue;
        }

        // 2. [^\r\n\p{L}\p{N}]? \p{L}+   (optional leading non-letter/digit/newline, then letters)
        if (!is_letter_cp(cps[i]) && cps[i] != '\r' && cps[i] != '\n' && !is_digit_cp(cps[i]) && !is_space_cp(cps[i]) && i + 1 < n &&
            is_letter_cp(cps[i + 1])) {
            std::size_t start = i;
            ++i;
            while (i < n && is_letter_cp(cps[i]))
                ++i;
            emit(start, i);
            continue;
        }
        if (is_letter_cp(cps[i])) {
            std::size_t start = i;
            while (i < n && is_letter_cp(cps[i]))
                ++i;
            emit(start, i);
            continue;
        }

        // 3. \p{N}{1,3}  (digit runs, up to 3 at a time)
        if (is_digit_cp(cps[i])) {
            std::size_t start = i;
            std::size_t count = 0;
            while (i < n && is_digit_cp(cps[i]) && count < 3) {
                ++i;
                ++count;
            }
            emit(start, i);
            continue;
        }

        // 4. " ?[^\s\p{L}\p{N}]+[\r\n]*"  (optional leading space, punctuation run, trailing newlines)
        if (!is_space_cp(cps[i]) && !is_letter_cp(cps[i]) && !is_digit_cp(cps[i])) {
            std::size_t start = i;
            ++i;
            while (i < n && !is_space_cp(cps[i]) && !is_letter_cp(cps[i]) && !is_digit_cp(cps[i]))
                ++i;
            while (i < n && (cps[i] == '\r' || cps[i] == '\n'))
                ++i;
            emit(start, i);
            continue;
        }
        if (i > 0 && cps[i - 1] == ' ' && !is_space_cp(cps[i])) {
            // unreachable given above branches, kept for pattern fidelity
        }

        // 5. \s*[\r\n]+  or  \s+(?!\S)  or  \s+   (whitespace runs)
        if (is_space_cp(cps[i])) {
            std::size_t start = i;
            while (i < n && is_space_cp(cps[i]))
                ++i;
            // if followed by more non-space content and this run is >1 char,
            // GPT2 keeps the last space attached to the next chunk; approximate
            // by giving back one trailing space when followed by non-space.
            if (i < n && i > start + 1 && !is_space_cp(cps[i])) {
                --i;
            }
            emit(start, i);
            continue;
        }

        // fallback: shouldn't hit, but avoid infinite loop
        emit(i, i + 1);
        ++i;
    }

    return chunks;
}
} // namespace

Tokenizer::Tokenizer(const GGUFLoader& loader) {
    init_byte_unicode_table();

    load_model(loader);
    load_vocab(loader);
    load_special_tokens(loader);

    if (model_ == TokenizerModel::BPE)
        load_merges(loader);
}

void Tokenizer::init_byte_unicode_table() {
    std::vector<uint32_t> bs;
    std::vector<uint32_t> cs;

    for (uint32_t b = static_cast<uint32_t>('!'); b <= static_cast<uint32_t>('~'); ++b) {
        bs.push_back(b);
        cs.push_back(b);
    }
    for (uint32_t b = 0xA1; b <= 0xAC; ++b) {
        bs.push_back(b);
        cs.push_back(b);
    }
    for (uint32_t b = 0xAE; b <= 0xFF; ++b) {
        bs.push_back(b);
        cs.push_back(b);
    }

    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; ++b) {
        bool found = false;
        for (uint32_t existing : bs) {
            if (existing == b) {
                found = true;
                break;
            }
        }
        if (!found) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    for (std::size_t i = 0; i < bs.size(); ++i) {
        byte_to_unicode_[bs[i]] = static_cast<char32_t>(cs[i]);
        unicode_to_byte_[static_cast<char32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
    }
}

void Tokenizer::load_special_token_strings() {
    special_tokens_.clear();
    for (TokenId id = 0; id < static_cast<TokenId>(id_to_piece_.size()); ++id) {
        const std::string& piece = id_to_piece_[static_cast<size_t>(id)];
        if (piece.size() >= 3 && piece.front() == '<' && piece.back() == '>' && piece.find('|') != std::string::npos) {
            special_tokens_.emplace_back(piece, id);
        }
    }
    // longest-match-first so e.g. "<|im_start|>" matches before any shorter overlapping token
    std::sort(special_tokens_.begin(), special_tokens_.end(), [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
}

void Tokenizer::load_special_tokens(const GGUFLoader& loader) {
    auto try_load = [&](const char* key, TokenId& out) {
        try {
            out = static_cast<TokenId>(loader.get_meta_int(key));
        } catch (...) {
            out = -1;
        }
    };

    try_load("tokenizer.ggml.bos_token_id", bos_token_);
    try_load("tokenizer.ggml.eos_token_id", eos_token_);
    try_load("tokenizer.ggml.unknown_token_id", unk_token_);
    try_load("tokenizer.ggml.padding_token_id", pad_token_);

    special_tokens_.clear();

    for (TokenId id = 0; id < static_cast<TokenId>(id_to_piece_.size()); ++id) {
        const std::string& piece = id_to_piece_[static_cast<size_t>(id)];

        if (piece.starts_with("<|") && piece.ends_with("|>")) {
            special_tokens_.push_back({piece, id});
        }
    }
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

    init_byte_unicode_table();
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

void Tokenizer::load_merges(const GGUFLoader& loader) {
    merge_ranks_.clear();

    const auto& merges = loader.get_meta_array("tokenizer.ggml.merges");

    int rank = 0;
    for (const auto& value : merges.values) {
        if (!std::holds_alternative<std::string>(value))
            throw std::runtime_error("merge entry is not a string");

        const std::string& entry = std::get<std::string>(value);
        std::size_t space = entry.find(' ');
        if (space == std::string::npos)
            throw std::runtime_error("malformed merge entry: " + entry);

        std::string left = entry.substr(0, space);
        std::string right = entry.substr(space + 1);

        merge_ranks_.emplace(std::make_pair(std::move(left), std::move(right)), rank);
        ++rank;
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

std::string Tokenizer::decode(std::span<const TokenId> tokens) const {

    std::string encoded;

    for (TokenId id : tokens) {
        encoded += decode(id);
    }

    std::string result;

    std::size_t i = 0;

    while (i < encoded.size()) {

        char32_t cp;
        std::size_t len = utf8_decode(encoded, i, cp);

        auto it = unicode_to_byte_.find(cp);

        if (it != unicode_to_byte_.end()) {

            result.push_back(static_cast<char>(it->second));

        } else {

            result.append(encoded, i, len);
        }

        i += len;
    }

    return result;
}

std::vector<TokenId> Tokenizer::encode(std::string_view text) const {
    std::vector<TokenId> output;

    if (bos_token_ >= 0)
        output.push_back(bos_token_);

    std::size_t pos = 0;
    std::string buffer; // accumulates plain text between special tokens

    auto flush_buffer = [&]() {
        if (buffer.empty())
            return;
        std::vector<TokenId> body;
        switch (model_) {
        case TokenizerModel::SentencePiece:
            body = encode_sentencepiece(buffer);
            break;
        case TokenizerModel::BPE:
            body = encode_bpe(buffer);
            break;
        case TokenizerModel::Unigram:
            body = encode_unigram(buffer);
            break;
        default:
            throw std::runtime_error("unsupported tokenizer model");
        }
        output.insert(output.end(), body.begin(), body.end());
        buffer.clear();
    };

    while (pos < text.size()) {
        bool matched = false;

        for (const auto& [tok_str, tok_id] : special_tokens_) {
            if (text.compare(pos, tok_str.size(), tok_str) == 0) {
                flush_buffer();
                output.push_back(tok_id);
                pos += tok_str.size();
                matched = true;
                break;
            }
        }

        if (!matched) {
            buffer.push_back(text[pos]);
            ++pos;
        }
    }

    flush_buffer();
    return output;
}

std::vector<TokenId> Tokenizer::encode_bpe(std::string_view text) const {
    std::vector<TokenId> tokens;

    size_t pos = 0;

    while (pos < text.size()) {

        // ----------------------------------------
        // 1. Check special tokens first
        // ----------------------------------------
        bool matched_special = false;

        for (const auto& [special, id] : special_tokens_) {

            if (pos + special.size() <= text.size() && text.compare(pos, special.size(), special) == 0) {

                tokens.push_back(id);
                pos += special.size();

                matched_special = true;
                break;
            }
        }

        if (matched_special)
            continue;

        // ----------------------------------------
        // 2. Encode normal text chunk with BPE
        // ----------------------------------------

        size_t end = pos;

        while (end < text.size()) {

            bool next_special = false;

            for (const auto& [special, id] : special_tokens_) {

                if (end + special.size() <= text.size() && text.compare(end, special.size(), special) == 0) {

                    next_special = true;
                    break;
                }
            }

            if (next_special)
                break;

            end++;
        }

        std::string chunk(text.substr(pos, end - pos));

        if (!chunk.empty()) {

            // Convert UTF-8 bytes to GPT byte unicode representation
            std::string bpe_input;
            for (unsigned char c : chunk) {
                char32_t cp = byte_to_unicode_[c];

                char buf[5];
                int len = 0;

                if (cp <= 0x7F) {
                    buf[len++] = static_cast<char>(cp);
                } else if (cp <= 0x7FF) {
                    buf[len++] = 0xC0 | (cp >> 6);
                    buf[len++] = 0x80 | (cp & 0x3F);
                } else if (cp <= 0xFFFF) {
                    buf[len++] = 0xE0 | (cp >> 12);
                    buf[len++] = 0x80 | ((cp >> 6) & 0x3F);
                    buf[len++] = 0x80 | (cp & 0x3F);
                } else {
                    buf[len++] = 0xF0 | (cp >> 18);
                    buf[len++] = 0x80 | ((cp >> 12) & 0x3F);
                    buf[len++] = 0x80 | ((cp >> 6) & 0x3F);
                    buf[len++] = 0x80 | (cp & 0x3F);
                }

                bpe_input.append(buf, len);
            }

            // Split into symbols
            std::vector<std::string> symbols;
            for (size_t i = 0; i < bpe_input.size();) {

                char32_t cp;
                size_t len = utf8_decode(bpe_input, i, cp);

                symbols.emplace_back(bpe_input.substr(i, len));

                i += len;
            }

            // BPE merge loop
            while (symbols.size() > 1) {

                int best_rank = INT_MAX;
                size_t best_index = SIZE_MAX;

                for (size_t i = 0; i + 1 < symbols.size(); i++) {

                    auto it = merge_ranks_.find({symbols[i], symbols[i + 1]});

                    if (it != merge_ranks_.end() && it->second < best_rank) {

                        best_rank = it->second;
                        best_index = i;
                    }
                }

                if (best_index == SIZE_MAX)
                    break;

                std::string merged = symbols[best_index] + symbols[best_index + 1];

                symbols[best_index] = merged;

                symbols.erase(symbols.begin() + best_index + 1);
            }

            // Convert pieces to token ids
            for (const auto& s : symbols) {

                auto it = piece_to_id_.find(s);

                if (it != piece_to_id_.end()) {
                    tokens.push_back(it->second);
                } else {
                    // fallback to byte tokens
                    for (unsigned char c : s) {

                        auto uni = byte_to_unicode_[c];

                        std::string piece;
                        piece.push_back(static_cast<char>(uni));

                        auto byte_it = piece_to_id_.find(piece);

                        if (byte_it != piece_to_id_.end())
                            tokens.push_back(byte_it->second);
                        else
                            tokens.push_back(unk_token_);
                    }
                }
            }
        }

        pos = end;
    }

    return tokens;
}

std::vector<TokenId> Tokenizer::encode_sentencepiece(std::string_view text) const {
    std::vector<TokenId> output;

    if (text.empty())
        return output;

    std::string remapped;
    remapped.reserve(text.size() + 4);

    for (char c : text) {
        if (c == ' ')
            remapped.append("\xE2\x96\x81"); // U+2581 lower one eighth block
        else
            remapped.push_back(c);
    }

    std::vector<std::string> pieces;
    {
        std::size_t i = 0;
        while (i < remapped.size()) {
            char32_t cp;
            std::size_t len = utf8_decode(remapped, i, cp);
            pieces.emplace_back(remapped, i, len);
            i += len;
        }
    }

    while (pieces.size() > 1) {
        std::size_t best = pieces.size();

        for (std::size_t i = 0; i + 1 < pieces.size(); ++i) {
            std::string merged = pieces[i] + pieces[i + 1];

            if (piece_to_id_.contains(merged)) {
                best = i;
                break;
            }
        }

        if (best == pieces.size())
            break;

        pieces[best] += pieces[best + 1];
        pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(best + 1));
    }

    for (const auto& piece : pieces) {
        auto it = piece_to_id_.find(piece);

        if (it != piece_to_id_.end()) {
            output.push_back(it->second);
            continue;
        }

        if (unk_token_ >= 0) {
            output.push_back(unk_token_);
            continue;
        }

        throw std::runtime_error("SentencePiece encoding failed");
    }

    return output;
}

std::vector<TokenId> Tokenizer::encode_unigram(std::string_view text) const {
    throw std::runtime_error("unigram tokenizer model is not implemented");
}
} // namespace llmengine