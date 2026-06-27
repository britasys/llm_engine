#include "llmengine/tokenizer.hpp"

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace llmengine {

namespace {

constexpr char kSpaceMarker[] = "\xE2\x96\x81";

inline bool starts_with(std::string_view s, std::string_view prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

} // namespace

std::vector<TokenId> Tokenizer::encode_sentencepiece(std::string_view text) const {
    std::vector<TokenId> output;

    if (text.empty())
        return output;

    std::string normalized;
    normalized.reserve(text.size() + 8);

    normalized += kSpaceMarker;

    for (char c : text) {
        if (c == ' ')
            normalized += kSpaceMarker;
        else
            normalized.push_back(c);
    }

    std::size_t pos = 0;

    while (pos < normalized.size()) {
        TokenId best_id = -1;
        std::size_t best_len = 0;

        for (const auto& [piece, id] : piece_to_id_) {
            if (piece.size() <= best_len)
                continue;

            if (starts_with(std::string_view(normalized).substr(pos), piece)) {
                best_len = piece.size();
                best_id = id;
            }
        }

        if (best_id != -1) {
            output.push_back(best_id);
            pos += best_len;
            continue;
        }

        std::string byte_piece(1, normalized[pos]);

        auto it = piece_to_id_.find(byte_piece);

        if (it != piece_to_id_.end()) {
            output.push_back(it->second);
            ++pos;
            continue;
        }

        if (unk_token_ >= 0) {
            output.push_back(unk_token_);
            ++pos;
            continue;
        }

        throw std::runtime_error("cannot tokenize input");
    }

    return output;
}

} // namespace llmengine