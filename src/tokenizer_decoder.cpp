#include "llmengine/tokenizer.hpp"

namespace llmengine {

namespace {

bool is_sentencepiece_space(std::string_view s, std::size_t pos) {
    return pos + 3 <= s.size() && static_cast<unsigned char>(s[pos]) == 0xE2 && static_cast<unsigned char>(s[pos + 1]) == 0x96 &&
           static_cast<unsigned char>(s[pos + 2]) == 0x81;
}

} // namespace

std::string Tokenizer::decode(std::span<const TokenId> tokens) const {
    std::string result;

    for (TokenId id : tokens) {
        const auto& piece = decode(id);

        for (std::size_t i = 0; i < piece.size();) {
            if (is_sentencepiece_space(piece, i)) {
                result.push_back(' ');
                i += 3;
            } else {
                result.push_back(piece[i]);
                ++i;
            }
        }
    }

    return result;
}

} // namespace llmengine