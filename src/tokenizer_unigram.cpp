#include "llmengine/tokenizer.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace llmengine {

std::vector<TokenId> Tokenizer::encode_unigram(std::string_view text) const {
    std::vector<TokenId> output;

    if (text.empty())
        return output;

    std::size_t pos = 0;

    while (pos < text.size()) {
        TokenId best_id = -1;
        std::size_t best_len = 0;

        std::string_view remaining = text.substr(pos);

        for (const auto& [piece, id] : piece_to_id_) {
            if (piece.size() <= best_len)
                continue;

            if (remaining.starts_with(piece)) {
                best_id = id;
                best_len = piece.size();
            }
        }

        if (best_id >= 0) {
            output.push_back(best_id);
            pos += best_len;
            continue;
        }

        if (unk_token_ >= 0) {
            output.push_back(unk_token_);
            ++pos;
            continue;
        }

        throw std::runtime_error("unigram encoding failed");
    }

    return output;
}

}