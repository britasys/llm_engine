#include "llmengine/tokenizer.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace llmengine {

namespace {

struct PairHash {
    std::size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
        return std::hash<std::string>{}(p.first) ^ (std::hash<std::string>{}(p.second) << 1);
    }
};

} // namespace

std::vector<TokenId> Tokenizer::encode_bpe(std::string_view text) const {
    std::vector<TokenId> output;

    if (text.empty())
        return output;

    std::vector<std::string> pieces;

    pieces.reserve(text.size());

    for (char c : text) {
        pieces.emplace_back(1, c);
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

        pieces[best] = pieces[best] + pieces[best + 1];

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

        throw std::runtime_error("BPE encoding failed");
    }

    return output;
}

} // namespace llmengine