#include "llmengine/tokenizer_utf8.hpp"

#include <stdexcept>

namespace llmengine::utf8 {

Codepoint decode_next(std::string_view text, std::size_t offset) {
    if (offset >= text.size())
        throw std::out_of_range("utf8 offset");

    const auto* bytes = reinterpret_cast<const unsigned char*>(text.data());

    uint8_t c = bytes[offset];

    if ((c & 0x80) == 0) {
        return {static_cast<uint32_t>(c), 1};
    }

    if ((c & 0xE0) == 0xC0) {
        if (offset + 1 >= text.size())
            throw std::runtime_error("invalid utf8");

        return {((c & 0x1F) << 6) | (bytes[offset + 1] & 0x3F), 2};
    }

    if ((c & 0xF0) == 0xE0) {
        if (offset + 2 >= text.size())
            throw std::runtime_error("invalid utf8");

        return {((c & 0x0F) << 12) | ((bytes[offset + 1] & 0x3F) << 6) | (bytes[offset + 2] & 0x3F), 3};
    }

    if ((c & 0xF8) == 0xF0) {
        if (offset + 3 >= text.size())
            throw std::runtime_error("invalid utf8");

        return {((c & 0x07) << 18) | ((bytes[offset + 1] & 0x3F) << 12) | ((bytes[offset + 2] & 0x3F) << 6) | (bytes[offset + 3] & 0x3F), 4};
    }

    throw std::runtime_error("invalid utf8 leading byte");
}

std::vector<uint32_t> decode(std::string_view text) {
    std::vector<uint32_t> result;

    std::size_t offset = 0;

    while (offset < text.size()) {
        auto cp = decode_next(text, offset);
        result.push_back(cp.value);
        offset += cp.size;
    }

    return result;
}

bool valid(std::string_view text) {
    try {
        std::size_t offset = 0;

        while (offset < text.size()) {
            offset += decode_next(text, offset).size;
        }

        return true;
    } catch (...) {
        return false;
    }
}

} // namespace llmengine::utf8