#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace llmengine::utf8 {

struct Codepoint {
    uint32_t value;
    std::size_t size;
};

Codepoint decode_next(std::string_view text, std::size_t offset);

std::vector<uint32_t> decode(std::string_view text);

bool valid(std::string_view text);

}