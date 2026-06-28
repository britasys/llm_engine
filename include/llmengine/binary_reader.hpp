#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace llmengine {

class BinaryReader {
public:
    explicit BinaryReader(std::span<const std::byte> data) noexcept : data_(data) {}

    [[nodiscard]] size_t position() const noexcept { return offset_; }
    [[nodiscard]] size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] size_t remaining() const noexcept { return data_.size() - offset_; }

    void seek(size_t position) {
        if (position > data_.size())
            throw std::runtime_error("BinaryReader seek out of range");

        offset_ = position;
    }

    template<typename T> T read() {
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

        ensure(sizeof(T));

        T value{};

        std::memcpy(&value, data_.data() + offset_, sizeof(T));

        offset_ += sizeof(T);

        return value;
    }

    template<typename T> T read_le() {
        T value = read<T>();

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        if constexpr (sizeof(T) == 2)
            value = std::byteswap(value);

        else if constexpr (sizeof(T) == 4)
            value = std::byteswap(value);

        else if constexpr (sizeof(T) == 8)
            value = std::byteswap(value);
#endif

        return value;
    }

    std::string read_string() {
        uint64_t length = read<uint64_t>();

        constexpr uint64_t MAX_STRING = 1ull << 24;

        if (length > MAX_STRING)
            throw std::runtime_error("string too large");

        ensure(length);

        std::string result(reinterpret_cast<const char*>(data_.data() + offset_), static_cast<size_t>(length));

        offset_ += static_cast<size_t>(length);

        return result;
    }

    std::span<const std::byte> read_bytes(size_t count) {
        ensure(count);

        auto result = data_.subspan(offset_, count);

        offset_ += count;

        return result;
    }

    [[nodiscard]] bool eof() const noexcept { return offset_ >= data_.size(); }

private:
    void ensure(size_t count) const {
        if (count > remaining()) {
            throw std::runtime_error("unexpected end of binary data");
        }
    }

private:
    std::span<const std::byte> data_;
    size_t offset_ = 0;
};

} // namespace llmengine