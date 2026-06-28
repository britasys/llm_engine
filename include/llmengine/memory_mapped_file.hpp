#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <system_error>

namespace llmengine {

/**
 * @brief Cross-platform memory-mapped file.
 *
 * Supports:
 *  - Windows (CreateFileMapping / MapViewOfFile)
 *  - Linux/macOS (mmap)
 *
 * The class is move-only and uses RAII.
 */
class MemoryMappedFile {
public:
    enum class Mode { ReadOnly, ReadWrite };

    MemoryMappedFile() noexcept;

    explicit MemoryMappedFile(const std::filesystem::path& path, Mode mode = Mode::ReadOnly);

    MemoryMappedFile(const std::filesystem::path& path, Mode mode, std::uint64_t length);

    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;

    MemoryMappedFile(MemoryMappedFile&& other) noexcept;
    MemoryMappedFile& operator=(MemoryMappedFile&& other) noexcept;

    ~MemoryMappedFile();

    /**
     * @brief Opens an existing file.
     *
     * @throws std::system_error on failure.
     */
    void open(const std::filesystem::path& path, Mode mode = Mode::ReadOnly);

    /**
     * @brief Opens or creates a file with the specified size.
     *
     * If the file already exists it will be resized.
     *
     * @throws std::system_error on failure.
     */
    void open(const std::filesystem::path& path, Mode mode, std::uint64_t length);

    /**
     * @brief Unmaps the file and closes all handles.
     */
    void close() noexcept;

    /**
     * @brief Flushes modified pages.
     *
     * No-op for read-only mappings.
     *
     * @throws std::system_error on failure.
     */
    void flush();

    [[nodiscard]] bool is_open() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] Mode mode() const noexcept;
    [[nodiscard]] std::uint64_t size() const noexcept;
    [[nodiscard]] const std::byte* data() const noexcept;
    [[nodiscard]] std::byte* data() noexcept;
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;
    [[nodiscard]] std::span<std::byte> writable_bytes();
    [[nodiscard]] const std::byte& operator[](std::size_t index) const noexcept;
    [[nodiscard]] std::byte& operator[](std::size_t index);
    [[nodiscard]] const std::byte& front() const noexcept;
    [[nodiscard]] std::byte& front();
    [[nodiscard]] const std::byte& back() const noexcept;
    [[nodiscard]] std::byte& back();
    [[nodiscard]] const std::byte* begin() const noexcept;
    [[nodiscard]] const std::byte* end() const noexcept;
    [[nodiscard]] std::byte* begin() noexcept;
    [[nodiscard]] std::byte* end() noexcept;

private:
    void map_existing_file(const std::filesystem::path& path, Mode mode);
    void create_and_map_file(const std::filesystem::path& path, Mode mode, std::uint64_t length);
    void move_from(MemoryMappedFile&& other) noexcept;

private:
#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>

private:
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else

    int fd_ = -1;

#endif

    void* data_ = nullptr;
    std::uint64_t size_ = 0;
    Mode mode_ = Mode::ReadOnly;
};

// -----------------------------------------------------------------------------
// Inline implementation
// -----------------------------------------------------------------------------

inline MemoryMappedFile::MemoryMappedFile() noexcept = default;

inline MemoryMappedFile::MemoryMappedFile(const std::filesystem::path& path, Mode mode) {
    open(path, mode);
}

inline MemoryMappedFile::MemoryMappedFile(const std::filesystem::path& path, Mode mode, std::uint64_t length) {
    open(path, mode, length);
}

inline MemoryMappedFile::~MemoryMappedFile() {
    close();
}

inline bool MemoryMappedFile::empty() const noexcept {
    return size_ == 0;
}

inline MemoryMappedFile::Mode MemoryMappedFile::mode() const noexcept {
    return mode_;
}

inline std::uint64_t MemoryMappedFile::size() const noexcept {
    return size_;
}

inline const std::byte* MemoryMappedFile::data() const noexcept {
    return static_cast<const std::byte*>(data_);
}

inline std::byte* MemoryMappedFile::data() noexcept {
    return static_cast<std::byte*>(data_);
}

inline std::span<const std::byte> MemoryMappedFile::bytes() const noexcept {
    return {data(), static_cast<std::size_t>(size_)};
}

inline std::span<std::byte> MemoryMappedFile::writable_bytes() {
    if (mode_ == Mode::ReadOnly) {
        throw std::system_error(std::make_error_code(std::errc::operation_not_permitted), "MemoryMappedFile is read-only");
    }

    return {data(), static_cast<std::size_t>(size_)};
}

inline const std::byte& MemoryMappedFile::operator[](std::size_t index) const noexcept {
    return data()[index];
}

inline std::byte& MemoryMappedFile::operator[](std::size_t index) {
    return writable_bytes()[index];
}

inline const std::byte& MemoryMappedFile::front() const noexcept {
    return data()[0];
}

inline std::byte& MemoryMappedFile::front() {
    return writable_bytes()[0];
}

inline const std::byte& MemoryMappedFile::back() const noexcept {
    return data()[size_ - 1];
}

inline std::byte& MemoryMappedFile::back() {
    return writable_bytes()[size_ - 1];
}

inline const std::byte* MemoryMappedFile::begin() const noexcept {
    return data();
}

inline const std::byte* MemoryMappedFile::end() const noexcept {
    return data() + size_;
}

inline std::byte* MemoryMappedFile::begin() noexcept {
    return data();
}

inline std::byte* MemoryMappedFile::end() noexcept {
    return data() + size_;
}

} // namespace llmengine
