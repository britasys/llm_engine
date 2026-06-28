#include "llmengine/memory_mapped_file.hpp"

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <Windows.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <system_error>

namespace llmengine {

namespace {

[[noreturn]]
void throw_last_error(const char* message) {
    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), message);
}

std::wstring to_utf16(const std::filesystem::path& path) {
    return path.wstring();
}

std::uint64_t file_size(HANDLE file) {
    LARGE_INTEGER size{};

    if (!::GetFileSizeEx(file, &size))
        throw_last_error("GetFileSizeEx failed");

    return static_cast<std::uint64_t>(size.QuadPart);
}

void resize_file(HANDLE file, std::uint64_t size) {
    LARGE_INTEGER pos{};
    pos.QuadPart = static_cast<LONGLONG>(size);

    if (!::SetFilePointerEx(file, pos, nullptr, FILE_BEGIN))
        throw_last_error("SetFilePointerEx failed");

    if (!::SetEndOfFile(file))
        throw_last_error("SetEndOfFile failed");
}

DWORD protection(MemoryMappedFile::Mode mode) {
    return mode == MemoryMappedFile::Mode::ReadOnly ? PAGE_READONLY : PAGE_READWRITE;
}

DWORD access(MemoryMappedFile::Mode mode) {
    return mode == MemoryMappedFile::Mode::ReadOnly ? FILE_MAP_READ : FILE_MAP_WRITE;
}

DWORD desired_access(MemoryMappedFile::Mode mode) {
    return mode == MemoryMappedFile::Mode::ReadOnly ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE);
}

} // namespace

MemoryMappedFile::MemoryMappedFile(MemoryMappedFile&& other) noexcept {
    move_from(std::move(other));
}

MemoryMappedFile& MemoryMappedFile::operator=(MemoryMappedFile&& other) noexcept {
    if (this != &other) {
        close();
        move_from(std::move(other));
    }

    return *this;
}

void MemoryMappedFile::move_from(MemoryMappedFile&& other) noexcept {
    file_ = other.file_;
    mapping_ = other.mapping_;
    data_ = other.data_;
    size_ = other.size_;
    mode_ = other.mode_;

    other.file_ = INVALID_HANDLE_VALUE;
    other.mapping_ = nullptr;
    other.data_ = nullptr;
    other.size_ = 0;
}

void MemoryMappedFile::open(const std::filesystem::path& path, Mode mode) {
    close();
    map_existing_file(path, mode);
}

void MemoryMappedFile::open(const std::filesystem::path& path, Mode mode, std::uint64_t length) {
    close();
    create_and_map_file(path, mode, length);
}

void MemoryMappedFile::map_existing_file(const std::filesystem::path& path, Mode mode) {
    mode_ = mode;

    file_ = ::CreateFileW(to_utf16(path).c_str(), desired_access(mode), FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);

    if (file_ == INVALID_HANDLE_VALUE)
        throw_last_error("CreateFileW failed");

    size_ = file_size(file_);

    if (size_ == 0)
        return;

    mapping_ = ::CreateFileMappingW(file_, nullptr, protection(mode), 0, 0, nullptr);

    if (!mapping_) {
        close();
        throw_last_error("CreateFileMappingW failed");
    }

    data_ = ::MapViewOfFile(mapping_, access(mode), 0, 0, 0);

    if (!data_) {
        close();
        throw_last_error("MapViewOfFile failed");
    }
}

void MemoryMappedFile::create_and_map_file(const std::filesystem::path& path, Mode mode, std::uint64_t length) {
    mode_ = mode;

    file_ = ::CreateFileW(to_utf16(path).c_str(), desired_access(mode), 0, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);

    if (file_ == INVALID_HANDLE_VALUE)
        throw_last_error("CreateFileW failed");

    resize_file(file_, length);
    size_ = length;

    if (size_ == 0)
        return;

    DWORD size_high = static_cast<DWORD>(size_ >> 32);
    DWORD size_low = static_cast<DWORD>(size_ & 0xffffffff);

    mapping_ = ::CreateFileMappingW(file_, nullptr, protection(mode), size_high, size_low, nullptr);

    if (!mapping_) {
        close();
        throw_last_error("CreateFileMappingW failed");
    }

    data_ = ::MapViewOfFile(mapping_, access(mode), 0, 0, 0);

    if (!data_) {
        close();
        throw_last_error("MapViewOfFile failed");
    }
}

void MemoryMappedFile::flush() {
    if (!data_)
        return;

    if (mode_ == Mode::ReadOnly)
        return;

    if (!::FlushViewOfFile(data_, 0))
        throw_last_error("FlushViewOfFile failed");

    if (!::FlushFileBuffers(file_))
        throw_last_error("FlushFileBuffers failed");
}

void MemoryMappedFile::close() noexcept {
    if (data_) {
        ::UnmapViewOfFile(data_);
        data_ = nullptr;
    }

    if (mapping_) {
        ::CloseHandle(mapping_);
        mapping_ = nullptr;
    }

    if (file_ != INVALID_HANDLE_VALUE) {
        ::CloseHandle(file_);
        file_ = INVALID_HANDLE_VALUE;
    }

    size_ = 0;
    mode_ = Mode::ReadOnly;
}

bool MemoryMappedFile::is_open() const noexcept {
    return data_ != nullptr || file_ != INVALID_HANDLE_VALUE;
}

} // namespace llmengine

#endif // _WIN32
