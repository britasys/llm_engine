#include "llmengine/memory_mapped_file.hpp"

#ifndef _WIN32

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <filesystem>
#include <system_error>
#include <utility>

namespace llmengine {

namespace {

[[noreturn]]
void throw_errno(const char* message) {
    throw std::system_error(errno, std::generic_category(), message);
}

int open_flags(MemoryMappedFile::Mode mode) {
    if (mode == MemoryMappedFile::Mode::ReadOnly)
        return O_RDONLY;

    return O_RDWR | O_CREAT;
}

int protection(MemoryMappedFile::Mode mode) {
    if (mode == MemoryMappedFile::Mode::ReadOnly)
        return PROT_READ;

    return PROT_READ | PROT_WRITE;
}

std::uint64_t get_file_size(int fd) {
    struct stat st{};

    if (::fstat(fd, &st) != 0)
        throw_errno("fstat failed");

    return static_cast<std::uint64_t>(st.st_size);
}

void resize_file(int fd, std::uint64_t size) {
    if (::ftruncate(fd, static_cast<off_t>(size)) != 0) {
        throw_errno("ftruncate failed");
    }
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
    fd_ = other.fd_;
    data_ = other.data_;
    size_ = other.size_;
    mode_ = other.mode_;

    other.fd_ = -1;
    other.data_ = nullptr;
    other.size_ = 0;
}

void MemoryMappedFile::open(const std::filesystem::path& path, Mode mode) {
    close();

    fd_ = ::open(path.c_str(), open_flags(mode), 0644);

    if (fd_ < 0)
        throw_errno("open failed");

    mode_ = mode;

    size_ = get_file_size(fd_);

    if (size_ == 0)
        return;

    data_ = ::mmap(nullptr, size_, protection(mode), MAP_SHARED, fd_, 0);

    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        close();
        throw_errno("mmap failed");
    }
}

void MemoryMappedFile::open(const std::filesystem::path& path, Mode mode, std::uint64_t length) {
    close();

    fd_ = ::open(path.c_str(), open_flags(mode), 0644);

    if (fd_ < 0)
        throw_errno("open failed");

    resize_file(fd_, length);

    size_ = length;
    mode_ = mode;

    if (size_ == 0)
        return;

    data_ = ::mmap(nullptr, size_, protection(mode), MAP_SHARED, fd_, 0);

    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        close();
        throw_errno("mmap failed");
    }
}

void MemoryMappedFile::flush() {
    if (!data_)
        return;

    if (mode_ == Mode::ReadOnly)
        return;

    if (::msync(data_, size_, MS_SYNC) != 0) {
        throw_errno("msync failed");
    }
}

void MemoryMappedFile::close() noexcept {
    if (data_) {
        ::munmap(data_, size_);

        data_ = nullptr;
    }

    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }

    size_ = 0;
    mode_ = Mode::ReadOnly;
}

bool MemoryMappedFile::is_open() const noexcept {
    return data_ != nullptr || fd_ >= 0;
}

} // namespace llmengine

#endif // !_WIN32