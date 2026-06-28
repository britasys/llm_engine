#pragma once

#include "llmengine/memory_mapped_file.hpp"

#include <ggml.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace llmengine {

struct GGUFTensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    ggml_type dtype;
    uint64_t offset;
    uint64_t nbytes;
};

struct MetadataArray;
using MetadataValue = std::variant<uint64_t, int64_t, float, bool, std::string, std::shared_ptr<MetadataArray>>;
struct MetadataArray {
    std::vector<MetadataValue> values;
};

using TensorMap = std::unordered_map<std::string, GGUFTensorInfo>;
using MetaMap = std::unordered_map<std::string, MetadataValue>;

// Abstract byte source for testability
class ByteSource {
public:
    virtual ~ByteSource() = default;

    virtual std::span<const std::byte> bytes() const noexcept = 0;
};

// Production implementation
class MMapByteSource final : public ByteSource {
public:
    explicit MMapByteSource(const std::filesystem::path& path) : mmap_(path) {}

    std::span<const std::byte> bytes() const noexcept override { return mmap_.bytes(); }

private:
    MemoryMappedFile mmap_;
};

// Test implementation
class MemoryByteSource final : public ByteSource {
public:
    explicit MemoryByteSource(std::vector<std::byte> data) : data_(std::move(data)) {}

    std::span<const std::byte> bytes() const noexcept override { return data_; }

private:
    std::vector<std::byte> data_;
};

class GGUFLoader {
public:
    explicit GGUFLoader(const std::filesystem::path& path);
    explicit GGUFLoader(std::shared_ptr<ByteSource> source);

    void load();

    bool has_tensor(const std::string& name) const;
    const GGUFTensorInfo& tensor_info(const std::string& name) const;
    const void* tensor_data(const std::string& name) const;

    [[nodiscard]] const TensorMap& tensors() const noexcept { return tensors_; }
    [[nodiscard]] const MetaMap& metadata() const noexcept { return metadata_; }
    [[nodiscard]] uint32_t version() const noexcept { return version_; }
    [[nodiscard]] uint64_t tensor_count() const noexcept { return tensor_count_; }
    [[nodiscard]] uint64_t metadata_count() const noexcept { return metadata_count_; }
    [[nodiscard]] uint64_t tensor_data_offset() const noexcept { return tensor_data_offset_; }
    [[nodiscard]] std::span<const std::byte> file_data() const noexcept { return source_->bytes(); }
    [[nodiscard]] const MetadataArray& get_meta_array(std::string_view key) const;
    [[nodiscard]] std::string get_meta_string(std::string_view key) const;
    [[nodiscard]] int64_t get_meta_int(std::string_view key) const;
    [[nodiscard]] float get_meta_float(std::string_view key) const;

private:
    void parse();
    static ggml_type gguf_type_to_ggml(uint32_t t);

private:
    std::shared_ptr<ByteSource> source_;

    uint32_t version_ = 0;
    uint64_t tensor_count_ = 0;
    uint64_t metadata_count_ = 0;
    uint64_t tensor_data_offset_ = 0;

    TensorMap tensors_;
    MetaMap metadata_;
};

} // namespace llmengine