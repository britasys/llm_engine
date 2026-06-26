#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.hpp"

namespace llmengine {

enum class GGUFValueType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};
struct GGUFTensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    DType dtype;
    uint64_t offset;
};

class GGUFLoader {
public:
    explicit GGUFLoader(const std::filesystem::path& path);

    ~GGUFLoader() = default;
    GGUFLoader(const GGUFLoader&) = delete;
    GGUFLoader& operator=(const GGUFLoader&) = delete;

    GGUFLoader(GGUFLoader&&) noexcept = default;
    GGUFLoader& operator=(GGUFLoader&&) noexcept = default;

    void load();

    [[nodiscard]] uint32_t version() const noexcept;
    [[nodiscard]] uint64_t tensor_count() const noexcept;
    [[nodiscard]] uint64_t metadata_count() const noexcept;
    [[nodiscard]] bool has_tensor(const std::string& name) const;
    [[nodiscard]] const GGUFTensorInfo& tensor_info(const std::string& name) const;
    [[nodiscard]] Tensor tensor(const std::string& name);
    [[nodiscard]] const std::unordered_map<std::string, std::string>& metadata() const noexcept;
    [[nodiscard]] const std::unordered_map<std::string, GGUFTensorInfo>& tensors() const noexcept;

private:
    std::filesystem::path path_;

    uint64_t tensor_data_offset_ = 0;

    std::vector<uint8_t> file_data_;

    uint32_t version_ = 0;
    uint64_t tensor_count_ = 0;
    uint64_t metadata_count_ = 0;

    std::unordered_map<std::string, std::string> metadata_;
    std::unordered_map<std::string, GGUFTensorInfo> tensors_;

    template<typename T> static T read(std::istream& in);
    static std::string read_string(std::istream& in);
    static std::string read_array_as_string(std::istream& in);
    static DType gguf_type_to_dtype(uint32_t gguf_type);
};

} // namespace llmengine