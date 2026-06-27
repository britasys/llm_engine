#pragma once

#include <cstdint>
#include <filesystem>
#include <ggml.h>
#include <istream>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmengine {

struct GGUFTensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    ggml_type dtype;
    uint64_t offset;
    uint64_t nbytes;
};

class GGUFLoader {
public:
    explicit GGUFLoader(const std::filesystem::path& path);

    void load();

    bool has_tensor(const std::string& name) const;
    const GGUFTensorInfo& tensor_info(const std::string& name) const;
    const void* tensor_data(const std::string& name) const;

    [[nodiscard]] const std::unordered_map<std::string, GGUFTensorInfo>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] const std::unordered_map<std::string, std::string>& metadata() const noexcept { return metadata_; }
    [[nodiscard]] uint32_t version() const noexcept { return version_; }
    [[nodiscard]] uint64_t tensor_count() const noexcept { return tensor_count_; }
    [[nodiscard]] uint64_t metadata_count() const noexcept { return metadata_count_; }
    [[nodiscard]] uint64_t tensor_data_offset() const noexcept { return tensor_data_offset_; }
    [[nodiscard]] const std::vector<uint8_t>& file_data() const noexcept { return file_data_; }

private:
    template<typename T> T read(std::istream& in) {
        ensure_remaining(in, sizeof(T));
        T value{};
        in.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!in)
            throw std::runtime_error("failed to read from gguf file");
        return value;
    }

    std::string read_string(std::istream& in);
    std::string read_array_as_string(std::istream& in, uint32_t type);
    static ggml_type gguf_type_to_ggml(uint32_t t);

    void ensure_remaining(std::istream& in, uint64_t n);

private:
    std::filesystem::path path_;
    uint32_t version_ = 0;
    uint64_t tensor_count_ = 0;
    uint64_t metadata_count_ = 0;
    uint64_t tensor_data_offset_ = 0;

    std::unordered_map<std::string, std::string> metadata_;
    std::unordered_map<std::string, GGUFTensorInfo> tensors_;
    std::vector<uint8_t> file_data_;
};

} // namespace llmengine