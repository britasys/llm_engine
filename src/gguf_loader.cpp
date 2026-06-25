#include "llmengine/gguf_loader.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace llmengine {

namespace {

constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF"

} // namespace

template<typename T> T GGUFLoader::read(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));

    if (!in) {
        throw std::runtime_error("GGUF: unexpected EOF");
    }

    return value;
}

std::string GGUFLoader::read_string(std::istream& in) {
    const auto len = read<uint64_t>(in);

    std::string str(len, '\0');
    in.read(str.data(), static_cast<std::streamsize>(len));

    if (!in) {
        throw std::runtime_error("GGUF: failed reading string");
    }

    return str;
}

std::string GGUFLoader::read_array_as_string(std::istream& in) {
    auto elem_type = static_cast<GGUFValueType>(read<uint32_t>(in));
    uint64_t count = read<uint64_t>(in);

    std::string out = "[";
    for (uint64_t i = 0; i < count; ++i) {
        if (i > 0)
            out += ",";

        switch (elem_type) {
        case GGUFValueType::UINT8:
            out += std::to_string(read<uint8_t>(in));
            break;

        case GGUFValueType::INT8:
            out += std::to_string(read<int8_t>(in));
            break;

        case GGUFValueType::UINT16:
            out += std::to_string(read<uint16_t>(in));
            break;

        case GGUFValueType::INT16:
            out += std::to_string(read<int16_t>(in));
            break;

        case GGUFValueType::UINT32:
            out += std::to_string(read<uint32_t>(in));
            break;

        case GGUFValueType::INT32:
            out += std::to_string(read<int32_t>(in));
            break;

        case GGUFValueType::UINT64:
            out += std::to_string(read<uint64_t>(in));
            break;

        case GGUFValueType::INT64:
            out += std::to_string(read<int64_t>(in));
            break;

        case GGUFValueType::FLOAT32:
            out += std::to_string(read<float>(in));
            break;

        case GGUFValueType::FLOAT64:
            out += std::to_string(read<double>(in));
            break;

        case GGUFValueType::BOOL:
            out += read<uint8_t>(in) ? "true" : "false";
            break;

        case GGUFValueType::STRING:
            out += "\"" + read_string(in) + "\"";
            break;

        case GGUFValueType::ARRAY:
            out += read_array_as_string(in);
            break;
            
        default:
            throw std::runtime_error("Unsupported GGUF array element type");
        }
    }
    out += "]";
    return out;
}

DType GGUFLoader::gguf_type_to_dtype(uint32_t type) {
    switch (type) {
    case 0:
        return DType::F32;
    case 1:
        return DType::F16;
    case 8:
        return DType::Q8_0;
    default:
        throw std::runtime_error("Unsupported GGUF tensor type");
    }
}

GGUFLoader::GGUFLoader(const std::filesystem::path& path) : path_(path) {}

void GGUFLoader::load() {
    std::ifstream file(path_, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Failed to open GGUF file");
    }

    const auto magic = read<uint32_t>(file);
    if (magic != GGUF_MAGIC) {
        throw std::runtime_error("Invalid GGUF magic");
    }

    version_ = read<uint32_t>(file);
    tensor_count_ = read<uint64_t>(file);
    metadata_count_ = read<uint64_t>(file);

    for (uint64_t i = 0; i < metadata_count_; ++i) {
        const auto key = read_string(file);
        const auto type = static_cast<GGUFValueType>(read<uint32_t>(file));

        switch (type) {

        case GGUFValueType::STRING:
            metadata_[key] = read_string(file);
            break;

        case GGUFValueType::ARRAY:
            metadata_[key] = read_array_as_string(file);
            break;

        case GGUFValueType::UINT8:
            metadata_[key] = std::to_string(read<uint8_t>(file));
            break;

        case GGUFValueType::INT8:
            metadata_[key] = std::to_string(read<int8_t>(file));
            break;

        case GGUFValueType::UINT16:
            metadata_[key] = std::to_string(read<uint16_t>(file));
            break;

        case GGUFValueType::INT16:
            metadata_[key] = std::to_string(read<int16_t>(file));
            break;

        case GGUFValueType::UINT32:
            metadata_[key] = std::to_string(read<uint32_t>(file));
            break;

        case GGUFValueType::INT32:
            metadata_[key] = std::to_string(read<int32_t>(file));
            break;

        case GGUFValueType::UINT64:
            metadata_[key] = std::to_string(read<uint64_t>(file));
            break;

        case GGUFValueType::INT64:
            metadata_[key] = std::to_string(read<int64_t>(file));
            break;

        case GGUFValueType::FLOAT32:
            metadata_[key] = std::to_string(read<float>(file));
            break;

        case GGUFValueType::FLOAT64:
            metadata_[key] = std::to_string(read<double>(file));
            break;

        case GGUFValueType::BOOL:
            metadata_[key] = read<uint8_t>(file) ? "true" : "false";
            break;

        default:
            throw std::runtime_error("Unsupported GGUF metadata type");
        }
    }

    for (uint64_t i = 0; i < tensor_count_; ++i) {
        GGUFTensorInfo info;

        info.name = read_string(file);
        const auto ndim = read<uint32_t>(file);
        info.shape.resize(ndim);

        for (uint32_t d = 0; d < ndim; ++d) {
            info.shape[d] = static_cast<int64_t>(read<uint64_t>(file));
        }

        const auto gguf_type = read<uint32_t>(file);
        info.dtype = gguf_type_to_dtype(gguf_type);

        info.offset = read<uint64_t>(file);

        tensors_.emplace(info.name, std::move(info));
    }

    file.seekg(0, std::ios::end);
    const auto size = static_cast<size_t>(file.tellg());
    file.seekg(0);

    file_data_.resize(size);

    file.read(reinterpret_cast<char*>(file_data_.data()), static_cast<std::streamsize>(size));

    if (!file) {
        throw std::runtime_error("Failed reading GGUF file");
    }
}

uint32_t GGUFLoader::version() const noexcept {
    return version_;
}

uint64_t GGUFLoader::tensor_count() const noexcept {
    return tensor_count_;
}

uint64_t GGUFLoader::metadata_count() const noexcept {
    return metadata_count_;
}

bool GGUFLoader::has_tensor(const std::string& name) const {
    return tensors_.contains(name);
}

const GGUFTensorInfo& GGUFLoader::tensor_info(const std::string& name) const {
    return tensors_.at(name);
}

Tensor GGUFLoader::tensor(const std::string& name) {
    const auto& info = tensors_.at(name);

    auto* ptr = file_data_.data() + info.offset;

    return Tensor::view(ptr, info.shape, info.dtype);
}

const std::unordered_map<std::string, std::string>& GGUFLoader::metadata() const noexcept {
    return metadata_;
}

} // namespace llmengine