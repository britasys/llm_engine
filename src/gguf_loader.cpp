#include "llmengine/gguf_loader.hpp"
#include "llmengine/binary_reader.hpp"

#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace llmengine {

namespace {
constexpr uint32_t GGUF_MAGIC = 0x46554747;
constexpr uint64_t MAX_STRING_LEN = 1ull << 24;
constexpr uint64_t MAX_ARRAY_LEN = 1ull << 24;
constexpr uint64_t MAX_TENSOR_COUNT = 1ull << 20;
constexpr uint64_t MAX_METADATA_COUNT = 1ull << 20;
constexpr uint32_t MAX_TENSOR_DIMS = 8;

bool mul_overflow(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
        return true;
    out = a * b;
    return false;
}

bool add_overflow(uint64_t a, uint64_t b, uint64_t& out) {
    if (b > std::numeric_limits<uint64_t>::max() - a)
        return true;
    out = a + b;
    return false;
}

} // namespace

ggml_type GGUFLoader::gguf_type_to_ggml(uint32_t t) {
    switch (t) {
    case 0:
        return GGML_TYPE_F32;
    case 1:
        return GGML_TYPE_F16;
    case 2:
        return GGML_TYPE_Q4_0;
    case 3:
        return GGML_TYPE_Q4_1;
    case 6:
        return GGML_TYPE_Q5_0;
    case 7:
        return GGML_TYPE_Q5_1;
    case 8:
        return GGML_TYPE_Q8_0;
    case 10:
        return GGML_TYPE_Q2_K;
    case 11:
        return GGML_TYPE_Q3_K;
    case 12:
        return GGML_TYPE_Q4_K;
    case 13:
        return GGML_TYPE_Q5_K;
    case 14:
        return GGML_TYPE_Q6_K;
    case 15:
        return GGML_TYPE_Q8_K;
    default:
        throw std::runtime_error("unsupported GGUF type");
    }
}

GGUFLoader::GGUFLoader(const std::filesystem::path& path) : path_(path) {}

void GGUFLoader::load() {
    mmap_.open(path_, MemoryMappedFile::Mode::ReadOnly);
    if (!mmap_.is_open())
        throw std::runtime_error("failed to open gguf file: " + path_.string());
    if (mmap_.size() < 24)
        throw std::runtime_error("gguf file too small: " + path_.string());

    BinaryReader reader(mmap_.bytes());
    if (reader.read<uint32_t>() != GGUF_MAGIC)
        throw std::runtime_error("invalid magic");
        
    version_ = reader.read<uint32_t>();
    tensor_count_ = reader.read<uint64_t>();
    metadata_count_ = reader.read<uint64_t>();

    if (tensor_count_ > MAX_TENSOR_COUNT)
        throw std::runtime_error("gguf tensor count exceeds limit");
    if (metadata_count_ > MAX_METADATA_COUNT)
        throw std::runtime_error("gguf metadata count exceeds limit");

    for (uint64_t i = 0; i < metadata_count_; i++) {
        std::string key = reader.read_string();
        uint32_t type = reader.read<uint32_t>();

        switch (type) {
        case 0:
            metadata_[key] = static_cast<uint64_t>(reader.read<uint8_t>());
            break;

        case 1:
            metadata_[key] = static_cast<int64_t>(reader.read<int8_t>());
            break;

        case 2:
            metadata_[key] = static_cast<uint64_t>(reader.read<uint16_t>());
            break;

        case 3:
            metadata_[key] = static_cast<int64_t>(reader.read<int16_t>());
            break;

        case 4:
            metadata_[key] = static_cast<uint64_t>(reader.read<uint32_t>());
            break;

        case 5:
            metadata_[key] = static_cast<int64_t>(reader.read<int32_t>());
            break;

        case 6:
            metadata_[key] = reader.read<float>();
            break;

        case 7:
            metadata_[key] = reader.read<uint8_t>() != 0;
            break;

        case 8:
            metadata_[key] = reader.read_string();
            break;

        case 9: {
            uint32_t array_type = reader.read<uint32_t>();
            uint64_t len = reader.read<uint64_t>();

            if (len > MAX_ARRAY_LEN)
                throw std::runtime_error("gguf array too large");

            auto arr = std::make_shared<MetadataArray>();

            arr->values.reserve(static_cast<size_t>(len));

            for (uint64_t j = 0; j < len; ++j) {

                switch (array_type) {

                case 0:
                    arr->values.push_back(static_cast<uint64_t>(reader.read<uint8_t>()));
                    break;

                case 1:
                    arr->values.push_back(static_cast<int64_t>(reader.read<int8_t>()));
                    break;

                case 2:
                    arr->values.push_back(static_cast<uint64_t>(reader.read<uint16_t>()));
                    break;

                case 3:
                    arr->values.push_back(static_cast<int64_t>(reader.read<int16_t>()));
                    break;

                case 4:
                    arr->values.push_back(static_cast<uint64_t>(reader.read<uint32_t>()));
                    break;

                case 5:
                    arr->values.push_back(static_cast<int64_t>(reader.read<int32_t>()));
                    break;

                case 6:
                    arr->values.push_back(reader.read<float>());
                    break;

                case 7:
                    arr->values.push_back(reader.read<uint8_t>() != 0);
                    break;

                case 8:
                    arr->values.push_back(reader.read_string());
                    break;

                default:
                    throw std::runtime_error("unsupported array type");
                }
            }

            metadata_[key] = arr;
            break;
        }

        case 10:
            metadata_[key] = reader.read<uint64_t>();
            break;

        case 11:
            metadata_[key] = reader.read<int64_t>();
            break;

        case 12:
            metadata_[key] = static_cast<float>(reader.read<double>());
            break;

        default:
            throw std::runtime_error("unsupported metadata type");
        }
    }

    for (uint64_t i = 0; i < tensor_count_; i++) {
        GGUFTensorInfo info;
        info.name = reader.read_string();
        if (tensors_.find(info.name) != tensors_.end())
            throw std::runtime_error("duplicate tensor name in gguf file: " + info.name);

        uint32_t ndim = reader.read<uint32_t>();
        if (ndim > MAX_TENSOR_DIMS)
            throw std::runtime_error("tensor has too many dimensions: " + info.name);

        info.shape.resize(ndim);
        uint64_t element_count = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            uint64_t dim = reader.read<uint64_t>();
            if (dim > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
                throw std::runtime_error("tensor dimension out of range: " + info.name);
            info.shape[d] = static_cast<int64_t>(dim);
            uint64_t next;
            if (mul_overflow(element_count, dim == 0 ? 1 : dim, next))
                throw std::runtime_error("tensor element count overflow: " + info.name);
            element_count = next;
        }

        uint32_t type = reader.read<uint32_t>();
        info.dtype = gguf_type_to_ggml(type);
        info.offset = reader.read<uint64_t>();

        int64_t blck_size = ggml_blck_size(info.dtype);
        size_t type_size = ggml_type_size(info.dtype);
        if (blck_size <= 0)
            throw std::runtime_error("invalid block size for tensor: " + info.name);
        if (element_count % static_cast<uint64_t>(blck_size) != 0)
            throw std::runtime_error("tensor element count not divisible by block size: " + info.name);

        uint64_t num_blocks = element_count / static_cast<uint64_t>(blck_size);
        uint64_t byte_size;
        if (mul_overflow(num_blocks, static_cast<uint64_t>(type_size), byte_size))
            throw std::runtime_error("tensor byte size overflow: " + info.name);
        info.nbytes = byte_size;

        tensors_.emplace(info.name, std::move(info));
    }

    uint64_t alignment = 32;

    auto it = metadata_.find("general.alignment");

    if (it != metadata_.end()) {
        uint64_t parsed = 0;

        try {
            parsed = std::stoull(std::get<std::string>(it->second));
        } catch (const std::exception&) {
            throw std::runtime_error("invalid general.alignment value in gguf file");
        }

        if (parsed == 0 || (parsed & (parsed - 1)) != 0)
            throw std::runtime_error("general.alignment must be a nonzero power of two");

        alignment = parsed;
    }

    const uint64_t header_end = static_cast<uint64_t>(reader.position());

    uint64_t padded;
    if (add_overflow(header_end, alignment - 1, padded))
        throw std::runtime_error("tensor data offset overflow");
    tensor_data_offset_ = padded & ~(alignment - 1);

    if (tensor_data_offset_ > mmap_.size())
        throw std::runtime_error("tensor data offset exceeds file size");

    for (const auto& [name, info] : tensors_) {
        uint64_t end;
        if (add_overflow(tensor_data_offset_, info.offset, end) || add_overflow(end, info.nbytes, end))
            throw std::runtime_error("tensor data range overflow: " + name);
        if (end > mmap_.size())
            throw std::runtime_error("tensor data out of bounds: " + name);
    }
}

bool GGUFLoader::has_tensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

const GGUFTensorInfo& GGUFLoader::tensor_info(const std::string& name) const {
    return tensors_.at(name);
}

const void* GGUFLoader::tensor_data(const std::string& name) const {
    const auto& t = tensors_.at(name);
    uint64_t start;
    if (add_overflow(tensor_data_offset_, t.offset, start) || start > mmap_.size() || t.nbytes > mmap_.size() - start)
        throw std::runtime_error("tensor data out of bounds: " + name);
    return mmap_.data() + start;
}

const MetadataArray& GGUFLoader::get_meta_array(std::string_view key) const {
    auto it = metadata_.find(std::string(key));

    if (it == metadata_.end())
        throw std::runtime_error("missing metadata");

    if (!std::holds_alternative<std::shared_ptr<MetadataArray>>(it->second))
        throw std::runtime_error("metadata is not array");

    const auto& array = std::get<std::shared_ptr<MetadataArray>>(it->second);

    if (!array)
        throw std::runtime_error("null metadata array");

    return *array;
}

std::string GGUFLoader::get_meta_string(std::string_view key) const {
    auto it = metadata_.find(std::string(key));

    if (it == metadata_.end())
        throw std::runtime_error("missing metadata");

    const auto& value = it->second;

    if (std::holds_alternative<std::string>(value))
        return std::get<std::string>(value);

    if (std::holds_alternative<uint64_t>(value))
        return std::to_string(std::get<uint64_t>(value));

    if (std::holds_alternative<int64_t>(value))
        return std::to_string(std::get<int64_t>(value));

    if (std::holds_alternative<float>(value))
        return std::to_string(std::get<float>(value));

    if (std::holds_alternative<bool>(value))
        return std::get<bool>(value) ? "true" : "false";

    if (std::holds_alternative<std::shared_ptr<MetadataArray>>(value))
        return "[array]";

    throw std::runtime_error("unsupported metadata type");
}

int64_t GGUFLoader::get_meta_int(std::string_view key) const {
    auto it = metadata_.find(std::string(key));

    if (it == metadata_.end())
        throw std::runtime_error("missing metadata: " + std::string(key));

    if (std::holds_alternative<int64_t>(it->second))
        return std::get<int64_t>(it->second);

    if (std::holds_alternative<uint64_t>(it->second))
        return static_cast<int64_t>(std::get<uint64_t>(it->second));

    throw std::runtime_error("metadata is not integer: " + std::string(key));
}

float GGUFLoader::get_meta_float(std::string_view key) const {
    auto it = metadata_.find(std::string(key));

    if (it == metadata_.end())
        throw std::runtime_error("missing metadata: " + std::string(key));

    if (std::holds_alternative<float>(it->second))
        return std::get<float>(it->second);

    if (std::holds_alternative<int64_t>(it->second))
        return static_cast<float>(std::get<int64_t>(it->second));

    if (std::holds_alternative<uint64_t>(it->second))
        return static_cast<float>(std::get<uint64_t>(it->second));

    throw std::runtime_error("metadata is not float: " + std::string(key));
}

} // namespace llmengine