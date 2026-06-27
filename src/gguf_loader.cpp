#include "llmengine/gguf_loader.hpp"
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

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

void GGUFLoader::ensure_remaining(std::istream& in, uint64_t n) {
    auto cur = in.tellg();
    if (cur < 0)
        throw std::runtime_error("invalid stream position");
    uint64_t pos = static_cast<uint64_t>(cur);
    if (pos > file_data_.size() || n > file_data_.size() - pos)
        throw std::runtime_error("unexpected end of gguf file");
}

std::string GGUFLoader::read_string(std::istream& in) {
    uint64_t len = read<uint64_t>(in);
    if (len > MAX_STRING_LEN)
        throw std::runtime_error("gguf string length exceeds limit");
    ensure_remaining(in, len);
    std::string s(len, '\0');
    if (len > 0)
        in.read(s.data(), static_cast<std::streamsize>(len));
    if (!in)
        throw std::runtime_error("failed to read string from gguf file");
    return s;
}

std::string GGUFLoader::read_array_as_string(std::istream& in, uint32_t elem_type) {
    uint64_t n = read<uint64_t>(in);
    if (n > MAX_ARRAY_LEN)
        throw std::runtime_error("gguf array length exceeds limit");
    std::string out = "[";
    for (uint64_t i = 0; i < n; i++) {
        if (i)
            out += ",";
        switch (elem_type) {
        case 0:
            out += std::to_string(read<uint8_t>(in));
            break;
        case 1:
            out += std::to_string(read<int8_t>(in));
            break;
        case 2:
            out += std::to_string(read<uint16_t>(in));
            break;
        case 3:
            out += std::to_string(read<int16_t>(in));
            break;
        case 4:
            out += std::to_string(read<uint32_t>(in));
            break;
        case 5:
            out += std::to_string(read<int32_t>(in));
            break;
        case 6:
            out += std::to_string(read<float>(in));
            break;
        case 7:
            out += read<uint8_t>(in) ? "true" : "false";
            break;
        case 8:
            out += "\"" + read_string(in) + "\"";
            break;
        case 10:
            out += std::to_string(read<uint64_t>(in));
            break;
        case 11:
            out += std::to_string(read<int64_t>(in));
            break;
        case 12:
            out += std::to_string(read<double>(in));
            break;
        default:
            throw std::runtime_error("unsupported array element type");
        }
    }
    out += "]";
    return out;
}

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
    std::ifstream f;
    f.open(path_, std::ios::binary | std::ios::in | std::ios::ate);
    if (!f)
        throw std::runtime_error("failed to open gguf file: " + path_.string());

    auto tell = f.tellg();
    if (tell < 0)
        throw std::runtime_error("failed to determine gguf file size: " + path_.string());
    size_t size = static_cast<size_t>(tell);
    f.seekg(0, std::ios::beg);
    if (!f)
        throw std::runtime_error("failed to seek gguf file: " + path_.string());

    if (size < 24)
        throw std::runtime_error("gguf file too small: " + path_.string());

    file_data_.resize(size);
    f.read(reinterpret_cast<char*>(file_data_.data()), static_cast<std::streamsize>(size));
    if (!f)
        throw std::runtime_error("failed to read gguf file: " + path_.string());
    f.close();

    struct VectorBuffer : std::streambuf {
        VectorBuffer(uint8_t* base, size_t size) {
            char* p = reinterpret_cast<char*>(base);
            setg(p, p, p + size);
        }

        pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override {
            if (!(which & std::ios_base::in))
                return pos_type(off_type(-1));

            char* base = eback();
            char* cur = gptr();
            char* end = egptr();
            off_type new_off;

            switch (dir) {
            case std::ios_base::beg:
                new_off = off;
                break;
            case std::ios_base::cur:
                new_off = (cur - base) + off;
                break;
            case std::ios_base::end:
                new_off = (end - base) + off;
                break;
            default:
                return pos_type(off_type(-1));
            }

            if (new_off < 0 || new_off > (end - base))
                return pos_type(off_type(-1));

            setg(base, base + new_off, end);
            return pos_type(new_off);
        }

        pos_type seekpos(pos_type pos, std::ios_base::openmode which) override { return seekoff(off_type(pos), std::ios_base::beg, which); }
    };
    VectorBuffer buf(file_data_.data(), file_data_.size());
    std::istream stream(&buf);

    if (read<uint32_t>(stream) != GGUF_MAGIC)
        throw std::runtime_error("invalid magic");
    version_ = read<uint32_t>(stream);
    tensor_count_ = read<uint64_t>(stream);
    metadata_count_ = read<uint64_t>(stream);

    if (tensor_count_ > MAX_TENSOR_COUNT)
        throw std::runtime_error("gguf tensor count exceeds limit");
    if (metadata_count_ > MAX_METADATA_COUNT)
        throw std::runtime_error("gguf metadata count exceeds limit");

    for (uint64_t i = 0; i < metadata_count_; i++) {
        std::string key = read_string(stream);
        uint32_t type = read<uint32_t>(stream);
        switch (type) {
        case 0:
            metadata_[key] = std::to_string(read<uint8_t>(stream));
            break;
        case 1:
            metadata_[key] = std::to_string(read<int8_t>(stream));
            break;
        case 2:
            metadata_[key] = std::to_string(read<uint16_t>(stream));
            break;
        case 3:
            metadata_[key] = std::to_string(read<int16_t>(stream));
            break;
        case 4:
            metadata_[key] = std::to_string(read<uint32_t>(stream));
            break;
        case 5:
            metadata_[key] = std::to_string(read<int32_t>(stream));
            break;
        case 6:
            metadata_[key] = std::to_string(read<float>(stream));
            break;
        case 7:
            metadata_[key] = read<uint8_t>(stream) ? "true" : "false";
            break;
        case 8:
            metadata_[key] = read_string(stream);
            break;
        case 9:
            metadata_[key] = read_array_as_string(stream, read<uint32_t>(stream));
            break;
        case 10:
            metadata_[key] = std::to_string(read<uint64_t>(stream));
            break;
        case 11:
            metadata_[key] = std::to_string(read<int64_t>(stream));
            break;
        case 12:
            metadata_[key] = std::to_string(read<double>(stream));
            break;
        default:
            throw std::runtime_error("unsupported metadata type");
        }
    }

    for (uint64_t i = 0; i < tensor_count_; i++) {
        GGUFTensorInfo info;
        info.name = read_string(stream);
        if (tensors_.find(info.name) != tensors_.end())
            throw std::runtime_error("duplicate tensor name in gguf file: " + info.name);

        uint32_t ndim = read<uint32_t>(stream);
        if (ndim > MAX_TENSOR_DIMS)
            throw std::runtime_error("tensor has too many dimensions: " + info.name);

        info.shape.resize(ndim);
        uint64_t element_count = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            uint64_t dim = read<uint64_t>(stream);
            if (dim > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
                throw std::runtime_error("tensor dimension out of range: " + info.name);
            info.shape[d] = static_cast<int64_t>(dim);
            uint64_t next;
            if (mul_overflow(element_count, dim == 0 ? 1 : dim, next))
                throw std::runtime_error("tensor element count overflow: " + info.name);
            element_count = next;
        }

        uint32_t type = read<uint32_t>(stream);
        info.dtype = gguf_type_to_ggml(type);
        info.offset = read<uint64_t>(stream);

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
    if (auto it = metadata_.find("general.alignment"); it != metadata_.end()) {
        uint64_t parsed;
        try {
            parsed = std::stoull(it->second);
        } catch (const std::exception&) {
            throw std::runtime_error("invalid general.alignment value in gguf file");
        }
        if (parsed == 0 || (parsed & (parsed - 1)) != 0)
            throw std::runtime_error("general.alignment must be a nonzero power of two");
        alignment = parsed;
    }

    auto cur = stream.tellg();
    if (cur < 0)
        throw std::runtime_error("invalid stream position computing tensor data offset");
    uint64_t header_end = static_cast<uint64_t>(cur);

    uint64_t padded;
    if (add_overflow(header_end, alignment - 1, padded))
        throw std::runtime_error("tensor data offset overflow");
    tensor_data_offset_ = padded & ~(alignment - 1);

    if (tensor_data_offset_ > file_data_.size())
        throw std::runtime_error("tensor data offset exceeds file size");

    for (const auto& [name, info] : tensors_) {
        uint64_t end;
        if (add_overflow(tensor_data_offset_, info.offset, end) || add_overflow(end, info.nbytes, end))
            throw std::runtime_error("tensor data range overflow: " + name);
        if (end > file_data_.size())
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
    if (add_overflow(tensor_data_offset_, t.offset, start) || start > file_data_.size() || t.nbytes > file_data_.size() - start)
        throw std::runtime_error("tensor data out of bounds: " + name);
    return file_data_.data() + start;
}

} // namespace llmengine