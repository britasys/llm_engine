#include <gtest/gtest.h>

#include "llmengine/gguf_loader.hpp"

#include <cstring>
#include <vector>

namespace llmengine::test {

class GGUFBuilder {
public:
    std::vector<std::byte> data;

    template<typename T> void write(T value) {
        auto ptr = reinterpret_cast<const std::byte*>(&value);

        data.insert(data.end(), ptr, ptr + sizeof(T));
    }

    void write_string(const std::string& s) {
        write<uint64_t>(s.size());

        auto ptr = reinterpret_cast<const std::byte*>(s.data());

        data.insert(data.end(), ptr, ptr + s.size());
    }

    void magic() { write<uint32_t>(0x46554747); }

    void header(uint64_t tensors, uint64_t metadata) {
        write<uint32_t>(3); // version
        write<uint64_t>(tensors);
        write<uint64_t>(metadata);
    }

    void metadata_string(const std::string& key, const std::string& value) {
        write_string(key);

        write<uint32_t>(8); // STRING

        write_string(value);
    }

    void metadata_uint(const std::string& key, uint64_t value) {
        write_string(key);

        write<uint32_t>(4); // UINT32

        write<uint32_t>(static_cast<uint32_t>(value));
    }

    void pad_to(size_t size) {
        if (data.size() < size)
            data.resize(size);
    }

    void tensor(const std::string& name, uint64_t offset) {
        write_string(name);

        write<uint32_t>(2);

        write<uint64_t>(2);
        write<uint64_t>(2);

        write<uint32_t>(GGML_TYPE_F32);

        write<uint64_t>(offset);

        // 2*2*f32
        pad_to(32 + offset + 16);
    }
};

static GGUFLoader load(std::vector<std::byte> bytes) {
    // Align after header + metadata + tensor info
    size_t aligned = (bytes.size() + 31) & ~size_t(31);

    bytes.resize(aligned + 4096);

    auto src = std::make_shared<MemoryByteSource>(std::move(bytes));

    GGUFLoader loader(src);

    loader.load();

    return loader;
}

TEST(GGUFLoaderTest, LoadsEmptyFile) {
    GGUFBuilder b;

    b.magic();
    b.header(0, 0);

    auto loader = load(std::move(b.data));

    EXPECT_EQ(loader.tensor_count(), 0);

    EXPECT_EQ(loader.metadata_count(), 0);
}

TEST(GGUFLoaderTest, ReadsStringMetadata) {
    GGUFBuilder b;

    b.magic();

    b.header(0, 1);

    b.metadata_string("general.name", "test-model");

    auto loader = load(std::move(b.data));

    EXPECT_EQ(loader.get_meta_string("general.name"), "test-model");
}

TEST(GGUFLoaderTest, ReadsIntegerMetadata) {
    GGUFBuilder b;

    b.magic();

    b.header(0, 1);

    b.metadata_uint("context_length", 4096);

    auto loader = load(std::move(b.data));

    EXPECT_EQ(loader.get_meta_int("context_length"), 4096);
}

TEST(GGUFLoaderTest, MissingMetadataThrows) {
    GGUFBuilder b;

    b.magic();
    b.header(0, 0);

    auto loader = load(std::move(b.data));

    EXPECT_THROW(loader.get_meta_string("missing"), std::runtime_error);
}

TEST(GGUFLoaderTest, TensorExists) {
    GGUFBuilder b;

    b.magic();

    b.header(1, 0);

    b.tensor("weight", 0);

    auto loader = load(std::move(b.data));

    EXPECT_TRUE(loader.has_tensor("weight"));

    EXPECT_FALSE(loader.has_tensor("unknown"));
}

TEST(GGUFLoaderTest, TensorInfoIsCorrect) {
    GGUFBuilder b;

    b.magic();

    b.header(1, 0);

    b.tensor("layer.weight", 128);

    auto loader = load(std::move(b.data));

    auto& info = loader.tensor_info("layer.weight");

    EXPECT_EQ(info.name, "layer.weight");

    EXPECT_EQ(info.shape.size(), 2);

    EXPECT_EQ(info.shape[0], 2);

    EXPECT_EQ(info.shape[1], 2);

    EXPECT_EQ(info.dtype, GGML_TYPE_F32);

    EXPECT_EQ(info.offset, 128);
}

TEST(GGUFLoaderTest, InvalidMagicThrows) {
    GGUFBuilder b;

    b.write<uint32_t>(0x12345678);

    b.header(0, 0);

    EXPECT_THROW(load(std::move(b.data)), std::runtime_error);
}

TEST(GGUFLoaderTest, TensorLookupThrows) {
    GGUFBuilder b;

    b.magic();
    b.header(0, 0);

    auto loader = load(std::move(b.data));

    EXPECT_THROW(loader.tensor_info("nope"), std::out_of_range);
}

TEST(GGUFLoaderTest, DuplicateTensorNamesFail) {
    GGUFBuilder b;

    b.magic();

    b.header(2, 0);

    b.tensor("same", 0);

    b.tensor("same", 4);

    EXPECT_THROW(load(std::move(b.data)), std::runtime_error);
}

TEST(GGUFLoaderTest, MetadataArrayTypeCheck) {
    GGUFBuilder b;

    b.magic();

    b.header(0, 0);

    auto loader = load(std::move(b.data));

    EXPECT_THROW({ (void)loader.get_meta_array("missing"); }, std::runtime_error);
}

} // namespace llmengine::test