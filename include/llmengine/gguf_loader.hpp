#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.hpp"

namespace llmengine {

/**
 * @brief GGUFValueType represents the internal metadata value types supported by the GGUF spec.
 * * GGUF files separate metadata keys from their values. Every metadata value is prefixed by 
 * this 32-bit integer identifier so the parser knows how many bytes to read next or how to 
 * parse structured data (like arrays or strings).
 */
enum class GGUFValueType : uint32_t {
    // --- Primitive Scalar Types ---
    // These are read as direct binary values from the stream.
    UINT8 = 0,    // 1 byte unsigned int. Often used for flag sets or small enumerations.
    INT8 = 1,     // 1 byte signed int.
    UINT16 = 2,   // 2 bytes unsigned int. 
    INT16 = 3,    // 2 bytes signed int. Usually used for legacy structural IDs.
    UINT32 = 4,   // 4 bytes unsigned int. Used frequently for sizes or type indexing.
    INT32 = 5,    // 4 bytes signed int.
    FLOAT32 = 6,  // 4 bytes single-precision float. Standard scalar metadata format (e.g., rope_theta).
    BOOL = 7,     // 1 byte boolean value (0 = false, 1 = true). Used for toggles like causal masks.

    // --- Complex / Structured Types ---
    // These require multi-stage parsing because they have variable sizes.
    STRING = 8,   // Prefixed by a uint64_t length, followed immediately by that many raw bytes.
                  // It does NOT have a null terminator '\0', so you must use std::string(ptr, length).

    ARRAY = 9,    // A nested structure. It is immediately followed by:
                  // 1. A 32-bit GGUFValueType denoting what type *every* element in this array is.
                  // 2. A 64-bit uint64_t denoting the number of elements in the array.
                  // 3. The raw element sequence packed tightly back-to-back.
                  // Useful for tracking token sequences or architectural layer configurations.

    // --- Large Precision Scalar Types ---
    UINT64 = 10,  // 8 bytes unsigned int. Crucial for massive file sizes, offsets, or token counts.
    INT64 = 11,   // 8 bytes signed int. Standard for tracking shape dimensions and tensor indices.
    FLOAT64 = 12, // 8 bytes double-precision float. Used when extreme hyperparameter precision is needed.
};

/**
 * @brief Metadata tracking where and how a tensor's weights are stored within the binary file.
 * * The GGUF file structure has a header index containing a list of these descriptors *before* * the actual tensor binary data block. This allows the loader to map out the memory layout 
 * of all model parameters without reading gigabytes of raw weights into RAM prematurely.
 */
struct GGUFTensorInfo {
    std::string name;             // The weight name (e.g., "blk.0.attn_q.weight").
    std::vector<int64_t> shape;   // Dimensions of the tensor. Note: GGUF stores dimensions in REVERSE order (typically column-major, e.g., [columns, rows]).
    DType dtype;                  // Engine-specific data type (FP32, FP16, or quantized formats like Q4_0, Q8_0).
    uint64_t offset;              // Absolute or relative byte offset pointing to where the raw tensor data starts in the data block.
};

/**
 * @brief High-performance weight loader optimized for GGUF model files.
 * * The loader reads the file header, parses global hyper-parameters (like context length, 
 * embedding dimension, attention heads) into a metadata map, maps tensor boundaries, 
 * and handles raw weight allocation/zero-copy loading into the engine's `Tensor` objects.
 */
class GGUFLoader {
public:
    /**
     * @brief Instantiates the loader with a target file path.
     * @param path Path to the valid `.gguf` file.
     */
    explicit GGUFLoader(const std::filesystem::path& path);

    /**
     * @brief Parses the GGUF file header and fills internal metadata and tensor maps.
     * * This performs the structural discovery phase. It validates the "GGUF" magic bytes, 
     * reads the version, loops through the `metadata_count` to parse configuration KVs, 
     * and loops through `tensor_count` to populate the `tensors_` directory.
     */
    void load();

    // --- Basic Getters ---
    [[nodiscard]] uint32_t version() const noexcept;
    [[nodiscard]] uint64_t tensor_count() const noexcept;
    [[nodiscard]] uint64_t metadata_count() const noexcept;

    /**
     * @brief Checks if a specific weight matrix exists in the file index.
     * @param name The layer weight identifier (e.g., "token_embd.weight").
     */
    [[nodiscard]] bool has_tensor(const std::string& name) const;

    /**
     * @brief Retrieves the structural information (shape, layout, offset) of a tensor without loading its data.
     */
    [[nodiscard]] const GGUFTensorInfo& tensor_info(const std::string& name) const;

    /**
     * @brief Constructs and returns an executable Tensor object for inference.
     * * In an optimized engine, this either allocates memory and copies the raw weights, 
     * or uses `mmap` to map the weight data directly from disk into memory, preventing high 
     * RAM spikes during model instantiation.
     */
    [[nodiscard]] Tensor tensor(const std::string& name);

    /**
     * @brief Exposes parsed hyper-parameters (e.g., "llama.context_length", "llama.embedding_length").
     */
    [[nodiscard]] const std::unordered_map<std::string, std::string>& metadata() const noexcept;

private:
    std::filesystem::path path_;

    /**
     * @brief Buffer containing file bytes. 
     * @note For research engines, reading into a vector is simple. For performance on large models, 
     * this is typically replaced by OS-level memory mapping (mmap on Unix, CreateFileMapping on Windows).
     */
    std::vector<std::byte> file_data_;

    // Parsed structural values directly from the GGUF file header.
    uint32_t version_ = 0;
    uint64_t tensor_count_ = 0;
    uint64_t metadata_count_ = 0;

    // Faster lookups for configuration values and weight allocations.
    std::unordered_map<std::string, std::string> metadata_;
    std::unordered_map<std::string, GGUFTensorInfo> tensors_;

    /**
     * @brief Helper to deserialize trivial/primitive types directly from the binary stream.
     * Handles little-endian conversions required by the GGUF spec.
     */
    template<typename T> static T read(std::istream& in);

    /**
     * @brief Helper to read GGUF-formatted strings.
     * GGUF strings are encoded as a uint64_t length prefix followed by string data bytes.
     */
    static std::string read_string(std::istream& in);

    /**
     * @brief Maps the native GGUF type enumeration token to your engine's internal DType format.
     * * This is vital because GGUF handles highly complex block-quantized formats (like Q4_K_M, Q5_K_S),
     * and your tensor engine needs to know how to map these integers to the correct compute kernels.
     */
    static DType gguf_type_to_dtype(uint32_t gguf_type);
};

} // namespace llmengine