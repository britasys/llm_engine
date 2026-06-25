#pragma once
//
// tensor.hpp
// ---------------------------------------------------------------------------
// CONCEPT: The Tensor is the single data structure that flows through the
// entire engine -- token embeddings, attention scores, weight matrices, the
// final logits. Get this abstraction right and everything downstream (ops,
// model, kv-cache) becomes simple. Get it wrong and you'll fight aliasing
// and shape bugs for the rest of the project.
//
// Design decisions worth understanding:
//
//  1. ROW-MAJOR, CONTIGUOUS STORAGE.
//     A Tensor of shape {rows, cols} stores element (r, c) at
//     data[r * cols + c]. This matches how llama.cpp (and numpy, and
//     PyTorch by default) lays out 2D weight matrices, which matters when
//     we eventually parse real GGUF files written by those tools.
//
//  2. OWNING vs NON-OWNING ("views").
//     Loading a multi-gigabyte model file and copying every weight matrix
//     into a fresh heap buffer would double your memory usage and be slow.
//     Real engines memory-map the file and construct *views* directly into
//     mapped memory. We model that here with an explicit owns_ flag rather
//     than hiding it in a smart pointer, so the lifetime story stays
//     visible in the type's contract: a view must not outlive the memory
//     it points into.
//
//  3. NO TEMPLATES (YET).
//     llama.cpp's ggml uses a single tagged-union-style tensor that can
//     hold f32, f16, or k-bit quantized data, switched on at runtime via an
//     enum. We follow that pattern (rather than templating Tensor<T>)
//     because quantized formats (Q4_0, Q8_0, ...) don't correspond to any
//     C++ arithmetic type -- they're packed bytes with a custom layout.
//     A runtime dtype tag is what lets one piece of model-loading code
//     handle "whatever quantization the file uses" uniformly.
//
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace llmengine {

enum class DType : uint8_t {
    F32,
    F16,

    Q4_0,
    Q4_1,

    Q5_0,
    Q5_1,

    Q8_0,
    Q8_1,

    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,

    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    IQ1_M
};

// Returns the size in bytes of one *block* for a given dtype, and how many
// elements that block encodes. For F32/F16 a "block" is just one element.
// Quantized types group elements (e.g. 32 at a time) sharing one scale
// factor, which is why we separate "block size in bytes" from "elements
// per block" instead of assuming size = count * sizeof(T).
struct DTypeInfo {
    std::size_t block_bytes;
    std::size_t block_elems;
};
[[nodiscard]] DTypeInfo dtype_info(DType dt);
[[nodiscard]] const char* dtype_name(DType dt);

// Tensors here are at most 4D (batch, sequence, heads, head_dim style
// usage covers everything a decoder-only transformer needs). A fixed-size
// std::array of dims avoids a heap allocation just to describe a shape.
inline constexpr std::size_t kMaxDims = 4;

class Tensor {
public:
    Tensor() = default;

    // Allocates owned, zero-initialized storage for `shape` elements of `dt`.
    static Tensor zeros(std::vector<int64_t> shape, DType dt = DType::F32);

    // Constructs a NON-OWNING view over externally managed memory, e.g. a
    // memory-mapped GGUF file region. Caller guarantees `data` stays valid
    // for the view's lifetime -- the Tensor does not extend it.
    static Tensor view(void* data, std::vector<int64_t> shape, DType dt);

    // --- shape / metadata -------------------------------------------------
    [[nodiscard]] const std::vector<int64_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] int64_t dim(std::size_t i) const { return shape_.at(i); }
    [[nodiscard]] std::size_t ndim() const noexcept { return shape_.size(); }
    [[nodiscard]] int64_t numel() const noexcept { return numel_; }
    [[nodiscard]] DType dtype() const noexcept { return dtype_; }
    [[nodiscard]] bool owns_data() const noexcept { return owns_; }

    // --- raw data access ----------------------------------------------------
    // Returned as bytes because quantized dtypes don't map to a C++ type;
    // typed accessors (as_f32, etc.) live here for the common case.
    [[nodiscard]] std::byte* data() noexcept { return data_; }
    [[nodiscard]] const std::byte* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size_bytes() const noexcept { return size_bytes_; }

    // Typed view helper for the common case (F32 contiguous data).
    // Throws if the tensor isn't actually F32 -- this is a defensive check,
    // not a place to silently reinterpret bytes.
    [[nodiscard]] std::span<float> as_f32();
    [[nodiscard]] std::span<const float> as_f32() const;

    // 2D convenience accessor: row r of a [rows, cols] F32 tensor.
    // Demonstrates the row-major layout math directly: row r starts at
    // offset r * cols.
    [[nodiscard]] std::span<float> row(int64_t r);
    [[nodiscard]] std::span<const float> row(int64_t r) const;

    [[nodiscard]] std::string shape_string() const;

    ~Tensor();
    Tensor(const Tensor&) = delete;            // owning copies would be
    Tensor& operator=(const Tensor&) = delete; // surprisingly expensive;
    Tensor(Tensor&&) noexcept;                 // force callers to be
    Tensor& operator=(Tensor&&) noexcept;      // explicit (use clone()).

    [[nodiscard]] Tensor clone() const; // explicit deep copy when you really need one

private:
    std::vector<int64_t> shape_{};
    int64_t numel_ = 0;
    DType dtype_ = DType::F32;
    std::byte* data_ = nullptr;
    std::size_t size_bytes_ = 0;
    bool owns_ = false;

    void free_if_owned() noexcept;
};

} // namespace llmengine
