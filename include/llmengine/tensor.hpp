#pragma once

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

struct DTypeInfo {
    std::size_t block_bytes;
    std::size_t block_elems;
};
[[nodiscard]] DTypeInfo dtype_info(DType dt);
[[nodiscard]] const char* dtype_name(DType dt);

inline constexpr std::size_t kMaxDims = 4;

class Tensor {
public:
    Tensor() = default;

    static Tensor zeros(std::vector<int64_t> shape, DType dt = DType::F32);

    static Tensor view(void* data, std::vector<int64_t> shape, DType dt);

    // --- shape / metadata -------------------------------------------------
    [[nodiscard]] const std::vector<int64_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] int64_t dim(std::size_t i) const { return shape_.at(i); }
    [[nodiscard]] std::size_t ndim() const noexcept { return shape_.size(); }
    [[nodiscard]] int64_t numel() const noexcept { return numel_; }
    [[nodiscard]] DType dtype() const noexcept { return dtype_; }
    [[nodiscard]] bool owns_data() const noexcept { return owns_; }

    // --- raw data access ----------------------------------------------------
    [[nodiscard]] std::byte* data() noexcept { return data_; }
    [[nodiscard]] const std::byte* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size_bytes() const noexcept { return size_bytes_; }

    [[nodiscard]] std::span<float> as_f32();
    [[nodiscard]] std::span<const float> as_f32() const;

    [[nodiscard]] std::span<float> row(int64_t r);
    [[nodiscard]] std::span<const float> row(int64_t r) const;

    [[nodiscard]] std::string shape_string() const;

    ~Tensor();
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

    [[nodiscard]] Tensor clone() const;

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
