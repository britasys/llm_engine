// src/model.cpp
#include "llmengine/model.hpp"
#include "llmengine/ops.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace llmengine {

namespace {

float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // subnormal half -> normalized float
            exp = 1;
            while (!(mant & 0x400u)) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FFu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13); // inf/nan
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

Tensor dequantize_q4_0_to_f32(const Tensor& t) {
    constexpr int64_t kBlockElems = 32;
    constexpr std::size_t kBlockBytes = 2 + 16; // d(f16) + qs[16]

    Tensor out = Tensor::zeros(t.shape(), DType::F32);
    auto y = out.as_f32();
    const std::byte* src = t.data();
    std::size_t n_blocks = static_cast<std::size_t>(t.numel()) / kBlockElems;

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::byte* block = src + b * kBlockBytes;
        uint16_t d_bits;
        std::memcpy(&d_bits, block, sizeof(uint16_t));
        float d = f16_to_f32(d_bits);
        const auto* qs = reinterpret_cast<const uint8_t*>(block + 2);
        float* yb = y.data() + b * static_cast<std::size_t>(kBlockElems);
        for (int j = 0; j < 16; ++j) {
            int x0 = (qs[j] & 0x0F) - 8;
            int x1 = (qs[j] >> 4) - 8;
            yb[j] = static_cast<float>(x0) * d;
            yb[j + 16] = static_cast<float>(x1) * d;
        }
    }
    return out;
}

Tensor dequantize_q4_1_to_f32(const Tensor& t) {
    constexpr int64_t kBlockElems = 32;
    constexpr std::size_t kBlockBytes = 2 + 2 + 16; // d(f16) + m(f16) + qs[16]

    Tensor out = Tensor::zeros(t.shape(), DType::F32);
    auto y = out.as_f32();
    const std::byte* src = t.data();
    std::size_t n_blocks = static_cast<std::size_t>(t.numel()) / kBlockElems;

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::byte* block = src + b * kBlockBytes;
        uint16_t d_bits, m_bits;
        std::memcpy(&d_bits, block, sizeof(uint16_t));
        std::memcpy(&m_bits, block + 2, sizeof(uint16_t));
        float d = f16_to_f32(d_bits), m = f16_to_f32(m_bits);
        const auto* qs = reinterpret_cast<const uint8_t*>(block + 4);
        float* yb = y.data() + b * static_cast<std::size_t>(kBlockElems);
        for (int j = 0; j < 16; ++j) {
            int x0 = qs[j] & 0x0F;
            int x1 = qs[j] >> 4;
            yb[j] = static_cast<float>(x0) * d + m;
            yb[j + 16] = static_cast<float>(x1) * d + m;
        }
    }
    return out;
}

Tensor dequantize_q5_1_to_f32(const Tensor& t) {
    constexpr int64_t kBlockElems = 32;
    constexpr std::size_t kBlockBytes = 2 + 2 + 4 + 16; // d + m + qh(u32) + qs[16]

    Tensor out = Tensor::zeros(t.shape(), DType::F32);
    auto y = out.as_f32();
    const std::byte* src = t.data();
    std::size_t n_blocks = static_cast<std::size_t>(t.numel()) / kBlockElems;

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::byte* block = src + b * kBlockBytes;
        uint16_t d_bits, m_bits;
        std::memcpy(&d_bits, block, sizeof(uint16_t));
        std::memcpy(&m_bits, block + 2, sizeof(uint16_t));
        float d = f16_to_f32(d_bits), m = f16_to_f32(m_bits);
        uint32_t qh;
        std::memcpy(&qh, block + 4, sizeof(uint32_t));
        const auto* qs = reinterpret_cast<const uint8_t*>(block + 8);
        float* yb = y.data() + b * static_cast<std::size_t>(kBlockElems);
        for (int j = 0; j < 16; ++j) {
            uint8_t xh0 = static_cast<uint8_t>((qh >> (j + 0)) << 4) & 0x10;
            uint8_t xh1 = static_cast<uint8_t>((qh >> (j + 12))) & 0x10;
            int x0 = (qs[j] & 0x0F) | xh0;
            int x1 = (qs[j] >> 4) | xh1;
            yb[j] = static_cast<float>(x0) * d + m;
            yb[j + 16] = static_cast<float>(x1) * d + m;
        }
    }
    return out;
}

Tensor dequantize_q2_k_to_f32(const Tensor& t) {
    constexpr int64_t kBlockElems = 256;
    constexpr std::size_t kBlockBytes = 16 + 64 + 2 + 2; // scales+qs+d+dmin

    Tensor out = Tensor::zeros(t.shape(), DType::F32);
    auto y = out.as_f32();
    const std::byte* src = t.data();
    std::size_t n_blocks = static_cast<std::size_t>(t.numel()) / kBlockElems;

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::byte* block = src + b * kBlockBytes;
        const auto* scales = reinterpret_cast<const uint8_t*>(block);
        const auto* q = reinterpret_cast<const uint8_t*>(block + 16);
        uint16_t d_bits, dmin_bits;
        std::memcpy(&d_bits, block + 80, sizeof(uint16_t));
        std::memcpy(&dmin_bits, block + 82, sizeof(uint16_t));
        float d = f16_to_f32(d_bits), dmin = f16_to_f32(dmin_bits);

        float* yp = y.data() + b * static_cast<std::size_t>(kBlockElems);
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            const uint8_t* qn = q + (n / 128) * 32;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = scales[is++];
                float dl = d * static_cast<float>(sc & 0x0F);
                float ml = dmin * static_cast<float>(sc >> 4);
                for (int l = 0; l < 16; ++l)
                    *yp++ = dl * static_cast<float>((qn[l] >> shift) & 3) - ml;

                sc = scales[is++];
                dl = d * static_cast<float>(sc & 0x0F);
                ml = dmin * static_cast<float>(sc >> 4);
                for (int l = 0; l < 16; ++l)
                    *yp++ = dl * static_cast<float>((qn[l + 16] >> shift) & 3) - ml;

                shift += 2;
            }
        }
    }
    return out;
}

Tensor dequantize_q3_k_to_f32(const Tensor& t) {
    constexpr int64_t kBlockElems = 256;
    constexpr std::size_t kBlockBytes = 32 + 64 + 12 + 2; // hmask+qs+scales+d

    Tensor out = Tensor::zeros(t.shape(), DType::F32);
    auto y = out.as_f32();
    const std::byte* src = t.data();
    std::size_t n_blocks = static_cast<std::size_t>(t.numel()) / kBlockElems;
    const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::byte* block = src + b * kBlockBytes;
        const auto* hm = reinterpret_cast<const uint8_t*>(block);
        const auto* q = reinterpret_cast<const uint8_t*>(block + 32);
        const auto* raw_scales = block + 96;
        uint16_t d_bits;
        std::memcpy(&d_bits, block + 108, sizeof(uint16_t));
        float d_all = f16_to_f32(d_bits);

        // Unpack 16 signed 6-bit scales out of the 12-byte packed array
        // (ggml's bit-trick: 4 groups of 4-byte words combined with
        // shifted/masked copies of a shared third word).
        uint32_t aux[4];
        std::memcpy(aux, raw_scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int8_t scales[16];
        std::memcpy(scales, aux, 16);
        for (auto& s : scales)
            s = static_cast<int8_t>(s - 32);

        float* yp = y.data() + b * static_cast<std::size_t>(kBlockElems);
        int is = 0;
        const uint8_t* qn = q;
        uint8_t m = 1;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * static_cast<float>(scales[is++]);
                for (int l = 0; l < 16; ++l)
                    *yp++ = dl * (static_cast<float>((qn[l] >> shift) & 3) -
                                  ((hm[l] & m) ? 0.0f : 4.0f));

                dl = d_all * static_cast<float>(scales[is++]);
                for (int l = 0; l < 16; ++l)
                    *yp++ = dl * (static_cast<float>((qn[l + 16] >> shift) & 3) -
                                  ((hm[l + 16] & m) ? 0.0f : 4.0f));

                shift += 2;
                m = static_cast<uint8_t>(m << 1);
            }
            qn += 32;
        }
    }
    return out;
}

void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = static_cast<uint8_t>((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

Tensor dequantize_q4_k_to_f32(const Tensor& t) {
    constexpr int64_t kBlockElems = 256;
    constexpr std::size_t kBlockBytes = 2 + 2 + 12 + 128; // d+dmin+scales+qs

    Tensor out = Tensor::zeros(t.shape(), DType::F32);
    auto y = out.as_f32();
    const std::byte* src = t.data();
    std::size_t n_blocks = static_cast<std::size_t>(t.numel()) / kBlockElems;

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::byte* block = src + b * kBlockBytes;
        uint16_t d_bits, dmin_bits;
        std::memcpy(&d_bits, block, sizeof(uint16_t));
        std::memcpy(&dmin_bits, block + 2, sizeof(uint16_t));
        float d = f16_to_f32(d_bits), dmin = f16_to_f32(dmin_bits);
        const auto* scales = reinterpret_cast<const uint8_t*>(block + 4);
        const auto* q = reinterpret_cast<const uint8_t*>(block + 16);

        float* yp = y.data() + b * static_cast<std::size_t>(kBlockElems);
        int is = 0;
        const uint8_t* qn = q;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, sc, m);
            float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m);
            float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l)
                *yp++ = d1 * static_cast<float>(qn[l] & 0x0F) - m1;
            for (int l = 0; l < 32; ++l)
                *yp++ = d2 * static_cast<float>(qn[l] >> 4) - m2;
            qn += 32;
            is += 2;
        }
    }
    return out;
}

Tensor dequantize_q5_k_to_f32(const Tensor& t) {
    constexpr int64_t kBlockElems = 256;
    constexpr std::size_t kBlockBytes = 2 + 2 + 12 + 32 + 128; // d+dmin+scales+qh+qs

    Tensor out = Tensor::zeros(t.shape(), DType::F32);
    auto y = out.as_f32();
    const std::byte* src = t.data();
    std::size_t n_blocks = static_cast<std::size_t>(t.numel()) / kBlockElems;

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::byte* block = src + b * kBlockBytes;
        uint16_t d_bits, dmin_bits;
        std::memcpy(&d_bits, block, sizeof(uint16_t));
        std::memcpy(&dmin_bits, block + 2, sizeof(uint16_t));
        float d = f16_to_f32(d_bits), dmin = f16_to_f32(dmin_bits);
        const auto* scales = reinterpret_cast<const uint8_t*>(block + 4);
        const auto* qh = reinterpret_cast<const uint8_t*>(block + 16);
        const auto* ql = reinterpret_cast<const uint8_t*>(block + 48);

        float* yp = y.data() + b * static_cast<std::size_t>(kBlockElems);
        int is = 0;
        const uint8_t* qn = ql;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, sc, m);
            float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m);
            float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l)
                *yp++ = d1 * static_cast<float>((qn[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l)
                *yp++ = d2 * static_cast<float>((qn[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            qn += 32;
            is += 2;
            u1 = static_cast<uint8_t>(u1 << 2);
            u2 = static_cast<uint8_t>(u2 << 2);
        }
    }
    return out;
}

Tensor dequantize_q8_k_to_f32(const Tensor& t) {
    constexpr int64_t kBlockElems = 256;
    constexpr std::size_t kBlockBytes = 4 + 256 + 32; // d(f32) + qs[256] + bsums[16] i16

    Tensor out = Tensor::zeros(t.shape(), DType::F32);
    auto y = out.as_f32();
    const std::byte* src = t.data();
    std::size_t n_blocks = static_cast<std::size_t>(t.numel()) / kBlockElems;

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::byte* block = src + b * kBlockBytes;
        float d;
        std::memcpy(&d, block, sizeof(float));
        const auto* qs = reinterpret_cast<const int8_t*>(block + 4);
        float* yp = y.data() + b * static_cast<std::size_t>(kBlockElems);
        for (int j = 0; j < 256; ++j)
            yp[j] = d * static_cast<float>(qs[j]);
    }
    return out;
}

Tensor dequantize_q6_k_to_f32(const Tensor& t) {
    constexpr int64_t kBlockElems = 256;
    constexpr std::size_t kBlockBytes = 128 + 64 + 16 + 2; // ql + qh + scales + d

    Tensor out = Tensor::zeros(t.shape(), DType::F32);
    auto out_span = out.as_f32();

    const std::byte* src = t.data();
    std::size_t n_blocks = static_cast<std::size_t>(t.numel()) / kBlockElems;

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::byte* block = src + b * kBlockBytes;
        const auto* ql = reinterpret_cast<const uint8_t*>(block);       // 128 bytes
        const auto* qh = reinterpret_cast<const uint8_t*>(block + 128); // 64 bytes
        const auto* sc = reinterpret_cast<const int8_t*>(block + 192);  // 16 bytes

        uint16_t d_bits;
        std::memcpy(&d_bits, block + 208, sizeof(uint16_t));
        float d = f16_to_f32(d_bits); // see the Q5_0 fix for this helper

        float* y = out_span.data() + b * static_cast<std::size_t>(kBlockElems);

        // Each 256-element block splits into two 128-element halves, each
        // processed identically against its own slice of ql/qh/scales --
        // this mirrors ggml's reference dequantize_row_q6_K exactly.
        for (int half = 0; half < 2; ++half) {
            const uint8_t* ql_h = ql + half * 64;
            const uint8_t* qh_h = qh + half * 32;
            const int8_t* sc_h = sc + half * 8;
            float* y_h = y + half * 128;

            for (int l = 0; l < 32; ++l) {
                int is = l / 16; // which of the two 16-wide scale groups in this half

                int q1c = (ql_h[l] & 0x0F) | (((qh_h[l] >> 0) & 3) << 4);
                int q2c = (ql_h[l + 32] & 0x0F) | (((qh_h[l] >> 2) & 3) << 4);
                int q3c = (ql_h[l] >> 4) | (((qh_h[l] >> 4) & 3) << 4);
                int q4c = (ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3) << 4);

                // 6-bit unsigned (0..63) centered to signed (-32..31).
                int8_t q1 = static_cast<int8_t>(q1c - 32);
                int8_t q2 = static_cast<int8_t>(q2c - 32);
                int8_t q3 = static_cast<int8_t>(q3c - 32);
                int8_t q4 = static_cast<int8_t>(q4c - 32);

                y_h[l] = d * static_cast<float>(sc_h[is + 0]) * static_cast<float>(q1);
                y_h[l + 32] = d * static_cast<float>(sc_h[is + 2]) * static_cast<float>(q2);
                y_h[l + 64] = d * static_cast<float>(sc_h[is + 4]) * static_cast<float>(q3);
                y_h[l + 96] = d * static_cast<float>(sc_h[is + 6]) * static_cast<float>(q4);
            }
        }
    }
    return out;
}

Tensor dequantize_q5_0_to_f32(const Tensor& t) {
    constexpr int64_t kBlockElems = 32;
    constexpr std::size_t kBlockBytes = 2 + 4 + 16; // scale(f16) + qh(u32) + qs[16]

    Tensor out = Tensor::zeros(t.shape(), DType::F32);
    auto out_span = out.as_f32();

    const std::byte* src = t.data();
    std::size_t n_blocks = static_cast<std::size_t>(t.numel()) / kBlockElems;

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::byte* block = src + b * kBlockBytes;

        uint16_t scale_bits;
        std::memcpy(&scale_bits, block, sizeof(uint16_t));
        float scale = f16_to_f32(scale_bits);

        uint32_t qh;
        std::memcpy(&qh, block + 2, sizeof(uint32_t));

        const auto* qs = reinterpret_cast<const uint8_t*>(block + 6); // 16 bytes, 2 values/byte

        for (int64_t i = 0; i < kBlockElems; ++i) {
            // low 4 bits live in qs; the 5th (high) bit for element i
            // lives in bit i of qh.
            uint8_t nibble = (i % 2 == 0) ? (qs[i / 2] & 0x0Fu) : (qs[i / 2] >> 4);
            uint8_t high_bit = (qh >> i) & 0x1u;
            int32_t q5 = static_cast<int32_t>(nibble) | (static_cast<int32_t>(high_bit) << 4);
            // Q5_0 has no bias term: values are centered by subtracting 16
            // (the midpoint of the 5-bit unsigned range), matching Q4_0's
            // symmetric-zero convention one bit wider.
            float val = static_cast<float>(q5 - 16) * scale;
            out_span[b * static_cast<std::size_t>(kBlockElems) + static_cast<std::size_t>(i)] = val;
        }
    }
    return out;
}

Tensor dequantize_q8_0_to_f32(const Tensor& t) {
    // Q8_0 block layout: 32 int8 values followed by 1 f32 scale
    // (block_bytes=36, block_elems=32 -- see dtype_info() in tensor.hpp).
    constexpr int64_t kBlockElems = 32;
    constexpr std::size_t kBlockBytes = 32 + sizeof(float);

    Tensor out = Tensor::zeros(t.shape(), DType::F32);
    auto out_span = out.as_f32();

    const std::byte* src = t.data();
    std::size_t n_blocks = static_cast<std::size_t>(t.numel()) / kBlockElems;

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::byte* block = src + b * kBlockBytes;
        const auto* qvals = reinterpret_cast<const int8_t*>(block);
        float scale;
        std::memcpy(&scale, block + kBlockElems, sizeof(float));
        for (int64_t i = 0; i < kBlockElems; ++i) {
            out_span[b * static_cast<std::size_t>(kBlockElems) + static_cast<std::size_t>(i)] =
                static_cast<float>(qvals[i]) * scale;
        }
    }
    return out;
}

Tensor load_f32(GGUFLoader& loader, const std::string& name) {
    Tensor t = loader.tensor(name);
    switch (t.dtype()) {
    case DType::F32:
        return t;
    case DType::Q8_0:
        return dequantize_q8_0_to_f32(t);
    case DType::Q4_0:
        return dequantize_q4_0_to_f32(t);
    case DType::Q4_1:
        return dequantize_q4_1_to_f32(t);
    case DType::Q5_0:
        return dequantize_q5_0_to_f32(t);
    case DType::Q5_1:
        return dequantize_q5_1_to_f32(t);
    case DType::Q2_K:
        return dequantize_q2_k_to_f32(t);
    case DType::Q3_K:
        return dequantize_q3_k_to_f32(t);
    case DType::Q4_K:
        return dequantize_q4_k_to_f32(t);
    case DType::Q5_K:
        return dequantize_q5_k_to_f32(t);
    case DType::Q6_K:
        return dequantize_q6_k_to_f32(t);
    case DType::Q8_K:
        return dequantize_q8_k_to_f32(t);
    default:
        throw std::runtime_error("load_f32: tensor '" + name + "' has unsupported dtype " +
                                 std::string(dtype_name(t.dtype())));
    }
}

int64_t meta_int(const std::unordered_map<std::string, std::string>& meta, const std::string& key,
                 int64_t fallback) {
    auto it = meta.find(key);
    if (it == meta.end())
        return fallback;
    try {
        return std::stoll(it->second);
    } catch (...) {
        return fallback;
    }
}
float meta_float(const std::unordered_map<std::string, std::string>& meta, const std::string& key,
                 float fallback) {
    auto it = meta.find(key);
    if (it == meta.end())
        return fallback;
    try {
        return std::stof(it->second);
    } catch (...) {
        return fallback;
    }
}

} // namespace

ModelConfig ModelConfig::from_metadata(const std::unordered_map<std::string, std::string>& meta) {
    ModelConfig c;

    auto arch_it = meta.find("general.architecture");
    std::string arch = (arch_it != meta.end()) ? arch_it->second : "llama";

    c.vocab_size = meta_int(meta, arch + ".vocab_size", 0);
    if (c.vocab_size == 0) {
        c.vocab_size = meta_int(meta, "tokenizer.ggml.bos_token_id", 0);
    }

    c.n_embd = meta_int(meta, arch + ".embedding_length", 0);
    c.n_layers = meta_int(meta, arch + ".block_count", 0);
    c.n_heads = meta_int(meta, arch + ".attention.head_count", 0);
    c.n_kv_heads = meta_int(meta, arch + ".attention.head_count_kv", c.n_heads);
    c.n_ff = meta_int(meta, arch + ".feed_forward_length", 0);
    c.max_seq_len = meta_int(meta, arch + ".context_length", 2048);
    c.rope_theta = meta_float(meta, arch + ".rope.freq_base", 10000.0f);
    c.rms_eps = meta_float(meta, arch + ".attention.layer_norm_rms_epsilon", 1e-5f);

    if (c.vocab_size <= 0 || c.n_embd <= 0 || c.n_layers <= 0 || c.n_heads <= 0) {
        throw std::runtime_error(
            "ModelConfig::from_metadata: missing required architecture metadata "
            "(vocab_size/embedding_length/block_count/attention.head_count) "
            "for architecture '" +
            arch + "'");
    }
    return c;
}

Tensor linear(const Tensor& x, const Tensor& w) {
    int64_t in_features = x.dim(1);

    if (w.dim(1) == in_features) {
        int64_t out_features = w.dim(0);
        Tensor out = Tensor::zeros({1, out_features});
        auto x_span = x.as_f32();
        auto out_span = out.as_f32();
        for (int64_t o = 0; o < out_features; ++o) {
            auto wrow = w.row(o);
            float acc = 0.0f;
            for (int64_t i = 0; i < in_features; ++i)
                acc += x_span[i] * wrow[i];
            out_span[o] = acc;
        }
        return out;
    }
    if (w.dim(0) == in_features) {
        return ops::matmul(x, w);
    }
    throw std::runtime_error("linear: weight shape " + w.shape_string() +
                             " incompatible with input feature count " +
                             std::to_string(in_features));
}

Model::Model(GGUFLoader& loader) {
    config_ = ModelConfig::from_metadata(loader.metadata());

    token_embd_ = load_f32(loader, "token_embd.weight");
    output_norm_ = load_f32(loader, "output_norm.weight");
    rope_cache_.initialize(config_.max_seq_len, config_.head_dim(), config_.rope_theta);
    output_weight_ = loader.has_tensor("output.weight") ? load_f32(loader, "output.weight")
                                                        : load_f32(loader, "token_embd.weight");

    layers_.reserve(static_cast<std::size_t>(config_.n_layers));
    for (int64_t i = 0; i < config_.n_layers; ++i) {
        std::string p = "blk." + std::to_string(i) + ".";
        LayerWeights lw;
        lw.attn_norm = load_f32(loader, p + "attn_norm.weight");
        lw.wq = load_f32(loader, p + "attn_q.weight");
        lw.wk = load_f32(loader, p + "attn_k.weight");
        lw.wv = load_f32(loader, p + "attn_v.weight");
        lw.wo = load_f32(loader, p + "attn_output.weight");
        lw.ffn_norm = load_f32(loader, p + "ffn_norm.weight");
        lw.w_gate = load_f32(loader, p + "ffn_gate.weight");
        lw.w_up = load_f32(loader, p + "ffn_up.weight");
        lw.w_down = load_f32(loader, p + "ffn_down.weight");
        layers_.push_back(std::move(lw));
    }
}

Tensor Model::attention(int64_t layer_idx, const Tensor& x_norm, int64_t pos,
                        KVCache& kv_cache) const {
    const auto& lw = layers_[static_cast<std::size_t>(layer_idx)];
    const int64_t head_dim = config_.head_dim();
    const int64_t n_heads = config_.n_heads;
    const int64_t n_kv_heads = config_.n_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    Tensor q = linear(x_norm, lw.wq);
    Tensor k = linear(x_norm, lw.wk);
    Tensor v = linear(x_norm, lw.wv);

    auto q_span = q.as_f32();
    auto k_span = k.as_f32();

    for (int64_t h = 0; h < n_heads; ++h) {
        apply_rope(q_span.subspan(static_cast<std::size_t>(h * head_dim),
                                  static_cast<std::size_t>(head_dim)),
                   pos, rope_cache_);
    }
    for (int64_t h = 0; h < n_kv_heads; ++h) {
        apply_rope(k_span.subspan(static_cast<std::size_t>(h * head_dim),
                                  static_cast<std::size_t>(head_dim)),
                   pos, rope_cache_);
    }

    Tensor k_view = Tensor::view(k.data(), {n_kv_heads, head_dim}, DType::F32);
    Tensor v_view = Tensor::view(v.data(), {n_kv_heads, head_dim}, DType::F32);
    kv_cache.append(layer_idx, k_view, v_view);

    Tensor cached_keys = kv_cache.keys(layer_idx);
    Tensor cached_values = kv_cache.values(layer_idx);
    const int64_t seq_len = pos + 1;

    auto ck_span = cached_keys.as_f32();
    auto cv_span = cached_values.as_f32();

    const int64_t group_size = n_heads / n_kv_heads;

    Tensor attn_out = Tensor::zeros({1, n_heads * head_dim});
    auto out_span = attn_out.as_f32();

    std::vector<float> scores(static_cast<std::size_t>(seq_len));

    for (int64_t h = 0; h < n_heads; ++h) {
        int64_t kv_h = h / group_size;
        auto q_head = q_span.subspan(static_cast<std::size_t>(h * head_dim),
                                     static_cast<std::size_t>(head_dim));

        for (int64_t t = 0; t < seq_len; ++t) {
            std::size_t key_off = static_cast<std::size_t>((t * n_kv_heads + kv_h) * head_dim);
            float dot = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) {
                dot += q_head[static_cast<std::size_t>(d)] *
                       ck_span[key_off + static_cast<std::size_t>(d)];
            }
            scores[static_cast<std::size_t>(t)] = dot * scale;
        }

        {
            float max_s = scores[0];
            for (float s : scores)
                max_s = std::max(max_s, s);
            float sum = 0.0f;
            for (float& s : scores) {
                s = std::exp(s - max_s);
                sum += s;
            }
            for (float& s : scores)
                s /= sum;
        }

        auto out_head = out_span.subspan(static_cast<std::size_t>(h * head_dim),
                                         static_cast<std::size_t>(head_dim));
        for (int64_t t = 0; t < seq_len; ++t) {
            std::size_t val_off = static_cast<std::size_t>((t * n_kv_heads + kv_h) * head_dim);
            float w = scores[static_cast<std::size_t>(t)];
            for (int64_t d = 0; d < head_dim; ++d) {
                out_head[static_cast<std::size_t>(d)] +=
                    w * cv_span[val_off + static_cast<std::size_t>(d)];
            }
        }
    }

    return linear(attn_out, lw.wo);
}

Tensor Model::feed_forward(int64_t layer_idx, const Tensor& x_norm) const {
    const auto& lw = layers_[static_cast<std::size_t>(layer_idx)];
    Tensor gate = ops::silu(linear(x_norm, lw.w_gate));
    Tensor up = linear(x_norm, lw.w_up);

    auto gate_span = gate.as_f32();
    auto up_span = up.as_f32();
    for (std::size_t i = 0; i < gate_span.size(); ++i)
        gate_span[i] *= up_span[i];

    return linear(gate, lw.w_down);
}

Tensor Model::forward_layer(int64_t layer_idx, const Tensor& x, int64_t pos,
                            KVCache& kv_cache) const {
    const auto& lw = layers_[static_cast<std::size_t>(layer_idx)];

    Tensor attn_in = ops::rms_norm(x, lw.attn_norm, config_.rms_eps);
    Tensor attn_out = attention(layer_idx, attn_in, pos, kv_cache);
    Tensor x1 = ops::add(x, attn_out);

    Tensor ffn_in = ops::rms_norm(x1, lw.ffn_norm, config_.rms_eps);
    Tensor ffn_out = feed_forward(layer_idx, ffn_in);
    Tensor x2 = ops::add(x1, ffn_out);

    return x2;
}

Tensor Model::forward(TokenId token, int64_t pos, KVCache& kv_cache) const {
    if (token < 0 || token >= config_.vocab_size) {
        throw std::out_of_range("Model::forward: token id " + std::to_string(token) +
                                " is out of vocabulary range [0, " +
                                std::to_string(config_.vocab_size) + ")");
    }

    Tensor x = Tensor::zeros({1, config_.n_embd});
    auto embd_row = token_embd_.row(token);
    auto x_span = x.as_f32();
    for (std::size_t i = 0; i < x_span.size(); ++i)
        x_span[i] = embd_row[i];

    for (int64_t l = 0; l < config_.n_layers; ++l) {
        x = forward_layer(l, x, pos, kv_cache);
    }

    Tensor x_norm = ops::rms_norm(x, output_norm_, config_.rms_eps);
    Tensor logits = linear(x_norm, output_weight_);
    return logits;
}

} // namespace llmengine