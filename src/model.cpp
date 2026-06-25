// src/model.cpp
#include "llmengine/model.hpp"

#include <cmath>
#include <stdexcept>

#include "llmengine/ops.hpp"

namespace llmengine {

namespace {

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

namespace {

Tensor linear(const Tensor& x, const Tensor& w) {
    if (x.ndim() != 2 || w.ndim() != 2)
        throw std::runtime_error("linear: expected 2D tensors, got x=" + x.shape_string() +
                                 " w=" + w.shape_string());
    int64_t in_features = x.dim(1);

    if (w.dim(1) == in_features) {
        return ops::matmul(x, ops::transpose(w));
    }
    if (w.dim(0) == in_features) {
        return ops::matmul(x, w);
    }
    throw std::runtime_error("linear: weight shape " + w.shape_string() +
                             " is incompatible with input feature count " +
                             std::to_string(in_features));
}

void apply_rope(std::span<float> v, int64_t pos, float theta_base) {
    const std::size_t dim = v.size();
    for (std::size_t i = 0; i < dim / 2; ++i) {
        float exponent = static_cast<float>(2 * i) / static_cast<float>(dim);
        float theta_i = std::pow(theta_base, -exponent);
        float angle = static_cast<float>(pos) * theta_i;
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);
        float v0 = v[2 * i];
        float v1 = v[2 * i + 1];
        v[2 * i] = v0 * cos_a - v1 * sin_a;
        v[2 * i + 1] = v0 * sin_a + v1 * cos_a;
    }
}

} // namespace

Model::Model(GGUFLoader& loader) {
    config_ = ModelConfig::from_metadata(loader.metadata());

    token_embd_ = loader.tensor("token_embd.weight");
    output_norm_ = loader.tensor("output_norm.weight");
    output_weight_ = loader.has_tensor("output.weight") ? loader.tensor("output.weight")
                                                        : loader.tensor("token_embd.weight");

    layers_.reserve(static_cast<std::size_t>(config_.n_layers));

    for (int64_t i = 0; i < config_.n_layers; ++i) {
        std::string p = "blk." + std::to_string(i) + ".";
        LayerWeights lw;
        lw.attn_norm = loader.tensor(p + "attn_norm.weight");
        lw.wq = loader.tensor(p + "attn_q.weight");
        lw.wk = loader.tensor(p + "attn_k.weight");
        lw.wv = loader.tensor(p + "attn_v.weight");
        lw.wo = loader.tensor(p + "attn_output.weight");
        lw.ffn_norm = loader.tensor(p + "ffn_norm.weight");
        lw.w_gate = loader.tensor(p + "ffn_gate.weight");
        lw.w_up = loader.tensor(p + "ffn_up.weight");
        lw.w_down = loader.tensor(p + "ffn_down.weight");
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
                   pos, config_.rope_theta);
    }
    for (int64_t h = 0; h < n_kv_heads; ++h) {
        apply_rope(k_span.subspan(static_cast<std::size_t>(h * head_dim),
                                  static_cast<std::size_t>(head_dim)),
                   pos, config_.rope_theta);
    }

    Tensor k_view = Tensor::view(k.data(), {n_kv_heads, head_dim}, DType::F32);
    Tensor v_view = Tensor::view(v.data(), {n_kv_heads, head_dim}, DType::F32);
    kv_cache.append(layer_idx, k_view, v_view);

    Tensor cached_keys = kv_cache.keys(layer_idx);
    Tensor cached_values = kv_cache.values(layer_idx);
    const int64_t seq_len = kv_cache.size();

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