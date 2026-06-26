// src/model.cpp
#include "llmengine/model.hpp"
#include "llmengine/model_helper.hpp"
#include "llmengine/ops.hpp"

#include <cmath>
#include <stdexcept>

namespace llmengine {

namespace {
std::size_t calculate_arena_size(const ModelConfig& config) {
    std::size_t attn_mem = (2 * config.n_embd) + (2 * config.n_kv_heads * config.head_dim());
    attn_mem *= sizeof(float);

    std::size_t ffn_mem = 3 * config.n_ff * sizeof(float);
    std::size_t peak_bytes = std::max(attn_mem, ffn_mem);
    std::size_t raw_estimate = peak_bytes * 8;
    constexpr std::size_t kMinFloor = 4 * 1024 * 1024; // 4 MB

    return std::max(raw_estimate, kMinFloor);
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

void linear(const Tensor& x, const Tensor& w, Tensor& out) {
    int64_t in_features = x.dim(1);

    if (w.dim(1) == in_features) {
        int64_t out_features = w.dim(0);

        auto x_span = x.as_f32();
        auto out_span = out.as_f32(); // Write directly into the destination

        for (int64_t o = 0; o < out_features; ++o) {
            auto wrow = w.row(o);
            float acc = 0.0f;
            for (int64_t i = 0; i < in_features; ++i) {
                acc += x_span[i] * wrow[i];
            }
            out_span[o] = acc;
        }
        return;
    }

    if (w.dim(0) == in_features) {
        return ops::matmul(x, w, out);
    }

    throw std::runtime_error("linear: weight shape " + w.shape_string() +
                             " incompatible with input feature count " +
                             std::to_string(in_features));
}

Model::Model(GGUFLoader& loader)
    : config_{ModelConfig::from_metadata(loader.metadata())},
      scratch_arena_{calculate_arena_size(config_)} {
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

void Model::attention(int64_t layer_idx, const Tensor& x_norm, int64_t pos, KVCache& kv_cache,
                      Tensor& out) const {
    const auto& lw = layers_[static_cast<std::size_t>(layer_idx)];
    const int64_t head_dim = config_.head_dim();
    const int64_t n_heads = config_.n_heads;
    const int64_t n_kv_heads = config_.n_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // FIX: Dynamically read out_features from the weight tensors themselves
    // instead of multiplying config parameters.
    int64_t q_out_features = (lw.wq.dim(0) == x_norm.dim(1)) ? lw.wq.dim(1) : lw.wq.dim(0);
    int64_t k_out_features = (lw.wk.dim(0) == x_norm.dim(1)) ? lw.wk.dim(1) : lw.wk.dim(0);
    int64_t v_out_features = (lw.wv.dim(0) == x_norm.dim(1)) ? lw.wv.dim(1) : lw.wv.dim(0);

    int64_t q_shape[] = {1, q_out_features};
    int64_t k_shape[] = {1, k_out_features};
    int64_t v_shape[] = {1, v_out_features};

    Tensor q, k, v;
    const_cast<ScratchArena&>(scratch_arena_).alloc(q, q_shape, DType::F32);
    const_cast<ScratchArena&>(scratch_arena_).alloc(k, k_shape, DType::F32);
    const_cast<ScratchArena&>(scratch_arena_).alloc(v, v_shape, DType::F32);

    linear(x_norm, lw.wq, q);
    linear(x_norm, lw.wk, k);
    linear(x_norm, lw.wv, v);

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

    Tensor attn_out;
    const_cast<ScratchArena&>(scratch_arena_).alloc(attn_out, q_shape, DType::F32);
    std::fill_n(attn_out.as_f32().data(), attn_out.numel(), 0.0f);
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

    linear(attn_out, lw.wo, out);
}

Tensor Model::feed_forward(int64_t layer_idx, const Tensor& x_norm) const {
    const auto& lw = layers_[static_cast<std::size_t>(layer_idx)];

    Tensor gate_linear, up;
    int64_t ff_shape[] = {1, config_.n_ff};
    const_cast<ScratchArena&>(scratch_arena_).alloc(gate_linear, ff_shape, DType::F32);
    const_cast<ScratchArena&>(scratch_arena_).alloc(up, ff_shape, DType::F32);

    linear(x_norm, lw.w_gate, gate_linear);
    linear(x_norm, lw.w_up, up);

    Tensor gate;
    const_cast<ScratchArena&>(scratch_arena_).alloc(gate, ff_shape, DType::F32);
    ops::silu(gate_linear, gate);

    auto gate_span = gate.as_f32();
    auto up_span = up.as_f32();
    for (std::size_t i = 0; i < gate_span.size(); ++i)
        gate_span[i] *= up_span[i];

    Tensor out;
    int64_t out_shape[] = {1, config_.n_embd};
    const_cast<ScratchArena&>(scratch_arena_).alloc(out, out_shape, DType::F32);
    linear(gate, lw.w_down, out);
    return out;
}

Tensor Model::forward_layer(int64_t layer_idx, const Tensor& x, int64_t pos,
                            KVCache& kv_cache) const {
    const auto& lw = layers_[static_cast<std::size_t>(layer_idx)];

    const_cast<ScratchArena&>(scratch_arena_).reset();

    int64_t layer_shape[] = {1, config_.n_embd}; // [1, 896]

    Tensor attn_in;
    const_cast<ScratchArena&>(scratch_arena_).alloc(attn_in, layer_shape, DType::F32);
    ops::rms_norm(x, lw.attn_norm, attn_in, config_.rms_eps);

    Tensor attn_out;
    const_cast<ScratchArena&>(scratch_arena_).alloc(attn_out, layer_shape, DType::F32);
    attention(layer_idx, attn_in, pos, kv_cache, attn_out);

    Tensor x1;
    const_cast<ScratchArena&>(scratch_arena_).alloc(x1, layer_shape, DType::F32);
    ops::add(x, attn_out, x1);

    Tensor ffn_in;
    const_cast<ScratchArena&>(scratch_arena_).alloc(ffn_in, layer_shape, DType::F32);
    ops::rms_norm(x1, lw.ffn_norm, ffn_in, config_.rms_eps);

    Tensor ffn_out = feed_forward(layer_idx, ffn_in);

    Tensor x2;
    const_cast<ScratchArena&>(scratch_arena_).alloc(x2, layer_shape, DType::F32);
    ops::add(x1, ffn_out, x2);

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

    Tensor x_norm;
    int64_t final_shape[] = {1, config_.n_embd};
    const_cast<ScratchArena&>(scratch_arena_).alloc(x_norm, final_shape, DType::F32);
    ops::rms_norm(x, output_norm_, x_norm, config_.rms_eps);

    x_norm = Tensor::view(x_norm.data(), {1, config_.n_embd}, DType::F32);
    int64_t logits_shape[] = {1, config_.vocab_size};
    Tensor logits;
    const_cast<ScratchArena&>(scratch_arena_).alloc(logits, logits_shape, DType::F32);
    linear(x_norm, output_weight_, logits);

    return logits;
}

} // namespace llmengine