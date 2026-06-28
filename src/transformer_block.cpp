// transformer_block.cpp
#include "llmengine/transformer_block.hpp"

#include <cmath>

namespace llmengine {

namespace {

inline void assign_i32(ggml_tensor* tensor, int32_t value) {
    static_cast<int32_t*>(tensor->data)[0] = value;
}

} // namespace

TransformerBlock::TransformerBlock(const ModelConfig& config, const LayerWeights& weights) : config_(config), weights_(weights) {}

ggml_tensor* TransformerBlock::forward(ggml_context* ctx, ggml_cgraph* graph, ggml_tensor* x, int64_t pos, ggml_tensor* k_cache, ggml_tensor* v_cache,
                                       int64_t layer_index) const {

    const int64_t head_dim = config_.head_dim();
    const int64_t n_heads = config_.n_heads;
    const int64_t n_kv_heads = config_.n_kv_heads;
    const int64_t kv_len = pos + 1;

    const int64_t layer_stride = config_.max_seq_len * n_kv_heads * head_dim;

    const size_t k_elem_size = ggml_element_size(k_cache);
    const size_t v_elem_size = ggml_element_size(v_cache);

    ggml_tensor* cur = ggml_rms_norm(ctx, x, config_.rms_eps);
    cur = ggml_mul(ctx, cur, weights_.attn_norm);

    ggml_tensor* q = ggml_add(ctx, ggml_mul_mat(ctx, weights_.query_weight, cur), weights_.query_bias);
    ggml_tensor* k = ggml_add(ctx, ggml_mul_mat(ctx, weights_.key_weight, cur), weights_.key_bias);
    ggml_tensor* v = ggml_add(ctx, ggml_mul_mat(ctx, weights_.value_weight, cur), weights_.value_bias);

    ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    static_cast<int32_t*>(pos_tensor->data)[0] = static_cast<int32_t>(pos);

    q = ggml_rope_ext(ctx, ggml_reshape_3d(ctx, q, head_dim, n_heads, 1), pos_tensor, nullptr, static_cast<int>(head_dim), GGML_ROPE_TYPE_NEOX,
                      static_cast<int>(config_.max_seq_len), config_.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    k = ggml_rope_ext(ctx, ggml_reshape_3d(ctx, k, head_dim, n_kv_heads, 1), pos_tensor, nullptr, static_cast<int>(head_dim), GGML_ROPE_TYPE_NEOX,
                      static_cast<int>(config_.max_seq_len), config_.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    const size_t layer_offset_k = static_cast<size_t>(layer_index * layer_stride) * k_elem_size;
    const size_t layer_offset_v = static_cast<size_t>(layer_index * layer_stride) * v_elem_size;
    const size_t pos_offset_k = static_cast<size_t>(pos * n_kv_heads * head_dim) * k_elem_size;
    const size_t pos_offset_v = static_cast<size_t>(pos * n_kv_heads * head_dim) * v_elem_size;

    ggml_tensor* k_slice = ggml_view_1d(ctx, k_cache, n_kv_heads * head_dim, layer_offset_k + pos_offset_k);
    ggml_tensor* v_slice = ggml_view_1d(ctx, v_cache, n_kv_heads * head_dim, layer_offset_v + pos_offset_v);

    ggml_tensor* k_write = ggml_cpy(ctx, k, k_slice);
    ggml_tensor* v_write = ggml_cpy(ctx, v, v_slice);

    ggml_build_forward_expand(graph, k_write);
    ggml_build_forward_expand(graph, v_write);

    q = ggml_scale_inplace(ctx, q, 1.0f / std::sqrt(static_cast<float>(head_dim)));

    ggml_tensor* k_active =
        ggml_view_3d(ctx, k_cache, head_dim, kv_len, n_kv_heads, n_kv_heads * head_dim * k_elem_size, head_dim * k_elem_size, layer_offset_k);
    ggml_tensor* v_raw =
        ggml_view_3d(ctx, v_cache, head_dim, kv_len, n_kv_heads, n_kv_heads * head_dim * v_elem_size, head_dim * v_elem_size, layer_offset_v);

    ggml_tensor* v_active = ggml_cont(ctx, ggml_permute(ctx, v_raw, 1, 0, 2, 3));
    ggml_tensor* q_perm = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));

    ggml_tensor* scores = ggml_mul_mat(ctx, k_active, q_perm);
    scores = ggml_soft_max_inplace(ctx, scores);

    ggml_tensor* attn_out = ggml_mul_mat(ctx, v_active, scores);
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
    attn_out = ggml_reshape_1d(ctx, attn_out, config_.n_embd);

    x = ggml_add(ctx, x, ggml_mul_mat(ctx, weights_.output_weight, attn_out));

    cur = ggml_rms_norm(ctx, x, config_.rms_eps);
    cur = ggml_mul(ctx, cur, weights_.ffn_norm);

    ggml_tensor* gate = ggml_silu_inplace(ctx, ggml_mul_mat(ctx, weights_.gate_weight, cur));
    ggml_tensor* up = ggml_mul_mat(ctx, weights_.up_weight, cur);
    ggml_tensor* ffn_out = ggml_mul_mat(ctx, weights_.down_weight, ggml_mul(ctx, gate, up));

    return ggml_add(ctx, x, ffn_out);
}

} // namespace llmengine