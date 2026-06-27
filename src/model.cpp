#include "llmengine/model.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llmengine {

namespace {
ggml_tensor* load_tensor_native(ggml_context* ctx, GGUFLoader& loader, const std::string& name) {
    if (!loader.has_tensor(name))
        throw std::runtime_error("Missing tensor: " + name);
    const auto& info = loader.tensor_info(name);

    ggml_tensor* t = nullptr;
    if (info.shape.size() == 1)
        t = ggml_new_tensor_1d(ctx, info.dtype, info.shape[0]);
    else if (info.shape.size() == 2)
        t = ggml_new_tensor_2d(ctx, info.dtype, info.shape[0], info.shape[1]);
    else if (info.shape.size() == 3)
        t = ggml_new_tensor_3d(ctx, info.dtype, info.shape[0], info.shape[1], info.shape[2]);
    else if (info.shape.size() == 4)
        t = ggml_new_tensor_4d(ctx, info.dtype, info.shape[0], info.shape[1], info.shape[2], info.shape[3]);

    if (!t)
        throw std::runtime_error("Failed to allocate weight layout: " + name);
    t->data = const_cast<void*>(loader.tensor_data(name));
    return t;
}

std::size_t calculate_scratch_size(const ModelConfig& config) {
    (void)config;
    return 128ULL * 1024ULL * 1024ULL;
}

int64_t count_tokenizer_vocab(const GGUFLoader& loader) {
    try {
        const auto& tokens = loader.get_meta_array("tokenizer.ggml.tokens");

        return static_cast<int64_t>(tokens.values.size());
    } catch (const std::exception&) {
        return 0;
    }
}

void inline_assign_i32(ggml_tensor* t, size_t index, int32_t val) {
    if (t && t->data) {
        static_cast<int32_t*>(t->data)[index] = val;
    }
}
} // namespace

ModelConfig ModelConfig::from_metadata(const GGUFLoader& loader) {
    ModelConfig c;

    std::string arch = loader.get_meta_string("general.architecture");

    try {
        c.vocab_size = loader.get_meta_int(arch + ".vocab_size");
    } catch (...) {
        c.vocab_size = count_tokenizer_vocab(loader);
    }

    c.n_embd = loader.get_meta_int(arch + ".embedding_length");
    c.n_layers = loader.get_meta_int(arch + ".block_count");
    c.n_heads = loader.get_meta_int(arch + ".attention.head_count");
    c.n_kv_heads = loader.get_meta_int(arch + ".attention.head_count_kv");
    c.n_ff = loader.get_meta_int(arch + ".feed_forward_length");
    c.max_seq_len = loader.get_meta_int(arch + ".context_length");
    c.rope_theta = loader.get_meta_float(arch + ".rope.freq_base");
    c.rms_eps = loader.get_meta_float(arch + ".attention.layer_norm_rms_epsilon");

    if (c.vocab_size <= 0 || c.n_embd <= 0 || c.n_layers <= 0 || c.n_heads <= 0) {
        throw std::runtime_error("ModelConfig::from_metadata: architecture configuration missing.");
    }
    if (c.n_kv_heads <= 0 || c.n_heads % c.n_kv_heads != 0) {
        throw std::runtime_error("ModelConfig::from_metadata: invalid head_count_kv.");
    }
    if (c.n_embd % c.n_heads != 0) {
        throw std::runtime_error("ModelConfig::from_metadata: embedding_length not divisible by head_count.");
    }

    return c;
}

Model::Model(GGUFLoader& loader) : config_{ModelConfig::from_metadata(loader)}, scratch_arena_{calculate_scratch_size(config_)} {
    backend_ = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
    if (!backend_)
        throw std::runtime_error("Failed to initialize CPU backend.");

    ggml_init_params params = {.mem_size = loader.tensors().size() * ggml_tensor_overhead() + 4096, .mem_buffer = nullptr, .no_alloc = true};
    weights_ctx_ = ggml_init(params);
    if (!weights_ctx_)
        throw std::runtime_error("Failed to initialize weights context.");

    token_embd_ = load_tensor_native(weights_ctx_, loader, "token_embd.weight");
    output_norm_ = load_tensor_native(weights_ctx_, loader, "output_norm.weight");
    output_weight_ = loader.has_tensor("output.weight") ? load_tensor_native(weights_ctx_, loader, "output.weight") : token_embd_;

    layers_.reserve(static_cast<std::size_t>(config_.n_layers));
    for (int64_t i = 0; i < config_.n_layers; ++i) {
        std::string p = "blk." + std::to_string(i) + ".";
        LayerWeights lw;
        lw.attn_norm = load_tensor_native(weights_ctx_, loader, p + "attn_norm.weight");
        lw.wq = load_tensor_native(weights_ctx_, loader, p + "attn_q.weight");
        lw.bq = load_tensor_native(weights_ctx_, loader, p + "attn_q.bias");
        lw.wk = load_tensor_native(weights_ctx_, loader, p + "attn_k.weight");
        lw.bk = load_tensor_native(weights_ctx_, loader, p + "attn_k.bias");
        lw.wv = load_tensor_native(weights_ctx_, loader, p + "attn_v.weight");
        lw.bv = load_tensor_native(weights_ctx_, loader, p + "attn_v.bias");
        lw.wo = load_tensor_native(weights_ctx_, loader, p + "attn_output.weight");
        lw.ffn_norm = load_tensor_native(weights_ctx_, loader, p + "ffn_norm.weight");
        lw.w_gate = load_tensor_native(weights_ctx_, loader, p + "ffn_gate.weight");
        lw.w_up = load_tensor_native(weights_ctx_, loader, p + "ffn_up.weight");
        lw.w_down = load_tensor_native(weights_ctx_, loader, p + "ffn_down.weight");
        layers_.push_back(lw);
    }
}

Model::~Model() {
    if (weights_ctx_)
        ggml_free(weights_ctx_);
}

ggml_tensor* Model::forward(TokenId token, int64_t pos, ggml_tensor* k_cache, ggml_tensor* v_cache) const {
    scratch_arena_.reset();
    ggml_context* ctx = scratch_arena_.ctx();

    ggml_tensor* token_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    inline_assign_i32(token_tensor, 0, token);

    ggml_tensor* x = ggml_get_rows(ctx, token_embd_, token_tensor);

    const int64_t head_dim = config_.head_dim();
    const int64_t n_heads = config_.n_heads;
    const int64_t n_kv_heads = config_.n_kv_heads;
    const int64_t kv_len = pos + 1;
    const int64_t layer_stride = config_.max_seq_len * n_kv_heads * head_dim;
    const size_t k_elem_size = ggml_element_size(k_cache);
    const size_t v_elem_size = ggml_element_size(v_cache);

    ggml_cgraph* graph = ggml_new_graph(ctx);

    for (int64_t l = 0; l < config_.n_layers; ++l) {
        const auto& lw = layers_[static_cast<std::size_t>(l)];

        ggml_tensor* cur = ggml_rms_norm(ctx, x, config_.rms_eps);
        cur = ggml_mul(ctx, cur, lw.attn_norm);

        // ggml_tensor* q = ggml_mul_mat(ctx, lw.wq, cur);
        // ggml_tensor* k = ggml_mul_mat(ctx, lw.wk, cur);
        // ggml_tensor* v = ggml_mul_mat(ctx, lw.wv, cur);

        ggml_tensor* q = ggml_add(ctx, ggml_mul_mat(ctx, lw.wq, cur), lw.bq);
        ggml_tensor* k = ggml_add(ctx, ggml_mul_mat(ctx, lw.wk, cur), lw.bk);
        ggml_tensor* v = ggml_add(ctx, ggml_mul_mat(ctx, lw.wv, cur), lw.bv);

        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        inline_assign_i32(pos_tensor, 0, static_cast<int32_t>(pos));

        q = ggml_rope_ext(ctx, ggml_reshape_3d(ctx, q, head_dim, n_heads, 1), pos_tensor, nullptr, static_cast<int>(head_dim), GGML_ROPE_TYPE_NEOX,
                          static_cast<int>(config_.max_seq_len), config_.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        k = ggml_rope_ext(ctx, ggml_reshape_3d(ctx, k, head_dim, n_kv_heads, 1), pos_tensor, nullptr, static_cast<int>(head_dim), GGML_ROPE_TYPE_NEOX,
                          static_cast<int>(config_.max_seq_len), config_.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        const size_t layer_offset_k = static_cast<size_t>(l * layer_stride) * k_elem_size;
        const size_t layer_offset_v = static_cast<size_t>(l * layer_stride) * v_elem_size;
        const size_t pos_offset_k = static_cast<size_t>(pos * n_kv_heads * head_dim) * k_elem_size;
        const size_t pos_offset_v = static_cast<size_t>(pos * n_kv_heads * head_dim) * v_elem_size;

        ggml_tensor* k_slice = ggml_view_1d(ctx, k_cache, n_kv_heads * head_dim, layer_offset_k + pos_offset_k);
        ggml_tensor* v_slice = ggml_view_1d(ctx, v_cache, n_kv_heads * head_dim, layer_offset_v + pos_offset_v);

        ggml_tensor* k_write = ggml_cpy(ctx, k, k_slice);
        ggml_tensor* v_write = ggml_cpy(ctx, v, v_slice);
        ggml_build_forward_expand(graph, k_write);
        ggml_build_forward_expand(graph, v_write);

        q = ggml_scale_inplace(ctx, q, 1.0f / std::sqrt(static_cast<float>(head_dim)));

        ggml_tensor* K_active = ggml_view_3d(ctx, k_cache, head_dim, kv_len, n_kv_heads,
                                             /*nb1=*/n_kv_heads * head_dim * k_elem_size, // stride between positions
                                             /*nb2=*/head_dim * k_elem_size,              // stride between heads
                                             layer_offset_k);

        ggml_tensor* V_raw = ggml_view_3d(ctx, v_cache, head_dim, kv_len, n_kv_heads,
                                          /*nb1=*/n_kv_heads * head_dim * v_elem_size,
                                          /*nb2=*/head_dim * v_elem_size, layer_offset_v);

        ggml_tensor* V_active = ggml_cont(ctx, ggml_permute(ctx, V_raw, 1, 0, 2, 3));

        ggml_tensor* q_perm = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));

        ggml_tensor* scores = ggml_mul_mat(ctx, K_active, q_perm);
        scores = ggml_soft_max_inplace(ctx, scores);

        ggml_tensor* attn_out = ggml_mul_mat(ctx, V_active, scores);
        attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
        attn_out = ggml_reshape_1d(ctx, attn_out, config_.n_embd);

        x = ggml_add(ctx, x, ggml_mul_mat(ctx, lw.wo, attn_out));

        cur = ggml_rms_norm(ctx, x, config_.rms_eps);
        cur = ggml_mul(ctx, cur, lw.ffn_norm);

        ggml_tensor* gate = ggml_silu_inplace(ctx, ggml_mul_mat(ctx, lw.w_gate, cur));
        ggml_tensor* up = ggml_mul_mat(ctx, lw.w_up, cur);
        ggml_tensor* ffn_out = ggml_mul_mat(ctx, lw.w_down, ggml_mul(ctx, gate, up));

        x = ggml_add(ctx, x, ffn_out);
    }

    x = ggml_rms_norm(ctx, x, config_.rms_eps);
    x = ggml_mul(ctx, x, output_norm_);

    ggml_tensor* logits = ggml_mul_mat(ctx, output_weight_, x);

    ggml_build_forward_expand(graph, logits);
    ggml_backend_graph_compute(backend_, graph);

    return logits;
}

} // namespace llmengine