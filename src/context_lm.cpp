#include "swipe_decoder/context_lm.hpp"
#include "swipe_decoder/fp16_utils.hpp"
#include "swipe_decoder/layout.hpp"  // utf8_to_codepoints, utf8_encode_one, utf8_tolower
#include "swipe_decoder/vocab_hash.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::from_blob;
using executorch::runtime::EValue;
using executorch::runtime::Error;

namespace swipe_decoder {

struct ContextLMScorer::Impl {
    ContextLMMetadata metadata;
    std::unique_ptr<Module> module;

    std::unordered_map<std::string, int> word2id;
    std::unordered_map<std::string, int> word2id_lower;
    std::vector<std::string> id2word;

    // Embedding weights (extracted from PTE get_embeddings method)
    int emb_num_exact = 0;
    int emb_num_buckets = 0;
    int emb_embed_dim = 0;
    std::vector<float> exact_embed;  // [num_exact * embed_dim]
    std::vector<float> exact_bias;   // [num_exact]
    std::vector<float> hash_embed;   // [num_buckets * embed_dim]
    std::vector<float> hash_bias;    // [num_buckets]

    // Precomputed scores for all exact embeddings (h·emb + bias)
    // Reused across predict_topk calls (recomputed each call with new h)
    std::vector<float> exact_scores;

    int cfg_max_context_len = 0;
    int cfg_num_hashes = 0;
    bool loaded = false;

    // Reusable buffers for context tensor data
    std::vector<int64_t> context_ids_buf;
    std::vector<int64_t> context_hashes_buf;

    void cleanup() {
        module.reset();
        loaded = false;
    }

    // Look up a word, returning (word_id, hash_indices)
    void lookup(const std::string& word, uint32_t& id_out,
                const uint32_t*& hashes_out, uint32_t* oov_buf,
                bool allow_lowercase_fallback = true) const {
        
        static uint32_t hash[2];

        auto it = word2id.find(word);
        if(it == word2id.end() && allow_lowercase_fallback) {
            auto it2 = word2id_lower.find(utf8_tolower(word));
            if(it2 != word2id_lower.end()) {
                it = it2;
            }
        }

        if(it != word2id.end()) {
            id_out = it->second;
            hashes_out = oov_buf;
        } else {
            compute_hash_indices_wy(word.c_str(), word.length(), emb_num_buckets, hash);
            hashes_out = hash;
            id_out = emb_num_exact;
        }
    }

    // Build context tensors (right-padded: words at positions 0..N-1, OOV at N..L-1).
    // Returns the number of context words placed (clamped to L).
    int build_context_tensors(const std::vector<std::string>& context_words) {
        int L = cfg_max_context_len;
        int K = cfg_num_hashes;

        std::fill(context_ids_buf.begin(), context_ids_buf.end(), 0);
        std::fill(context_hashes_buf.begin(), context_hashes_buf.end(), 0);

        // Take the last L words if context is longer than max_context_len
        int n_ctx = std::min(static_cast<int>(context_words.size()), L);
        int src_start = static_cast<int>(context_words.size()) - n_ctx;
        std::vector<uint32_t> oov_buf(K, 0);

        for (int i = 0; i < n_ctx; ++i) {
            uint32_t wid;
            const uint32_t* hashes;
            lookup(context_words[src_start + i], wid, hashes, oov_buf.data());
            context_ids_buf[i] = static_cast<int64_t>(wid);
            for (int k = 0; k < K; ++k) {
                context_hashes_buf[i * K + k] = static_cast<int64_t>(hashes[k]);
            }
        }

        return n_ctx;
    }

    // Run backbone to get context vector h [E] at position (context_len - 1).
    // PTE returns [1, L, E]; we index into the right position.
    // Returns nullptr on error.
    float* run_backbone(float* h_buf, int context_len) {
        int L = cfg_max_context_len;
        int K = cfg_num_hashes;
        int E = emb_embed_dim;

        // Position to read from the output — last valid context position
        int pos = std::max(0, context_len - 1);

        auto t_ctx_ids = from_blob(
            context_ids_buf.data(), {1, L},
            executorch::aten::ScalarType::Long);
        auto t_ctx_hashes = from_blob(
            context_hashes_buf.data(), {1, L, K},
            executorch::aten::ScalarType::Long);

        auto result = module->forward({
            EValue(std::move(t_ctx_ids)),
            EValue(std::move(t_ctx_hashes)),
        });

        if (!result.ok()) {
            fprintf(stderr, "ContextLMScorer: Forward failed (err=0x%x)\n",
                    static_cast<unsigned>(result.error()));
            return nullptr;
        }

        auto& output_values = result.get();
        if (output_values.empty() || !output_values[0].isTensor()) {
            fprintf(stderr, "ContextLMScorer: Unexpected output format\n");
            return nullptr;
        }

        auto& out_tensor = output_values[0].toTensor();

        // Validate output shape [1, L, E]
        if (out_tensor.dim() != 3 || out_tensor.size(2) != E ||
            pos >= out_tensor.size(1)) {
            fprintf(stderr, "ContextLMScorer: Unexpected backbone output shape "
                    "(dim=%zu, expected [1, %d, %d], pos=%d)\n",
                    static_cast<size_t>(out_tensor.dim()), L, E, pos);
            return nullptr;
        }

        if (out_tensor.scalar_type() == executorch::aten::ScalarType::Half) {
            const uint16_t* fp16_ptr = out_tensor.const_data_ptr<uint16_t>();
            // Output is [1, L, E], index at [0, pos, :]
            fp16_ptr += pos * E;
            for (int i = 0; i < E; ++i) {
                h_buf[i] = fp16_to_fp32(fp16_ptr[i]);
            }
        } else {
            const float* ptr = out_tensor.const_data_ptr<float>();
            // Output is [1, L, E], index at [0, pos, :]
            std::memcpy(h_buf, ptr + pos * E, E * sizeof(float));
        }

        return h_buf;
    }

    // Score a single word given h. Returns h·emb(word) + bias(word).
    float score_word(const float* h, uint32_t word_id,
                     const uint32_t* hash_indices) const {
        int E = emb_embed_dim;
        float dot = 0.0f;
        float bias = 0.0f;

        if (static_cast<int>(word_id) < emb_num_exact) {
            // Exact embedding
            const float* emb = exact_embed.data() + word_id * E;
            for (int d = 0; d < E; ++d) {
                dot += h[d] * emb[d];
            }
            bias = exact_bias[word_id];
        } else {
            // Hash embedding: sum over K hash functions
            int K = cfg_num_hashes;
            std::vector<float> emb_sum(E, 0.0f);
            float bias_sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                uint32_t bucket = hash_indices[k];
                if (static_cast<int>(bucket) >= emb_num_buckets) continue;
                const float* bucket_emb = hash_embed.data() + bucket * E;
                for (int d = 0; d < E; ++d) {
                    emb_sum[d] += bucket_emb[d];
                }
                bias_sum += hash_bias[bucket];
            }
            for (int d = 0; d < E; ++d) {
                dot += h[d] * emb_sum[d];
            }
            bias = bias_sum;
        }

        return dot + bias;
    }
};

ContextLMScorer::ContextLMScorer() : impl_(std::make_unique<Impl>()) {}
ContextLMScorer::~ContextLMScorer() = default;
ContextLMScorer::ContextLMScorer(ContextLMScorer&&) noexcept = default;
ContextLMScorer& ContextLMScorer::operator=(ContextLMScorer&&) noexcept = default;

// ---- Extract embeddings from PTE get_embeddings method ----

static bool extract_embeddings_from_pte(
    Module& module,
    int& num_exact, int& num_buckets, int& embed_dim,
    std::vector<float>& exact_embed, std::vector<float>& exact_bias,
    std::vector<float>& hash_embed, std::vector<float>& hash_bias
) {
    auto load_err = module.load_method("get_embeddings");
    if (load_err != Error::Ok) {
        fprintf(stderr, "ContextLMScorer: Failed to load get_embeddings method (err=0x%x)\n",
                static_cast<unsigned>(load_err));
        return false;
    }

    auto result = module.execute("get_embeddings", std::vector<EValue>{});
    if (!result.ok()) {
        fprintf(stderr, "ContextLMScorer: get_embeddings execution failed (err=0x%x)\n",
                static_cast<unsigned>(result.error()));
        return false;
    }

    auto& outputs = result.get();
    if (outputs.size() < 4) {
        fprintf(stderr, "ContextLMScorer: get_embeddings returned %zu outputs (expected 4)\n",
                outputs.size());
        return false;
    }

    // Extract 4 tensors: exact_embed [T, E], exact_bias [T], hash_embed [B, E], hash_bias [B]
    for (int i = 0; i < 4; ++i) {
        if (!outputs[i].isTensor()) {
            fprintf(stderr, "ContextLMScorer: get_embeddings output %d is not a tensor\n", i);
            return false;
        }
    }

    auto& t_exact_embed = outputs[0].toTensor();
    auto& t_exact_bias = outputs[1].toTensor();
    auto& t_hash_embed = outputs[2].toTensor();
    auto& t_hash_bias = outputs[3].toTensor();

    num_exact = static_cast<int>(t_exact_embed.size(0));
    embed_dim = static_cast<int>(t_exact_embed.size(1));
    num_buckets = static_cast<int>(t_hash_embed.size(0));

    if (num_exact <= 0 || embed_dim <= 0 || num_buckets <= 0) {
        fprintf(stderr, "ContextLMScorer: Invalid embedding dimensions "
                "(exact=%d, dim=%d, buckets=%d)\n", num_exact, embed_dim, num_buckets);
        return false;
    }

    // Validate tensor element counts match expected sizes
    if (t_exact_embed.numel() != num_exact * embed_dim ||
        t_exact_bias.numel() != num_exact ||
        t_hash_embed.numel() != num_buckets * embed_dim ||
        t_hash_bias.numel() != num_buckets) {
        fprintf(stderr, "ContextLMScorer: Embedding tensor size mismatch\n");
        return false;
    }

    // Helper to copy tensor data, handling FP16 to FP32 conversion
    auto copy_tensor = [](const executorch::aten::Tensor& tensor,
                          std::vector<float>& out, int count) {
        out.resize(count);
        if (tensor.scalar_type() == executorch::aten::ScalarType::Half) {
            const uint16_t* src = tensor.const_data_ptr<uint16_t>();
            for (int i = 0; i < count; ++i) {
                out[i] = fp16_to_fp32(src[i]);
            }
        } else {
            const float* src = tensor.const_data_ptr<float>();
            std::memcpy(out.data(), src, count * sizeof(float));
        }
    };

    copy_tensor(t_exact_embed, exact_embed, num_exact * embed_dim);
    copy_tensor(t_exact_bias, exact_bias, num_exact);
    copy_tensor(t_hash_embed, hash_embed, num_buckets * embed_dim);
    copy_tensor(t_hash_bias, hash_bias, num_buckets);

    fprintf(stderr, "ContextLMScorer: Extracted embeddings from PTE "
            "(exact=[%d, %d], hash=[%d, %d])\n",
            num_exact, embed_dim, num_buckets, embed_dim);

    return true;
}

// ---- Public API ----

// Prepare the PTE side of ContextLMScorer: runtime init, module load,
// embedding extraction, forward-method load, max_context_len derivation.
// Leaves impl in a state where impl->vocab has NOT been built yet.
static bool prepare_pte(ContextLMScorer::Impl& impl, const std::string& pte_path,
                        int num_threads) {
    executorch::runtime::runtime_init();

    if (num_threads > 0) {
        auto* tp = executorch::extension::threadpool::get_threadpool();
        if (tp) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            tp->_unsafe_reset_threadpool(num_threads);
#pragma GCC diagnostic pop
        }
    }

    impl.module = std::make_unique<Module>(pte_path, Module::LoadMode::Mmap);

    if (!extract_embeddings_from_pte(
            *impl.module,
            impl.emb_num_exact, impl.emb_num_buckets, impl.emb_embed_dim,
            impl.exact_embed, impl.exact_bias,
            impl.hash_embed, impl.hash_bias)) {
        return false;
    }

    auto load_err = impl.module->load_forward();
    if (load_err != Error::Ok) {
        fprintf(stderr, "ContextLMScorer: Failed to load forward method '%s' (err=0x%x)\n",
                pte_path.c_str(), static_cast<unsigned>(load_err));
        return false;
    }

    auto meta = impl.module->method_meta("forward");
    if (!meta.ok()) {
        fprintf(stderr, "ContextLMScorer: Cannot read forward method metadata\n");
        return false;
    }
    auto input_meta = meta->input_tensor_meta(0);
    if (!input_meta.ok() || input_meta->sizes().size() < 2) {
        fprintf(stderr, "ContextLMScorer: Cannot read forward input tensor shape\n");
        return false;
    }
    impl.cfg_max_context_len = input_meta->sizes()[1];
    impl.cfg_num_hashes = 2;

    return true;
}

// Finish setup after vocab has been built: fallback map + buffers.
static void finalize_after_vocab(ContextLMScorer::Impl& impl) {
    impl.exact_scores.resize(impl.emb_num_exact);
    int L = impl.cfg_max_context_len;
    int K = impl.cfg_num_hashes;
    impl.context_ids_buf.resize(L);
    impl.context_hashes_buf.resize(L * K);
    impl.loaded = true;
}

bool ContextLMScorer::load(
    const std::string& pte_path,
    const std::string& vocab_path,
    int num_threads
) {
    impl_->cleanup();
    
    std::string meta_path = metadata_path_for(pte_path);
    if (!load_contextlm_metadata(meta_path, impl_->metadata)) {
        fprintf(stderr, "ContextLMScorer: no metadata.json at '%s'!\n", meta_path.c_str());
        return false;
    }

    if(impl_->metadata.kind != "contextlm") {
        fprintf(stderr, "ContextLMScorer: wrong metadata loaded! it's for kind '%s'...\n",
            impl_->metadata.kind.c_str());
        return false;
    }

    if (!prepare_pte(*impl_, pte_path, num_threads)) {
        impl_->cleanup();
        return false;
    }

    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        fprintf(stderr, "ContextLMScorer: Failed to open vocab '%s'\n",
            vocab_path.c_str());
        impl_->cleanup();
        return false;
    }

    impl_->word2id.clear();
    impl_->id2word.clear();
    impl_->id2word.emplace_back("<OOV>");
    
    std::string word;
    std::size_t line_idx = 0;
    while (std::getline(file, word)) {
        if (!word.empty() && word.back() == '\r') {
            word.pop_back();
        }
        
        impl_->word2id[word] = line_idx + 1;

        std::string lowered = utf8_tolower(word);
        if(lowered != word) {
            impl_->word2id_lower[lowered] = line_idx + 1;
        }
        impl_->id2word.emplace_back(word);
        
        ++line_idx;
    }
    
    finalize_after_vocab(*impl_);
    return true;
}

bool ContextLMScorer::is_loaded() const {
    return impl_->loaded;
}

std::vector<float> ContextLMScorer::score(
    const std::vector<std::string>& context_words,
    const std::vector<std::string>& candidate_words
) {
    if (!impl_->loaded || candidate_words.empty()) {
        return {};
    }

    int E = impl_->emb_embed_dim;
    int K = impl_->cfg_num_hashes;

    // Run backbone
    int n_ctx = impl_->build_context_tensors(context_words);
    std::vector<float> h(E);
    if (!impl_->run_backbone(h.data(), n_ctx)) {
        return {};
    }

    // Score each candidate via C++ dot product
    int n = static_cast<int>(candidate_words.size());
    std::vector<float> scores(n);
    std::vector<uint32_t> oov_buf(K, 0);

    for (int i = 0; i < n; ++i) {
        uint32_t wid;
        const uint32_t* hashes;
        impl_->lookup(candidate_words[i], wid, hashes, oov_buf.data());
        scores[i] = impl_->score_word(h.data(), wid, hashes);
    }

    return scores;
}

std::vector<std::pair<std::string, float>> ContextLMScorer::predict_topk(
    const std::vector<std::string>& context_words,
    int k
) {
    if (!impl_->loaded || k <= 0) {
        return {};
    }

    int E = impl_->emb_embed_dim;
    int T = impl_->emb_num_exact;
    k = std::min(k, T);

    // Run backbone
    int n_ctx = impl_->build_context_tensors(context_words);
    std::vector<float> h(E);
    if (!impl_->run_backbone(h.data(), n_ctx)) {
        return {};
    }

    // Score all exact embeddings: h·emb[i] + bias[i] for i in 1..T-1
    // Skip index 0 (<oov>) — not a real prediction target.
    // This is a dense matmul: [1, E] × [E, T] + [T] -> [T]
    float* scores = impl_->exact_scores.data();
    const float* emb = impl_->exact_embed.data();
    const float* bias = impl_->exact_bias.data();

    scores[0] = -1e30f;  // Ensure <oov> never appears in top-k
    for (int i = 1; i < T; ++i) {
        float dot = 0.0f;
        for (int d = 0; d < E; ++d) {
            dot += h[d] * emb[i * E + d];
        }
        scores[i] = dot + bias[i];
    }

    // Partial sort for top-k (starting from index 1 to skip <oov>)
    int n_candidates = T - 1;
    k = std::min(k, n_candidates);
    std::vector<int> indices(n_candidates);
    std::iota(indices.begin(), indices.end(), 1);  // 1..T-1

    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [scores](int a, int b) {
            return scores[a] > scores[b];
        });

    std::vector<std::pair<std::string, float>> result;
    result.reserve(k);
    for (int i = 0; i < k; ++i) {
        int idx = indices[i];
        const std::string& word = (idx < static_cast<int>(impl_->id2word.size()))
            ? impl_->id2word[idx] : "";
        if (!word.empty()) {
            result.emplace_back(word, scores[idx]);
        }
    }

    return result;
}

ContextLMScorer::WordLookup ContextLMScorer::lookup_word(const std::string& word, bool allow_lowercase_fallback) const {
    WordLookup result;
    result.word_id = 0;
    result.hash_indices.resize(impl_->cfg_num_hashes, 0);
    result.is_exact = false;

    if (!impl_->loaded) return result;

    std::vector<uint32_t> oov_buf(impl_->cfg_num_hashes, 0);
    uint32_t wid;
    const uint32_t* hashes;
    impl_->lookup(word, wid, hashes, oov_buf.data(), allow_lowercase_fallback);

    result.word_id = wid;
    for (int k = 0; k < impl_->cfg_num_hashes; ++k) {
        result.hash_indices[k] = hashes[k];
    }

    result.is_exact = wid != impl_->emb_num_exact;
    return result;
}

int ContextLMScorer::embed_dim() const {
    return impl_->emb_embed_dim;
}

int ContextLMScorer::num_exact() const {
    return impl_->emb_num_exact;
}

int ContextLMScorer::num_buckets() const {
    return impl_->emb_num_buckets;
}

const ContextLMMetadata& ContextLMScorer::metadata() const {
    return impl_->metadata;
}

}  // namespace swipe_decoder
