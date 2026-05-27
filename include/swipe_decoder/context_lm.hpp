/**
 * ContextLMScorer - ExecuTorch-based context language model.
 *
 * The ContextLM depends on a paired vocab.txt file which was used during its
 * training. This is distinct from the trie vocab.combined. This one contains
 * the corresponding words in its exact embedding vocab.
 *
 * For words that don't have an exact embedding, they are approximated via
 * hashing.
 *
 * score(): looks up embeddings for requested candidates, computes h·emb + bias
 * predict_topk(): scores all exact embeddings (top ~32K words), returns top-k
 */

#ifndef SWIPE_DECODER_CONTEXT_LM_HPP
#define SWIPE_DECODER_CONTEXT_LM_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "swipe_decoder/model_metadata.hpp"

namespace swipe_decoder {

class ContextLMScorer {
public:
    ContextLMScorer();
    ~ContextLMScorer();

    // Non-copyable
    ContextLMScorer(const ContextLMScorer&) = delete;
    ContextLMScorer& operator=(const ContextLMScorer&) = delete;

    // Movable
    ContextLMScorer(ContextLMScorer&&) noexcept;
    ContextLMScorer& operator=(ContextLMScorer&&) noexcept;

    /**
     * Load backbone model and build vocab from combined file.
     *
     * The PTE must contain two methods:
     *   - "forward": backbone (context_ids, context_hashes -> h)
     *   - "get_embeddings": returns (exact_embed, exact_bias, hash_embed, hash_bias)
     *
     * All model parameters (num_exact, num_buckets, max_context_len) are
     * derived from the PTE file itself.
     *
     * @param pte_path Path to backbone .pte (multi-method)
     * @param vocab_path Path to vocab.txt file
     * @param num_threads CPU threads for inference
     * @return true if loaded successfully
     */
    bool load(const std::string& pte_path,
              const std::string& vocab_path,
              int num_threads = 1);

    bool is_loaded() const;

    /**
     * Score candidate words given context.
     *
     * Runs backbone to get h, then scores each candidate via dot product
     * with its embedding + bias.
     */
    std::vector<float> score(
        const std::vector<std::string>& context_words,
        const std::vector<std::string>& candidate_words);

    /**
     * Predict top-k next words given context.
     *
     * Runs backbone to get h, then scores all exact embeddings (0..num_exact-1)
     * and returns the top-k. Hash-embedded words are excluded.
     */
    std::vector<std::pair<std::string, float>> predict_topk(
        const std::vector<std::string>& context_words,
        int k = 10);

    int embed_dim() const;
    int num_exact() const;
    int num_buckets() const;

    /**
     * Look up a word in the vocab table.
     *
     * Returns (word_id, [hash_idx_0, hash_idx_1], is_exact)
     *
     * Words that don't have an exact embedding will have word_id of num_exact()
     */
    struct WordLookup {
        uint32_t word_id;
        std::vector<uint32_t> hash_indices;
        bool is_exact;
    };
    WordLookup lookup_word(const std::string& word, bool allow_lowercase_fallback = true) const;


    const ContextLMMetadata& metadata() const;

    struct Impl;
private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace swipe_decoder

#endif  // SWIPE_DECODER_CONTEXT_LM_HPP
