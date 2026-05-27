#include "swipe_decoder/beam_search.hpp"
#include "swipe_decoder/trie.hpp"
#include "swipe_decoder/itrie.h"
#include "swipe_decoder/layout.hpp"
#include "swipe_decoder/vocab_hash.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#ifdef __linux__
#include <cstdio>
#endif

namespace swipe_decoder {

// ============================================================================
// Flat hash map - open addressing, linear probing
// ============================================================================

/**
 * Minimal open-addressing hash map for (uint64_t -> uint32_t).
 *
 * Designed for the beam search dedup map where:
 * - Keys are (trie_node << 1) | blank_ended
 * - Working set is small (beam_width * ~15 entries per timestep)
 * - Cleared every timestep via generation counter - O(1) clear
 * - Load factor kept <= 0.5 for low collision rate
 *
 * Uses a generation counter for clear(). Each entry stores the generation
 * it was written in, and stale entries are treated as empty. This avoids
 * clearing 128KB+ of memory 32 times per decode at wide beam widths.
 *
 * Entry is 16 bytes (uint64 key + uint32 value + uint32 generation)
 */
struct FlatHashMap {
    struct Entry {
        uint64_t key;
        uint32_t value;       // index into new_beam_vec
        uint32_t generation;  // valid when == current gen
    };

    Entry* table = nullptr;
    size_t mask = 0;
    size_t capacity = 0;
    uint32_t gen = 0;

    FlatHashMap() = default;
    ~FlatHashMap() { std::free(table); }

    // Non-copyable
    FlatHashMap(const FlatHashMap&) = delete;
    FlatHashMap& operator=(const FlatHashMap&) = delete;

    void resize(size_t min_capacity) {
        // Round up to power of 2
        size_t cap = 64;
        while (cap < min_capacity) cap <<= 1;
        if (cap == capacity) return;
        std::free(table);
        capacity = cap;
        mask = cap - 1;
        table = static_cast<Entry*>(std::calloc(cap, sizeof(Entry)));
        gen = 1;  // generation 0 = all entries stale after calloc
    }

    void clear() {
        ++gen;
        if (gen == 0) {
            // Wraparound (every ~4 billion clears) - full reset
            std::memset(table, 0, capacity * sizeof(Entry));
            gen = 1;
        }
    }

    // MurmurHash3 64-bit finalizer
    static uint64_t hash(uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }

    /**
     * Find existing entry or insert new one.
     *
     * Returns (pointer to value slot, true if newly inserted).
     * On insert, the value is set to `default_value`.
     */
    std::pair<uint32_t*, bool> find_or_insert(uint64_t key, uint32_t default_value) {
        size_t idx = hash(key) & mask;
        while (true) {
            auto& e = table[idx];
            if (e.generation == gen && e.key == key) {
                return {&e.value, false};
            }
            if (e.generation != gen) {
                e.key = key;
                e.value = default_value;
                e.generation = gen;
                return {&e.value, true};
            }
            idx = (idx + 1) & mask;
        }
    }
};

// ============================================================================
// Beam search implementation
// ============================================================================

struct BeamHypothesis {
    float score;        // Log probability score
    float prune_score;  // Length-aware log probability score for smarter pruning
    uint32_t trie_node; // Current trie node index

    // For two-finger swipe typing
    // float time_score; (TODO)
    unsigned left_be : 1;
    unsigned right_be : 1;
    uint16_t left_idx, right_idx;

    // Sort by prune_score descending so nth_element keeps the top-K
    // under the (optionally length-aware) partition key.
    bool operator<(const BeamHypothesis& other) const {
        return prune_score > other.prune_score;
    }
};

/**
 * Length-aware pruning key.
 *  prune_score = score / max(depth, 1)^gamma_prune + beta_prune * depth
 */
static inline float length_prune_score(
    float score, uint16_t depth, float gamma_prune, float beta_prune)
{
    if (gamma_prune == 0.0f && beta_prune == 0.0f) {
        return score;
    }

    int d_eff = depth > 0 ? static_cast<int>(depth) : 1;
    return score / std::pow(static_cast<float>(d_eff), gamma_prune)
         + beta_prune * static_cast<float>(depth);
}

struct TrieBeamSearch::Impl {
    // Reusable buffers to avoid allocations during decode
    std::vector<BeamHypothesis> beam;
    std::vector<BeamHypothesis> new_beam_vec;
    FlatHashMap dedup_map;

    void init(int beam_width, int num_classes) {
        beam.reserve(beam_width * 2);
        new_beam_vec.reserve(beam_width * num_classes);
        // Table size: 2x max possible candidates for ~0.5 load factor
        dedup_map.resize(static_cast<size_t>(beam_width) * num_classes * 2);
    }

    void clear() {
        beam.clear();
        new_beam_vec.clear();
        dedup_map.clear();
    }
};

TrieBeamSearch::TrieBeamSearch(
    int max_beam_width,
    const ScoringParams &scoring,
    int num_chars
)
    : max_beam_width_(max_beam_width)
    , num_chars_(num_chars)
    , num_classes_(num_chars_ + 1)
    , blank_idx_(num_chars_)
    , impl_(std::make_unique<Impl>())
{
    set_scoring(scoring);
    impl_->init(max_beam_width_, num_classes_);
}

TrieBeamSearch::~TrieBeamSearch() = default;


void TrieBeamSearch::advance_beam_search(
    const ITrie *trie,
    const BeamHypothesis &hyp,
    const float *probs_ts,
    float advance_lag, float advance_delta,
    uint16_t left_advance, uint16_t right_advance
) {
    const int bi = blank_idx_;
    const float gp = single_mode_ ? gamma_prune_ : 0.0f;
    const float bp = single_mode_ ? beta_prune_  : 0.0f;

    // TODO: Parametrize and optimize
    float sc = 600.0f;
    const float delta_ts = -((advance_lag * advance_lag) / (sc * sc));
    //const float new_time_score = hyp.time_score + delta_ts;

    const TrieId node = hyp.trie_node;
    const uint16_t node_depth = static_cast<uint16_t>(trie->get_depth(node));

    const uint16_t new_left_idx = hyp.left_idx + left_advance;
    const uint16_t new_right_idx = hyp.right_idx + right_advance;

    // A: emit blank
    {
        float blank_score = hyp.score + probs_ts[bi] + delta_ts;
        float blank_prune = length_prune_score(
            blank_score, node_depth, gp, bp);
        uint64_t key = (static_cast<uint64_t>(hyp.trie_node) << 1) | 1ULL;

        bool new_left_be = hyp.left_be   || (left_advance > 0);
        bool new_right_be = hyp.right_be || (right_advance > 0);

        auto [val_ptr, inserted] = impl_->dedup_map.find_or_insert(
            key, static_cast<uint32_t>(impl_->new_beam_vec.size()));
        
        BeamHypothesis new_hyp = {
            blank_score,
            blank_prune,
            hyp.trie_node,
            //new_time_score,
            new_left_be, new_right_be,
            new_left_idx, new_right_idx
        };

        if (inserted) {
            impl_->new_beam_vec.push_back(new_hyp);
        } else if (impl_->new_beam_vec[*val_ptr].score < blank_score) {
            impl_->new_beam_vec[*val_ptr] = new_hyp;
        }
    }

    // B: emit character. Unlike typical CTC, we don't require blank_ended for
    // emitting the same character.
    const int num_children = trie->get_child_count(node);
    for (int child_idx = num_children - 1; child_idx >= 0; child_idx--) {
        const TrieId child_node = trie->get_child(node, child_idx);
        int c = trie->get_char_idx(child_node);
        if(c == -1) continue;
        
        float char_score = hyp.score + probs_ts[c] + delta_ts;
        const uint16_t child_depth = node_depth + 1;
        float char_prune = length_prune_score(
            char_score, child_depth, gp, bp);

        bool new_left_be  = hyp.left_be  && (left_advance == 0);
        bool new_right_be = hyp.right_be && (right_advance == 0);

        uint64_t key = static_cast<uint64_t>(child_node) << 1;
        auto [val_ptr, inserted] = impl_->dedup_map.find_or_insert(
            key, static_cast<uint32_t>(impl_->new_beam_vec.size()));

        BeamHypothesis new_hyp = {
            char_score,
            char_prune,
            child_node,
            //new_time_score,
            new_left_be, new_right_be,
            new_left_idx, new_right_idx
        };
        
        if (inserted) {
            impl_->new_beam_vec.push_back(new_hyp);
        } else if (impl_->new_beam_vec[*val_ptr].score < char_score) {
            impl_->new_beam_vec[*val_ptr] = new_hyp;
        }
    }


    // C: emit the same character but stay on the same node
    const bool blank_ended = left_advance > 0 ? hyp.left_be : hyp.right_be;
    if (!blank_ended && hyp.trie_node != 0) {
        int pc = trie->get_char_idx(hyp.trie_node);
        if(pc != -1) {
            float same_score = hyp.score + probs_ts[pc] + delta_ts;
            float same_prune = length_prune_score(
                same_score, node_depth, gp, bp);
            uint64_t key = static_cast<uint64_t>(hyp.trie_node) << 1;

            bool new_left_be  = hyp.left_be  && (left_advance == 0);
            bool new_right_be = hyp.right_be && (right_advance == 0);

            BeamHypothesis new_hyp = {
                same_score,
                same_prune,
                hyp.trie_node,
                //new_time_score,
                new_left_be, new_right_be,
                new_left_idx, new_right_idx
            };

            auto [val_ptr, inserted] = impl_->dedup_map.find_or_insert(
                key, static_cast<uint32_t>(impl_->new_beam_vec.size()));
            if (inserted) {
                impl_->new_beam_vec.push_back(new_hyp);
            } else if (impl_->new_beam_vec[*val_ptr].score < same_score) {
                impl_->new_beam_vec[*val_ptr] = new_hyp;
            }
        }
    }
}

/**
 * Run the core beam search loop (shared by decode, decode_with_scores).
 *
 * After this returns, impl->beam contains the final hypotheses.
 */
void TrieBeamSearch::run_beam_search(
    const ITrie *trie,
    const float* left_logprobs,
    const float* left_times,
    size_t left_time_steps,
    const float* right_logprobs,
    const float* right_times,
    size_t right_time_steps,
    int beam_width
) {
    impl_->clear();
    const int ncl = num_classes_;

    // Guard against crazy timesteps because Hypothesis left/right idx are 16-bit
    if(left_time_steps >= std::numeric_limits<uint16_t>::max()
            || right_time_steps >= std::numeric_limits<uint16_t>::max()) {
        fprintf(stderr, "Too many timesteps specified! %zu:%zu\n", left_time_steps, right_time_steps);
        return;
    }

    // Initialize with empty hypothesis at trie root
    impl_->beam.push_back({0.0f, 0.0f, trie->root(), false, false, 0, 0});
    for (int t=0; t < left_time_steps + right_time_steps; ++t) {
        impl_->new_beam_vec.clear();
        impl_->dedup_map.clear();

        for (const auto& hyp : impl_->beam) {
            const float* l_probs = (left_logprobs  != nullptr) ? (left_logprobs  + hyp.left_idx  * ncl) : nullptr;
            const float* r_probs = (right_logprobs != nullptr) ? (right_logprobs + hyp.right_idx * ncl) : nullptr;

            float dt, cdt, adt;
            bool can_advance_left  = hyp.left_idx  < left_time_steps  && l_probs != nullptr;
            bool can_advance_right = hyp.right_idx < right_time_steps && r_probs != nullptr;

            if(can_advance_left && can_advance_right) dt = left_times[hyp.left_idx] - right_times[hyp.right_idx];
            else dt = 0.0f;

            // 1. Advance left
            if(can_advance_left) {
                if(hyp.left_idx + 1 < left_time_steps && hyp.right_idx < right_time_steps) {
                    cdt = left_times[hyp.left_idx + 1] - right_times[hyp.right_idx];
                    adt = left_times[hyp.left_idx + 1] - left_times[hyp.left_idx];
                } else {
                    cdt = dt;
                    adt = 0.0f;
                }
                advance_beam_search(trie, hyp, l_probs, cdt, adt, 1, 0);
            }

            // 2. Advance right
            if(can_advance_right) {
                if(hyp.left_idx < left_time_steps && hyp.right_idx + 1 < right_time_steps) {
                    cdt = left_times[hyp.left_idx] - right_times[hyp.right_idx + 1];
                    adt = right_times[hyp.right_idx + 1] - right_times[hyp.right_idx];
                } else {
                    cdt = dt;
                    adt = 0.0f;
                }
                advance_beam_search(trie, hyp, r_probs, cdt, adt, 0, 1);
            }
        }

        // Prune to beam width
        if (impl_->new_beam_vec.size() > static_cast<size_t>(beam_width)) {
            std::nth_element(
                impl_->new_beam_vec.begin(),
                impl_->new_beam_vec.begin() + beam_width,
                impl_->new_beam_vec.end()
            );
            impl_->new_beam_vec.resize(beam_width);
        }

        std::swap(impl_->beam, impl_->new_beam_vec);

        if (impl_->beam.empty()) break;
    }
}

// TODO: Remove this method completely, in favor of with_scores?
std::vector<std::pair<std::string, float>> TrieBeamSearch::decode_multi(
    const ITrie *itrie,
    const float* left_logprobs,
    const float* left_times,
    size_t left_time_steps,
    const float* right_logprobs,
    const float* right_times,
    size_t right_time_steps,
    int top_k
) {
    std::vector<DecodedWord> words;
    decode_with_scores_multi(itrie, words, left_logprobs, left_times, left_time_steps, right_logprobs, right_times, right_time_steps, 1.0f);
    finish_decode(words, top_k);

    std::vector<std::pair<std::string, float>> results;
    for(const auto &v : words) {
        results.emplace_back(v.word, v.final_score);
    }

    return results;
}

void TrieBeamSearch::decode_with_scores_multi(
    const ITrie *itrie,
    std::vector<DecodedWord> &words_out,
    const float* left_logprobs,
    const float* left_times,
    size_t left_time_steps,
    const float* right_logprobs,
    const float* right_times,
    size_t right_time_steps,
    float weight,
    int beam_width
) {
    if(beam_width > max_beam_width_) {
        fprintf(stderr, "Beam width %d exceeds maximum %d!\n", beam_width, max_beam_width_);
        beam_width = max_beam_width_;
    }
    run_beam_search(itrie, left_logprobs, left_times, left_time_steps, right_logprobs, right_times, right_time_steps, beam_width);

    for (const auto& hyp : impl_->beam) {
        const TrieId node = hyp.trie_node;
        if (!itrie->is_word(node)) continue;

        float word_len = static_cast<float>(itrie->get_depth(node));

        DecodedWord dw;
        dw.ctc_score = hyp.score;
        dw.freq_score = freq_bonus_ * itrie->get_log_frequency(node);
        dw.word_length = word_len;
        float L = std::max(word_len, 1.0f);
        dw.final_score = dw.ctc_score / std::pow(L, length_norm_exp_) + weight * length_bonus_ * word_len + dw.freq_score;
        dw.frequency = itrie->get_log_frequency(node);
        dw.word = itrie->get_word(node);

        words_out.emplace_back(dw);
    }
}

void TrieBeamSearch::finish_decode(
    std::vector<DecodedWord> &words_out,
    int top_k
) {
    std::sort(words_out.begin(), words_out.end(),
              [](const auto& a, const auto& b) {
                  return a.final_score > b.final_score;
              });

    // Remove duplicates
    {
        std::unordered_set<std::string> seen_words;

        for (int i = 0; i<words_out.size(); i++) {
            std::string word = words_out[i].word;
            std::transform(word.begin(), word.end(), word.begin(),
                        [](unsigned char c) { return std::tolower(c); });

            if (seen_words.find(word) != seen_words.end()) {
                words_out.erase(words_out.begin() + i);
                i--;
            } else {
                seen_words.insert(word);
            }
        }
    }

    if (words_out.size() > static_cast<size_t>(top_k)) {
        words_out.resize(top_k);
    }
}

}  // namespace swipe_decoder
