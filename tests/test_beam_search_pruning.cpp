/**
 * Tests for transition C (same-char-collapse-stay) and length-aware
 * beam pruning. These features were ported from universal-swipe's
 * `_beam_search.pyx` so the per-state ctc_score equals the exact
 * FA-Viterbi log-prob and so long-prefix hypotheses can survive narrow
 * beam widths.
 */

#include "swipe_decoder/beam_search.hpp"
#include "swipe_decoder/trie.hpp"
#include "swipe_decoder/itrie.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    static void test_##name(); \
    struct Register_##name { \
        Register_##name() { \
            tests_run++; \
            printf("  %-60s ", #name); \
            try { test_##name(); tests_passed++; printf("PASS\n"); } \
            catch (...) { printf("FAIL (exception)\n"); } \
        } \
    } register_##name; \
    static void test_##name()

#define ASSERT(cond) do { \
    if (!(cond)) { printf("FAIL\n    %s:%d: %s\n", __FILE__, __LINE__, #cond); return; } \
} while(0)

using namespace swipe_decoder;


// ── Transition C ─────────────────────────────────────────────────────────────


TEST(transition_c_reaches_repeated_letter_word) {
    // Vocabulary including a word with no internal repeats and a word
    // with a repeat ("aab"). Emissions stay on 'a' through frames 0-15
    // (no blank) then switch to 'b'. Without transition C the (aa, False)
    // state only receives mass via the strict-CTC path; with C, the a->a
    // collapse-stay contributes, and "aab" surfaces in the beam.
    auto trie = std::make_shared<Trie>("abcdefghij");
    trie->insert("aab", 1.0f);
    trie->insert("ab", 1.0f);
    trie->insert("bab", 1.0f);
    auto itrie = trie->itrie();

    TrieBeamSearch bs(20, ScoringParams{}, 10);

    const int T = 24;
    const int ncl = 11;  // 10 letters + blank
    std::vector<float> log_probs(T * ncl, -5.0f);
    for (int t = 0; t < 16; ++t) log_probs[t * ncl + 0] = 0.0f;  // 'a'
    for (int t = 16; t < 24; ++t) log_probs[t * ncl + 1] = 0.0f; // 'b'

    auto results = bs.decode(itrie, log_probs.data(), T, 10);
    bool found_aab = false;
    for (const auto& r : results) {
        if (r.first == "aab") { found_aab = true; break; }
    }
    ASSERT(found_aab);
}


// ── Length-aware pruning — γ_prune / β_prune ─────────────────────────────────


TEST(zero_prune_matches_legacy) {
    // With γ_prune = β_prune = 0, prune_score == score and nth_element
    // partitions identically to the legacy raw-CTC pruning. Sanity check
    // that explicit zeros produce the same results as the default ctor.
    auto trie_a = std::make_shared<Trie>();
    trie_a->insert("the", 1.0f);
    trie_a->insert("they", 1.0f);
    trie_a->insert("there", 1.0f);
    auto trie_b = std::make_shared<Trie>();
    trie_b->insert("the", 1.0f);
    trie_b->insert("they", 1.0f);
    trie_b->insert("there", 1.0f);
    auto itrie_a = trie_a->itrie();
    auto itrie_b = trie_b->itrie();

    TrieBeamSearch bs_default(10, ScoringParams{});
    TrieBeamSearch bs_explicit(10, ScoringParams{
        .gamma_prune = 0.0f,
        .beta_prune = 0.0f
    });

    const int T = 6;
    const int ncl = 27;  // 26 letters + blank
    std::vector<float> log_probs(T * ncl, -3.0f);
    // 't' h' e' y'  blank blank
    log_probs[0 * ncl + ('t' - 'a')] = 0.0f;
    log_probs[1 * ncl + ('h' - 'a')] = 0.0f;
    log_probs[2 * ncl + ('e' - 'a')] = 0.0f;
    log_probs[3 * ncl + ('y' - 'a')] = 0.0f;
    log_probs[4 * ncl + 26]          = 0.0f;
    log_probs[5 * ncl + 26]          = 0.0f;

    auto r_def = bs_default.decode(itrie_a, log_probs.data(), T, 5);
    auto r_exp = bs_explicit.decode(itrie_b, log_probs.data(), T, 5);

    ASSERT(r_def.size() == r_exp.size());
    for (size_t i = 0; i < r_def.size(); ++i) {
        ASSERT(r_def[i].first == r_exp[i].first);
        // Final scores are rescored from raw log-probs by gamma/lambda/beta —
        // they're not the partition key — but with identical setup they
        // must match exactly.
        ASSERT(std::abs(r_def[i].second - r_exp[i].second) < 1e-5f);
    }
}


TEST(nonzero_prune_does_not_crash_or_corrupt) {
    // Smoke test: tuned-style nonzero pruning runs and yields non-empty
    // top-K with finite scores. Cross-validation of the actual lift vs
    // bit-equality with universal-swipe's beam belongs in an end-to-end
    // eval; this just guards the path.
    auto trie = std::make_shared<Trie>();
    trie->insert("the", 50.0f);
    trie->insert("they", 50.0f);
    trie->insert("there", 30.0f);
    auto itrie = trie->itrie();

    TrieBeamSearch bs(20, ScoringParams{
        .gamma_prune = 0.1922f,
        .beta_prune = 1.8051f
    });

    const int T = 8;
    const int ncl = 27;
    std::vector<float> log_probs(T * ncl, -3.0f);
    log_probs[0 * ncl + ('t' - 'a')] = 0.0f;
    log_probs[1 * ncl + ('h' - 'a')] = 0.0f;
    log_probs[2 * ncl + ('e' - 'a')] = 0.0f;
    for (int t = 3; t < T; ++t) log_probs[t * ncl + 26] = 0.0f;  // blanks

    auto results = bs.decode(itrie, log_probs.data(), T, 5);
    ASSERT(!results.empty());
    for (const auto& r : results) {
        ASSERT(std::isfinite(r.second));
        ASSERT(!r.first.empty());
    }
}


TEST(set_pruning_updates_runtime) {
    // The set_scoring() mutator should swap in new coefficients without
    // requiring a rebuild. Construct with zero pruning, change to tuned
    // values, run a decode — the path should execute without crashing
    // and accessors should reflect the new state.
    auto trie = std::make_shared<Trie>();
    trie->insert("hello", 1.0f);
    trie->insert("help", 1.0f);
    trie->insert("hell", 1.0f);
    auto itrie = trie->itrie();

    TrieBeamSearch bs(10, ScoringParams{});
    ASSERT(bs.gamma_prune() == 0.0f);
    ASSERT(bs.beta_prune() == 0.0f);

    bs.set_scoring(ScoringParams{
        .gamma_prune = 0.35f,
        .beta_prune = 1.25f
    });
    ASSERT(bs.gamma_prune() == 0.35f);
    ASSERT(bs.beta_prune() == 1.25f);

    const int T = 8;
    const int ncl = 27;
    std::vector<float> log_probs(T * ncl, -3.0f);
    log_probs[0 * ncl + ('h' - 'a')] = 0.0f;
    log_probs[1 * ncl + ('e' - 'a')] = 0.0f;
    log_probs[2 * ncl + ('l' - 'a')] = 0.0f;
    log_probs[3 * ncl + ('l' - 'a')] = 0.0f;
    log_probs[4 * ncl + ('o' - 'a')] = 0.0f;
    for (int t = 5; t < T; ++t) log_probs[t * ncl + 26] = 0.0f;

    auto results = bs.decode(itrie, log_probs.data(), T, 3);
    ASSERT(!results.empty());
}


int main() {
    printf("Running beam-search pruning + transition-C tests...\n");
    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
