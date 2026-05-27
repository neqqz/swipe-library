/**
 * LM vocab coverage test - checks that trie words are reachable by the context LM.
 *
 * Usage:
 *   ./test_lm_vocab <context_lm.pte> <lm_vocab.txt> <vocab.combined>
 *
 * Parses the combined vocab file to extract the same lowercased alpha forms
 * the trie produces, then looks each up in the LM's VocabTable. Reports OOV
 * count and rate.  Fails if OOV rate exceeds a threshold.
 *
 * This catches systemic mismatches like the decoder outputting lowercase words
 * while the LM vocab is case-sensitive (trained with capitalization preserved).
 *
 * Run via: make test-sanity
 */

#include "swipe_decoder/context_lm.hpp"
#include "swipe_decoder/beam_search.hpp"  // ParsedCombined, parse_combined_vocab, flatten_parsed_to_entries
#include "swipe_decoder/layout.hpp"  // utf8_to_codepoints, utf8_encode_one, simple_tolower

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace swipe_decoder;

// ============================================================================
// Minimal test framework (same pattern as test_sanity.cpp)
// ============================================================================

static int tests_run = 0;
static int tests_passed = 0;

struct TestFailed {};

#define ASSERT(cond) do { \
    if (!(cond)) { printf("FAIL\n    %s:%d: %s\n", __FILE__, __LINE__, #cond); throw TestFailed{}; } \
} while(0)

#define ASSERT_MSG(cond, msg) do { \
    if (!(cond)) { printf("FAIL\n    %s:%d: %s (%s)\n", __FILE__, __LINE__, #cond, msg); throw TestFailed{}; } \
} while(0)

static void run_test(const char* name, void (*func)()) {
    tests_run++;
    printf("  %-50s ", name);
    try {
        func();
        tests_passed++;
        printf("PASS\n");
    } catch (TestFailed&) {
        // already printed
    } catch (const std::exception& e) {
        printf("FAIL (exception: %s)\n", e.what());
    } catch (...) {
        printf("FAIL (unknown exception)\n");
    }
}

// ============================================================================
// Vocab parsing helpers (mirrors beam_search.cpp logic)
// ============================================================================

/// Normalize right single quotation mark (U+2019) to ASCII apostrophe
static std::string normalize_apostrophes(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ) {
        unsigned char c = s[i];
        // U+2019 = E2 80 99 in UTF-8
        if (c == 0xE2 && i + 2 < s.size()
            && (unsigned char)s[i+1] == 0x80
            && (unsigned char)s[i+2] == 0x99) {
            out += '\'';
            i += 3;
            continue;
        }
        // U+2018 = E2 80 98
        if (c == 0xE2 && i + 2 < s.size()
            && (unsigned char)s[i+1] == 0x80
            && (unsigned char)s[i+2] == 0x98) {
            out += '\'';
            i += 3;
            continue;
        }
        out += static_cast<char>(c);
        i += 1;
    }
    return out;
}

/// Represents a word from the combined vocab file
struct VocabEntry {
    std::string surface;    // original surface form from file (e.g. "China")
    std::string alpha;      // lowercased alpha-only form (e.g. "china")
    float freq;
};

/// Parse combined vocab file and return entries sorted by frequency (descending)
static std::vector<VocabEntry> parse_combined_vocab(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return {};

    // Deduplicate by alpha form, keeping the entry with highest freq
    std::unordered_map<std::string, VocabEntry> best;
    std::string line;
    while (std::getline(file, line)) {
        size_t start = line.find("word=");
        if (start == std::string::npos) continue;

        size_t word_start = start + 5;
        size_t comma = line.find(',', word_start);
        if (comma == std::string::npos) continue;

        std::string surface = line.substr(word_start, comma - word_start);
        surface = normalize_apostrophes(surface);

        // Extract frequency (,f=NNN)
        std::string needle = ",f=";
        size_t fpos = line.find(needle);
        if (fpos == std::string::npos) continue;
        size_t fstart = fpos + needle.size();
        size_t fend = fstart;
        while (fend < line.size() && line[fend] >= '0' && line[fend] <= '9') fend++;
        if (fend == fstart) continue;
        float freq = 0;
        try { freq = std::stof(line.substr(fstart, fend - fstart)); } catch (...) { continue; }

        std::string alpha = to_alpha_form(surface);
        if (alpha.empty()) continue;

        auto it = best.find(alpha);
        if (it == best.end() || freq > it->second.freq) {
            best[alpha] = {surface, alpha, freq};
        }
    }

    std::vector<VocabEntry> entries;
    entries.reserve(best.size());
    for (auto& [_, e] : best) {
        entries.push_back(std::move(e));
    }
    std::sort(entries.begin(), entries.end(),
              [](const VocabEntry& a, const VocabEntry& b) { return a.freq > b.freq; });
    return entries;
}

// ============================================================================
// Globals
// ============================================================================

static ContextLMScorer* g_lm = nullptr;
static std::vector<VocabEntry>* g_entries = nullptr;

// ============================================================================
// Test: overall OOV rate
// ============================================================================

static void test_overall_oov_rate() {
    const uint32_t MIN_EXACT_WORDS = 30000; // We have about 32k exact embeddings

    int total = 0;
    int inexact = 0;

    for (const auto& entry : *g_entries) {
        total++;
        auto surface_lookup = g_lm->lookup_word(entry.surface);
        if (!surface_lookup.is_exact) inexact++;
    }


    printf("\n");
    printf("    Total trie words:    %d\n", total);
    printf("    Inexact words:       %d\n", inexact);
    printf("    Exact words:         %d\n", (total-inexact));

    ASSERT_MSG((total-inexact) > MIN_EXACT_WORDS, "The number of exact embeddings should pass threshold");
    printf("    ");
}

// ============================================================================
// Test: top-N frequent words should have near-zero OOV
// ============================================================================

static void test_top_freq_words_coverage() {
    const int TOP_N = 500;
    const int MAX_OOV = 5; // at most 5 of the top 500 can be OOV

    int n = std::min(TOP_N, (int)g_entries->size());
    int oov_count = 0;
    std::vector<std::string> oov_words;

    for (int i = 0; i < n; i++) {
        const auto& entry = (*g_entries)[i];
        // Check both forms, if either matches, the LM can score this word
        auto surface_lookup = g_lm->lookup_word(entry.surface);
        auto alpha_lookup = g_lm->lookup_word(entry.alpha);
        if (!surface_lookup.is_exact && !alpha_lookup.is_exact) {
            oov_count++;
            if (oov_words.size() < 20) {
                oov_words.push_back(entry.alpha + " (" + entry.surface + ")");
            }
        }
    }

    printf("\n");
    printf("    Top %d words: %d OOV (max allowed: %d)\n", n, oov_count, MAX_OOV);
    if (!oov_words.empty()) {
        printf("    OOV examples:");
        for (const auto& w : oov_words) printf(" %s", w.c_str());
        printf("\n");
    }

    ASSERT_MSG(oov_count <= MAX_OOV,
        (std::to_string(oov_count) + " of top " + std::to_string(n)
         + " words are OOV (max " + std::to_string(MAX_OOV) + ")").c_str());
    printf("    ");
}

// ============================================================================
// Test: case mismatch diagnostic
// ============================================================================

static void test_case_mismatch_diagnostic() {
    // Count words where the surface form IS in the LM but the alpha form is NOT.
    // This is expected for a case-sensitive LM - not a runtime problem because
    // expansion sends surface forms to the LM, not alpha forms.
    int total = 0;
    int case_mismatch = 0;
    std::vector<std::string> examples;

    for (const auto& entry : *g_entries) {
        if (entry.surface == entry.alpha) continue;  // no case difference
        total++;

        auto surface_lookup = g_lm->lookup_word(entry.surface);
        auto alpha_lookup = g_lm->lookup_word(entry.alpha);

        if (surface_lookup.is_exact && !alpha_lookup.is_exact) {
            case_mismatch++;
            if (examples.size() < 10) {
                examples.push_back(entry.surface + " -> " + entry.alpha);
            }
        }
    }

    printf("\n");
    printf("    Words with case difference: %d\n", total);
    printf("    Surface found, alpha NOT:   %d\n", case_mismatch);
    if (!examples.empty()) {
        printf("    Examples:\n");
        for (const auto& ex : examples) {
            printf("      %s\n", ex.c_str());
        }
    }

    // Informational only - the overall surface OOV test is the hard gate.
    // A case-sensitive LM is expected to have high mismatch here.
    printf("    ");
}

// ============================================================================
// Test: context words case sensitivity
// ============================================================================

static void test_context_words_case_sensitivity() {
    // Test that the OOV case fallback works for context words.
    // Words like "British", "French" only exist capitalized in the LM vocab.
    // "british", "french" are OOV. The fallback should map lowercase OOV
    // context words to their capitalized forms so the LM gets real embeddings
    // instead of OOV zeros.
    //
    // Use context with words whose lowercase forms are OOV in the vocab:
    // "british" -> OOV, but "British" -> found
    // "french" -> OOV, but "French" -> found

    // Verify our test words are actually OOV lowercase / found capitalized
    auto british_lower = g_lm->lookup_word("british", false);
    auto british_cap = g_lm->lookup_word("British", false);
    ASSERT_MSG(!british_lower.is_exact, "\"british\" should be OOV for this test");
    ASSERT_MSG(british_cap.is_exact, "\"British\" should be in vocab");

    auto french_lower = g_lm->lookup_word("french", false);
    auto french_cap = g_lm->lookup_word("French", false);
    ASSERT_MSG(!french_lower.is_exact, "\"french\" should be OOV for this test");
    ASSERT_MSG(french_cap.is_exact, "\"French\" should be in vocab");

    // "the British and French" -> predict next
    // With caps: LM sees "British" and "French" (found -> exact embeddings)
    // With lower: "british" and "french" are OOV, fallback should recover them
    std::vector<std::string> context_caps = {"the", "British", "and", "French"};
    std::vector<std::string> context_lower = {"the", "british", "and", "french"};

    auto results_caps = g_lm->predict_topk(context_caps, 10);
    auto results_lower = g_lm->predict_topk(context_lower, 10);

    ASSERT_MSG(!results_caps.empty(), "predict_topk with caps context returned nothing");
    ASSERT_MSG(!results_lower.empty(), "predict_topk with lower context returned nothing");

    printf("\n");
    printf("    predict_topk(\"the British and French\"):\n");
    for (int i = 0; i < (int)results_caps.size(); i++) {
        printf("      %d. %s: %.4f\n", i + 1,
               results_caps[i].first.c_str(), results_caps[i].second);
    }
    printf("    predict_topk(\"the british and french\"):\n");
    for (int i = 0; i < (int)results_lower.size(); i++) {
        printf("      %d. %s: %.4f\n", i + 1,
               results_lower[i].first.c_str(), results_lower[i].second);
    }

    // With OOV fallback working, both contexts should produce the same results
    // because "british" -> "British" and "french" -> "French" via the fallback.
    // Check that the top-3 words match between caps and lower contexts.
    int matches = 0;
    int n = std::min(3, std::min((int)results_caps.size(), (int)results_lower.size()));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (results_caps[i].first == results_lower[j].first) {
                matches++;
                break;
            }
        }
    }
    ASSERT_MSG(matches >= 2,
        ("only " + std::to_string(matches) + "/3 top words overlap between caps and lower context."
         " OOV case fallback may not be working").c_str());
    printf("    ");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <context_lm.pte> <vocab.combined>\n", argv[0]);
        fprintf(stderr, "\nChecks that trie words are reachable in the context LM vocab.\n");
        return 1;
    }

    const std::string lm_path = argv[1];
    const std::string lm_vocab_path = argv[2];
    const std::string vocab_path = argv[3];

    // Load LM
    fprintf(stderr, "Loading context LM: %s\n", lm_path.c_str());
    fprintf(stderr, "LM Vocab: %s\n", lm_vocab_path.c_str());
    fprintf(stderr, "Ref vocab: %s\n", vocab_path.c_str());
    ContextLMScorer lm;
    if (!lm.load(lm_path, lm_vocab_path, 1)) {
        fprintf(stderr, "Error: Failed to load context LM\n");
        return 1;
    }
    g_lm = &lm;

    fprintf(stderr, "LM loaded: num_exact=%d, num_buckets=%d, embed_dim=%d\n",
            lm.num_exact(), lm.num_buckets(), lm.embed_dim());

    // Parse vocab
    auto entries = parse_combined_vocab(vocab_path);
    if (entries.empty()) {
        fprintf(stderr, "Error: No words parsed from %s\n", vocab_path.c_str());
        return 1;
    }
    g_entries = &entries;
    fprintf(stderr, "Parsed %zu unique trie words from vocab\n", entries.size());

    printf("Running LM vocab coverage tests...\n\n");

    run_test("overall_oov_rate", test_overall_oov_rate);
    run_test("top_freq_words_coverage", test_top_freq_words_coverage);
    run_test("case_mismatch_diagnostic", test_case_mismatch_diagnostic);
    run_test("context_words_case_sensitivity", test_context_words_case_sensitivity);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
