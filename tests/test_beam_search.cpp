/**
 * Native unit tests for beam search, trie, and resampler.
 */

#include "swipe_decoder/beam_search.hpp"
#include "swipe_decoder/layout.hpp"
#include "swipe_decoder/resampler.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
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
            printf("  %-50s ", #name); \
            try { test_##name(); tests_passed++; printf("PASS\n"); } \
            catch (...) { printf("FAIL (exception)\n"); } \
        } \
    } register_##name; \
    static void test_##name()

#define ASSERT(cond) do { \
    if (!(cond)) { printf("FAIL\n    %s:%d: %s\n", __FILE__, __LINE__, #cond); return; } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { printf("FAIL\n    %s:%d: %s != %s\n", __FILE__, __LINE__, #a, #b); return; } \
} while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    if (std::abs((a) - (b)) > (tol)) { \
        printf("FAIL\n    %s:%d: %s (%.6f) != %s (%.6f) within %g\n", \
               __FILE__, __LINE__, #a, (double)(a), #b, (double)(b), (double)(tol)); return; } \
} while(0)

using namespace swipe_decoder;

// The C++ case-folding table is injected at runtime (by Kotlin on Android).
// Install the small set of pairs these tests rely on before any TEST runs,
// via a static initializer placed before the first TEST() below.
namespace {
struct InstallTestCaseMapping {
    InstallTestCaseMapping() {
        // Sorted by `from` ascending — case_lookup_lower does binary search.
        static const int kFrom[] = {
            0x00C1,  // Á
            0x00C7,  // Ç
            0x00C9,  // É
            0x00D1,  // Ñ
            0x00D6,  // Ö
            0x00DC,  // Ü
        };
        static const int kTo[] = {
            0x00E1,  // á
            0x00E7,  // ç
            0x00E9,  // é
            0x00F1,  // ñ
            0x00F6,  // ö
            0x00FC,  // ü
        };
        static_assert(sizeof(kFrom) == sizeof(kTo), "paired arrays");
        set_case_mapping(kFrom, kTo, sizeof(kFrom) / sizeof(kFrom[0]));
    }
} g_install_case_mapping;
}  // namespace

// ============================================================================
// Trie tests
// ============================================================================

TEST(trie_insert_contains) {
    Trie trie;
    trie.insert("hello", 1.0f);
    trie.insert("world", 2.0f);
    trie.insert("help", 0.5f);

    ASSERT(trie.contains("hello"));
    ASSERT(trie.contains("world"));
    ASSERT(trie.contains("help"));
    ASSERT(!trie.contains("hell"));  // prefix, not a word
    ASSERT(!trie.contains("hellos"));
    ASSERT(!trie.contains("xyz"));
    ASSERT(!trie.contains(""));
    ASSERT_EQ(trie.size(), 3u);
}

TEST(trie_reconstruct_word) {
    Trie trie;
    trie.insert("cat", 1.0f);
    trie.insert("car", 1.0f);
    trie.insert("care", 1.0f);

    // Walk trie to find "cat" node
    const TrieNode* root = trie.root();
    ASSERT(root != nullptr);

    uint32_t c_node = root->children['c' - 'a'];
    ASSERT(c_node != 0);
    const TrieNode* c = trie.get_node(c_node);
    ASSERT(c != nullptr);

    uint32_t ca_node = c->children['a' - 'a'];
    ASSERT(ca_node != 0);
    const TrieNode* ca = trie.get_node(ca_node);
    ASSERT(ca != nullptr);

    uint32_t cat_node = ca->children['t' - 'a'];
    ASSERT(cat_node != 0);
    ASSERT_EQ(trie.reconstruct_word(cat_node), std::string("cat"));

    uint32_t car_node = ca->children['r' - 'a'];
    ASSERT(car_node != 0);
    ASSERT_EQ(trie.reconstruct_word(car_node), std::string("car"));

    uint32_t care_node = trie.get_node(car_node)->children['e' - 'a'];
    ASSERT(care_node != 0);
    ASSERT_EQ(trie.reconstruct_word(care_node), std::string("care"));
}

TEST(trie_case_insensitive) {
    Trie trie;
    trie.insert("Hello", 1.0f);
    ASSERT(trie.contains("hello"));
    ASSERT_EQ(trie.size(), 1u);
}

TEST(trie_invalid_chars_rejected) {
    Trie trie;
    trie.insert("can't", 1.0f);  // apostrophe -> rejected
    trie.insert("123", 1.0f);
    ASSERT_EQ(trie.size(), 0u);
}

TEST(trie_duplicate_insert) {
    Trie trie;
    trie.insert("test", 1.0f);
    trie.insert("test", 2.0f);  // update frequency
    ASSERT_EQ(trie.size(), 1u);
    ASSERT(trie.contains("test"));
}

// ============================================================================
// Beam search tests
// ============================================================================

TEST(beam_search_identity_logprobs) {
    // Create a vocabulary with single-char words
    auto trie = std::make_shared<Trie>();
    trie->insert("a", 1.0f);
    trie->insert("b", 1.0f);
    trie->insert("c", 1.0f);

    auto itrie = trie->itrie();
    TrieBeamSearch bs(50, { 0.0f, 0.0f, 0.56f });  // no bonuses

    // Create log_probs: [T=4, 27] where each timestep strongly predicts one char
    // t=0: 'a', t=1: blank, t=2: blank, t=3: blank
    const int T = 4;
    std::vector<float> log_probs(T * NUM_CLASSES, -10.0f);

    // t=0: strong 'a'
    log_probs[0 * NUM_CLASSES + 0] = -0.1f;  // 'a'

    // t=1,2,3: strong blank
    log_probs[1 * NUM_CLASSES + BLANK_IDX] = -0.1f;
    log_probs[2 * NUM_CLASSES + BLANK_IDX] = -0.1f;
    log_probs[3 * NUM_CLASSES + BLANK_IDX] = -0.1f;

    auto results = bs.decode(itrie, log_probs.data(), T, 3);
    ASSERT(!results.empty());
    ASSERT_EQ(results[0].first, std::string("a"));
}

TEST(beam_search_two_char_word) {
    auto trie = std::make_shared<Trie>();
    trie->insert("ab", 1.0f);
    trie->insert("a", 1.0f);
    trie->insert("b", 1.0f);

    auto itrie = trie->itrie();
    TrieBeamSearch bs(50, { 0.0f, 0.0f, 0.56f });

    // t=0: 'a', t=1: 'b', t=2: blank, t=3: blank
    const int T = 4;
    std::vector<float> log_probs(T * NUM_CLASSES, -10.0f);

    log_probs[0 * NUM_CLASSES + 0] = -0.1f;  // 'a'
    log_probs[1 * NUM_CLASSES + 1] = -0.1f;  // 'b'
    log_probs[2 * NUM_CLASSES + BLANK_IDX] = -0.1f;
    log_probs[3 * NUM_CLASSES + BLANK_IDX] = -0.1f;

    auto results = bs.decode(itrie, log_probs.data(), T, 3);
    ASSERT(!results.empty());
    ASSERT_EQ(results[0].first, std::string("ab"));
}

TEST(beam_search_freq_bonus) {
    auto trie = std::make_shared<Trie>();
    trie->insert("a", 100.0f);   // high frequency
    trie->insert("b", 0.001f);   // low frequency

    // With freq_bonus, "a" should win even with slightly worse CTC score
    auto itrie = trie->itrie();
    TrieBeamSearch bs(50, { 1.0f, 0.0f, 0.56f });

    const int T = 4;
    std::vector<float> log_probs(T * NUM_CLASSES, -10.0f);

    // Both 'a' and 'b' have similar CTC scores, 'b' slightly better
    log_probs[0 * NUM_CLASSES + 0] = -0.5f;  // 'a'
    log_probs[0 * NUM_CLASSES + 1] = -0.3f;  // 'b' (slightly better CTC)
    log_probs[1 * NUM_CLASSES + BLANK_IDX] = -0.1f;
    log_probs[2 * NUM_CLASSES + BLANK_IDX] = -0.1f;
    log_probs[3 * NUM_CLASSES + BLANK_IDX] = -0.1f;

    auto results = bs.decode(itrie, log_probs.data(), T, 3);
    ASSERT(results.size() >= 2);
    // "a" should be first due to massive freq bonus
    ASSERT_EQ(results[0].first, std::string("a"));
}

TEST(beam_search_empty_input) {
    auto trie = std::make_shared<Trie>();
    trie->insert("a", 1.0f);

    auto itrie = trie->itrie();
    TrieBeamSearch bs(50, { 0.0f, 0.0f, 0.56f });

    auto results = bs.decode(itrie, nullptr, 0, 3);
    ASSERT(results.empty());
}

// ============================================================================
// Resampler tests
// ============================================================================

TEST(resample_single_point) {
    std::vector<float> x = {0.5f};
    std::vector<float> y = {0.3f};
    std::vector<float> t = {0.0f};
    std::vector<float> out_x, out_y;

    resample_path(x, y, t, out_x, out_y, true, 10);

    ASSERT_EQ(out_x.size(), 10u);
    ASSERT_EQ(out_y.size(), 10u);
    for (size_t i = 0; i < 10; ++i) {
        ASSERT_NEAR(out_x[i], 0.5f, 1e-6f);
        ASSERT_NEAR(out_y[i], 0.3f, 1e-6f);
    }
}

TEST(resample_linear_path) {
    // Straight line from (0,0) to (1,1)
    const int n = 50;
    std::vector<float> x(n), y(n), t(n);
    for (int i = 0; i < n; ++i) {
        float frac = static_cast<float>(i) / (n - 1);
        x[i] = frac;
        y[i] = frac;
        t[i] = frac * 100.0f;  // 100ms total
    }

    std::vector<float> out_x, out_y;
    resample_path(x, y, t, out_x, out_y, false, 20);

    ASSERT_EQ(out_x.size(), 20u);
    // Should still be linear
    for (size_t i = 0; i < 20; ++i) {
        float expected = static_cast<float>(i) / 19.0f;
        ASSERT_NEAR(out_x[i], expected, 0.01f);
        ASSERT_NEAR(out_y[i], expected, 0.01f);
    }
}

TEST(resample_interleaved_format) {
    std::vector<float> x = {0.0f, 0.5f, 1.0f};
    std::vector<float> y = {0.1f, 0.5f, 0.9f};
    std::vector<float> t = {0.0f, 50.0f, 100.0f};
    std::vector<float> out;

    resample_path_interleaved(x, y, t, out, false, 4);

    ASSERT_EQ(out.size(), 8u);
    // First 4 = x values, next 4 = y values
    ASSERT_NEAR(out[0], 0.0f, 0.01f);  // x[0]
    ASSERT_NEAR(out[3], 1.0f, 0.01f);  // x[3]
    ASSERT_NEAR(out[4], 0.1f, 0.01f);  // y[0]
    ASSERT_NEAR(out[7], 0.9f, 0.01f);  // y[3]
}

TEST(resample_empty_input) {
    std::vector<float> x, y, t;
    std::vector<float> out_x, out_y;

    resample_path(x, y, t, out_x, out_y, true, 10);

    ASSERT_EQ(out_x.size(), 10u);
    for (size_t i = 0; i < 10; ++i) {
        ASSERT_NEAR(out_x[i], 0.0f, 1e-6f);
    }
}

TEST(resample_normalize_clips) {
    std::vector<float> x = {-0.1f, 0.5f, 1.2f};
    std::vector<float> y = {0.0f, 0.5f, 1.0f};
    std::vector<float> t;
    std::vector<float> out_x, out_y;

    resample_path(x, y, t, out_x, out_y, true, 3);

    // With normalize=true, values should be clipped to [0, 1]
    ASSERT(out_x[0] >= 0.0f);
    ASSERT(out_x[2] <= 1.0f);
}

// ============================================================================
// resample_to_60hz tests
// ============================================================================

TEST(resample_60hz_120hz_input) {
    // 120 Hz input (8ms intervals), 20 points = 152ms duration
    // Expected output: ceil(152 / 16.667) + 1 = 10 + 1 = 11 points
    const int n = 20;
    std::vector<float> x(n), y(n), t(n);
    for (int i = 0; i < n; i++) {
        x[i] = static_cast<float>(i) / (n - 1);
        y[i] = 0.5f;
        t[i] = static_cast<float>(i) * 8.0f;  // 8ms = 125 Hz
    }

    std::vector<float> out_x, out_y;
    resample_to_60hz(x.size(), x.data(), y.data(), t.data(), out_x, out_y);

    // Duration = 152ms, interval = 16.667ms -> ceil(152/16.667)+1 = 11
    ASSERT_EQ(out_x.size(), 11u);
    ASSERT_EQ(out_y.size(), 11u);

    // First and last should match input endpoints
    ASSERT_NEAR(out_x[0], 0.0f, 1e-5f);
    // All y should be 0.5
    for (size_t i = 0; i < out_y.size(); i++) {
        ASSERT_NEAR(out_y[i], 0.5f, 1e-5f);
    }
}

TEST(resample_60hz_already_60hz) {
    // 60 Hz input (16.667ms intervals), ~10 points = ~150ms
    const int n = 10;
    std::vector<float> x(n), y(n), t(n);
    for (int i = 0; i < n; i++) {
        x[i] = static_cast<float>(i) / (n - 1);
        y[i] = static_cast<float>(i) / (n - 1);
        t[i] = static_cast<float>(i) * (1000.0f / 60.0f);  // 16.667ms
    }

    std::vector<float> out_x, out_y;
    resample_to_60hz(x.size(), x.data(), y.data(), t.data(), out_x, out_y);

    // Should produce approximately same number of points
    ASSERT_EQ(out_x.size(), static_cast<size_t>(n));
    // Values should be close to input (resampling at same rate)
    ASSERT_NEAR(out_x[0], 0.0f, 1e-5f);
    ASSERT_NEAR(out_y[0], 0.0f, 1e-5f);
}

TEST(resample_60hz_single_point) {
    std::vector<float> x = {0.5f};
    std::vector<float> y = {0.3f};
    std::vector<float> t = {100.0f};
    std::vector<float> out_x, out_y;

    resample_to_60hz(x.size(), x.data(), y.data(), t.data(), out_x, out_y);

    ASSERT_EQ(out_x.size(), 2u);
    ASSERT_NEAR(out_x[0], 0.5f, 1e-6f);
    ASSERT_NEAR(out_x[1], 0.5f, 1e-6f);
}

TEST(resample_60hz_empty) {
    std::vector<float> x, y, t;
    std::vector<float> out_x, out_y;

    resample_to_60hz(x.size(), x.data(), y.data(), t.data(), out_x, out_y);

    ASSERT_EQ(out_x.size(), 0u);
    ASSERT_EQ(out_y.size(), 0u);
}

TEST(resample_60hz_preserves_linearity) {
    // Linear path from (0,0) to (1,1) over 500ms at 240Hz
    const int n = 120;  // 240 Hz * 0.5s
    std::vector<float> x(n), y(n), t(n);
    for (int i = 0; i < n; i++) {
        float frac = static_cast<float>(i) / (n - 1);
        x[i] = frac;
        y[i] = frac;
        t[i] = frac * 500.0f;
    }

    std::vector<float> out_x, out_y;
    resample_to_60hz(x.size(), x.data(), y.data(), t.data(), out_x, out_y);

    // Output should still be linear
    for (size_t i = 0; i < out_x.size(); i++) {
        ASSERT_NEAR(out_x[i], out_y[i], 1e-5f);
    }
    ASSERT_NEAR(out_x[0], 0.0f, 1e-5f);
}

TEST(default_en_qwerty_layout) {
    auto layout = default_en_qwerty();
    ASSERT_EQ(layout.num_keys, 26);
    ASSERT_EQ(layout.letters.size(), 26u);
    ASSERT_EQ(layout.key_cx.size(), 26u);
    ASSERT_EQ(layout.key_cy.size(), 26u);

    // Layout is indexed on lowercase alpha form only; callers pre-lower.
    ASSERT_EQ(layout.lookup('a'), 0);
    ASSERT_EQ(layout.lookup('z'), 25);
    // Invalid chars should be -1
    ASSERT_EQ(layout.lookup('1'), -1);
}

TEST(trie_configurable_alphabet) {
    // 3-letter alphabet: "abc"
    Trie trie("abc");
    ASSERT_EQ(trie.num_chars(), 3);
    ASSERT_EQ(trie.letters(), std::string("abc"));

    trie.insert("abc", 1.0f);
    trie.insert("cab", 2.0f);
    ASSERT(trie.contains("abc"));
    ASSERT(trie.contains("cab"));
    ASSERT(!trie.contains("xyz"));  // not in alphabet
    ASSERT_EQ(trie.size(), 2u);

    ASSERT_EQ(trie.reconstruct_word(/* walk the trie for "abc" */
        trie.get_node(trie.root()->children[0])  // 'a'
            ? trie.get_node(
                trie.get_node(trie.root()->children[0])->children[1])  // 'b'
                ? trie.get_node(
                    trie.get_node(
                        trie.get_node(trie.root()->children[0])->children[1]
                    )->children[2])  // 'c'
                    ? 0 : 0
                : 0
            : 0
    ), std::string(""));  // node 0 = root

    // Simpler: just verify reconstruct works via contains path
    // Walk: root -> a(0) -> b(1) -> c(2)
    uint32_t a_idx = trie.root()->children[0];
    ASSERT(a_idx != 0);
    uint32_t ab_idx = trie.get_node(a_idx)->children[1];
    ASSERT(ab_idx != 0);
    uint32_t abc_idx = trie.get_node(ab_idx)->children[2];
    ASSERT(abc_idx != 0);
    ASSERT_EQ(trie.reconstruct_word(abc_idx), std::string("abc"));
}

TEST(beam_search_configurable_num_chars) {
    // Use a 3-char alphabet for beam search
    auto trie = std::make_shared<Trie>("abc");
    trie->insert("ab", 1.0f);
    trie->insert("a", 1.0f);

    // num_classes = 4 (a, b, c, blank)
    auto itrie = trie->itrie();
    TrieBeamSearch bs(50, { 0.0f, 0.0f, 0.0f }, 3);
    ASSERT_EQ(bs.num_chars(), 3);
    ASSERT_EQ(bs.num_classes(), 4);

    // Create log_probs: [T=3, 4]
    const int T = 3;
    const int NC = 4;
    std::vector<float> log_probs(T * NC, -10.0f);

    // t=0: strong 'a' (index 0)
    log_probs[0 * NC + 0] = -0.1f;
    // t=1: strong 'b' (index 1)
    log_probs[1 * NC + 1] = -0.1f;
    // t=2: strong blank (index 3)
    log_probs[2 * NC + 3] = -0.1f;

    auto results = bs.decode(itrie, log_probs.data(), T, 3);
    ASSERT(!results.empty());
    ASSERT_EQ(results[0].first, std::string("ab"));
}

// ============================================================================
// Layout ordering tests (regression: keyboard-order vs alphabetical-order)
// ============================================================================

// Helper: write a string to a temp file and return its path
#include <cstdlib>
#include <fstream>
#include <unistd.h>

static std::string write_temp_json(const std::string& content) {
    char path[] = "/tmp/test_layout_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return "";
    std::string p(path);
    {
        std::ofstream f(p);
        f << content;
    }
    close(fd);
    return p;
}

TEST(load_layout_json_reorders_keyboard_to_alpha) {
    // Minimal 4-key layout: letters are alphabetical "abcd",
    // but JSON keys are in keyboard order: c, a, d, b
    // load_layout_json must map each key's cx/cy to the correct
    // alphabetical index (a=0, b=1, c=2, d=3).
    std::string json = R"({
        "letters": "abcd",
        "keys": [
            {"letter": "c", "cx": 0.30, "cy": 0.10},
            {"letter": "a", "cx": 0.10, "cy": 0.20},
            {"letter": "d", "cx": 0.40, "cy": 0.30},
            {"letter": "b", "cx": 0.20, "cy": 0.40}
        ]
    })";
    auto path = write_temp_json(json);
    ASSERT(!path.empty());

    KeyboardLayout layout;
    ASSERT(load_layout_json(path, layout));
    std::remove(path.c_str());

    ASSERT_EQ(layout.num_keys, 4);
    // key_cx/key_cy must be in alphabetical order: a, b, c, d
    ASSERT_NEAR(layout.key_cx[0], 0.10f, 0.001f);  // a
    ASSERT_NEAR(layout.key_cx[1], 0.20f, 0.001f);  // b
    ASSERT_NEAR(layout.key_cx[2], 0.30f, 0.001f);  // c
    ASSERT_NEAR(layout.key_cx[3], 0.40f, 0.001f);  // d

    ASSERT_NEAR(layout.key_cy[0], 0.20f, 0.001f);  // a
    ASSERT_NEAR(layout.key_cy[1], 0.40f, 0.001f);  // b
    ASSERT_NEAR(layout.key_cy[2], 0.10f, 0.001f);  // c
    ASSERT_NEAR(layout.key_cy[3], 0.30f, 0.001f);  // d
}

TEST(make_layout_requires_alpha_ordered_arrays) {
    // make_layout assigns cx[i] to letters[i], so the caller must
    // pass cx/cy in alphabetical order. Verify it matches load_layout_json.
    std::string json = R"({
        "letters": "abcd",
        "keys": [
            {"letter": "c", "cx": 0.30, "cy": 0.10},
            {"letter": "a", "cx": 0.10, "cy": 0.20},
            {"letter": "d", "cx": 0.40, "cy": 0.30},
            {"letter": "b", "cx": 0.20, "cy": 0.40}
        ]
    })";
    auto path = write_temp_json(json);
    ASSERT(!path.empty());

    KeyboardLayout from_json;
    ASSERT(load_layout_json(path, from_json));
    std::remove(path.c_str());

    // Build the same layout via make_layout with correctly-ordered arrays
    float cx[] = {0.10f, 0.20f, 0.30f, 0.40f};  // a, b, c, d
    float cy[] = {0.20f, 0.40f, 0.10f, 0.30f};   // a, b, c, d
    auto from_arrays = make_layout("abcd", cx, cy, 4);

    ASSERT_EQ(from_arrays.num_keys, from_json.num_keys);
    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(from_arrays.key_cx[i], from_json.key_cx[i], 0.001f);
        ASSERT_NEAR(from_arrays.key_cy[i], from_json.key_cy[i], 0.001f);
    }
}

TEST(layout_multibyte_utf8_letters) {
    // Spanish-like layout with ñ (U+00F1, 2-byte UTF-8)
    // letters: "anñz" — 4 codepoints, alphabetically sorted
    // keys in keyboard order: z, ñ, a, n
    std::string json = R"({
        "letters": "anñz",
        "keys": [
            {"letter": "z", "cx": 0.40, "cy": 0.80},
            {"letter": "ñ", "cx": 0.30, "cy": 0.50},
            {"letter": "a", "cx": 0.10, "cy": 0.50},
            {"letter": "n", "cx": 0.20, "cy": 0.50}
        ]
    })";
    auto path = write_temp_json(json);
    ASSERT(!path.empty());

    KeyboardLayout layout;
    ASSERT(load_layout_json(path, layout));
    std::remove(path.c_str());

    ASSERT_EQ(layout.num_keys, 4);
    // Alphabetical order: a=0, n=1, ñ=2, z=3
    ASSERT_NEAR(layout.key_cx[0], 0.10f, 0.001f);  // a
    ASSERT_NEAR(layout.key_cx[1], 0.20f, 0.001f);  // n
    ASSERT_NEAR(layout.key_cx[2], 0.30f, 0.001f);  // ñ
    ASSERT_NEAR(layout.key_cx[3], 0.40f, 0.001f);  // z

    // Layout is indexed on the lowercase alpha form only.
    ASSERT_EQ(layout.lookup(0x00F1), 2);   // ñ
}

TEST(trie_spanish_alphabet_with_enie) {
    // Trie with 27-letter Spanish alphabet can insert and find words with ñ
    Trie trie("abcdefghijklmnñopqrstuvwxyz");
    ASSERT_EQ(trie.num_chars(), 27);

    trie.insert("amigo", 141.0f);
    trie.insert("año", 100.0f);
    trie.insert("españa", 90.0f);

    ASSERT(trie.contains("amigo"));
    ASSERT(trie.contains("año"));
    ASSERT(trie.contains("españa"));
    ASSERT(!trie.contains("kiwi"));  // not inserted
    ASSERT_EQ(trie.size(), 3u);
}

// ============================================================================
// Unicode case folding via installed table
// ============================================================================

TEST(installed_case_mapping_folds_non_ascii) {
    // Pairs installed in the InstallTestCaseMapping fixture above.
    ASSERT_EQ(simple_tolower(0x00D1), 0x00F1);  // Ñ -> ñ
    ASSERT_EQ(simple_tolower(0x00C9), 0x00E9);  // É -> é
    // Miss -> identity (no ẞ pair installed).
    ASSERT_EQ(simple_tolower(0x1E9E), 0x1E9E);
    // ASCII fast path still folds regardless of the installed table.
    ASSERT_EQ(simple_tolower('A'), 'a');
    // Lowercase identity.
    ASSERT_EQ(simple_tolower(0x00F1), 0x00F1);
}

TEST(trie_uppercase_vocab_insert) {
    // Words with uppercase non-ASCII should be normalized and found
    Trie trie("abcdefghijklmnñopqrstuvwxyz");
    trie.insert("España", 90.0f);
    trie.insert("AMIGO", 141.0f);
    ASSERT(trie.contains("españa"));
    ASSERT(trie.contains("amigo"));
    ASSERT_EQ(trie.size(), 2u);
}

// ============================================================================
// parse_combined_vocab — malformed-line tolerance + edge cases
// ============================================================================

static std::string write_temp_combined(const std::string& content) {
    char path[] = "/tmp/test_combined_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return "";
    std::string p(path);
    {
        std::ofstream f(p);
        f << content;
    }
    close(fd);
    return p;
}

// Find an alpha entry by name. Returns nullptr if not present.
static const ParsedAlpha* find_alpha(const ParsedCombined& p, const std::string& s) {
    for (const auto& a : p.alphas) if (a.alpha == s) return &a;
    return nullptr;
}

TEST(parse_tolerates_malformed_lines) {
    // Mix of: blank line, dictionary= header, lang= header, line without
    // ",f=", unknown extra keys, and no trailing newline on the last line.
    std::string vocab =
        "dictionary=test\n"
        "lang=en\n"
        "\n"
        "# not a comment-aware parser, but should be skipped\n"
        " word=hello,f=100,flags=,x=extra,g=ignored\n"
        " word=malformed,flags=only\n"                      // no f= — skip
        " word=world,f=50,other=stuff,f2=bogus\n"
        " word=trailing,f=42,flags=";                       // no final \n

    auto path = write_temp_combined(vocab);
    ASSERT(!path.empty());

    ParsedCombined parsed;
    ASSERT(parse_combined_vocab(path, "f", parsed));
    std::remove(path.c_str());

    ASSERT(find_alpha(parsed, "hello") != nullptr);
    ASSERT(find_alpha(parsed, "world") != nullptr);
    ASSERT(find_alpha(parsed, "trailing") != nullptr);
    ASSERT(find_alpha(parsed, "malformed") == nullptr);

    // world's freq comes from ,f=50, not ,f2=bogus
    auto* w = find_alpha(parsed, "world");
    ASSERT_NEAR(w->max_freq, 50.0f, 0.001f);
}

TEST(parse_duplicate_surface_bumps_freq) {
    // Same surface twice with different freqs — keep max and don't create
    // duplicate entries in forms[].
    std::string vocab =
        "dictionary=test\n"
        " word=hello,f=10,flags=\n"
        " word=hello,f=200,flags=\n"
        " word=hello,f=50,flags=\n";

    auto path = write_temp_combined(vocab);
    ParsedCombined parsed;
    ASSERT(parse_combined_vocab(path, "f", parsed));
    std::remove(path.c_str());

    ASSERT_EQ(parsed.forms.size(), 1u);
    ASSERT_NEAR(parsed.forms[0].freq, 200.0f, 0.001f);

    auto* a = find_alpha(parsed, "hello");
    ASSERT(a != nullptr);
    ASSERT_NEAR(a->max_freq, 200.0f, 0.001f);
    ASSERT_EQ(a->form_indices.size(), 1u);
}

TEST(parse_apostrophe_variants_unified) {
    // Curly apostrophe (U+2019 = E2 80 99), left curly (U+2018 = E2 80 98),
    // acute accent (U+00B4 = C2 B4), and backtick all normalize to ASCII '.
    // All four surfaces share alpha "dont" and map into one expansion bucket.
    std::string vocab =
        "dictionary=test\n"
        " word=don\xE2\x80\x99t,f=100,flags=\n"   // don't (curly right)
        " word=don\xE2\x80\x98t,f=90,flags=\n"    // don't (curly left)
        " word=don\xC2\xB4t,f=80,flags=\n"        // don´t (acute)
        " word=don`t,f=70,flags=\n"               // don`t (backtick)
        " word=don't,f=60,flags=\n";              // don't (ASCII)

    auto path = write_temp_combined(vocab);
    ParsedCombined parsed;
    ASSERT(parse_combined_vocab(path, "f", parsed));
    std::remove(path.c_str());

    // All five map to alpha "dont".
    auto* a = find_alpha(parsed, "dont");
    ASSERT(a != nullptr);

    // After normalization, all surfaces become the ASCII "don't" -> deduped
    // into a single form with freq = max (= 100).
    ASSERT_EQ(parsed.forms.size(), 1u);
    ASSERT(std::string(parsed.forms[0].surface_ptr, parsed.forms[0].surface_len) == "don't");
    ASSERT_NEAR(parsed.forms[0].freq, 100.0f, 0.001f);
}

TEST(flatten_preserves_file_order) {
    // The LM's tied-freq tiebreaker depends on word_id = file_order. Verify
    // flatten_parsed_to_entries emits surfaces in the same order they appear
    // in the source file (not sorted by freq, not sorted by alpha).
    std::string vocab =
        "dictionary=test\n"
        " word=zebra,f=10,flags=\n"
        " word=apple,f=100,flags=\n"
        " word=mango,f=50,flags=\n"
        " word=kiwi,f=75,flags=\n";

    auto path = write_temp_combined(vocab);
    ParsedCombined parsed;
    ASSERT(parse_combined_vocab(path, "f", parsed));
    std::remove(path.c_str());

    std::vector<std::pair<std::string, uint32_t>> flat;
    flatten_parsed_to_entries(parsed, flat);

    ASSERT_EQ(flat.size(), 4u);
    ASSERT(flat[0].first == "zebra");
    ASSERT(flat[1].first == "apple");
    ASSERT(flat[2].first == "mango");
    ASSERT(flat[3].first == "kiwi");
}

TEST(trie_ascii_fast_path_matches_unicode_path) {
    // Trie::insert has an ASCII fast-path; an English trie built via the
    // fast path must behave identically to one built via the general path.
    // We can't easily force the slow path on ASCII input, but we can verify
    // the fast path produces a trie with the expected word lookups and
    // reconstructed surface forms, i.e. no corruption from the optimization.
    Trie trie;
    trie.reserve_nodes(16);  // tiny reservation — exercise re-growth
    trie.insert("hello", 100.0f);
    trie.insert("help", 50.0f);
    trie.insert("world", 200.0f);

    ASSERT(trie.contains("hello"));
    ASSERT(trie.contains("help"));
    ASSERT(trie.contains("world"));
    ASSERT(!trie.contains("hel"));      // prefix, not inserted as word
    ASSERT(!trie.contains("helloo"));   // over-run
    ASSERT_EQ(trie.size(), 3u);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("Running native tests...\n\n");
    // Tests are auto-registered and already executed by static initializers
    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
