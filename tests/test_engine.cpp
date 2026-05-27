/**
 * Integration tests for SwipeEngine - universal-swipe architecture.
 *
 * Covers the four shipping inference modes:
 *   1. encoder-only
 *   2. encoder + paired decoder           (guarded: runs iff a decoder .pte exists)
 *   3. encoder + context LM
 *   4. encoder + paired decoder + LM      (guarded: runs iff a decoder .pte exists)
 *
 * Also tests negative paths (no model files needed).
 *
 * Usage:
 *   test_engine                                        # negative paths only
 *   test_engine <encoder.pte> <vocab> <layout.json>    # + encoder-only mode
 *       [alt_layout.json]                              # + runtime layout swap
 *       [lm_model.pte] [lm_vocab]                      # + encoder+LM mode
 *       [decoder.pte]                                  # + encoder+decoder modes
 *       [mismatched_layout.json]                       # + setMode rejection test
 *                                                      #   (layout whose num_keys
 *                                                      #   != decoder.num_chars)
 */

#include "swipe_decoder/engine.hpp"
#include "swipe_decoder/trie.hpp"
#include "swipe_decoder/layout.hpp"
#include "swipe_decoder/model_metadata.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace swipe_decoder;

// ============================================================================
// Minimal test framework
// ============================================================================

static int tests_run = 0;
static int tests_passed = 0;
static int tests_skipped = 0;

struct TestFailed {};

#define ASSERT(cond) do { \
    if (!(cond)) { printf("FAIL\n    %s:%d: %s\n", __FILE__, __LINE__, #cond); throw TestFailed{}; } \
} while(0)

#define ASSERT_MSG(cond, msg) do { \
    if (!(cond)) { printf("FAIL\n    %s:%d: %s (%s)\n", __FILE__, __LINE__, #cond, msg); throw TestFailed{}; } \
} while(0)

static void run_test(const char* name, void (*func)()) {
    tests_run++;
    printf("  %-55s ", name);
    try {
        func();
        tests_passed++;
        printf("PASS\n");
    } catch (TestFailed&) {
    } catch (const std::exception& e) {
        printf("FAIL (exception: %s)\n", e.what());
    } catch (...) {
        printf("FAIL (unknown exception)\n");
    }
}

static void skip_test(const char* name, const char* reason) {
    tests_skipped++;
    printf("  %-55s SKIP (%s)\n", name, reason);
}

// ============================================================================
// Globals set from command line
// ============================================================================

static std::string g_encoder_path;

static std::string g_vocab_path;
static Trie g_vocab;

static std::string g_layout_path;
static KeyboardLayout g_layout;

static std::string g_alt_layout_path;
static KeyboardLayout g_alt_layout;

static std::string g_mismatched_layout_path;  // layout whose num_keys != decoder num_chars
static KeyboardLayout g_mismatched_layout;

static std::string g_lm_model_path;
static std::string g_lm_vocab_path;
static std::string g_decoder_path;

static bool approx_eq(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol;
}

// Small helpers that keep setMode test calls readable.
static SwipeEngine::Mode mode_layout(const KeyboardLayout& layout, const std::string& vocab_path) {
    SwipeEngine::Mode m;
    m.layout = layout;
    //m.vocab_path = vocab_path;
    return m;
}

static SwipeEngine::Mode mode_decoder(const std::string& decoder_path,
                                       const KeyboardLayout& layout,
                                       const std::string& vocab_path) {
    SwipeEngine::Mode m;
    m.layout = layout;
    //m.vocab_path = vocab_path;
    m.decoder_path = decoder_path;  // "" = unload
    return m;
}

static void make_swipe(std::vector<float>& x, std::vector<float>& y, std::vector<float>& t) {
    const int n = 72;
    x.resize(n);
    y.resize(n);
    t.resize(n);

    const float x0 = 0.12f, y0 = 0.72f;
    const float xm = 0.48f, ym = 0.30f;
    const float x1 = 0.86f, y1 = 0.66f;

    for (int i = 0; i < n; ++i) {
        float u = static_cast<float>(i) / static_cast<float>(n - 1);
        float v = 1.0f - u;
        x[i] = v * v * x0 + 2.0f * v * u * xm + u * u * x1;
        y[i] = v * v * y0 + 2.0f * v * u * ym + u * u * y1;
        t[i] = static_cast<float>(i) * 8.0f;
    }
}

// ============================================================================
// Negative-path tests (no model files required)
// ============================================================================

static void test_engine_not_ready() {
    SwipeEngine engine;
    ASSERT(!engine.is_ready());
    ASSERT(!engine.hasDecoder());

    float x[] = {0.5f};
    float y[] = {0.5f};
    float t[] = {0.0f};
    auto results = engine.recognize(x, y, t, 1);
    ASSERT(results.empty());
}

static void test_predict_next_no_lm() {
    SwipeEngine engine;
    std::vector<std::string> context = {"the"};
    auto results = engine.predict_next(context, 10);
    ASSERT(results.empty());
}

static void test_init_bad_encoder_path() {
    SwipeEngine engine;
    SwipeEngine::Config config;
    config.encoder_path = "/nonexistent/encoder.pte";
    ASSERT(!engine.init(config));
    ASSERT(!engine.is_ready());
}

static void test_recognize_after_failed_init() {
    SwipeEngine engine;
    SwipeEngine::Config config;
    config.encoder_path = "/nonexistent/encoder.pte";
    engine.init(config);

    float x[] = {0.1f, 0.5f, 0.9f};
    float y[] = {0.5f, 0.5f, 0.5f};
    float t[] = {0.0f, 16.67f, 33.33f};
    auto results = engine.recognize(x, y, t, 3);
    ASSERT(results.empty());
}

// ============================================================================
// Encoder-only mode
// ============================================================================

static void test_encoder_only_init_and_recognize() {
    SwipeEngine engine;
    SwipeEngine::Config cfg;
    cfg.encoder_path = g_encoder_path;
    cfg.dictionaries = { g_vocab.itrie() };
    cfg.layout = g_layout;
    cfg.num_threads = 1;
    //cfg.use_expansion = true;

    ASSERT_MSG(engine.init(cfg), "engine.init failed");
    ASSERT_MSG(engine.is_ready(), "engine not ready");
    ASSERT_MSG(!engine.hasDecoder(), "unexpected decoder loaded");

    std::vector<float> x, y, t;
    make_swipe(x, y, t);
    auto out = engine.recognize(x.data(), y.data(), t.data(), static_cast<int>(x.size()));
    ASSERT_MSG(!out.empty(), "recognize returned empty");
    ASSERT_MSG(!out[0].word.empty(), "top result has empty word");

    const auto& timing = engine.last_timing();
    ASSERT_MSG(timing.encoder_us > 0, "encoder timing not recorded");
    ASSERT_MSG(timing.decoder_us == 0, "decoder timing should be zero in encoder-only mode");
    ASSERT_MSG(timing.beam_us > 0, "beam timing not recorded");
}

static void test_encoder_only_runtime_setlayout() {
    SwipeEngine engine;
    SwipeEngine::Config cfg;
    cfg.encoder_path = g_encoder_path;
    cfg.dictionaries = { g_vocab.itrie() };
    cfg.layout = g_layout;
    cfg.num_threads = 1;
    //cfg.use_expansion = true;

    ASSERT_MSG(engine.init(cfg), "engine.init failed");

    KeyboardLayout alt;
    ASSERT_MSG(load_layout_json(g_alt_layout_path, alt), "failed loading alt layout");
    ASSERT_MSG(engine.setMode(mode_layout(alt, g_vocab_path)), "setMode(layout) failed");

    std::vector<float> x, y, t;
    make_swipe(x, y, t);
    auto out = engine.recognize(x.data(), y.data(), t.data(), static_cast<int>(x.size()));
    ASSERT_MSG(!out.empty(), "recognize after setMode(layout) returned empty");
}

static void test_setlayout_json_vs_array_equivalence() {
    // Regression test: layout built via load_layout_json (C++ path) must match
    // layout built via make_layout (JNI path) when arrays are in alphabetical order.
    SwipeEngine engine;
    SwipeEngine::Config cfg;
    cfg.encoder_path = g_encoder_path;
    cfg.dictionaries = { g_vocab.itrie() };
    cfg.layout = g_layout;
    cfg.num_threads = 1;
    //cfg.use_expansion = true;

    ASSERT_MSG(engine.init(cfg), "engine.init failed");

    KeyboardLayout alt_json;
    ASSERT_MSG(load_layout_json(g_alt_layout_path, alt_json), "failed loading alt layout JSON");
    ASSERT_MSG(engine.setMode(mode_layout(alt_json, g_vocab_path)), "setMode(layout json) failed");

    std::vector<float> x, y, t;
    make_swipe(x, y, t);
    auto out_json = engine.recognize(x.data(), y.data(), t.data(),
                                      static_cast<int>(x.size()));
    ASSERT_MSG(!out_json.empty(), "recognize after json layout returned empty");

    SwipeEngine engine2;
    ASSERT_MSG(engine2.init(cfg), "engine2.init failed");
    KeyboardLayout alt_array = make_layout(alt_json.letters,
                                            alt_json.key_cx.data(),
                                            alt_json.key_cy.data(),
                                            alt_json.num_keys);
    ASSERT_MSG(engine2.setMode(mode_layout(alt_array, g_vocab_path)),
               "setMode(layout arrays) failed");

    auto out_array = engine2.recognize(x.data(), y.data(), t.data(),
                                        static_cast<int>(x.size()));
    ASSERT_MSG(!out_array.empty(), "recognize after array layout returned empty");

    ASSERT_MSG(out_json[0].word == out_array[0].word,
               (std::string("top word mismatch: json='") + out_json[0].word +
                "' array='" + out_array[0].word + "'").c_str());
}

// ============================================================================
// Encoder + context LM
// ============================================================================

static void test_encoder_plus_lm() {
    SwipeEngine engine;
    SwipeEngine::Config cfg;
    cfg.encoder_path = g_encoder_path;
    cfg.dictionaries = { g_vocab.itrie() };
    cfg.layout = g_layout;
    cfg.num_threads = 1;
    //cfg.use_expansion = true;
    cfg.lm_model_path = g_lm_model_path;
    cfg.lm_vocab_path = g_lm_vocab_path.empty() ? g_vocab_path : g_lm_vocab_path;
    cfg.alpha = 0.3f;

    ASSERT_MSG(engine.init(cfg), "engine.init with LM failed");
    ASSERT_MSG(!engine.hasDecoder(), "unexpected decoder in encoder+LM mode");

    std::vector<std::string> ctx = {"the", "quick", "brown"};
    auto preds = engine.predict_next(ctx, 8);
    ASSERT_MSG(!preds.empty(), "predict_next returned empty");
    ASSERT_MSG(!preds[0].first.empty(), "top prediction empty");

    std::vector<float> x, y, t;
    make_swipe(x, y, t);
    auto out = engine.recognize(x.data(), y.data(), t.data(),
                                 static_cast<int>(x.size()), ctx, 4);
    ASSERT_MSG(!out.empty(), "recognize with LM context returned empty");

    const auto& timing = engine.last_timing();
    ASSERT_MSG(timing.lm_us > 0, "lm timing not recorded");
}

// Context-aware reranking must actually move scores around. Run the same
// swipe twice through the same engine with different context words and
// assert at least one candidate's final_score changed or the top-1 flipped.
// Without this, LM reranking could silently degrade to a no-op.
static void test_lm_reranking_context_dependent() {
    SwipeEngine engine;
    SwipeEngine::Config cfg;
    cfg.encoder_path = g_encoder_path;
    cfg.dictionaries = { g_vocab.itrie() };
    cfg.layout = g_layout;
    cfg.lm_model_path = g_lm_model_path;
    cfg.lm_vocab_path = g_lm_vocab_path.empty() ? g_vocab_path : g_lm_vocab_path;
    cfg.alpha = 0.5f;
    cfg.top_k = 10;
    ASSERT_MSG(engine.init(cfg), "engine.init with LM failed");

    std::vector<float> x, y, t;
    make_swipe(x, y, t);

    std::vector<std::string> ctx_a = {"we", "are"};
    std::vector<std::string> ctx_b = {"he", "says"};

    auto out_a = engine.recognize(x.data(), y.data(), t.data(),
                                   static_cast<int>(x.size()), ctx_a, 10);
    auto out_b = engine.recognize(x.data(), y.data(), t.data(),
                                   static_cast<int>(x.size()), ctx_b, 10);
    ASSERT_MSG(!out_a.empty() && !out_b.empty(), "empty reranked output");

    // LM scores must actually be populated (would be 0.0 if no rerank happened).
    bool any_lm_score_nonzero = false;
    for (const auto& r : out_a) {
        if (r.lm_score != 0.0f) { any_lm_score_nonzero = true; break; }
    }
    ASSERT_MSG(any_lm_score_nonzero, "lm_score never populated - LM rerank is a no-op");

    // Different contexts must produce different rerank outcomes (score or order).
    bool diverged = (out_a.size() != out_b.size());
    if (!diverged) {
        for (size_t i = 0; i < out_a.size(); ++i) {
            if (out_a[i].word != out_b[i].word
                || !approx_eq(out_a[i].final_score, out_b[i].final_score, 1e-4f)) {
                diverged = true;
                break;
            }
        }
    }
    ASSERT_MSG(diverged, "LM rerank produced identical scores across different contexts");
}

// lm_alpha=0 must be a true no-op: results must match an engine with no LM
// loaded at all (same order, same final_scores, lm_score left at 0).
static void test_lm_alpha_zero_is_no_op() {
    SwipeEngine::Config base_cfg;
    base_cfg.encoder_path = g_encoder_path;
    base_cfg.dictionaries = { g_vocab.itrie() };
    base_cfg.layout = g_layout;
    base_cfg.top_k = 8;

    SwipeEngine no_lm;
    ASSERT_MSG(no_lm.init(base_cfg), "no-LM init failed");

    SwipeEngine::Config lm_cfg = base_cfg;
    lm_cfg.lm_model_path = g_lm_model_path;
    lm_cfg.lm_vocab_path = g_lm_vocab_path.empty() ? g_vocab_path : g_lm_vocab_path;
    lm_cfg.alpha = 0.0f;  // explicit no-op
    SwipeEngine with_lm;
    ASSERT_MSG(with_lm.init(lm_cfg), "LM init failed");

    std::vector<float> x, y, t;
    make_swipe(x, y, t);
    std::vector<std::string> ctx = {"the", "quick"};

    auto a = no_lm  .recognize(x.data(), y.data(), t.data(), static_cast<int>(x.size()));
    auto b = with_lm.recognize(x.data(), y.data(), t.data(), static_cast<int>(x.size()), ctx, 8);
    ASSERT_MSG(!a.empty() && !b.empty(), "empty recognize");
    ASSERT_MSG(a.size() == b.size(), "result sizes diverged at alpha=0");

    for (size_t i = 0; i < a.size(); ++i) {
        ASSERT_MSG(a[i].word == b[i].word,
                   (std::string("alpha=0 reordered results @") + std::to_string(i)).c_str());
        ASSERT_MSG(approx_eq(a[i].final_score, b[i].final_score, 1e-4f),
                   "alpha=0 shifted final_score");
        ASSERT_MSG(b[i].lm_score == 0.0f, "alpha=0 populated lm_score");
    }
}

// ============================================================================
// Encoder + paired decoder (guarded)
// ============================================================================

static void test_encoder_plus_decoder() {
    SwipeEngine engine;
    SwipeEngine::Config cfg;
    cfg.encoder_path = g_encoder_path;
    cfg.decoder_path = g_decoder_path;
    cfg.dictionaries = { g_vocab.itrie() };
    cfg.layout = g_layout;
    cfg.num_threads = 1;
    //cfg.use_expansion = true;

    ASSERT_MSG(engine.init(cfg), "engine.init with decoder failed");
    ASSERT_MSG(engine.hasDecoder(), "decoder should be loaded");

    std::vector<float> x, y, t;
    make_swipe(x, y, t);
    auto out = engine.recognize(x.data(), y.data(), t.data(), static_cast<int>(x.size()));
    ASSERT_MSG(!out.empty(), "recognize with decoder returned empty");

    const auto& timing = engine.last_timing();
    ASSERT_MSG(timing.decoder_us > 0, "decoder timing not recorded");
}

static void test_decoder_runtime_swap() {
    // Start encoder-only, swap decoder in, recognize, swap it back out.
    SwipeEngine engine;
    SwipeEngine::Config cfg;
    cfg.encoder_path = g_encoder_path;
    cfg.dictionaries = { g_vocab.itrie() };
    cfg.layout = g_layout;
    cfg.num_threads = 1;
    //cfg.use_expansion = true;

    ASSERT_MSG(engine.init(cfg), "engine.init failed");
    ASSERT_MSG(!engine.hasDecoder(), "unexpected decoder after encoder-only init");

    KeyboardLayout layout;
    ASSERT_MSG(load_layout_json(g_layout_path, layout), "failed loading layout");

    ASSERT_MSG(engine.setMode(mode_decoder(g_decoder_path, layout, g_vocab_path)),
               "setMode(decoder) failed");
    ASSERT_MSG(engine.hasDecoder(), "decoder not loaded after setMode");

    std::vector<float> x, y, t;
    make_swipe(x, y, t);
    auto out_with = engine.recognize(x.data(), y.data(), t.data(), static_cast<int>(x.size()));
    ASSERT_MSG(!out_with.empty(), "recognize after setMode(decoder) returned empty");

    ASSERT_MSG(engine.setMode(mode_decoder("", layout, g_vocab_path)),
               "setMode(\"\") decoder unload failed");
    ASSERT_MSG(!engine.hasDecoder(), "decoder should be unloaded");

    auto out_without = engine.recognize(x.data(), y.data(), t.data(), static_cast<int>(x.size()));
    ASSERT_MSG(!out_without.empty(), "recognize after decoder unload returned empty");
}

static void test_encoder_decoder_plus_lm() {
    SwipeEngine engine;
    SwipeEngine::Config cfg;
    cfg.encoder_path = g_encoder_path;
    cfg.decoder_path = g_decoder_path;
    cfg.dictionaries = { g_vocab.itrie() };
    cfg.layout = g_layout;
    cfg.num_threads = 1;
    //cfg.use_expansion = true;
    cfg.lm_model_path = g_lm_model_path;
    cfg.lm_vocab_path = g_lm_vocab_path.empty() ? g_vocab_path : g_lm_vocab_path;
    cfg.alpha = 0.3f;

    ASSERT_MSG(engine.init(cfg), "engine.init failed");
    ASSERT_MSG(engine.hasDecoder(), "decoder should be loaded");

    std::vector<std::string> ctx = {"the", "quick", "brown"};
    std::vector<float> x, y, t;
    make_swipe(x, y, t);
    auto out = engine.recognize(x.data(), y.data(), t.data(),
                                 static_cast<int>(x.size()), ctx, 4);
    ASSERT_MSG(!out.empty(), "recognize returned empty");

    const auto& timing = engine.last_timing();
    ASSERT_MSG(timing.decoder_us > 0, "decoder timing missing");
    ASSERT_MSG(timing.lm_us > 0, "lm timing missing");
}

// Engine should modify scoring when the decoder is loaded. Regression guard:
// an earlier iteration of the decoder setter ignored the decoder's scoring block.
static void test_scoring_defaults_from_decoder_metadata() {
    DecoderMetadata dm;
    ASSERT_MSG(load_decoder_metadata(metadata_path_for(g_decoder_path), dm),
               "failed to load decoder metadata.json");

    float g0, l0, b0, g1, l1, b1;
    {
        SwipeEngine engine;
        SwipeEngine::Config cfg;
        cfg.encoder_path = g_encoder_path;
        cfg.dictionaries = { g_vocab.itrie() };
        cfg.layout = g_layout;
        ASSERT_MSG(engine.init(cfg), "engine.init failed");

        auto& bs = engine.beam_search();

        g0 = bs.length_norm_exp();
        l0 = bs.freq_bonus();
        b0 = bs.length_bonus();
    }

    {
        SwipeEngine engine;
        SwipeEngine::Config cfg;
        cfg.encoder_path = g_encoder_path;
        cfg.decoder_path = g_decoder_path;
        cfg.dictionaries = { g_vocab.itrie() };
        cfg.layout = g_layout;
        ASSERT_MSG(engine.init(cfg), "engine.init failed");

        auto& bs = engine.beam_search();

        g1 = bs.length_norm_exp();
        l1 = bs.freq_bonus();
        b1 = bs.length_bonus();
    }

    bool all_equal = approx_eq(g0, g1) && approx_eq(l0, l1) && approx_eq(b0, b1);
    ASSERT_MSG(!all_equal, "decoder and non-decoder scoring parameters are identical");
}

// cfg.gamma/lambda_/beta must take effect on the active beam search.
static void test_scoring_overrides_applied() {
    const float override_gamma  = 0.9f;
    const float override_lambda = 0.123f;
    const float override_beta   = 2.75f;

    SwipeEngine engine;
    SwipeEngine::Config cfg;
    cfg.encoder_path = g_encoder_path;
    cfg.decoder_path = g_decoder_path;
    cfg.dictionaries = { g_vocab.itrie() };
    cfg.layout = g_layout;
    cfg.gamma   = override_gamma;
    cfg.lambda_ = override_lambda;
    cfg.beta    = override_beta;

    ASSERT_MSG(engine.init(cfg), "engine.init with overrides failed");

    auto& bs = engine.beam_search();
    ASSERT_MSG(approx_eq(bs.length_norm_exp(), override_gamma),  "gamma override lost");
    ASSERT_MSG(approx_eq(bs.freq_bonus(),      override_lambda), "lambda override lost");
    ASSERT_MSG(approx_eq(bs.length_bonus(),    override_beta),   "beta override lost");
}

// Round-trip layout swap (en -> fr -> en) must return same results as the
// original encoder-only (en) recognize.
static void test_layout_swap_roundtrip() {
    SwipeEngine a;
    SwipeEngine::Config cfg;
    cfg.encoder_path = g_encoder_path;
    cfg.dictionaries = { g_vocab.itrie() };
    cfg.layout = g_layout;
    ASSERT_MSG(a.init(cfg), "engine A init failed");

    std::vector<float> x, y, t;
    make_swipe(x, y, t);
    auto baseline = a.recognize(x.data(), y.data(), t.data(), static_cast<int>(x.size()));
    ASSERT_MSG(!baseline.empty(), "baseline recognize empty");

    KeyboardLayout en, fr;
    ASSERT_MSG(load_layout_json(g_layout_path,     en), "load en layout failed");
    ASSERT_MSG(load_layout_json(g_alt_layout_path, fr), "load fr layout failed");

    ASSERT_MSG(a.setMode(mode_layout(fr, g_vocab_path)), "setMode(fr) failed");
    (void)a.recognize(x.data(), y.data(), t.data(), static_cast<int>(x.size()));
    ASSERT_MSG(a.setMode(mode_layout(en, g_vocab_path)), "setMode(en) failed");

    auto after = a.recognize(x.data(), y.data(), t.data(), static_cast<int>(x.size()));
    ASSERT_MSG(after.size() == baseline.size(),
               "result size changed across round-trip");
    for (size_t i = 0; i < baseline.size(); ++i) {
        ASSERT_MSG(after[i].word == baseline[i].word,
                   (std::string("word @") + std::to_string(i) + " changed: '"
                    + baseline[i].word + "' -> '" + after[i].word + "'").c_str());
        ASSERT_MSG(approx_eq(after[i].final_score, baseline[i].final_score, 1e-4f),
                   "final_score diverged across layout round-trip");
    }
}

// Loading a decoder mid-session must materially change the beam-search output
// vs. encoder-only for the same swipe (different final_scores at minimum -
// decoder reshapes log-probs and decoder-metadata scoring takes over).
static void test_decoder_swap_changes_output() {
    SwipeEngine base;
    SwipeEngine::Config cfg;
    cfg.encoder_path = g_encoder_path;
    cfg.dictionaries = { g_vocab.itrie() };
    cfg.layout = g_layout;
    ASSERT_MSG(base.init(cfg), "base init failed");

    std::vector<float> x, y, t;
    make_swipe(x, y, t);
    auto enc_only = base.recognize(x.data(), y.data(), t.data(), static_cast<int>(x.size()));
    ASSERT_MSG(!enc_only.empty(), "encoder-only recognize empty");

    KeyboardLayout layout;
    ASSERT_MSG(load_layout_json(g_layout_path, layout), "load layout failed");
    ASSERT_MSG(base.setMode(mode_decoder(g_decoder_path, layout, g_vocab_path)),
               "setMode(decoder) failed");
    ASSERT_MSG(base.hasDecoder(), "decoder not loaded");

    auto with_dec = base.recognize(x.data(), y.data(), t.data(), static_cast<int>(x.size()));
    ASSERT_MSG(!with_dec.empty(), "decoder recognize empty");

    bool changed = (with_dec.size() != enc_only.size());
    if (!changed) {
        for (size_t i = 0; i < enc_only.size(); ++i) {
            if (with_dec[i].word != enc_only[i].word
                || !approx_eq(with_dec[i].final_score, enc_only[i].final_score, 1e-3f)) {
                changed = true;
                break;
            }
        }
    }
    ASSERT_MSG(changed, "loading a decoder didn't change any output");
}

// Unloading the decoder must restore encoder-only behavior exactly - proves
// the decoder swap didn't mutate encoder buffers or trie state.
static void test_decoder_swap_does_not_corrupt_encoder() {
    SwipeEngine ref;
    SwipeEngine::Config cfg;
    cfg.encoder_path = g_encoder_path;
    cfg.dictionaries = { g_vocab.itrie() };
    cfg.layout = g_layout;
    ASSERT_MSG(ref.init(cfg), "reference init failed");

    std::vector<float> x, y, t;
    make_swipe(x, y, t);
    auto ref_out = ref.recognize(x.data(), y.data(), t.data(), static_cast<int>(x.size()));
    ASSERT_MSG(!ref_out.empty(), "reference recognize empty");

    SwipeEngine swap;
    ASSERT_MSG(swap.init(cfg), "swap engine init failed");

    KeyboardLayout layout;
    ASSERT_MSG(load_layout_json(g_layout_path, layout), "load layout failed");
    ASSERT_MSG(swap.setMode(mode_decoder(g_decoder_path, layout, g_vocab_path)), "setMode(decoder) failed");
    (void)swap.recognize(x.data(), y.data(), t.data(), static_cast<int>(x.size()));
    ASSERT_MSG(swap.setMode(mode_decoder("", layout, g_vocab_path)), "setMode decoder unload failed");
    ASSERT_MSG(!swap.hasDecoder(), "decoder still loaded after unload");

    auto after = swap.recognize(x.data(), y.data(), t.data(), static_cast<int>(x.size()));
    ASSERT_MSG(after.size() == ref_out.size(),
               "result size mismatch after decoder unload");
    for (size_t i = 0; i < ref_out.size(); ++i) {
        ASSERT_MSG(after[i].word == ref_out[i].word,
                   (std::string("word @") + std::to_string(i)
                    + " drift after decoder swap: '" + ref_out[i].word
                    + "' -> '" + after[i].word + "'").c_str());
        ASSERT_MSG(approx_eq(after[i].final_score, ref_out[i].final_score, 1e-4f),
                   "final_score drift after decoder swap");
    }
}

// ============================================================================
// Cross-layout coverage - the universal encoder must handle every shipping
// layout without crashing. Runs one bezier swipe per layout and verifies the
// encoder produced finite emissions + beam search returned cleanly.
// ============================================================================

static void test_cross_layout_coverage() {
    static const char* kLayouts[] = {
        "models/layouts/en_qwerty.json",
        "models/layouts/fr_azerty.json",
        "models/layouts/de_qwertz.json",
        "models/layouts/es_spanish.json",
        "models/layouts/ru_east_slavic.json",
        "models/layouts/lv_latvian_qwerty.json",
    };

    std::vector<float> x, y, t;
    make_swipe(x, y, t);

    for (const char* path : kLayouts) {
        SwipeEngine engine;
        SwipeEngine::Config cfg;
        cfg.encoder_path = g_encoder_path;
        cfg.dictionaries = { g_vocab.itrie() };  // en vocab; non-en layouts yield empty trie

        KeyboardLayout layout;
        ASSERT_MSG(load_layout_json(path, layout), "couldn't load a layout");
        cfg.layout = layout;
        cfg.num_threads = 1;

        ASSERT_MSG(engine.init(cfg),
                   (std::string("init failed for layout ") + path).c_str());

        (void)engine.recognize(x.data(), y.data(), t.data(), static_cast<int>(x.size()));

        const auto& tm = engine.last_timing();
        ASSERT_MSG(tm.encoder_us > 0,
                   (std::string("encoder didn't run for layout ") + path).c_str());
        ASSERT_MSG(std::isfinite(tm.encoder_us),
                   (std::string("non-finite encoder timing for ") + path).c_str());
    }
}

// setMode with a layout whose key count doesn't match the decoder's num_chars
// must reject and leave the engine untouched (still in encoder-only mode with
// the original layout).
static void test_setdecoder_layout_mismatch_rejection() {
    SwipeEngine engine;
    SwipeEngine::Config cfg;
    cfg.encoder_path = g_encoder_path;
    cfg.dictionaries = { g_vocab.itrie() };
    cfg.layout = g_layout;
    ASSERT_MSG(engine.init(cfg), "engine.init failed");
    ASSERT_MSG(!engine.hasDecoder(), "unexpected decoder after encoder-only init");

    KeyboardLayout bad;
    ASSERT_MSG(load_layout_json(g_mismatched_layout_path, bad),
               "failed to load mismatched layout");

    const bool ok = engine.setMode(mode_decoder(g_decoder_path, bad, g_vocab_path));
    ASSERT_MSG(!ok, "setMode should reject layout/decoder mismatch");
    ASSERT_MSG(!engine.hasDecoder(), "decoder should not be loaded after rejection");

    // Engine must still be usable after the rejected setMode call.
    std::vector<float> x, y, t;
    make_swipe(x, y, t);
    auto out = engine.recognize(x.data(), y.data(), t.data(), static_cast<int>(x.size()));
    ASSERT_MSG(!out.empty(), "recognize broken after rejected setMode");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc > 1) g_encoder_path = argv[1];
    if (argc > 2) g_vocab_path = argv[2];
    if (argc > 3) g_layout_path = argv[3];
    if (argc > 4) g_alt_layout_path = argv[4];
    if (argc > 5) g_lm_model_path = argv[5];
    if (argc > 6) g_lm_vocab_path = argv[6];
    if (argc > 7) g_decoder_path = argv[7];
    if (argc > 8) g_mismatched_layout_path = argv[8];

    if(!g_vocab_path.empty()) g_vocab = *load_trie_simple(g_vocab_path);

    if(!g_layout_path.empty()) load_layout_json(g_layout_path, g_layout);
    else g_layout = default_en_qwerty();

    if(!g_alt_layout_path.empty()) load_layout_json(g_alt_layout_path, g_alt_layout);
    if(!g_mismatched_layout_path.empty()) load_layout_json(g_mismatched_layout_path, g_mismatched_layout);

    printf("Running engine integration tests...\n\n");
    if(false) {
        run_test("encoder_plus_decoder",                 test_encoder_plus_decoder);
        return 0;
    }

    run_test("engine_not_ready",                     test_engine_not_ready);
    run_test("predict_next_no_lm",                   test_predict_next_no_lm);
    run_test("init_bad_encoder_path",                test_init_bad_encoder_path);
    run_test("recognize_after_failed_init",          test_recognize_after_failed_init);

    bool have_encoder = !g_encoder_path.empty() && !g_vocab_path.empty() && !g_layout_path.empty();

    if (have_encoder) {
        run_test("encoder_only_init_and_recognize",      test_encoder_only_init_and_recognize);
    } else {
        skip_test("encoder_only_init_and_recognize",      "no encoder/vocab/layout args");
    }

    if (have_encoder && !g_alt_layout_path.empty()) {
        run_test("encoder_only_runtime_setlayout",       test_encoder_only_runtime_setlayout);
        run_test("setlayout_json_vs_array_equivalence",  test_setlayout_json_vs_array_equivalence);
        run_test("layout_swap_roundtrip",                test_layout_swap_roundtrip);
    } else {
        skip_test("encoder_only_runtime_setlayout",       "no alt layout path");
        skip_test("setlayout_json_vs_array_equivalence",  "no alt layout path");
        skip_test("layout_swap_roundtrip",                "no alt layout path");
    }

    if (have_encoder) {
        run_test("cross_layout_coverage",                test_cross_layout_coverage);
    } else {
        skip_test("cross_layout_coverage",                "no encoder/vocab args");
    }

    if (have_encoder && !g_lm_model_path.empty()) {
        run_test("encoder_plus_lm",                      test_encoder_plus_lm);
        run_test("lm_reranking_context_dependent",       test_lm_reranking_context_dependent);
        run_test("lm_alpha_zero_is_no_op",               test_lm_alpha_zero_is_no_op);
    } else {
        skip_test("encoder_plus_lm",                      "no LM model path");
        skip_test("lm_reranking_context_dependent",       "no LM model path");
        skip_test("lm_alpha_zero_is_no_op",               "no LM model path");
    }

    if (have_encoder && !g_decoder_path.empty()) {
        run_test("encoder_plus_decoder",                 test_encoder_plus_decoder);
        run_test("decoder_runtime_swap",                 test_decoder_runtime_swap);
        run_test("decoder_swap_changes_output",          test_decoder_swap_changes_output);
        run_test("decoder_swap_does_not_corrupt_encoder",
                 test_decoder_swap_does_not_corrupt_encoder);
        run_test("scoring_defaults_from_decoder_metadata", test_scoring_defaults_from_decoder_metadata);
        run_test("scoring_overrides_applied",            test_scoring_overrides_applied);
        if (!g_lm_model_path.empty()) {
            run_test("encoder_decoder_plus_lm",          test_encoder_decoder_plus_lm);
        } else {
            skip_test("encoder_decoder_plus_lm",          "no LM model path");
        }
        if (!g_mismatched_layout_path.empty()) {
            run_test("setdecoder_layout_mismatch_rejection",
                     test_setdecoder_layout_mismatch_rejection);
        } else {
            skip_test("setdecoder_layout_mismatch_rejection",
                      "no mismatched layout path");
        }
    } else {
        skip_test("encoder_plus_decoder",                 "no paired decoder .pte");
        skip_test("decoder_runtime_swap",                 "no paired decoder .pte");
        skip_test("decoder_swap_changes_output",          "no paired decoder .pte");
        skip_test("decoder_swap_does_not_corrupt_encoder","no paired decoder .pte");
        skip_test("scoring_defaults_from_decoder_metadata", "no paired decoder .pte");
        skip_test("scoring_overrides_applied",            "no paired decoder .pte");
        skip_test("encoder_decoder_plus_lm",              "no paired decoder .pte");
        skip_test("setdecoder_layout_mismatch_rejection", "no paired decoder .pte");
    }

    printf("\n%d/%d tests passed", tests_passed, tests_run);
    if (tests_skipped > 0) printf(", %d skipped", tests_skipped);
    printf("\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
