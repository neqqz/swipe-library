/**
 * SwipeEngine is the main API that performs the full swipe recognition
 * pipeline for you. Specify the model paths, dictionaries, and layout.
 *
 * Supported modes:
 *   - encoder only
 *   - encoder + decoder
 *   - encoder + context LM
 *   - encoder + decoder + context LM
 *
 * A single encoder works for any layout/language. A decoder is paired to
 * a specific encoder, layout, and language. A context LM is only paired
 * to a specific language. These can be swapped via setMode, for example if
 * the keyboard layout changes.
 */

#ifndef SWIPE_DECODER_ENGINE_HPP
#define SWIPE_DECODER_ENGINE_HPP

#include "swipe_decoder/beam_search.hpp"
#include "swipe_decoder/context_lm.hpp"
#include "swipe_decoder/decoder.hpp"
#include "swipe_decoder/encoder.hpp"
#include "swipe_decoder/layout.hpp"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

class ITrie;

namespace swipe_decoder {

struct SwipeSeg {
    size_t num_points;
    const float *x;
    const float *y;
    const float *t;
};

struct SwipeSegOwned {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> t;
    
    inline SwipeSeg view() const {
        return {x.size(), x.data(), y.data(), t.data()};
    }
};

class SwipeEngine {
public:
    struct Timing {
        float resample_us = 0;
        float encoder_us = 0;
        float decoder_us = 0;
        float beam_us = 0;
        float lm_us = 0;
        float total_us = 0;
    };

    struct Config {
        int num_threads = 1;  // thread count for running models
        int beam_width = 100; // maximum beam width value
        int top_k = 4;        // default top_k value

        // Path to the encoder .pte model. It must have a metadata.json next to it.
        std::string encoder_path;

        // Keyboard layout, for example `default_en_qwerty()` from layout.hpp
        KeyboardLayout layout;

        // Dictionaries to use initially
        std::vector<ITrie *> dictionaries;

        // Optional decoder, it's up to you to make sure this matches the given
        // layout and encoder. It must have a metadata.json next to it.
        std::string decoder_path;

        // Optional context LM. It must have a metadata.json next to it.
        std::string lm_model_path;
        std::string lm_vocab_path;

        // Scoring overrides, recommended to not touch
        std::optional<float> gamma;
        std::optional<float> lambda_;
        std::optional<float> beta;
        std::optional<float> alpha;
        std::string scoring_path;
    };

    /**
     * Atomic state-mutation request passed to setMode().
     *
     * Every field is three-state:
     *   - nullopt      :  keep current (no-op for that slot)
     *   - empty string :  unload (decoder_path / lm_model_path only)
     *   - non-empty    :  set / load (no-op if already matches current)
     *
     * Loads are staged in scratch so a failure leaves prior state intact.
     */
    struct Mode {
        std::optional<KeyboardLayout> layout;
        std::optional<std::vector<ITrie *>> dictionaries;
        std::optional<std::string>    decoder_path;     // "" = unload
        std::optional<std::string>    lm_model_path;    // "" = unload
        std::optional<std::string>    lm_vocab_path;
    };

    SwipeEngine();
    ~SwipeEngine();

    SwipeEngine(const SwipeEngine&) = delete;
    SwipeEngine& operator=(const SwipeEngine&) = delete;
    SwipeEngine(SwipeEngine&&) noexcept;
    SwipeEngine& operator=(SwipeEngine&&) noexcept;

    /**
     * Load encoder, vocab, layout, and (optionally) paired decoder + LM.
     */
    bool init(const Config& config);

    bool is_ready() const;

    /** True iff a paired decoder is currently loaded. */
    bool hasDecoder() const;

    /** True iff a context LM is currently loaded. */
    bool hasLm() const;

    /**
     * Apply any subset of layout / vocab / decoder / LM changes atomically.
     * See `Mode` for per-field semantics. Fields matching current state are
     * no-ops (e.g. passing the already-loaded decoder path does not reload).
     * Returns false and leaves prior state intact on any load/validation
     * failure.
     */
    bool setMode(const Mode& mode);

    /**
     * Recognize a multi-swipe. Takes in two sequences of (x, y, t) swipes and
     * outputs top-k word candidates. You don't need to use this for typical
     * swipe typing recognition.
     */
    std::vector<DecodedWord> recognize_multi(
        std::vector<SwipeSeg> left,
        std::vector<SwipeSeg> right = {},
        const std::vector<std::string>& context_words = {},
        int top_k = 0,  // 0 = use config.top_k
        int beam_width = 0, // 0 = use config.beam_width
        const std::vector<float> &trie_weights = {}
    );

    /**
     * Recognize a typical swipe, takes in (x, y, t) points where t is
     * milliseconds since the first point. Outputs top-k word candidates
     */
    inline std::vector<DecodedWord> recognize(
        const float *x,
        const float *y,
        const float *t,
        size_t num_points,
        const std::vector<std::string>& context_words = {},
        int top_k = 0,  // 0 = use config.top_k
        int beam_width = 0, // 0 = use config.beam_width
        const std::vector<float> &trie_weights = {}
    ) {
        return recognize_multi({{num_points, x, y, t}}, {}, context_words, top_k, beam_width, trie_weights);
    } 

    /**
     * Context-only next-word prediction (requires LM).
     */
    std::vector<std::pair<std::string, float>> predict_next(
        const std::vector<std::string>& context_words,
        int k = 10
    );

    const Timing& last_timing() const;

    const ScoringParams &scoring() const;

    TrieBeamSearch& beam_search();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace swipe_decoder

#endif  // SWIPE_DECODER_ENGINE_HPP
