#include "swipe_decoder/engine.hpp"
#include "swipe_decoder/resampler.hpp"
#include "swipe_decoder/trie.hpp"
#include "swipe_decoder/itrie.h"

#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <cmath>

#ifdef __ANDROID__
#include <android/log.h>
#define ELOG(...) __android_log_print(ANDROID_LOG_ERROR, "SwipeEngine", __VA_ARGS__)
#else
#define ELOG(...) fprintf(stderr, __VA_ARGS__)
#endif

using Clock = std::chrono::high_resolution_clock;

constexpr const char *DEFAULT_SCORING = 
#include "generated/scoring.json"
;

typedef uint8_t MultiModelMode;
#define MMM_DECODER_BIT 1
#define MMM_LM_BIT 2

namespace swipe_decoder {

namespace {

// Populate the padded encoder layout inputs from a `KeyboardLayout`.
//
// `max_keys` is a fixed tensor dimension baked into the encoder export (64 for
// honorable_sturgeon). Real keys occupy slots [0..K-1]; padded slots get
// (0, 0) coords and mask=false. Mirrors `sample_layout_inputs()` in
// universal-swipe's export pipeline.
bool build_padded_layout_inputs(const KeyboardLayout& layout,
                                int max_keys,
                                std::vector<float>& keys_out,
                                std::vector<bool>& mask_out) {
    if (layout.num_keys > max_keys) {
        ELOG("SwipeEngine: layout has %d keys, exceeds encoder max_keys=%d\n",
             layout.num_keys, max_keys);
        return false;
    }
    keys_out.assign(static_cast<size_t>(max_keys) * 2, 0.0f);
    mask_out.assign(static_cast<size_t>(max_keys), false);
    for (int k = 0; k < layout.num_keys; ++k) {
        keys_out[2 * k + 0] = layout.key_cx[k];
        keys_out[2 * k + 1] = layout.key_cy[k];
        mask_out[k] = true;
    }
    return true;
}

void trim_begin(std::string &v) {
    size_t first = v.find_first_not_of(" \t");
    if(first != std::string::npos) v = v.substr(first);
}

void trim_end(std::string &v) {
    size_t last = v.find_last_not_of(" \t");
    if(last != std::string::npos) v.erase(last + 1);
}

}  // namespace

struct SwipeEngine::Impl {
    Config config;
    SwipeEncoder encoder;
    SwipeDecoder decoder;

    std::vector<ITrie *> itries;
    std::unique_ptr<TrieBeamSearch> beam_search;
    std::unique_ptr<ContextLMScorer> lm_scorer;

    std::unordered_map<std::string, ScoringParams> scoring_map;
    std::unordered_map<MultiModelMode, ScoringParams> curr_scoring;
    bool load_scoring_map(const std::string &path) {
        std::string contents;
        if(path.empty()) {
            contents = DEFAULT_SCORING;
        } else {
            std::ifstream in(path, std::ios::binary);
            if(!in) {
                return false;
            }
            std::ostringstream oss;
            oss << in.rdbuf();
            contents = oss.str();
        }

        std::istringstream in(contents);
        std::string line;

        ScoringParams curr_params{};
        std::string   curr_key;

        while (std::getline(in, line)) {
            if (line.empty()) continue;

            trim_begin(line);
            trim_end(line);

            if (!line.empty() && line.front() == '"' && line.back() == '{') {
                if (!curr_key.empty())
                    scoring_map.emplace(curr_key, curr_params);

                size_t q1 = line.find('"');
                size_t q2 = line.find('"', q1 + 1);
                curr_key  = line.substr(q1 + 1, q2 - q1 - 1);
                curr_params = {};
                continue;
            }

            if (!curr_key.empty()) {
                size_t q1 = line.find('"');
                if (q1 == std::string::npos) continue;
                size_t q2 = line.find('"', q1 + 1);
                if (q2 == std::string::npos) continue;

                std::string subkey = line.substr(q1 + 1, q2 - q1 - 1);

                size_t colon = line.find(':', q2);
                if (colon == std::string::npos) continue;
                size_t comma = line.find(',', colon + 1);
                float value = std::stof(line.substr(colon + 1, comma - colon - 1));

                if      (subkey == "gamma")       curr_params.gamma       = value;
                else if (subkey == "lambda")      curr_params.lambda      = value;
                else if (subkey == "beta")        curr_params.beta        = value;
                else if (subkey == "alpha")       curr_params.alpha       = value;
                else if (subkey == "gamma_prune") curr_params.gamma_prune = value;
                else if (subkey == "beta_prune")  curr_params.beta_prune  = value;
            }
        }

        if (!curr_key.empty())
            scoring_map.emplace(curr_key, curr_params);
        
        return scoring_map.size() > 0 && scoring_map.find("fallback") != scoring_map.end();
    }

    std::optional<ScoringParams> get_scoring_for_key(const std::string &key) {
        auto scoring_ = scoring_map.find(key);
        if(scoring_ == scoring_map.end()) {
            ELOG("Failed to find scoring for [%s]\n", key.c_str());
            return std::nullopt;
        }

        return scoring_->second;
    }

    std::string key_for_flag(MultiModelMode mode) {
        std::ostringstream tgt_key;
        tgt_key << encoder.metadata().kind << ":" << encoder.metadata().codename << " ";
        
        if((mode & MMM_DECODER_BIT) != 0) {
            if(decoder.is_loaded()
                    && decoder.metadata().kind.size() > 0
                    && decoder.metadata().codename.size() > 0) {
                tgt_key << decoder.metadata().kind << ":" << decoder.metadata().codename << " ";
            }
        }

        if((mode & MMM_LM_BIT) != 0) {
            if(lm_scorer && lm_scorer->is_loaded()
                    && lm_scorer->metadata().kind.size() > 0
                    && lm_scorer->metadata().codename.size() > 0) {
                tgt_key << lm_scorer->metadata().kind << ":" << lm_scorer->metadata().codename;
            }
        }

        std::string result = tgt_key.str();
        trim_end(result);

        return result;
    }

    ScoringParams closest_params_for_flag(MultiModelMode mode) {
        auto key = key_for_flag(mode);
        auto scoring = get_scoring_for_key(key);
        if(scoring.has_value()) {
            return scoring.value();
        } else {
            // First try removing the decoder
            if((mode & MMM_DECODER_BIT) != 0) {
                ELOG("Missing decoder scoring, will try without\n");
                return closest_params_for_flag(mode & ~MMM_DECODER_BIT);
            }

            // Then try removing the LM
            if((mode & MMM_LM_BIT) != 0) {
                ELOG("Missing LM scoring, will try without\n");
                return closest_params_for_flag(mode & ~MMM_LM_BIT);
            }

            // Encoder is missing, forget about it
            ELOG("Missing encoder scoring!!!\n");
            return scoring_map.find("fallback")->second;
        }
    }
    
    bool update_scoring() {
        curr_scoring.clear();
        for(MultiModelMode i=0; i<=(MMM_DECODER_BIT | MMM_LM_BIT); i++) {
            curr_scoring.emplace(i, closest_params_for_flag(i));
        }

        quick_update_scoring(MMM_DECODER_BIT | MMM_LM_BIT);
        return rebuild_beam_search(layout.num_keys);
    }

    MultiModelMode curr_mode;
    void quick_update_scoring(MultiModelMode mode) {
        if(!beam_search) return;

        auto find_result = curr_scoring.find(mode);
        if(find_result != curr_scoring.end()) {
            scoring = find_result->second;
        } else {
            ELOG("quick_update_scoring has unknown mode %d\n", mode);
            scoring = scoring_map.find("fallback")->second;
        }

        if(config.gamma.has_value())   scoring.gamma  = *config.gamma;
        if(config.lambda_.has_value()) scoring.lambda = *config.lambda_;
        if(config.beta.has_value())    scoring.beta   = *config.beta;
        if(config.alpha.has_value())   scoring.alpha  = *config.alpha;

        curr_mode = mode;
        beam_search->set_scoring(scoring);
        beam_search->set_single_mode((mode & MMM_DECODER_BIT) != 0);
    }

    std::vector<float> make_gaussian(float x, float y) {
        const int K  = layout.num_keys;
        const int Kp1 = K + 1;
        const int blank_dst = K;

        std::vector<float> result;
        for(int j=0; j<16; j++)
        for(int i=0; i<Kp1; i++) {
            result.emplace_back(i == blank_dst ? 1.0f : 0.0f);
        }

        float sum = 0.0f;
        int start_idx = result.size();
        for(int i=0; i<Kp1; i++) {
            if(i == blank_dst) {
                result.emplace_back(0.0f);
            } else {
                float cx = layout.key_cx[i];
                float cy = layout.key_cy[i];
                
                float sigma_x = (1.0f / 10.0f) / 3.5f;
                float sigma_y = (1.0f / 3.0f)  / 3.5f;

                float gx = x - cx;
                float gy = y - cy;
                
                float gauss = std::exp(-((gx*gx) / (2*(sigma_x*sigma_x)) + (gy*gy) / (2*(sigma_y*sigma_y))));
                sum += gauss;
                result.emplace_back(gauss);
            }
        }
        for(int i=0; i<Kp1; i++) {
            result[i + start_idx] = result[i + start_idx] / sum;
        }

        for(int j=0; j<15; j++)
        for(int i=0; i<Kp1; i++) {
            result.emplace_back(i == blank_dst ? 1.0f : 0.0f);
        }

        for(auto &v : result) {
            v = std::log(std::max(1e-10f, v));
        }

        return result;
    }

    std::vector<float> predict_segment(const SwipeSeg &seg) {
        const int T  = encoder.input_length();
        const int Tp = encoder.output_length();
        const int Km = encoder.max_keys();
        const int C  = encoder.coeff_dim();
        const int K  = layout.num_keys;
        const int Kp1 = K + 1;
        const int blank_src = Km;  // blank lives at `max_keys` in the encoder emissions
        const int blank_dst = K;   // and moves to slot K in the sliced view


        auto t0 = Clock::now();
        resample_to_60hz(seg.num_points, seg.x, seg.y, seg.t, hz60_x, hz60_y);
        resample_path_interleaved(hz60_x, hz60_y, {},
                                resampled, true,
                                static_cast<size_t>(T));

        float deviation = 0.0f;
        for(int i=1; i<hz60_x.size(); i++) {
            float px = hz60_x[i-1];
            float py = hz60_y[i-1];
            float cx = hz60_x[i];
            float cy = hz60_y[i];

            float dx = cx-px;
            float dy = cy-py;

            deviation += std::sqrt(dx * dx + dy * dy);
        }
        if(false && deviation < 0.1f) {
            auto result = make_gaussian(hz60_x[0], hz60_y[0]);
            auto t1 = Clock::now();
            resample_us_val += std::chrono::duration<double, std::micro>(t1 - t0).count();
            return result;
        }
        auto t1 = Clock::now();
        resample_us_val += std::chrono::duration<double, std::micro>(t1 - t0).count();

        if (!build_padded_layout_inputs(layout, Km, padded_keys, padded_mask)) {
            return {};
        }
        std::vector<uint8_t> mask_bytes(Km);
        for (int i = 0; i < Km; ++i) mask_bytes[i] = padded_mask[i] ? 1 : 0;

        bool enc_ok = encoder.forward(
            resampled.data(),
            padded_keys.data(),
            reinterpret_cast<const bool*>(mask_bytes.data()),
            full_emissions.data(),
            coefficients.data(),
            lambda_out.data());

        auto t2 = Clock::now();
        encoder_us_val += std::chrono::duration<double, std::micro>(t2 - t1).count();

        if (!enc_ok) return {};

        {
            const int stride_src = Km + 1;
            for (int ti = 0; ti < Tp; ++ti) {
                const float* src = &full_emissions[static_cast<size_t>(ti) * stride_src];
                float* dst = &sliced_emissions[static_cast<size_t>(ti) * Kp1];
                for (int k = 0; k < K; ++k) dst[k] = src[k];
                dst[blank_dst] = src[blank_src];
            }
        }

        auto t3 = Clock::now();
        encoder_us_val += std::chrono::duration<double, std::micro>(t3 - t2).count();

        if (decoder.is_loaded() && ((curr_mode & MMM_DECODER_BIT) != 0)) {
            const int D = decoder.input_dim();
            // Expected D = (K+1) + coeff_dim + 1 for this layout.
            if (D != Kp1 + C + 1) {
                ELOG("SwipeEngine: decoder input_dim=%d mismatches (K+1)+C+1=%d; skipping decoder\n",
                    D, Kp1 + C + 1);
            } else {
                auto t3a = Clock::now();
                decoder_input.resize(static_cast<size_t>(Tp) * D);
                for (int ti = 0; ti < Tp; ++ti) {
                    float* dst = &decoder_input[static_cast<size_t>(ti) * D];
                    const float* emit = &sliced_emissions[static_cast<size_t>(ti) * Kp1];
                    const float* coef = &coefficients[static_cast<size_t>(ti) * C];
                    const float lam = lambda_out[ti];
                    std::memcpy(dst, emit, Kp1 * sizeof(float));
                    std::memcpy(dst + Kp1, coef, C * sizeof(float));
                    dst[Kp1 + C] = lam;
                }

                bool success = decoder.forward(decoder_input.data(), log_probs.data());

                auto t3b = Clock::now();
                decoder_us_val += std::chrono::duration<double, std::micro>(t3b - t3a).count();

                if(success) {
                    return log_probs;
                }
            }
        }

        return sliced_emissions;
    }

    double resample_us_val = 0.0;
    double encoder_us_val = 0.0;
    double decoder_us_val = 0.0;


    bool ready = false;
    KeyboardLayout layout;
    ScoringParams scoring;
    Timing timing;

    // Reusable per-inference buffers.
    std::vector<float> resampled;             // [2 * T] interleaved
    std::vector<float> hz60_x, hz60_y;
    std::vector<float> padded_keys;           // [max_keys * 2]
    std::vector<bool>  padded_mask;           // [max_keys]
    std::vector<float> full_emissions;        // [T' * (max_keys+1)]
    std::vector<float> coefficients;          // [T' * coeff_dim]
    std::vector<float> lambda_out;            // [T']
    std::vector<float> sliced_emissions;      // [T' * (K+1)]
    std::vector<float> decoder_input;         // [T' * ((K+1)+coeff+1)]
    std::vector<float> log_probs;             // [T' * (K+1)] - fed to beam search

    // Load (or reload) vocab + beam search using current `trie` / `layout` /
    // scoring. Caller must have already populated `trie` before calling.
    bool rebuild_beam_search(int num_keys) {
        beam_search = std::make_unique<TrieBeamSearch>(
            config.beam_width,
            scoring,
            num_keys);
        return beam_search != nullptr;
    }
};

SwipeEngine::SwipeEngine() : impl_(std::make_unique<Impl>()) {}
SwipeEngine::~SwipeEngine() = default;
SwipeEngine::SwipeEngine(SwipeEngine&&) noexcept = default;
SwipeEngine& SwipeEngine::operator=(SwipeEngine&&) noexcept = default;

bool SwipeEngine::init(const Config& config) {
    impl_->config = config;
    impl_->ready = false;

    if (config.encoder_path.empty()) {
        ELOG("SwipeEngine: encoder_path is required\n");
        return false;
    }

    // 1. Load scoring (loads provided or default)
    if (!impl_->load_scoring_map(config.scoring_path)) {
        ELOG("SwipeEngine: invalid scoring file passed\n");
        return false;
    }

    // 2. Load encoder.
    if (!impl_->encoder.load(config.encoder_path, config.num_threads)) {
        ELOG("SwipeEngine: failed to load encoder '%s'\n", config.encoder_path.c_str());
        return false;
    }

    // 3. Load layout.
    impl_->layout = config.layout;
    if (impl_->layout.num_keys > impl_->encoder.max_keys()) {
        ELOG("SwipeEngine: layout has %d keys, exceeds encoder max_keys=%d\n",
             impl_->layout.num_keys, impl_->encoder.max_keys());
        return false;
    }

    // 4. Optionally load paired decoder.
    if (!config.decoder_path.empty()) {
        if (!impl_->decoder.load(config.decoder_path, config.num_threads)) {
            ELOG("SwipeEngine: failed to load decoder '%s'\n", config.decoder_path.c_str());
            return false;
        }
    }

    // 5. Optionally load context LM and its fixed embedding vocabulary.
    if (!config.lm_model_path.empty() && !config.lm_vocab_path.empty()) {
        impl_->lm_scorer = std::make_unique<ContextLMScorer>();
        bool lm_ok;
        lm_ok = impl_->lm_scorer->load(config.lm_model_path, config.lm_vocab_path,
                                        config.num_threads);
        if (!lm_ok) {
            ELOG("SwipeEngine: failed to load context LM\n");
            return false;
        }
    }

    // 6. Update the scoring (builds beam search internally).
    if(!impl_->update_scoring()) {
        ELOG("SwipeEngine: failed to update scoring\n");
    }
    impl_->quick_update_scoring(MMM_DECODER_BIT
        | (config.lm_model_path.empty() ? 0 : MMM_LM_BIT));

    // 7. Pre-allocate inference buffers.
    const int T  = impl_->encoder.input_length();
    const int Tp = impl_->encoder.output_length();
    const int Km = impl_->encoder.max_keys();
    const int C  = impl_->encoder.coeff_dim();
    const int K  = impl_->layout.num_keys;
    impl_->resampled.resize(static_cast<size_t>(2) * T);
    impl_->padded_keys.resize(static_cast<size_t>(Km) * 2);
    impl_->padded_mask.resize(Km);
    impl_->full_emissions.resize(static_cast<size_t>(Tp) * (Km + 1));
    impl_->coefficients.resize(static_cast<size_t>(Tp) * C);
    impl_->lambda_out.resize(Tp);
    impl_->sliced_emissions.resize(static_cast<size_t>(Tp) * (K + 1));
    impl_->decoder_input.resize(static_cast<size_t>(Tp) * ((K + 1) + C + 1));
    impl_->log_probs.resize(static_cast<size_t>(Tp) * (K + 1));

    // 8. Set initial layout and dictionaries
    impl_->itries = config.dictionaries;

    impl_->ready = true;
    return true;
}

bool SwipeEngine::is_ready() const { return impl_->ready; }
bool SwipeEngine::hasDecoder() const { return impl_->decoder.is_loaded(); }
bool SwipeEngine::hasLm() const { return impl_->lm_scorer && impl_->lm_scorer->is_loaded(); }

namespace {

bool layout_equals(const KeyboardLayout& a, const KeyboardLayout& b) {
    if (a.num_keys != b.num_keys) return false;
    if (a.letters != b.letters) return false;
    if (a.key_cx.size() != b.key_cx.size() || a.key_cy.size() != b.key_cy.size()) return false;
    for (int i = 0; i < a.num_keys; ++i) {
        if (a.key_cx[i] != b.key_cx[i]) return false;
        if (a.key_cy[i] != b.key_cy[i]) return false;
    }
    return true;
}

}  // namespace

bool SwipeEngine::setMode(const Mode& mode) {
    if (!impl_->ready) {
        ELOG("SwipeEngine::setMode: engine not ready\n");
        return false;
    }

    // ---- 1. Resolve what is actually changing (skip no-op slots) ------
    const bool want_layout =
        mode.layout.has_value() && !layout_equals(*mode.layout, impl_->layout);

    bool want_decoder_change = false;
    bool want_decoder_unload = false;
    if (mode.decoder_path.has_value()) {
        if (mode.decoder_path->empty()) {
            if (impl_->decoder.is_loaded()) {
                want_decoder_change = true;
                want_decoder_unload = true;
            }
        } else if (!impl_->decoder.is_loaded()
                   || *mode.decoder_path != impl_->config.decoder_path) {
            want_decoder_change = true;
        }
    }

    bool want_lm_change = false;
    bool want_lm_unload = false;
    const bool lm_currently_loaded = impl_->lm_scorer && impl_->lm_scorer->is_loaded();
    if (mode.lm_model_path.has_value()) {
        if (mode.lm_model_path->empty()) {
            if (lm_currently_loaded) {
                want_lm_change = true;
                want_lm_unload = true;
            }
        } else if (!lm_currently_loaded
                   || *mode.lm_model_path != impl_->config.lm_model_path
                   || (mode.lm_vocab_path.has_value()
                       && *mode.lm_vocab_path != impl_->config.lm_vocab_path)) {
            want_lm_change = true;
        }
    }

    // ---- 2. Validate prospective layout against encoder capacity ------
    const KeyboardLayout* effective_layout =
        want_layout ? &*mode.layout : &impl_->layout;
    if (effective_layout->num_keys > impl_->encoder.max_keys()) {
        ELOG("SwipeEngine::setMode: layout has %d keys, exceeds encoder max_keys=%d\n",
             effective_layout->num_keys, impl_->encoder.max_keys());
        return false;
    }

    // ---- 3. Scratch-load decoder / LM so a failure leaves prior state intact ----
    SwipeDecoder fresh_decoder;
    if (want_decoder_change && !want_decoder_unload) {
        if (!fresh_decoder.load(*mode.decoder_path, impl_->config.num_threads)) {
            ELOG("SwipeEngine::setMode: failed to load decoder '%s'\n",
                 mode.decoder_path->c_str());
            return false;
        }
        const auto& dm = fresh_decoder.metadata();
        if (dm.num_chars > 0 && dm.num_chars != effective_layout->num_keys) {
            ELOG("SwipeEngine::setMode: layout has %d keys but decoder expects %d ('%s')\n",
                 effective_layout->num_keys, dm.num_chars, dm.layout.c_str());
            return false;
        }
    }

    std::unique_ptr<ContextLMScorer> fresh_lm;
    std::string effective_lm_vocab;
    if (want_lm_change && !want_lm_unload) {
        effective_lm_vocab = mode.lm_vocab_path.has_value()
            ? *mode.lm_vocab_path
            : impl_->config.lm_vocab_path;
        fresh_lm = std::make_unique<ContextLMScorer>();
        if (!fresh_lm->load(*mode.lm_model_path, effective_lm_vocab,
                            impl_->config.num_threads)) {
            ELOG("SwipeEngine::setMode: failed to load LM '%s' with vocab '%s'\n",
                 mode.lm_model_path->c_str(), effective_lm_vocab.c_str());
            return false;
        }
    }

    // ---- 4. Commit decoder slot --------------------------------------
    if (want_decoder_change) {
        if (want_decoder_unload) {
            impl_->decoder.unload();
            impl_->config.decoder_path.clear();
        } else {
            impl_->decoder = std::move(fresh_decoder);
            impl_->config.decoder_path = *mode.decoder_path;
        }
    }

    // ---- 5. Commit layout / vocab ------------------------------------
    if(mode.dictionaries && (*mode.dictionaries).size() > 0) impl_->itries = *mode.dictionaries;
    if (want_layout) impl_->layout = *mode.layout;

    // Resize per-layout inference buffers.
    if (want_layout) {
        const int Tp = impl_->encoder.output_length();
        const int C  = impl_->encoder.coeff_dim();
        const int K  = impl_->layout.num_keys;
        impl_->sliced_emissions.resize(static_cast<size_t>(Tp) * (K + 1));
        impl_->decoder_input.resize(static_cast<size_t>(Tp) * ((K + 1) + C + 1));
        impl_->log_probs.resize(static_cast<size_t>(Tp) * (K + 1));
    }

    // ---- 6. Commit LM slot -------------------------------------------
    if (want_lm_change) {
        if (want_lm_unload) {
            impl_->lm_scorer.reset();
            impl_->config.lm_model_path.clear();
        } else {
            impl_->lm_scorer = std::move(fresh_lm);
            impl_->config.lm_model_path = *mode.lm_model_path;
            impl_->config.lm_vocab_path = effective_lm_vocab;
        }
    }

    // Rebuild beam search if layout/vocab/decoder changed (scoring or num_keys moved).
    if (want_layout || want_decoder_change || want_lm_change) {
        if(!impl_->update_scoring()) {  // rebuilds beam search automatically
            ELOG("SwipeEngine: failed to update scoring during update\n");
        }
    }

    return true;
}


std::vector<DecodedWord> SwipeEngine::recognize_multi(
    std::vector<SwipeSeg> left,
    std::vector<SwipeSeg> right,
    const std::vector<std::string>& context_words,
    int top_k,
    int beam_width,
    const std::vector<float> &trie_weights
) {
    if (!impl_->ready) return {};

    bool use_lm = impl_->lm_scorer && impl_->lm_scorer->is_loaded()
                  && !context_words.empty() && (impl_->config.alpha != 0.0f);
    bool use_decoder = (left.size() + right.size()) == 1;

    impl_->quick_update_scoring(0 
        | (use_lm ? MMM_LM_BIT : 0)
        | (use_decoder ? MMM_DECODER_BIT : 0)
    );

    use_lm = use_lm && impl_->scoring.alpha != 0.0f;

    auto& tm = impl_->timing;
    tm = Timing{};
    auto t0 = Clock::now();

    impl_->resample_us_val = 0.0;
    impl_->encoder_us_val = 0.0;
    impl_->decoder_us_val = 0.0;

    const int T  = impl_->encoder.input_length();
    const int Tp = impl_->encoder.output_length();
    const int Km = impl_->encoder.max_keys();
    const int C  = impl_->encoder.coeff_dim();
    const int K  = impl_->layout.num_keys;
    const int Kp1 = K + 1;
    const int blank_src = Km;  // blank lives at `max_keys` in the encoder emissions
    const int blank_dst = K;   // and moves to slot K in the sliced view


    std::vector<float> left_logprobs_m;
    std::vector<float> left_times;

    std::vector<float> right_logprobs_m;
    std::vector<float> right_times;

    for(const auto &seg : left) {
        if(seg.num_points == 0) continue;

        auto result = impl_->predict_segment(seg);
        left_logprobs_m.insert(left_logprobs_m.end(), result.begin(), result.end());
        
        int num_steps = result.size() / Kp1;

        float start_time = seg.t[0];
        float end_time = seg.t[seg.num_points - 1];
        float delta_time = (end_time - start_time) / (num_steps * 1.0f);
        for(int i=0; i<num_steps; i++) {
            left_times.emplace_back(start_time + delta_time * i);
        }
    }
    for(const auto &seg : right) {
        if(seg.num_points == 0) continue;

        auto result = impl_->predict_segment(seg);
        right_logprobs_m.insert(right_logprobs_m.end(), result.begin(), result.end());
        
        int num_steps = result.size() / Kp1;

        float start_time = seg.t[0];
        float end_time = seg.t[seg.num_points - 1];
        float delta_time = (end_time - start_time) / (num_steps * 1.0f);
        for(int i=0; i<num_steps; i++) {
            right_times.emplace_back(start_time + delta_time * i);
        }
    }

    auto t1 = Clock::now();

    // ---- 6. Beam search ------------------------------------------------
    if (top_k <= 0) top_k = impl_->config.top_k;
    int decode_top_k = use_lm ? std::max(top_k, 200) : top_k;

    if (beam_width <= 0) beam_width = impl_->config.beam_width;

    std::vector<DecodedWord> results;
    int i = 0;
    for(const auto &itrie : impl_->itries) {
        const float weight = i < trie_weights.size() ?
            trie_weights[i] : 1.0f;

        i++;

        if(weight <= 0.001f) continue; // Skip ignored tries

        impl_->beam_search->decode_with_scores_multi(
            itrie, results,
            left_logprobs_m.data(), left_times.data(), left_times.size(),
            right_logprobs_m.data(), right_times.data(), right_times.size(),
            weight, beam_width
        );
    }
    impl_->beam_search->finish_decode(results, decode_top_k);
    auto t2 = Clock::now();

    tm.resample_us = impl_->resample_us_val;
    tm.encoder_us  = impl_->encoder_us_val;
    tm.decoder_us  = impl_->decoder_us_val;
    tm.beam_us     = std::chrono::duration<float, std::micro>(t2 - t1).count();

    // ---- 7. LM rerank --------------------------------------------------
    auto t3 = Clock::now();
    if (use_lm && !results.empty()) {
        std::vector<std::string> words;
        words.reserve(results.size());
        for (const auto& r : results) words.push_back(r.word);

        auto lm_scores = impl_->lm_scorer->score(context_words, words);
        if (lm_scores.size() == results.size()) {
            for (size_t i = 0; i < results.size(); ++i) {
                results[i].lm_score = lm_scores[i];
                results[i].final_score += impl_->scoring.alpha * lm_scores[i];
            }
            auto ctc_top_it = std::max_element(results.begin(), results.end(),
                [](const DecodedWord& a, const DecodedWord& b) {
                    return a.ctc_score < b.ctc_score;
                });
            DecodedWord ctc_top = (ctc_top_it != results.end()) ? *ctc_top_it : DecodedWord{};

            std::sort(results.begin(), results.end(),
                [](const DecodedWord& a, const DecodedWord& b) {
                    return a.final_score > b.final_score;
                });
            if (static_cast<int>(results.size()) > top_k) results.resize(top_k);

            // Preserve CTC top-1 in the visible slate.
            if (!ctc_top.word.empty()) {
                bool present = false;
                for (const auto& r : results) if (r.word == ctc_top.word) { present = true; break; }
                if (!present && !results.empty()) results.back() = ctc_top;
            }
        } else {
            ELOG("SwipeEngine: LM score count mismatch (%zu vs %zu), skipping rerank\n",
                 lm_scores.size(), results.size());
        }
    }
    auto t4 = Clock::now();
    tm.lm_us = std::chrono::duration<float, std::micro>(t4 - t3).count();
    tm.total_us = std::chrono::duration<float, std::micro>(t4 - t0).count();

    return results;
}

std::vector<std::pair<std::string, float>> SwipeEngine::predict_next(
    const std::vector<std::string>& context_words, int k
) {
    if (!impl_->lm_scorer || !impl_->lm_scorer->is_loaded()) return {};
    return impl_->lm_scorer->predict_topk(context_words, k);
}

const ScoringParams &SwipeEngine::scoring() const {
    return impl_->scoring;
}

const SwipeEngine::Timing& SwipeEngine::last_timing() const { return impl_->timing; }
TrieBeamSearch& SwipeEngine::beam_search() { return *impl_->beam_search; }

}  // namespace swipe_decoder
