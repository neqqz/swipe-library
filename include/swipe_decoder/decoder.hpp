/**
 * SwipeDecoder - wraps a paired-decoder .pte (DFSMN refiner).
 *
 * Takes the concatenated encoder outputs `[T', (K+1) + coeff_dim + 1]`
 * and produces refined `log_probs [T', K+1]` suitable for CTC beam search.
 *
 * The decoder is per-layout (e.g. `magic_macaw` for `en_qwerty`). It is
 * optional. The engine can run encoder-only, or load a decoder for a
 * specific layout.
 */

#ifndef SWIPE_DECODER_DECODER_HPP
#define SWIPE_DECODER_DECODER_HPP

#include "swipe_decoder/model_metadata.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace swipe_decoder {

class SwipeDecoder {
public:
    SwipeDecoder();
    ~SwipeDecoder();

    SwipeDecoder(const SwipeDecoder&) = delete;
    SwipeDecoder& operator=(const SwipeDecoder&) = delete;
    SwipeDecoder(SwipeDecoder&&) noexcept;
    SwipeDecoder& operator=(SwipeDecoder&&) noexcept;

    /**
     * Load a paired-decoder .pte and its sibling metadata.json.
     */
    bool load(const std::string& model_path, int num_threads = 1);

    /** Release the loaded decoder */
    void unload();

    bool is_loaded() const;

    /**
     * Run the refiner forward pass.
     *
     * @param concat_features [1, T', input_dim]   = concat(log_emissions_slice(K+1), coefficients, lambda)
     * @param log_probs_out   [1, T', K+1]         output: refined log-probs for beam search
     */
    bool forward(const float* concat_features, float* log_probs_out);

    int input_length() const;   // T'
    int input_dim() const;      // (K+1) + coeff_dim + 1
    int num_chars() const;      // K - layout alphabet size this decoder was trained for
    int num_classes() const;    // K+1 - output width (chars + blank)

    const DecoderMetadata& metadata() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace swipe_decoder

#endif  // SWIPE_DECODER_DECODER_HPP
