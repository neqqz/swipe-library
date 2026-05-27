/**
 * SwipeEncoder - wraps the universal-swipe encoder .pte.
 *
 * The encoder takes a resampled trajectory + keyboard layout and emits three
 * tensors per output timestep: log_emissions [T, K_max+1], coefficients [T, C],
 * and lambda [T, 1]. K_max is padded (typically 64); the active layout's real
 * key count K is handled by the layout_mask input (padded positions come out
 * at −inf in log_emissions).
 *
 * A single encoder .pte works for any layout. Callers pass padded
 * layout_keys / layout_mask at inference time.
 */

#ifndef SWIPE_DECODER_ENCODER_HPP
#define SWIPE_DECODER_ENCODER_HPP

#include "swipe_decoder/model_metadata.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace swipe_decoder {

class SwipeEncoder {
public:
    SwipeEncoder();
    ~SwipeEncoder();

    SwipeEncoder(const SwipeEncoder&) = delete;
    SwipeEncoder& operator=(const SwipeEncoder&) = delete;
    SwipeEncoder(SwipeEncoder&&) noexcept;
    SwipeEncoder& operator=(SwipeEncoder&&) noexcept;

    /**
     * Load an encoder .pte. Also reads the sibling metadata.json.
     */
    bool load(const std::string& model_path, int num_threads = 1);

    bool is_loaded() const;

    /**
     * Run the encoder forward pass.
     *
     * @param features      [1, 2, T]        resampled (x, y) path
     * @param layout_keys   [1, max_keys, 2] padded key centers (real keys first)
     * @param layout_mask   [1, max_keys]    true for real keys, false for padding
     * @param log_emissions [1, T', max_keys+1] output - log-probs over K_max chars + blank
     * @param coefficients  [1, T', coeff_dim] output - DCT coefficients per timestep
     * @param lambda_out    [1, T', 1]       output - emission mixing weight per timestep
     *
     * Caller-allocated output buffers must be sized using output_length(),
     * max_keys(), and coeff_dim(). T' is discovered at load time and is
     * typically T/2 (adapter stride=2).
     */
    bool forward(const float* features,
                 const float* layout_keys,
                 const bool* layout_mask,
                 float* log_emissions,
                 float* coefficients,
                 float* lambda_out);

    // Shape getters (valid after load()).
    int input_length() const;   // T - length of resampled path
    int output_length() const;  // T' - encoder output temporal length
    int max_keys() const;       // K_max - padded layout slot count
    int coeff_dim() const;      // C - DCT coefficient dim per timestep

    const EncoderMetadata& metadata() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace swipe_decoder

#endif  // SWIPE_DECODER_ENCODER_HPP
