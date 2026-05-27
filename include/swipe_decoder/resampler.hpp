/**
 * Path resampling for swipe keyboard input.
 *
 * Resamples variable-length swipe paths to fixed [2, 64] tensor format
 * using time-uniform interpolation (matching transformer training data).
 */

#ifndef SWIPE_DECODER_RESAMPLER_HPP
#define SWIPE_DECODER_RESAMPLER_HPP

#include <cstddef>
#include <vector>

namespace swipe_decoder {

// Default output path length
constexpr size_t RESAMPLE_LENGTH = 64;

/**
 * Resample a variable-length path to fixed length using time-uniform interpolation.
 *
 * Takes x, y, t coordinate arrays and resamples them to uniformly-spaced
 * points in time. This matches how the transformer training data was prepared.
 *
 * If timestamps are not provided (empty vector), falls back to index-based
 * interpolation (equivalent to assuming uniform sampling rate).
 *
 * @param x Input x coordinates (must be same length as y and t)
 * @param y Input y coordinates
 * @param t Input timestamps in milliseconds (can be empty for index-based fallback)
 * @param out_x Output x coordinates (will be resized to resample_length)
 * @param out_y Output y coordinates (will be resized to resample_length)
 * @param normalize If true, clip output to [0, 1] range
 * @param resample_length Number of output points (default: RESAMPLE_LENGTH)
 */
void resample_path(
    const std::vector<float>& x,
    const std::vector<float>& y,
    const std::vector<float>& t,
    std::vector<float>& out_x,
    std::vector<float>& out_y,
    bool normalize = true,
    size_t resample_length = RESAMPLE_LENGTH
);

/**
 * Resample path and return interleaved [2, N] array.
 *
 * Output format: [x0, x1, ..., xN-1, y0, y1, ..., yN-1]
 * This matches the model's expected input layout.
 *
 * @param x Input x coordinates
 * @param y Input y coordinates
 * @param t Input timestamps in milliseconds (can be empty for index-based fallback)
 * @param out Output array of size 2*resample_length (will be resized)
 * @param normalize If true, clip output to [0, 1] range
 * @param resample_length Number of output points (default: RESAMPLE_LENGTH)
 */
void resample_path_interleaved(
    const std::vector<float>& x,
    const std::vector<float>& y,
    const std::vector<float>& t,
    std::vector<float>& out,
    bool normalize = true,
    size_t resample_length = RESAMPLE_LENGTH
);

/**
 * Resample a variable-rate path to uniform 60 Hz (16.667 ms intervals).
 *
 * The training data is predominantly sampled at ~60 Hz. This normalizes
 * higher-rate input (120/240 Hz) to match training conditions before
 * the final fixed-length resampling.
 *
 * @param n Number of input points
 * @param x Input x coordinates
 * @param y Input y coordinates
 * @param t Input timestamps in milliseconds
 * @param out_x Output x coordinates at 60 Hz (variable length)
 * @param out_y Output y coordinates at 60 Hz (variable length)
 */
void resample_to_60hz(
    const size_t n,
    const float *x,
    const float *y,
    const float *t,
    std::vector<float>& out_x,
    std::vector<float>& out_y
);

}  // namespace swipe_decoder

#endif  // SWIPE_DECODER_RESAMPLER_HPP
