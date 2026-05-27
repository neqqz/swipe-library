#include "swipe_decoder/resampler.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace swipe_decoder {

namespace {

// Linear interpolation helper
inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

}  // namespace

void resample_path(
    const std::vector<float>& x,
    const std::vector<float>& y,
    const std::vector<float>& t,
    std::vector<float>& out_x,
    std::vector<float>& out_y,
    bool normalize,
    size_t resample_length
) {
    const size_t n = x.size();
    assert(y.size() == n && "resample_path: x and y must have the same size");
    assert((t.empty() || t.size() == n) && "resample_path: t must be empty or match x size");
    out_x.resize(resample_length);
    out_y.resize(resample_length);

    // Handle degenerate cases
    if (n == 0) {
        std::fill(out_x.begin(), out_x.end(), 0.0f);
        std::fill(out_y.begin(), out_y.end(), 0.0f);
        return;
    }

    if (n == 1) {
        std::fill(out_x.begin(), out_x.end(), x[0]);
        std::fill(out_y.begin(), out_y.end(), y[0]);
        if (normalize) {
            for (size_t i = 0; i < resample_length; i++) {
                out_x[i] = std::clamp(out_x[i], 0.0f, 1.0f);
                out_y[i] = std::clamp(out_y[i], 0.0f, 1.0f);
            }
        }
        return;
    }

    constexpr float eps = 1e-12f;

    // Determine if we have valid timestamps
    bool has_time = !t.empty() && t.size() == n;
    float total_time = 0.0f;

    if (has_time) {
        total_time = t[n - 1] - t[0];
    }

    // Use time-uniform interpolation if we have timestamps with non-zero duration
    // Otherwise fall back to index-based interpolation (uniform sample rate assumption)
    if (has_time && total_time > eps) {
        // Time-uniform resampling (matches transformer training)
        float t0 = t[0];

        for (size_t i = 0; i < resample_length; i++) {
            float target_t = t0 + static_cast<float>(i) * total_time / static_cast<float>(resample_length - 1);

            // Binary search for the segment containing target_t
            auto it = std::lower_bound(t.begin(), t.end(), target_t);
            size_t idx1 = static_cast<size_t>(it - t.begin());

            if (idx1 == 0) {
                out_x[i] = x[0];
                out_y[i] = y[0];
            } else if (idx1 >= n) {
                out_x[i] = x[n - 1];
                out_y[i] = y[n - 1];
            } else {
                size_t idx0 = idx1 - 1;
                float seg_time = t[idx1] - t[idx0];
                float frac = (seg_time > eps) ? (target_t - t[idx0]) / seg_time : 0.0f;

                out_x[i] = lerp(x[idx0], x[idx1], frac);
                out_y[i] = lerp(y[idx0], y[idx1], frac);
            }
        }
    } else {
        // Index-based interpolation (assumes uniform sampling rate)
        // This is equivalent to time-uniform when samples are evenly spaced in time
        for (size_t i = 0; i < resample_length; i++) {
            float frac_pos = static_cast<float>(i) / static_cast<float>(resample_length - 1);
            float idx = frac_pos * static_cast<float>(n - 1);
            size_t idx0 = static_cast<size_t>(idx);
            size_t idx1 = std::min(idx0 + 1, n - 1);
            float frac = idx - static_cast<float>(idx0);

            out_x[i] = lerp(x[idx0], x[idx1], frac);
            out_y[i] = lerp(y[idx0], y[idx1], frac);
        }
    }

    // Clip to [0, 1] if requested (input should already be normalized)
    if (normalize) {
        for (size_t i = 0; i < resample_length; i++) {
            out_x[i] = std::clamp(out_x[i], 0.0f, 1.0f);
            out_y[i] = std::clamp(out_y[i], 0.0f, 1.0f);
        }
    }
}

void resample_path_interleaved(
    const std::vector<float>& x,
    const std::vector<float>& y,
    const std::vector<float>& t,
    std::vector<float>& out,
    bool normalize,
    size_t resample_length
) {
    std::vector<float> out_x, out_y;
    resample_path(x, y, t, out_x, out_y, normalize, resample_length);

    // Interleave as [x0..xN-1, y0..yN-1] for [2, N] tensor
    out.resize(2 * resample_length);
    for (size_t i = 0; i < resample_length; i++) {
        out[i] = out_x[i];
        out[resample_length + i] = out_y[i];
    }
}

void resample_to_60hz(
    const size_t n,
    const float *x,
    const float *y,
    const float *t,
    std::vector<float>& out_x,
    std::vector<float>& out_y
) {
    if (n == 0) {
        out_x.clear();
        out_y.clear();
        return;
    }

    if (n == 1) {
        out_x = {x[0], x[0]};
        out_y = {y[0], y[0]};
        return;
    }

    constexpr float interval_ms = 1000.0f / 60.0f;  // 16.667 ms
    constexpr float eps = 1e-12f;

    float t0 = t[0];
    float duration = t[n - 1] - t0;

    if (duration <= eps) {
        // Zero duration: return endpoints
        out_x = {x[0], x[n - 1]};
        out_y = {y[0], y[n - 1]};
        return;
    }

    // Choose the output count and spacing to match the training-time
    // resampler in `universal_swipe.data.swipe_dataset.resample_trajectory`,
    // which uses `np.linspace(0, duration, n)` semantics:
    //
    //   n_samples = max(2, round(duration_ms / 16.667ms) + 1)
    //   target_t  = np.linspace(0, duration_ms, n_samples)
    //
    // The earlier implementation used `ceil(...) + 1` with a constant 16.667ms
    // step and endpoint clamping, which produced a compressed final interval
    // (e.g. a 494ms swipe got 31 points at 16.667ms stride with the last gap
    // squeezed to 10.67ms; a 520ms swipe got a 3.33ms final gap). The encoder
    // never sees that compression at training time, so it's a small train/test
    // shift. Switching to linspace closes that shift and matches the Python
    // pipeline's encoder argmax sequence on 100% of swipe.futo.org samples
    // (vs ~63% under the old clamp), at a cost of ~0.05pp coverage and ~0.17pp
    // top-1 of measurable change end-to-end.
    size_t num_output = std::max(static_cast<size_t>(2),
        static_cast<size_t>(std::round(duration / interval_ms)) + 1);
    out_x.resize(num_output);
    out_y.resize(num_output);

    // Uniform spacing across [0, duration]. With num_output points the
    // interval is `duration / (num_output - 1)` - slightly under 16.667ms
    // when duration isn't a clean multiple of 60Hz, but uniformly so.
    float step = duration / static_cast<float>(num_output - 1);

    for (size_t i = 0; i < num_output; i++) {
        float target_t = t0 + static_cast<float>(i) * step;
        // Defensive clamp: float round-off may put the final target_t a hair
        // past t[n-1]. With uniform spacing this is a fraction of an
        // ULP of duration, not the 16.67ms overshoot the old impl needed.
        if (target_t > t[n - 1]) target_t = t[n - 1];

        auto it = std::lower_bound(t, t + n, target_t);
        size_t idx1 = static_cast<size_t>(it - t);

        if (idx1 == 0) {
            out_x[i] = x[0];
            out_y[i] = y[0];
        } else if (idx1 >= n) {
            out_x[i] = x[n - 1];
            out_y[i] = y[n - 1];
        } else {
            size_t idx0 = idx1 - 1;
            float seg_time = t[idx1] - t[idx0];
            float frac = (seg_time > eps) ? (target_t - t[idx0]) / seg_time : 0.0f;
            out_x[i] = lerp(x[idx0], x[idx1], frac);
            out_y[i] = lerp(y[idx0], y[idx1], frac);
        }
    }
}

}  // namespace swipe_decoder
