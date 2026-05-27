/**
 * Portable FP16 and FP32 conversion utilities
 * IEEE 754 half-precision: 1 sign + 5 exponent + 10 mantissa bits
 */

#ifndef SWIPE_DECODER_FP16_UTILS_HPP
#define SWIPE_DECODER_FP16_UTILS_HPP

#include <cstdint>
#include <cstring>

namespace swipe_decoder {

inline uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint16_t sign = (bits >> 16) & 0x8000;
    int exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = bits & 0x7FFFFF;
    if (exp <= 0) return sign;           // underflow -> zero
    if (exp >= 31) return sign | 0x7C00; // overflow -> inf
    return sign | (exp << 10) | (frac >> 13);
}

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    if (exp == 0) {
        if (frac == 0) {
            float f;
            uint32_t bits = sign;
            std::memcpy(&f, &bits, 4);
            return f;
        }
        // Denorm
        while (!(frac & 0x400)) { frac <<= 1; exp--; }
        exp++; frac &= 0x3FF;
    } else if (exp == 31) {
        uint32_t bits = sign | 0x7F800000 | (frac << 13);
        float f;
        std::memcpy(&f, &bits, 4);
        return f;
    }
    uint32_t bits = sign | ((exp + 112) << 23) | (frac << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace swipe_decoder

#endif  // SWIPE_DECODER_FP16_UTILS_HPP
