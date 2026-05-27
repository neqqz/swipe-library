#include "swipe_decoder/encoder.hpp"
#include "swipe_decoder/fp16_utils.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#ifdef __ANDROID__
#include <android/log.h>
#define ELOG(...) __android_log_print(ANDROID_LOG_ERROR, "SwipeEncoder", __VA_ARGS__)
#else
#define ELOG(...) fprintf(stderr, __VA_ARGS__)
#endif

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::from_blob;
using executorch::runtime::EValue;
using executorch::runtime::Error;

namespace swipe_decoder {

struct SwipeEncoder::Impl {
    std::unique_ptr<Module> module;
    EncoderMetadata metadata;
    bool loaded = false;

    int input_length = 0;       // T
    int output_length = 0;      // T' (typically T/2)
    int max_keys = 0;
    int coeff_dim = 0;
    bool is_fp16 = false;

    // Reusable FP16 input staging buffers.
    std::vector<uint16_t> feat_fp16;
    std::vector<uint16_t> keys_fp16;

    void cleanup() {
        module.reset();
        loaded = false;
        input_length = output_length = max_keys = coeff_dim = 0;
        is_fp16 = false;
    }
};

SwipeEncoder::SwipeEncoder() : impl_(std::make_unique<Impl>()) {}
SwipeEncoder::~SwipeEncoder() = default;
SwipeEncoder::SwipeEncoder(SwipeEncoder&&) noexcept = default;
SwipeEncoder& SwipeEncoder::operator=(SwipeEncoder&&) noexcept = default;

bool SwipeEncoder::load(const std::string& model_path, int num_threads) {
    impl_->cleanup();

    std::string meta_path = metadata_path_for(model_path);
    if (!load_encoder_metadata(meta_path, impl_->metadata)) {
        ELOG("SwipeEncoder: no metadata.json at '%s'!\n", meta_path.c_str());
        return false;
    }

    if(impl_->metadata.kind != "encoder") {
        ELOG("SwipeEncoder: wrong metadata loaded! it's for kind '%s'...\n",
            impl_->metadata.kind.c_str());
        return false;
    }


    executorch::runtime::runtime_init();

    if (num_threads > 0) {
        auto* tp = executorch::extension::threadpool::get_threadpool();
        if (tp) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            tp->_unsafe_reset_threadpool(num_threads);
#pragma GCC diagnostic pop
        }
    }

    impl_->module = std::make_unique<Module>(model_path, Module::LoadMode::Mmap);
    auto load_err = impl_->module->load_forward();
    if (load_err != Error::Ok) {
        ELOG("SwipeEncoder: load '%s' failed (err=0x%x)\n",
             model_path.c_str(), static_cast<unsigned>(load_err));
        impl_->cleanup();
        return false;
    }

    // Discover T and max_keys from forward() input metadata.
    auto meta_result = impl_->module->method_meta("forward");
    if (!meta_result.ok()) {
        ELOG("SwipeEncoder: method_meta('forward') failed\n");
        impl_->cleanup();
        return false;
    }
    auto& meta = meta_result.get();
    if (meta.num_inputs() < 3) {
        ELOG("SwipeEncoder: expected 3 inputs (features, layout_keys, layout_mask), got %zu\n",
             static_cast<size_t>(meta.num_inputs()));
        impl_->cleanup();
        return false;
    }

    auto feat_info = meta.input_tensor_meta(0);
    auto keys_info = meta.input_tensor_meta(1);
    auto mask_info = meta.input_tensor_meta(2);
    if (!feat_info.ok() || !keys_info.ok() || !mask_info.ok()) {
        ELOG("SwipeEncoder: cannot read input tensor metadata\n");
        impl_->cleanup();
        return false;
    }

    // features: [1, 2, T]; layout_keys: [1, max_keys, 2]
    if (feat_info->sizes().size() >= 3) {
        impl_->input_length = static_cast<int>(feat_info->sizes()[2]);
    }
    if (keys_info->sizes().size() >= 3) {
        impl_->max_keys = static_cast<int>(keys_info->sizes()[1]);
    }
    impl_->is_fp16 = (feat_info->scalar_type() == executorch::aten::ScalarType::Half);

    if (impl_->input_length <= 0 || impl_->max_keys <= 0) {
        ELOG("SwipeEncoder: invalid input shapes (T=%d, max_keys=%d)\n",
             impl_->input_length, impl_->max_keys);
        impl_->cleanup();
        return false;
    }

    // Dummy forward to discover output tensor shapes (T', coeff_dim).
    const size_t feat_elems = 1 * 2 * impl_->input_length;
    const size_t keys_elems = 1 * impl_->max_keys * 2;
    const size_t elem_bytes = impl_->is_fp16 ? 2 : 4;
    std::vector<uint8_t> feat_buf(feat_elems * elem_bytes, 0);
    std::vector<uint8_t> keys_buf(keys_elems * elem_bytes, 0);
    std::vector<uint8_t> mask_buf(impl_->max_keys, 1);  // all-true dummy

    auto dtype = impl_->is_fp16 ? executorch::aten::ScalarType::Half
                                : executorch::aten::ScalarType::Float;
    auto feat_t = from_blob(feat_buf.data(),
                            {1, 2, impl_->input_length}, dtype);
    auto keys_t = from_blob(keys_buf.data(),
                            {1, impl_->max_keys, 2}, dtype);
    auto mask_t = from_blob(mask_buf.data(),
                            {1, impl_->max_keys},
                            executorch::aten::ScalarType::Bool);

    std::vector<EValue> inputs;
    inputs.reserve(3);
    inputs.emplace_back(std::move(feat_t));
    inputs.emplace_back(std::move(keys_t));
    inputs.emplace_back(std::move(mask_t));

    auto result = impl_->module->forward(inputs);
    if (!result.ok()) {
        ELOG("SwipeEncoder: dummy forward failed (err=0x%x)\n",
             static_cast<unsigned>(result.error()));
        impl_->cleanup();
        return false;
    }
    auto& outs = result.get();
    if (outs.size() < 3 || !outs[0].isTensor() || !outs[1].isTensor() || !outs[2].isTensor()) {
        ELOG("SwipeEncoder: expected 3 output tensors, got %zu\n", outs.size());
        impl_->cleanup();
        return false;
    }

    // log_emissions: [1, T', K_max+1]
    auto& emit = outs[0].toTensor();
    auto& coef = outs[1].toTensor();
    if (emit.dim() != 3 || coef.dim() != 3) {
        ELOG("SwipeEncoder: unexpected output dims (emit=%zd coef=%zd)\n",
             (ssize_t)emit.dim(), (ssize_t)coef.dim());
        impl_->cleanup();
        return false;
    }
    impl_->output_length = static_cast<int>(emit.size(1));
    impl_->coeff_dim = static_cast<int>(coef.size(2));

    // Sanity check against K_max+1 on the emissions head.
    if (emit.size(2) != impl_->max_keys + 1) {
        ELOG("SwipeEncoder: emissions width %d != max_keys+1 %d\n",
             static_cast<int>(emit.size(2)), impl_->max_keys + 1);
        impl_->cleanup();
        return false;
    }

    if (impl_->is_fp16) {
        impl_->feat_fp16.resize(feat_elems);
        impl_->keys_fp16.resize(keys_elems);
    }

    impl_->loaded = true;
    return true;
}

bool SwipeEncoder::is_loaded() const { return impl_->loaded; }

bool SwipeEncoder::forward(const float* features,
                           const float* layout_keys,
                           const bool* layout_mask,
                           float* log_emissions,
                           float* coefficients,
                           float* lambda_out) {
    if (!impl_->loaded) return false;

    const int T = impl_->input_length;
    const int K = impl_->max_keys;
    const int Kp1 = K + 1;
    const int Tp = impl_->output_length;
    const int C = impl_->coeff_dim;

    const size_t feat_elems = 1 * 2 * T;
    const size_t keys_elems = 1 * K * 2;

    auto dtype = impl_->is_fp16 ? executorch::aten::ScalarType::Half
                                : executorch::aten::ScalarType::Float;

    TensorPtr feat_t, keys_t;
    if (impl_->is_fp16) {
        for (size_t i = 0; i < feat_elems; ++i) impl_->feat_fp16[i] = fp32_to_fp16(features[i]);
        for (size_t i = 0; i < keys_elems; ++i) impl_->keys_fp16[i] = fp32_to_fp16(layout_keys[i]);
        feat_t = from_blob(impl_->feat_fp16.data(), {1, 2, T}, dtype);
        keys_t = from_blob(impl_->keys_fp16.data(), {1, K, 2}, dtype);
    } else {
        feat_t = from_blob(const_cast<float*>(features), {1, 2, T}, dtype);
        keys_t = from_blob(const_cast<float*>(layout_keys), {1, K, 2}, dtype);
    }
    auto mask_t = from_blob(const_cast<bool*>(layout_mask), {1, K},
                            executorch::aten::ScalarType::Bool);

    std::vector<EValue> inputs;
    inputs.reserve(3);
    inputs.emplace_back(std::move(feat_t));
    inputs.emplace_back(std::move(keys_t));
    inputs.emplace_back(std::move(mask_t));

    auto result = impl_->module->forward(inputs);
    if (!result.ok()) {
        ELOG("SwipeEncoder: forward failed (err=0x%x)\n",
             static_cast<unsigned>(result.error()));
        return false;
    }
    auto& outs = result.get();
    if (outs.size() < 3) return false;

    auto& emit = outs[0].toTensor();
    auto& coef = outs[1].toTensor();
    auto& lam = outs[2].toTensor();

    const size_t emit_n = static_cast<size_t>(Tp) * Kp1;
    const size_t coef_n = static_cast<size_t>(Tp) * C;
    const size_t lam_n  = static_cast<size_t>(Tp) * 1;

    auto copy = [&](const executorch::aten::Tensor& t, float* dst, size_t n) {
        if (static_cast<size_t>(t.numel()) < n) return false;
        if (impl_->is_fp16) {
            const uint16_t* s = reinterpret_cast<const uint16_t*>(t.const_data_ptr());
            for (size_t i = 0; i < n; ++i) dst[i] = fp16_to_fp32(s[i]);
        } else {
            std::memcpy(dst, t.const_data_ptr<float>(), n * sizeof(float));
        }
        return true;
    };
    if (!copy(emit, log_emissions, emit_n)) return false;
    if (!copy(coef, coefficients, coef_n)) return false;
    if (!copy(lam, lambda_out, lam_n)) return false;
    return true;
}

int SwipeEncoder::input_length() const { return impl_->input_length; }
int SwipeEncoder::output_length() const { return impl_->output_length; }
int SwipeEncoder::max_keys() const { return impl_->max_keys; }
int SwipeEncoder::coeff_dim() const { return impl_->coeff_dim; }
const EncoderMetadata& SwipeEncoder::metadata() const { return impl_->metadata; }

}  // namespace swipe_decoder
