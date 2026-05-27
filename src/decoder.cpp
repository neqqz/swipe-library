#include "swipe_decoder/decoder.hpp"
#include "swipe_decoder/fp16_utils.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#ifdef __ANDROID__
#include <android/log.h>
#define DLOG(...) __android_log_print(ANDROID_LOG_ERROR, "SwipeDecoder", __VA_ARGS__)
#else
#define DLOG(...) fprintf(stderr, __VA_ARGS__)
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

struct SwipeDecoder::Impl {
    std::unique_ptr<Module> module;
    DecoderMetadata metadata;
    bool loaded = false;

    int input_length = 0;   // T'
    int input_dim = 0;      // (K+1) + coeff_dim + 1
    int num_classes = 0;    // K+1
    bool is_fp16 = false;

    std::vector<uint16_t> in_fp16;

    void cleanup() {
        module.reset();
        loaded = false;
        input_length = input_dim = num_classes = 0;
        is_fp16 = false;
        metadata = DecoderMetadata{};
    }
};

SwipeDecoder::SwipeDecoder() : impl_(std::make_unique<Impl>()) {}
SwipeDecoder::~SwipeDecoder() = default;
SwipeDecoder::SwipeDecoder(SwipeDecoder&&) noexcept = default;
SwipeDecoder& SwipeDecoder::operator=(SwipeDecoder&&) noexcept = default;

bool SwipeDecoder::load(const std::string& model_path, int num_threads) {
    impl_->cleanup();

    std::string meta_path = metadata_path_for(model_path);
    if (!load_decoder_metadata(meta_path, impl_->metadata)) {
        DLOG("SwipeDecoder: no metadata.json at '%s'!\n", meta_path.c_str());
        return false;
    }

    if(impl_->metadata.kind != "decoder") {
        DLOG("SwipeDecoder: wrong metadata loaded! it's for kind '%s'...\n",
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
        DLOG("SwipeDecoder: load '%s' failed (err=0x%x)\n",
             model_path.c_str(), static_cast<unsigned>(load_err));
        impl_->cleanup();
        return false;
    }

    auto meta_result = impl_->module->method_meta("forward");
    if (!meta_result.ok()) {
        DLOG("SwipeDecoder: method_meta('forward') failed\n");
        impl_->cleanup();
        return false;
    }
    auto& mm = meta_result.get();
    if (mm.num_inputs() < 1) {
        DLOG("SwipeDecoder: no inputs in forward() metadata\n");
        impl_->cleanup();
        return false;
    }
    auto in_info = mm.input_tensor_meta(0);
    if (!in_info.ok() || in_info->sizes().size() < 3) {
        DLOG("SwipeDecoder: unexpected input shape\n");
        impl_->cleanup();
        return false;
    }
    impl_->input_length = static_cast<int>(in_info->sizes()[1]);
    impl_->input_dim = static_cast<int>(in_info->sizes()[2]);
    impl_->is_fp16 = (in_info->scalar_type() == executorch::aten::ScalarType::Half);

    // Dummy forward to discover output width K+1.
    const size_t in_elems = 1 * impl_->input_length * impl_->input_dim;
    const size_t elem_bytes = impl_->is_fp16 ? 2 : 4;
    std::vector<uint8_t> in_buf(in_elems * elem_bytes, 0);
    auto dtype = impl_->is_fp16 ? executorch::aten::ScalarType::Half
                                : executorch::aten::ScalarType::Float;
    auto in_t = from_blob(in_buf.data(),
                          {1, impl_->input_length, impl_->input_dim}, dtype);
    auto result = impl_->module->forward(EValue(std::move(in_t)));
    if (!result.ok()) {
        DLOG("SwipeDecoder: dummy forward failed (err=0x%x)\n",
             static_cast<unsigned>(result.error()));
        impl_->cleanup();
        return false;
    }
    auto& outs = result.get();
    if (outs.empty() || !outs[0].isTensor()) {
        DLOG("SwipeDecoder: unexpected output\n");
        impl_->cleanup();
        return false;
    }
    auto& out = outs[0].toTensor();
    if (out.dim() != 3) {
        DLOG("SwipeDecoder: expected 3D output, got %zdD\n", (ssize_t)out.dim());
        impl_->cleanup();
        return false;
    }
    impl_->num_classes = static_cast<int>(out.size(2));

    // Backfill metadata fields discovered from model if missing.
    if (impl_->metadata.num_chars == 0) {
        impl_->metadata.num_chars = impl_->num_classes - 1;
    }
    if (impl_->metadata.input_length == 0) impl_->metadata.input_length = impl_->input_length;
    if (impl_->metadata.input_dim == 0) impl_->metadata.input_dim = impl_->input_dim;

    if (impl_->is_fp16) {
        impl_->in_fp16.resize(in_elems);
    }

    impl_->loaded = true;
    return true;
}

void SwipeDecoder::unload() {
    impl_->cleanup();
}

bool SwipeDecoder::is_loaded() const { return impl_->loaded; }

bool SwipeDecoder::forward(const float* concat_features, float* log_probs_out) {
    if (!impl_->loaded) return false;

    const int T = impl_->input_length;
    const int D = impl_->input_dim;
    const int Kp1 = impl_->num_classes;
    const size_t in_elems = 1 * T * D;
    const size_t out_elems = 1 * T * Kp1;

    auto dtype = impl_->is_fp16 ? executorch::aten::ScalarType::Half
                                : executorch::aten::ScalarType::Float;
    TensorPtr in_t;
    if (impl_->is_fp16) {
        for (size_t i = 0; i < in_elems; ++i) impl_->in_fp16[i] = fp32_to_fp16(concat_features[i]);
        in_t = from_blob(impl_->in_fp16.data(), {1, T, D}, dtype);
    } else {
        in_t = from_blob(const_cast<float*>(concat_features), {1, T, D}, dtype);
    }

    auto result = impl_->module->forward(EValue(std::move(in_t)));
    if (!result.ok()) {
        DLOG("SwipeDecoder: forward failed (err=0x%x)\n",
             static_cast<unsigned>(result.error()));
        return false;
    }
    auto& outs = result.get();
    if (outs.empty() || !outs[0].isTensor()) return false;
    auto& out = outs[0].toTensor();
    if (static_cast<size_t>(out.numel()) < out_elems) return false;

    if (impl_->is_fp16) {
        const uint16_t* s = reinterpret_cast<const uint16_t*>(out.const_data_ptr());
        for (size_t i = 0; i < out_elems; ++i) log_probs_out[i] = fp16_to_fp32(s[i]);
    } else {
        std::memcpy(log_probs_out, out.const_data_ptr<float>(), out_elems * sizeof(float));
    }
    return true;
}

int SwipeDecoder::input_length() const { return impl_->input_length; }
int SwipeDecoder::input_dim() const { return impl_->input_dim; }
int SwipeDecoder::num_chars() const { return impl_->metadata.num_chars; }
int SwipeDecoder::num_classes() const { return impl_->num_classes; }
const DecoderMetadata& SwipeDecoder::metadata() const { return impl_->metadata; }

}  // namespace swipe_decoder
