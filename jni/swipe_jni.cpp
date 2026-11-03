/**
 * JNI bindings for org.futo.ml.inference.SwipeDecoder.
 */

#include "swipe_decoder/engine.hpp"
#include "swipe_decoder/unicode_case.hpp"

#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <sched.h>
#include <sys/auxv.h>

#define LOGTAG "SwipeJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOGTAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOGTAG, __VA_ARGS__)

using namespace swipe_decoder;

static SwipeEngine* to_engine(jlong handle) {
    return reinterpret_cast<SwipeEngine*>(static_cast<uintptr_t>(handle));
}

// Helper: pull an optional JNI string into std::string. Empty jstring -> empty.
static std::string jstr_or_empty(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    if (!c) return {};
    std::string out(c);
    env->ReleaseStringUTFChars(s, c);
    return out;
}

// ---------------------------------------------------------------------------
// CPU affinity helpers
//
// Goals:
//  1. Prefer big (or little) cores when the cgroup allows them.
//  2. Never fight the cgroup: always intersect with the currently allowed mask.
//  3. Cache topology so we don't re-read /sys on every call (that was slow
//     enough to contribute to frame drops on input-start).
//  4. If no preferred cores are available right now, leave affinity alone
//     instead of forcing a useless / contested pin.
// ---------------------------------------------------------------------------

struct CoreInfo {
    int id;
    long max_freq;
};

struct Topology {
    std::vector<CoreInfo> cores;
    long min_freq = 0;
    long max_freq = 0;
    bool ready = false;
};

static Topology& topology() {
    static Topology t;
    if (t.ready) return t;

    for (int i = 0; i < CPU_SETSIZE && i < 64; ++i) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        FILE* f = fopen(path, "r");
        if (!f) {
            if (i < 16) continue;  // sparse online mask early on
            break;
        }
        long freq = 0;
        if (fscanf(f, "%ld", &freq) == 1 && freq > 0) {
            t.cores.push_back({i, freq});
        }
        fclose(f);
    }

    if (!t.cores.empty()) {
        t.min_freq = t.cores[0].max_freq;
        t.max_freq = t.cores[0].max_freq;
        for (auto& c : t.cores) {
            t.min_freq = std::min(t.min_freq, c.max_freq);
            t.max_freq = std::max(t.max_freq, c.max_freq);
        }
    }
    t.ready = true;
    return t;
}

static int cpu_count(const cpu_set_t& set, const std::vector<CoreInfo>& cores) {
    int n = 0;
    for (auto& c : cores) {
        if (CPU_ISSET(c.id, &set)) ++n;
    }
    return n;
}

// Preferred cores ∩ currently allowed (cgroup). Returns selected count.
static int build_preferred_mask(const char* affinity, cpu_set_t* out) {
    const Topology& topo = topology();
    if (topo.cores.empty()) return 0;

    const bool want_big = (std::string(affinity) == "big");
    const bool want_little = (std::string(affinity) == "little");
    if (!want_big && !want_little) return 0;

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return 0;
    }

    CPU_ZERO(out);
    int count = 0;
    const bool homogeneous = (topo.min_freq == topo.max_freq);

    for (auto& c : topo.cores) {
        if (!CPU_ISSET(c.id, &allowed)) continue;

        const bool is_little = (c.max_freq == topo.min_freq);
        const bool select = homogeneous
            || (want_big && !is_little)
            || (want_little && is_little);

        if (select) {
            CPU_SET(c.id, out);
            ++count;
        }
    }

    // Preferred cluster fully outside cgroup → use fastest allowed cores.
    if (count == 0 && want_big) {
        long best = -1;
        for (auto& c : topo.cores) {
            if (CPU_ISSET(c.id, &allowed) && c.max_freq > best) {
                best = c.max_freq;
            }
        }
        if (best > 0) {
            for (auto& c : topo.cores) {
                if (CPU_ISSET(c.id, &allowed) && c.max_freq == best) {
                    CPU_SET(c.id, out);
                    ++count;
                }
            }
        }
    }

    return count;
}

static int pin_to_cores(const char* affinity);

// Save the calling thread's CPU affinity on construction, pin to big cores for
// the lifetime of the scope, then restore the saved mask.
// If big cores are not currently allowed by the cgroup, leave affinity alone
// so we don't create unnecessary contention on a tiny allowed set.
class LoadOnBigCores {
  cpu_set_t saved_{};
  bool has_saved_ = false;
  bool did_pin_ = false;
public:
    LoadOnBigCores() {
        if (sched_getaffinity(0, sizeof(saved_), &saved_) == 0) {
            has_saved_ = true;
        }

        cpu_set_t preferred;
        const int n = build_preferred_mask("big", &preferred);
        if (n <= 0) return;

        const int current = has_saved_
            ? cpu_count(saved_, topology().cores)
            : 0;

        // Only call setaffinity when it actually restricts/improves the mask.
        if (!has_saved_ || n < current) {
            if (sched_setaffinity(0, sizeof(preferred), &preferred) == 0) {
                did_pin_ = true;
            }
        }
    }

    ~LoadOnBigCores() {
        if (has_saved_ && did_pin_) {
            sched_setaffinity(0, sizeof(saved_), &saved_);
        }
    }

    LoadOnBigCores(const LoadOnBigCores&) = delete;
    LoadOnBigCores& operator=(const LoadOnBigCores&) = delete;
};

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_futo_ml_inference_SwipeDecoder_nativeInit(
    JNIEnv* env, jobject /* obj */,
    jstring encoderPath, jstring decoderPath,
    jint threads, jint beamWidth, jint topK,
    jboolean useExpansion, jstring freqKeyStr,
    jstring lmModelPath, jstring lmVocabPath
) {
    SwipeEngine::Config config;
    config.encoder_path  = jstr_or_empty(env, encoderPath);
    config.decoder_path  = jstr_or_empty(env, decoderPath);
    config.num_threads   = threads;
    config.beam_width    = beamWidth;
    config.top_k         = topK;

    config.lm_model_path = jstr_or_empty(env, lmModelPath);
    config.lm_vocab_path = jstr_or_empty(env, lmVocabPath);

    LOGI("nativeInit: encoder=%s decoder=%s",
         config.encoder_path.c_str(),
         config.decoder_path.empty() ? "(none)" : config.decoder_path.c_str());

    auto* engine = new SwipeEngine();
    bool ok;
    {
        LoadOnBigCores pin;
        ok = engine->init(config);
    }
    if (!ok) {
        LOGE("nativeInit: engine->init() failed for encoder=%s", config.encoder_path.c_str());
        delete engine;
        return 0;
    }

    LOGI("nativeInit: success, hasDecoder=%d", engine->hasDecoder());
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(engine));
}

JNIEXPORT jobjectArray JNICALL
Java_org_futo_ml_inference_SwipeDecoder_nativeRecognize(
    JNIEnv* env, jobject /* obj */,
    jlong handle,
    jobjectArray leftSegs, jobjectArray rightSegs,
    jint topK, jobjectArray contextArray,
    jint beamWidth, jfloatArray trieWeightArray
) {
    auto* engine = to_engine(handle);
    if (!engine) return nullptr;

    auto extractSegs = [&](jobjectArray arr, std::vector<SwipeSegOwned>& out) -> bool {
        if (!arr) return true;
        jint n = env->GetArrayLength(arr);
        out.reserve(n);
        for (jint i = 0; i < n; ++i) {
            jobject seg = env->GetObjectArrayElement(arr, i);
            if (!seg) continue;
            jfieldID xFid = env->GetFieldID(env->GetObjectClass(seg), "x", "[F");
            jfieldID yFid = env->GetFieldID(env->GetObjectClass(seg), "y", "[F");
            jfieldID tFid = env->GetFieldID(env->GetObjectClass(seg), "t", "[F");
            jfloatArray xArr = (jfloatArray)env->GetObjectField(seg, xFid);
            jfloatArray yArr = (jfloatArray)env->GetObjectField(seg, yFid);
            jfloatArray tArr = (jfloatArray)env->GetObjectField(seg, tFid);

            SwipeSegOwned s;
            jint len = env->GetArrayLength(xArr);
            s.x.resize(len);
            s.y.resize(len);
            s.t.resize(len);
            env->GetFloatArrayRegion(xArr, 0, len, s.x.data());
            env->GetFloatArrayRegion(yArr, 0, len, s.y.data());
            env->GetFloatArrayRegion(tArr, 0, len, s.t.data());

            env->DeleteLocalRef(xArr);
            env->DeleteLocalRef(yArr);
            env->DeleteLocalRef(tArr);
            env->DeleteLocalRef(seg);
            out.push_back(std::move(s));
        }
        return true;
    };

    std::vector<SwipeSegOwned> left, right;
    if (!extractSegs(leftSegs, left) || !extractSegs(rightSegs, right)) return nullptr;


    std::vector<SwipeSeg> left_v, right_v;
    for(const auto &v : left)  left_v.emplace_back(v.view());
    for(const auto &v : right) right_v.emplace_back(v.view());


    std::vector<float> trieWeights;
    if(trieWeightArray) {
        jint numTrieWeights = env->GetArrayLength(trieWeightArray);
        jfloat * trieWeightsF = env->GetFloatArrayElements(trieWeightArray, nullptr);
        for(int i=0; i<numTrieWeights; i++) {
            trieWeights.emplace_back(trieWeightsF[i]);
        }

        env->ReleaseFloatArrayElements(trieWeightArray, trieWeightsF, JNI_ABORT);
    }

    std::vector<std::string> context_words;
    if (contextArray) {
        jint ctx_len = env->GetArrayLength(contextArray);
        context_words.reserve(ctx_len);
        for (jint i = 0; i < ctx_len; ++i) {
            auto jstr = static_cast<jstring>(env->GetObjectArrayElement(contextArray, i));
            if (jstr) {
                const char* cstr = env->GetStringUTFChars(jstr, nullptr);
                if (cstr) {
                    context_words.emplace_back(cstr);
                    env->ReleaseStringUTFChars(jstr, cstr);
                }
                env->DeleteLocalRef(jstr);
            }
        }
    }

    auto results = engine->recognize_multi(std::move(left_v), std::move(right_v), context_words, topK, beamWidth, trieWeights);

    jclass resultClass = env->FindClass("org/futo/ml/inference/SwipeDecoder$Result");
    if (!resultClass) return nullptr;
    jmethodID resultCtor = env->GetMethodID(resultClass, "<init>",
        "(Ljava/lang/String;FFF)V");
    if (!resultCtor) return nullptr;

    int count = static_cast<int>(std::min(results.size(), static_cast<size_t>(topK)));
    jobjectArray out = env->NewObjectArray(count, resultClass, nullptr);

    for (int i = 0; i < count; ++i) {
        jstring word = env->NewStringUTF(results[i].word.c_str());
        jobject result = env->NewObject(resultClass, resultCtor, word,
            results[i].final_score, results[i].ctc_score, results[i].lm_score);
        env->SetObjectArrayElement(out, i, result);
        env->DeleteLocalRef(word);
        env->DeleteLocalRef(result);
    }

    return out;
}

JNIEXPORT jobjectArray JNICALL
Java_org_futo_ml_inference_SwipeDecoder_nativePredictNext(
    JNIEnv* env, jobject /* obj */,
    jlong handle, jobjectArray contextArray, jint topK
) {
    auto* engine = to_engine(handle);
    if (!engine) return nullptr;

    std::vector<std::string> context_words;
    if (contextArray) {
        jint ctx_len = env->GetArrayLength(contextArray);
        context_words.reserve(ctx_len);
        for (jint i = 0; i < ctx_len; ++i) {
            auto jstr = static_cast<jstring>(env->GetObjectArrayElement(contextArray, i));
            if (jstr) {
                const char* cstr = env->GetStringUTFChars(jstr, nullptr);
                context_words.emplace_back(cstr);
                env->ReleaseStringUTFChars(jstr, cstr);
                env->DeleteLocalRef(jstr);
            }
        }
    }

    auto results = engine->predict_next(context_words, topK);

    jclass resultClass = env->FindClass("org/futo/ml/inference/SwipeDecoder$Result");
    if (!resultClass) return nullptr;
    jmethodID resultCtor = env->GetMethodID(resultClass, "<init>",
        "(Ljava/lang/String;FFF)V");
    if (!resultCtor) return nullptr;

    int count = static_cast<int>(results.size());
    jobjectArray out = env->NewObjectArray(count, resultClass, nullptr);

    for (int i = 0; i < count; ++i) {
        jstring word = env->NewStringUTF(results[i].first.c_str());
        jobject result = env->NewObject(resultClass, resultCtor, word,
            results[i].second, 0.0f, 0.0f);
        env->SetObjectArrayElement(out, i, result);
        env->DeleteLocalRef(word);
        env->DeleteLocalRef(result);
    }

    return out;
}

JNIEXPORT jfloatArray JNICALL
Java_org_futo_ml_inference_SwipeDecoder_nativeGetLastTiming(
    JNIEnv* env, jobject /* obj */, jlong handle
) {
    auto* engine = to_engine(handle);
    if (!engine) return nullptr;

    const auto& tm = engine->last_timing();
    jfloatArray arr = env->NewFloatArray(6);
    float values[6] = { tm.resample_us, tm.encoder_us, tm.decoder_us,
                        tm.beam_us, tm.lm_us, tm.total_us };
    env->SetFloatArrayRegion(arr, 0, 6, values);
    return arr;
}

JNIEXPORT jfloatArray JNICALL
Java_org_futo_ml_inference_SwipeDecoder_nativeGetScoring(
    JNIEnv* env, jobject /* obj */, jlong handle
) {
    auto* engine = to_engine(handle);
    if (!engine) return nullptr;

    const auto scoring = engine->scoring();
    jfloatArray arr = env->NewFloatArray(4);
    float values[4] = { scoring.gamma, scoring.lambda, scoring.beta, scoring.alpha };
    env->SetFloatArrayRegion(arr, 0, 4, values);
    return arr;
}

JNIEXPORT jboolean JNICALL
Java_org_futo_ml_inference_SwipeDecoder_nativeHasDecoder(
    JNIEnv* /* env */, jobject /* obj */, jlong handle
) {
    auto* engine = to_engine(handle);
    return engine ? static_cast<jboolean>(engine->hasDecoder()) : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_org_futo_ml_inference_SwipeDecoder_nativeHasLm(
    JNIEnv* /* env */, jobject /* obj */, jlong handle
) {
    auto* engine = to_engine(handle);
    return engine ? static_cast<jboolean>(engine->hasLm()) : JNI_FALSE;
}

// Unified state-mutation call mirroring SwipeEngine::Mode. Null jstring / null
// jfloatArray / NaN jfloat all mean "keep current". For decoderPath/lmModelPath
// a non-null empty string means "unload". Layout changes require all three of
// {letters, cx, cy} together; passing any subset is treated as no-op on layout.
JNIEXPORT jboolean JNICALL
Java_org_futo_ml_inference_SwipeDecoder_nativeSetMode(
    JNIEnv* env, jobject /* obj */,
    jlong handle,
    jstring letters, jfloatArray cxArray, jfloatArray cyArray,
    jlongArray trieArray,
    jstring decoderPath,
    jstring lmModelPath, jstring lmVocabPath
) {
    auto* engine = to_engine(handle);
    if (!engine) return JNI_FALSE;

    SwipeEngine::Mode mode;

    const char* letters_cstr = nullptr;
    jfloat* cx = nullptr;
    jfloat* cy = nullptr;
    if (letters && cxArray && cyArray) {
        letters_cstr = env->GetStringUTFChars(letters, nullptr);
        if (!letters_cstr) return JNI_FALSE;
        jint numKeys = env->GetArrayLength(cxArray);
        cx = env->GetFloatArrayElements(cxArray, nullptr);
        cy = env->GetFloatArrayElements(cyArray, nullptr);
        if (!cx || !cy) {
            if (cx) env->ReleaseFloatArrayElements(cxArray, cx, JNI_ABORT);
            if (cy) env->ReleaseFloatArrayElements(cyArray, cy, JNI_ABORT);
            env->ReleaseStringUTFChars(letters, letters_cstr);
            return JNI_FALSE;
        }
        mode.layout = make_layout(std::string(letters_cstr), cx, cy, numKeys);
    }

    std::vector<ITrie *> tries;
    if(trieArray) {
        jint numTries = env->GetArrayLength(trieArray);
        jlong *elements = env->GetLongArrayElements(trieArray, nullptr);
        if (elements != nullptr) {
            tries.reserve(static_cast<size_t>(numTries));
            for (jsize i = 0; i < numTries; ++i) {
                tries.push_back(reinterpret_cast<ITrie *>(elements[i]));
            }

            env->ReleaseLongArrayElements(trieArray, elements, JNI_ABORT);
        }
    }

    mode.dictionaries = tries;
    
    if (decoderPath) mode.decoder_path  = jstr_or_empty(env, decoderPath);
    if (lmModelPath) mode.lm_model_path = jstr_or_empty(env, lmModelPath);
    if (lmVocabPath) mode.lm_vocab_path = jstr_or_empty(env, lmVocabPath);

    bool ok;
    {
        // Any decoder/LM load benefits from big cores; unload/no-op is cheap.
        LoadOnBigCores pin;
        ok = engine->setMode(mode);
    }

    if (cx) env->ReleaseFloatArrayElements(cxArray, cx, JNI_ABORT);
    if (cy) env->ReleaseFloatArrayElements(cyArray, cy, JNI_ABORT);
    if (letters_cstr) env->ReleaseStringUTFChars(letters, letters_cstr);

    return static_cast<jboolean>(ok);
}

JNIEXPORT void JNICALL
Java_org_futo_ml_inference_SwipeDecoder_nativeSetCaseMapping(
    JNIEnv* env, jclass /* clazz */,
    jintArray fromArray, jintArray toArray
) {
    if (!fromArray || !toArray) {
        set_case_mapping(nullptr, nullptr, 0);
        return;
    }
    jsize n_from = env->GetArrayLength(fromArray);
    jsize n_to   = env->GetArrayLength(toArray);
    if (n_from != n_to) {
        LOGE("nativeSetCaseMapping: from.len=%d != to.len=%d", n_from, n_to);
        return;
    }
    jint* from = env->GetIntArrayElements(fromArray, nullptr);
    jint* to   = env->GetIntArrayElements(toArray,   nullptr);
    if (from && to) {
        static_assert(sizeof(jint) == sizeof(int), "jint/int size mismatch");
        set_case_mapping(reinterpret_cast<const int*>(from),
                         reinterpret_cast<const int*>(to),
                         static_cast<std::size_t>(n_from));
    }
    if (from) env->ReleaseIntArrayElements(fromArray, from, JNI_ABORT);
    if (to)   env->ReleaseIntArrayElements(toArray,   to,   JNI_ABORT);
}

JNIEXPORT void JNICALL
Java_org_futo_ml_inference_SwipeDecoder_nativeDestroy(
    JNIEnv* /* env */, jobject /* obj */, jlong handle
) {
    delete to_engine(handle);
}

// Pin the calling thread to big or little cores.
// Returns the number of cores in the resulting mask, or 0 on failure / no-op.
static int pin_to_cores(const char* affinity) {
    cpu_set_t mask;
    const int count = build_preferred_mask(affinity, &mask);
    if (count == 0) {
        // Nothing usable under current cgroup — leave affinity alone.
        return 0;
    }

    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) return 0;

    cpu_set_t actual;
    CPU_ZERO(&actual);
    if (sched_getaffinity(0, sizeof(actual), &actual) != 0) {
        return count;
    }
    const int actual_count = cpu_count(actual, topology().cores);

    // Log only once per process to avoid spam / extra cost on hot path.
    static bool logged_restriction = false;
    if (!logged_restriction && actual_count > 0 && actual_count < count) {
        LOGI("pinCores: preferred %d cores, cgroup allowed %d", count, actual_count);
        logged_restriction = true;
    }

#ifdef __aarch64__
    if (affinity && std::string(affinity) == "little") {
        unsigned long hwcap = getauxval(AT_HWCAP);
        if (hwcap & (1UL << 22)) {
            LOGE("pinCores: HWCAP reports SVE but pinned to little cores - ISA mismatch risk");
        }
    }
#endif

    return actual_count > 0 ? actual_count : count;
}

JNIEXPORT jint JNICALL
Java_org_futo_ml_inference_SwipeDecoder_nativePinCores(
    JNIEnv* env, jclass /* clazz */, jstring affinityStr
) {
    const char* aff_cstr = env->GetStringUTFChars(affinityStr, nullptr);
    if (!aff_cstr) return 0;
    int result = pin_to_cores(aff_cstr);
    if (result > 0) {
        LOGI("pinCores(%s): pinned to %d cores", aff_cstr, result);
    }
    env->ReleaseStringUTFChars(affinityStr, aff_cstr);
    return result;
}

}  // extern "C"
