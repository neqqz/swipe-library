#include "swipe_decoder/model_metadata.hpp"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#ifdef __ANDROID__
#include <android/log.h>
#define MLOG(...) __android_log_print(ANDROID_LOG_ERROR, "ModelMeta", __VA_ARGS__)
#else
#define MLOG(...) fprintf(stderr, __VA_ARGS__)
#endif

namespace swipe_decoder {

namespace {

std::string slurp(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Find `"key"` in `content`; returns start of value-region just past the ':'
// separator (whitespace skipped), or std::string::npos if key is absent.
size_t find_value(const std::string& content, const std::string& key,
                  size_t from = 0) {
    std::string needle = "\"" + key + "\"";
    size_t pos = content.find(needle, from);
    if (pos == std::string::npos) return std::string::npos;
    pos = content.find(':', pos + needle.size());
    if (pos == std::string::npos) return std::string::npos;
    ++pos;
    while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) ++pos;
    return pos;
}

bool extract_string(const std::string& content, const std::string& key,
                    std::string& out, size_t from = 0) {
    size_t p = find_value(content, key, from);
    if (p == std::string::npos || p >= content.size() || content[p] != '"') return false;
    ++p;
    size_t end = content.find('"', p);
    if (end == std::string::npos) return false;
    out = content.substr(p, end - p);
    return true;
}

bool extract_number(const std::string& content, const std::string& key,
                    double& out, size_t from = 0) {
    size_t p = find_value(content, key, from);
    if (p == std::string::npos || p >= content.size()) return false;
    char c = content[p];
    if (c != '-' && c != '+' && !std::isdigit(static_cast<unsigned char>(c))) return false;
    try {
        size_t consumed = 0;
        out = std::stod(content.substr(p), &consumed);
        return consumed > 0;
    } catch (...) {
        return false;
    }
}

bool extract_int(const std::string& content, const std::string& key,
                 int& out, size_t from = 0) {
    double d;
    if (!extract_number(content, key, d, from)) return false;
    out = static_cast<int>(d);
    return true;
}

// Return the substring for an object or array value: everything from the
// opening brace/bracket through the matching closing one (depth-counted, not
// string-aware since metadata.json has no braces-inside-strings).
bool extract_block(const std::string& content, const std::string& key,
                   std::string& out, char open = '{', char close = '}') {
    size_t p = find_value(content, key);
    if (p == std::string::npos || p >= content.size() || content[p] != open) return false;
    int depth = 0;
    for (size_t i = p; i < content.size(); ++i) {
        if (content[i] == open) ++depth;
        else if (content[i] == close) {
            --depth;
            if (depth == 0) {
                out = content.substr(p, i - p + 1);
                return true;
            }
        }
    }
    return false;
}

// Parse the first N integers from an array string like "[1, 2, 64]".
int parse_array_ints(const std::string& block, int* out, int max_n) {
    int n = 0;
    size_t i = 0;
    while (i < block.size() && n < max_n) {
        while (i < block.size()
               && !std::isdigit(static_cast<unsigned char>(block[i]))
               && block[i] != '-') ++i;
        if (i >= block.size()) break;
        try {
            size_t consumed = 0;
            double d = std::stod(block.substr(i), &consumed);
            if (consumed == 0) break;
            out[n++] = static_cast<int>(d);
            i += consumed;
        } catch (...) { break; }
    }
    return n;
}

// Parses "scoring": { gamma, lambda, beta }. Returns true iff all three fields
// were present and numeric — partial "scoring" blocks are treated as broken so
// we can't ship metadata that silently reverts to struct defaults.
bool parse_scoring(const std::string& content, ScoringParams& out) {
    std::string scoring_block;
    if (!extract_block(content, "scoring", scoring_block)) return false;
    double g, l, b;
    if (!extract_number(scoring_block, "gamma", g)) return false;
    if (!extract_number(scoring_block, "lambda", l)) return false;
    if (!extract_number(scoring_block, "beta", b)) return false;
    out.gamma  = static_cast<float>(g);
    out.lambda = static_cast<float>(l);
    out.beta   = static_cast<float>(b);
    return true;
}

}  // namespace

bool load_encoder_metadata(const std::string& metadata_path, EncoderMetadata& out) {
    std::string content = slurp(metadata_path);
    if (content.empty()) {
        MLOG("ModelMeta: cannot read '%s'\n", metadata_path.c_str());
        return false;
    }

    extract_string(content, "codename", out.codename);
    extract_string(content, "kind", out.kind);
    extract_string(content, "layout", out.layout);
    extract_int(content, "max_keys", out.max_keys);

    // input_shapes.features preferred, fall back to top-level input_shape.
    std::string shapes_block;
    std::string features_block;
    bool have_shape = false;
    if (extract_block(content, "input_shapes", shapes_block)
        && extract_block(shapes_block, "features", features_block, '[', ']')) {
        int dims[3] = {0, 0, 0};
        if (parse_array_ints(features_block, dims, 3) >= 3) {
            out.input_length = dims[2];
            have_shape = true;
        }
    }
    if (!have_shape) {
        std::string top_shape;
        if (extract_block(content, "input_shape", top_shape, '[', ']')) {
            int dims[3] = {0, 0, 0};
            if (parse_array_ints(top_shape, dims, 3) >= 3) {
                out.input_length = dims[2];
            }
        }
    }

    // coeff_dim from model_config.freq_x * freq_y.
    std::string model_cfg;
    if (extract_block(content, "model_config", model_cfg)) {
        int fx = 0, fy = 0;
        extract_int(model_cfg, "freq_x", fx);
        extract_int(model_cfg, "freq_y", fy);
        out.coeff_dim = fx * fy;
    }

    if (out.input_length <= 0 || out.max_keys <= 0 || out.coeff_dim <= 0) {
        MLOG("ModelMeta: '%s' missing required fields "
             "(input_length=%d max_keys=%d coeff_dim=%d)\n",
             metadata_path.c_str(),
             out.input_length, out.max_keys, out.coeff_dim);
        return false;
    }
    return true;
}

bool load_decoder_metadata(const std::string& metadata_path, DecoderMetadata& out) {
    std::string content = slurp(metadata_path);
    if (content.empty()) {
        MLOG("ModelMeta: cannot read '%s'\n", metadata_path.c_str());
        return false;
    }

    extract_string(content, "codename", out.codename);
    extract_string(content, "kind", out.kind);
    extract_string(content, "layout", out.layout);
    extract_int(content, "num_chars", out.num_chars);

    // Decoder input_shape = [1, T, feat_dim].
    std::string top_shape;
    if (extract_block(content, "input_shape", top_shape, '[', ']')) {
        int dims[3] = {0, 0, 0};
        if (parse_array_ints(top_shape, dims, 3) >= 3) {
            out.input_length = dims[1];
            out.input_dim = dims[2];
        }
    }
    // num_chars also lives inside model_config for some decoders.
    if (out.num_chars <= 0) {
        std::string model_cfg;
        if (extract_block(content, "model_config", model_cfg)) {
            extract_int(model_cfg, "num_chars", out.num_chars);
        }
    }

    if (out.input_length <= 0 || out.input_dim <= 0 || out.num_chars <= 0) {
        MLOG("ModelMeta: '%s' missing required fields "
             "(input_length=%d input_dim=%d num_chars=%d)\n",
             metadata_path.c_str(),
             out.input_length, out.input_dim, out.num_chars);
        return false;
    }
    return true;
}

bool load_contextlm_metadata(const std::string& metadata_path, ContextLMMetadata& out) {
    std::string content = slurp(metadata_path);
    if (content.empty()) {
        MLOG("ModelMeta: cannot read '%s'\n", metadata_path.c_str());
        return false;
    }

    extract_string(content, "codename", out.codename);
    extract_string(content, "kind", out.kind);
    return true;
}

std::string metadata_path_for(const std::string& model_path) {
    // Prefer <model_without_ext>.json if it exists; else <dir>/metadata.json.
    auto ends_with = [](const std::string& s, const std::string& suf) {
        return s.size() >= suf.size()
            && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
    };
    std::string stem = model_path;
    if (ends_with(stem, ".pte")) stem.resize(stem.size() - 4);
    {
        std::string per_model = stem + ".json";
        std::ifstream f(per_model);
        if (f.is_open()) return per_model;
    }
    size_t slash = model_path.find_last_of("/\\");
    std::string dir = (slash == std::string::npos) ? "." : model_path.substr(0, slash);
    return dir + "/metadata.json";
}

}  // namespace swipe_decoder
