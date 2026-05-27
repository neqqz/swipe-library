/**
 * Contract test for universal-swipe metadata.json sidecars.
 *
 * Every metadata.json shipped under models/ must parse cleanly with all
 * required fields populated. This is the gate that prevents a production
 * release from shipping metadata the engine would silently fall back on.
 * A missing scoring block or a renamed input_shape key would otherwise
 * manifest as a quiet accuracy regression, not a failure.
 *
 * Usage: test_metadata [models_dir]
 *   models_dir defaults to SWIPE_MODELS_DIR (repo-root/models) at build time.
 */

#include "swipe_decoder/model_metadata.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace swipe_decoder;

#ifndef SWIPE_MODELS_DIR
#define SWIPE_MODELS_DIR "models"
#endif

static std::string slurp(const fs::path& p) {
    std::ifstream f(p);
    if (!f.is_open()) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Heuristic: check which kind of metadata this is by looking for a
// top-level "kind": "<value>" field. LM/vocab metadata files without a kind
// field are skipped. This test only covers encoder + decoder sidecars.
static std::string detect_kind(const std::string& content) {
    size_t p = content.find("\"kind\"");
    if (p == std::string::npos) return {};
    p = content.find(':', p);
    if (p == std::string::npos) return {};
    p = content.find('"', p);
    if (p == std::string::npos) return {};
    size_t end = content.find('"', p + 1);
    if (end == std::string::npos) return {};
    return content.substr(p + 1, end - p - 1);
}

int main(int argc, char** argv) {
    fs::path models_dir = (argc >= 2) ? fs::path(argv[1]) : fs::path(SWIPE_MODELS_DIR);

    if (!fs::exists(models_dir) || !fs::is_directory(models_dir)) {
        fprintf(stderr, "test_metadata: models dir '%s' not found, skipping\n",
                models_dir.string().c_str());
        return 0;
    }

    std::vector<fs::path> files;
    for (auto const& entry : fs::recursive_directory_iterator(models_dir)) {
        if (entry.is_regular_file() && entry.path().filename() == "metadata.json") {
            files.push_back(entry.path());
        }
    }

    if (files.empty()) {
        fprintf(stderr, "test_metadata: no metadata.json found under '%s', skipping\n",
                models_dir.string().c_str());
        return 0;
    }

    printf("Validating %zu metadata.json file(s) under %s\n",
           files.size(), models_dir.string().c_str());

    int failures = 0;
    int encoder_count = 0;
    int decoder_count = 0;

    for (const auto& path : files) {
        std::string path_str = path.string();
        std::string content = slurp(path);
        std::string kind = detect_kind(content);

        printf("  %-60s ", path_str.c_str());

        if (kind == "encoder") {
            EncoderMetadata m;
            if (!load_encoder_metadata(path_str, m)) {
                printf("FAIL (parse)\n");
                failures++;
                continue;
            }
            if (m.layout.empty()) {
                printf("FAIL (empty layout)\n");
                failures++;
                continue;
            }
            printf("OK (encoder, layout=%s T=%d K=%d C=%d)\n",
                   m.layout.c_str(), m.input_length, m.max_keys, m.coeff_dim);
            encoder_count++;
        } else if (kind == "decoder") {
            DecoderMetadata m;
            if (!load_decoder_metadata(path_str, m)) {
                printf("FAIL (parse)\n");
                failures++;
                continue;
            }
            if (m.layout.empty()) {
                printf("FAIL (empty layout)\n");
                failures++;
                continue;
            }
            printf("OK (decoder, layout=%s T=%d D=%d K=%d)\n",
                   m.layout.c_str(), m.input_length, m.input_dim, m.num_chars);
            decoder_count++;
        } else {
            printf("FAIL (unknown or missing \"kind\" field: '%s')\n", kind.c_str());
            failures++;
        }
    }

    printf("\n%zu file(s) checked: %d encoder, %d decoder, %d failure(s)\n",
           files.size(), encoder_count, decoder_count, failures);

    return failures == 0 ? 0 : 1;
}
