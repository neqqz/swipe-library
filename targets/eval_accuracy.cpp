/**
 * Accuracy evaluation for swipe keyboard inference.
 *
 * Reads a labeled dataset of (ground_truth_word, swipe_path) pairs,
 * runs the full inference pipeline, and reports top-1 and top-4 accuracy.
 *
 * Dataset binary format (written by generate_eval_data.py):
 *   uint32_t num_samples
 *   For each sample:
 *     uint32_t word_len
 *     char[word_len] ground_truth_word  (no null terminator)
 *     uint32_t num_points
 *     float[num_points] x
 *     float[num_points] y
 *     float[num_points] t
 */

#include "swipe_decoder/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

using namespace swipe_decoder;
using Clock = std::chrono::high_resolution_clock;

struct EvalSample {
    std::string word;
    std::vector<float> x, y, t;
};

static bool load_dataset(const std::string& path, std::vector<EvalSample>& samples) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "Error: Cannot open dataset: %s\n", path.c_str());
        return false;
    }

    uint32_t num_samples;
    f.read(reinterpret_cast<char*>(&num_samples), 4);
    if (!f) {
        fprintf(stderr, "Error: Cannot read sample count\n");
        return false;
    }

    samples.resize(num_samples);
    for (uint32_t i = 0; i < num_samples; ++i) {
        uint32_t word_len;
        f.read(reinterpret_cast<char*>(&word_len), 4);
        if (!f || word_len > 100) {
            fprintf(stderr, "Error: Bad word length %u at sample %u\n", word_len, i);
            return false;
        }
        samples[i].word.resize(word_len);
        f.read(samples[i].word.data(), word_len);

        uint32_t num_points;
        f.read(reinterpret_cast<char*>(&num_points), 4);
        if (!f || num_points > 10000) {
            fprintf(stderr, "Error: Bad point count %u at sample %u\n", num_points, i);
            return false;
        }
        samples[i].x.resize(num_points);
        samples[i].y.resize(num_points);
        samples[i].t.resize(num_points);
        f.read(reinterpret_cast<char*>(samples[i].x.data()), num_points * 4);
        f.read(reinterpret_cast<char*>(samples[i].y.data()), num_points * 4);
        f.read(reinterpret_cast<char*>(samples[i].t.data()), num_points * 4);

        if (!f) {
            fprintf(stderr, "Error: Truncated data at sample %u\n", i);
            return false;
        }
    }

    return true;
}

static void usage(const char* argv0) {
    fprintf(stderr, "Usage: %s --encoder PATH --vocab PATH --layout PATH --dataset PATH [OPTIONS]\n", argv0);
    fprintf(stderr, "\nRequired:\n");
    fprintf(stderr, "  --encoder PATH     Universal encoder .pte (honorable_sturgeon)\n");
    fprintf(stderr, "  --vocab PATH       Combined vocab file\n");
    fprintf(stderr, "  --layout PATH      Layout JSON file\n");
    fprintf(stderr, "  --dataset PATH     Eval dataset binary\n");
    fprintf(stderr, "\nOptional:\n");
    fprintf(stderr, "  --decoder PATH     Paired decoder .pte\n");
    fprintf(stderr, "  --lm-model PATH    Context LM .pte\n");
    fprintf(stderr, "  --lm-vocab PATH    Context LM vocab (default: --vocab)\n");
    fprintf(stderr, "  --beam-width N     Beam width (default: 100)\n");
    fprintf(stderr, "  --top-k N          Candidates per sample (default: 4)\n");
    fprintf(stderr, "  --threads N        Threads (default: 1)\n");
    fprintf(stderr, "  --expansion        Enable surface-form expansion (default: on)\n");
    fprintf(stderr, "  --no-expansion     Disable surface-form expansion\n");
    fprintf(stderr, "  --gamma F          Length-norm exponent override\n");
    fprintf(stderr, "  --lambda F         Frequency bonus override\n");
    fprintf(stderr, "  --beta F           Per-char length bonus override\n");
    fprintf(stderr, "  --alpha F          Language model weight override\n");
    fprintf(stderr, "  --verbose          Per-sample results\n");
    fprintf(stderr, "  --json             JSON summary\n");
    fprintf(stderr, "  --limit N          First N samples only\n");
}

int main(int argc, char* argv[]) {
    std::string encoder_path, decoder_path, vocab_path, layout_path, dataset_path;
    std::string lm_model_path, lm_vocab_path;
    int beam_width = 100;
    int top_k = 4;
    int threads = 1;
    bool use_expansion = true;
    bool verbose = false;
    bool json_output = false;
    int limit = 0;
    std::optional<float> gamma_opt, lambda_opt, beta_opt, alpha_opt;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else if (arg == "--encoder" && i + 1 < argc) encoder_path = argv[++i];
        else if (arg == "--decoder" && i + 1 < argc) decoder_path = argv[++i];
        else if (arg == "--vocab" && i + 1 < argc) vocab_path = argv[++i];
        else if (arg == "--layout" && i + 1 < argc) layout_path = argv[++i];
        else if (arg == "--dataset" && i + 1 < argc) dataset_path = argv[++i];
        else if (arg == "--beam-width" && i + 1 < argc) beam_width = std::atoi(argv[++i]);
        else if (arg == "--top-k" && i + 1 < argc) top_k = std::atoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (arg == "--lm-model" && i + 1 < argc) lm_model_path = argv[++i];
        else if (arg == "--lm-vocab" && i + 1 < argc) lm_vocab_path = argv[++i];
        else if (arg == "--expansion") use_expansion = true;
        else if (arg == "--no-expansion") use_expansion = false;
        else if (arg == "--gamma" && i + 1 < argc) gamma_opt = (float)std::atof(argv[++i]);
        else if (arg == "--lambda" && i + 1 < argc) lambda_opt = (float)std::atof(argv[++i]);
        else if (arg == "--beta" && i + 1 < argc) beta_opt = (float)std::atof(argv[++i]);
        else if (arg == "--alpha" && i + 1 < argc) alpha_opt = (float)std::atof(argv[++i]);
        else if (arg == "--verbose") verbose = true;
        else if (arg == "--json") json_output = true;
        else if (arg == "--limit" && i + 1 < argc) limit = std::atoi(argv[++i]);
        else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            usage(argv[0]);
            return 1;
        }
    }

    if (encoder_path.empty() || vocab_path.empty() || layout_path.empty() || dataset_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    std::vector<EvalSample> samples;
    if (!load_dataset(dataset_path, samples)) return 1;
    fprintf(stderr, "Loaded %zu samples from %s\n", samples.size(), dataset_path.c_str());

    if (limit > 0 && limit < static_cast<int>(samples.size())) {
        samples.resize(limit);
        fprintf(stderr, "  Limited to %d samples\n", limit);
    }

    SwipeEngine engine;
    SwipeEngine::Config config;
    config.encoder_path = encoder_path;
    config.decoder_path = decoder_path;
    
    auto dict = *load_trie_simple(vocab_path);
    config.dictionaries = { dict.itrie() };
    load_layout_json(layout_path, config.layout);
    

    config.num_threads = threads;
    config.beam_width = beam_width;
    config.top_k = top_k;
    config.lm_model_path = lm_model_path;
    config.lm_vocab_path = lm_vocab_path.empty() ? vocab_path : lm_vocab_path;
    config.gamma = gamma_opt;
    config.lambda_ = lambda_opt;
    config.beta = beta_opt;
    config.alpha = alpha_opt;

    fprintf(stderr, "Initializing engine...\n");
    if (!engine.init(config)) {
        fprintf(stderr, "Error: Failed to initialize engine\n");
        return 1;
    }
    fprintf(stderr, "  beam_width=%d top_k=%d decoder=%s lm=%s\n",
            beam_width, top_k,
            engine.hasDecoder() ? "yes" : "no",
            !lm_model_path.empty() ? "yes" : "no");

    int top1_correct = 0;
    int top4_correct = 0;
    int total = static_cast<int>(samples.size());
    double total_time_ms = 0;

    struct LenBucket {
        int total = 0;
        int top1 = 0;
        int top4 = 0;
    };
    std::vector<LenBucket> by_length(20);

    fprintf(stderr, "Evaluating %d samples...\n", total);

    for (int i = 0; i < total; ++i) {
        const auto& sample = samples[i];

        auto t0 = Clock::now();
        auto results = engine.recognize(
            sample.x.data(), sample.y.data(), sample.t.data(),
            static_cast<int>(sample.x.size())
        );
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_time_ms += ms;

        bool in_top1 = false;
        bool in_top4 = false;

        if (!results.empty()) {
            if (results[0].word == sample.word) {
                in_top1 = true;
                in_top4 = true;
            } else {
                int check_k = std::min(top_k, static_cast<int>(results.size()));
                for (int j = 0; j < check_k; ++j) {
                    if (results[j].word == sample.word) {
                        in_top4 = true;
                        break;
                    }
                }
            }
        }

        if (in_top1) top1_correct++;
        if (in_top4) top4_correct++;

        size_t wlen = sample.word.size();
        if (wlen > 0 && wlen < by_length.size()) {
            by_length[wlen].total++;
            if (in_top1) by_length[wlen].top1++;
            if (in_top4) by_length[wlen].top4++;
        }

        if (verbose) {
            std::string top_words;
            int show_k = std::min(top_k, static_cast<int>(results.size()));
            for (int j = 0; j < show_k; ++j) {
                if (j > 0) top_words += ", ";
                top_words += results[j].word;
                char buf[32];
                snprintf(buf, sizeof(buf), "(%.2f)", results[j].final_score);
                top_words += buf;
            }
            const char* status = in_top1 ? "TOP1" : (in_top4 ? "TOP4" : "MISS");
            printf("[%4d] %-4s  %-20s -> [%s]  %.1f ms\n",
                   i, status, sample.word.c_str(), top_words.c_str(), ms);
        }

        if (!verbose && (i + 1) % 100 == 0) {
            fprintf(stderr, "  %d/%d (top1=%.1f%% top4=%.1f%%)\r",
                    i + 1, total,
                    100.0 * top1_correct / (i + 1),
                    100.0 * top4_correct / (i + 1));
        }
    }
    if (!verbose) fprintf(stderr, "\n");

    double top1_pct = 100.0 * top1_correct / total;
    double top4_pct = 100.0 * top4_correct / total;
    double avg_ms = total_time_ms / total;

    if (json_output) {
        printf("{\n");
        printf("  \"dataset\": \"%s\",\n", dataset_path.c_str());
        printf("  \"samples\": %d,\n", total);
        printf("  \"top1_correct\": %d,\n", top1_correct);
        printf("  \"top4_correct\": %d,\n", top4_correct);
        printf("  \"top1_accuracy\": %.4f,\n", top1_pct / 100.0);
        printf("  \"top4_accuracy\": %.4f,\n", top4_pct / 100.0);
        printf("  \"avg_inference_ms\": %.3f,\n", avg_ms);
        printf("  \"total_time_s\": %.1f,\n", total_time_ms / 1000.0);
        printf("  \"config\": {\n");
        printf("    \"beam_width\": %d,\n", beam_width);
        printf("    \"top_k\": %d,\n", top_k);
        printf("    \"has_decoder\": %s,\n", engine.hasDecoder() ? "true" : "false");
        printf("    \"expansion\": %s\n", use_expansion ? "true" : "false");
        printf("  }\n");
        printf("}\n");
    } else {
        printf("\nAccuracy Evaluation\n");
        printf("===================\n");
        printf("Dataset:      %s (%d samples)\n", dataset_path.c_str(), total);
        printf("Encoder:      %s\n", encoder_path.c_str());
        if (!decoder_path.empty()) {
            printf("Decoder:      %s\n", decoder_path.c_str());
        }
        if (!lm_model_path.empty()) {
            printf("LM:           %s\n", lm_model_path.c_str());
        }
        printf("Beam width:   %d\n", beam_width);
        if (use_expansion) printf("Expansion:    enabled\n");
        printf("\n");
        printf("  Top-1 accuracy:  %d / %d  = %.2f%%\n", top1_correct, total, top1_pct);
        printf("  Top-4 accuracy:  %d / %d  = %.2f%%\n", top4_correct, total, top4_pct);
        printf("\n");
        printf("  Avg inference:   %.3f ms\n", avg_ms);
        printf("  Total time:      %.1f s\n", total_time_ms / 1000.0);

        printf("\nAccuracy by word length:\n");
        printf("  Len   Count   Top-1    Top-4\n");
        printf("  ---  ------  ------   ------\n");
        for (size_t len = 2; len < by_length.size(); ++len) {
            const auto& b = by_length[len];
            if (b.total == 0) continue;
            printf("  %3zu   %5d   %5.1f%%   %5.1f%%\n",
                   len, b.total,
                   100.0 * b.top1 / b.total,
                   100.0 * b.top4 / b.total);
        }
    }

    return 0;
}
