/**
 * End-to-end inference benchmark for the universal-swipe pipeline.
 *
 * Measures each pipeline stage from raw swipe path to decoded words:
 *   1. Resampling       (variable-length path -> [2, T])
 *   2. Encoder          (log_emissions + coefficients + lambda)
 *   3. Decoder (opt.)   (refined [T', K+1] log-probs)
 *   4. Beam search      (top-k words)
 *   5. Context LM (opt.)
 *
 * Usage:
 *   ./benchmark_e2e --encoder <.pte> --vocab <file> --layout <.json> [OPTIONS]
 *
 * Options:
 *   --encoder PATH      Required - encoder .pte
 *   --decoder PATH      Optional - paired decoder .pte
 *   --vocab PATH        Required - vocab file
 *   --layout PATH       Required - layout JSON
 *   --beam-width N      (default: 100)
 *   --top-k N           (default: 4)
 *   --warmup N          (default: 50)
 *   --runs N            (default: 500)
 *   --threads N         (default: 1)
 *   --paths FILE        Load real swipe paths from binary file
 *   --path-points N     Synthetic swipe points if no --paths (default: 80)
 *   --lm-model PATH     Context LM .pte
 *   --lm-vocab PATH     Context LM vocab
 *   --json              JSON output
 *   --dump-times        CSV output per run
 *   --affinity MODE     big|little
 */

#include "swipe_decoder/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifdef __linux__
#include <sched.h>
#include <sys/auxv.h>
#endif

using namespace swipe_decoder;
using Clock = std::chrono::high_resolution_clock;

#ifdef __linux__
// Detect big/little cores by max frequency and pin the calling thread.
static int pin_to_cores(const std::string& affinity) {
    if (affinity != "big" && affinity != "little") return 0;

    struct CoreInfo { int id; long max_freq; };
    std::vector<CoreInfo> cores;
    for (int i = 0; i < CPU_SETSIZE && i < 64; ++i) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        FILE* f = fopen(path, "r");
        if (!f) { if (i < 16) continue; break; }
        long freq = 0;
        if (fscanf(f, "%ld", &freq) == 1 && freq > 0) cores.push_back({i, freq});
        fclose(f);
    }
    if (cores.empty()) return 0;

    long min_freq = cores[0].max_freq;
    long max_freq = cores[0].max_freq;
    for (auto& c : cores) {
        min_freq = std::min(min_freq, c.max_freq);
        max_freq = std::max(max_freq, c.max_freq);
    }
    if (min_freq == max_freq) {
        fprintf(stderr, "Note: all cores have the same max frequency, "
                "pinning to all %zu cores\n", cores.size());
    }

    cpu_set_t mask;
    CPU_ZERO(&mask);
    int count = 0;
    for (auto& c : cores) {
        bool is_little = (c.max_freq == min_freq);
        bool select = (min_freq == max_freq)
            || (affinity == "big" && !is_little)
            || (affinity == "little" && is_little);
        if (select) { CPU_SET(c.id, &mask); ++count; }
    }
    if (count == 0) return 0;
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
        fprintf(stderr, "Warning: sched_setaffinity failed\n");
        return 0;
    }

    cpu_set_t actual;
    CPU_ZERO(&actual);
    sched_getaffinity(0, sizeof(actual), &actual);
    int actual_count = 0;
    for (auto& c : cores) if (CPU_ISSET(c.id, &actual)) ++actual_count;
    if (actual_count != count) {
        fprintf(stderr, "Warning: requested %d cores but got %d "
                "(cgroup cpuset may be restricting affinity)\n",
                count, actual_count);
    }

#ifdef __aarch64__
    unsigned long hwcap = getauxval(AT_HWCAP);
    if (affinity == "little") {
        bool has_sve = (hwcap & (1UL << 22));
        bool has_sve2 = false;
#ifdef AT_HWCAP2
        unsigned long hwcap2 = getauxval(AT_HWCAP2);
        has_sve2 = (hwcap2 & (1UL << 1));
#endif
        if (has_sve || has_sve2) {
            fprintf(stderr, "Warning: HWCAP reports SVE%s but pinned to little cores\n",
                    has_sve2 ? "2" : "");
        }
    }
#endif

    return count;
}
#endif

static double get_rss_mb() {
    std::ifstream status("/proc/self/status");
    if (!status) return 0.0;
    std::string line;
    while (std::getline(status, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            long kb = 0;
            sscanf(line.c_str() + 6, "%ld", &kb);
            return kb / 1024.0;
        }
    }
    return 0.0;
}

struct Stats {
    double mean;
    double min_val;
    double p50;
    double p99;
    double max_val;
};

static Stats compute_stats(std::vector<double>& times) {
    if (times.empty()) return {0, 0, 0, 0, 0};
    std::sort(times.begin(), times.end());
    size_t n = times.size();
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    return {
        sum / n,
        times[0],
        times[n / 2],
        times[std::min(static_cast<size_t>(n * 0.99), n - 1)],
        times[n - 1],
    };
}

static void generate_swipe_path(
    std::mt19937& rng, int num_points,
    std::vector<float>& x, std::vector<float>& y, std::vector<float>& t
) {
    x.resize(num_points); y.resize(num_points); t.resize(num_points);
    std::uniform_real_distribution<float> pos_dist(0.1f, 0.9f);
    std::normal_distribution<float> noise(0.0f, 0.005f);
    float x0 = pos_dist(rng), y0 = pos_dist(rng);
    float x1 = pos_dist(rng), y1 = pos_dist(rng);
    float xm = (x0 + x1) / 2.0f + pos_dist(rng) * 0.2f - 0.1f;
    float ym = (y0 + y1) / 2.0f + pos_dist(rng) * 0.2f - 0.1f;
    for (int i = 0; i < num_points; ++i) {
        float frac = static_cast<float>(i) / (num_points - 1);
        float one_minus = 1.0f - frac;
        x[i] = one_minus * one_minus * x0 + 2.0f * one_minus * frac * xm + frac * frac * x1 + noise(rng);
        y[i] = one_minus * one_minus * y0 + 2.0f * one_minus * frac * ym + frac * frac * y1 + noise(rng);
        t[i] = static_cast<float>(i) * 8.0f;
        x[i] = std::max(0.0f, std::min(1.0f, x[i]));
        y[i] = std::max(0.0f, std::min(1.0f, y[i]));
    }
}

static void print_stage(const char* name, const Stats& s, const char* unit) {
    printf("  %-14s  %8.3f %s  %8.3f %s\n", name, s.p50, unit, s.p99, unit);
}

static void print_usage(const char* argv0) {
    fprintf(stderr, "Usage: %s --encoder <.pte> --vocab <file> --layout <.json> [OPTIONS]\n", argv0);
    fprintf(stderr, "\nRequired:\n");
    fprintf(stderr, "  --encoder PATH    Encoder .pte (honorable_sturgeon)\n");
    fprintf(stderr, "  --vocab PATH      Vocabulary file (.combined or plain)\n");
    fprintf(stderr, "  --layout PATH     Keyboard layout JSON\n");
    fprintf(stderr, "\nOptional:\n");
    fprintf(stderr, "  --decoder PATH    Paired decoder .pte\n");
    fprintf(stderr, "  --beam-width N    (default: 100)\n");
    fprintf(stderr, "  --top-k N         (default: 4)\n");
    fprintf(stderr, "  --warmup N        (default: 50)\n");
    fprintf(stderr, "  --runs N          (default: 500)\n");
    fprintf(stderr, "  --threads N       (default: 1)\n");
    fprintf(stderr, "  --paths FILE      Load real swipe paths from binary file\n");
    fprintf(stderr, "  --path-points N   Synthetic points if no --paths (default: 80)\n");
    fprintf(stderr, "  --lm-model PATH   Context LM .pte\n");
    fprintf(stderr, "  --lm-vocab PATH   Context LM vocab\n");
    fprintf(stderr, "  --json            JSON output\n");
    fprintf(stderr, "  --dump-times      CSV per-run output\n");
    fprintf(stderr, "  --affinity MODE   big|little\n");
}

int main(int argc, char* argv[]) {
    std::string encoder_path, decoder_path, vocab_path, layout_path;
    int beam_width = 300, top_k = 4, warmup = 50, runs = 500;
    int path_points = 80, threads = 1;
    bool json_output = false, dump_times = false;
    std::string paths_file;
    std::string lm_model_path, lm_vocab_path;
    std::string affinity;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto has_next = [&]() { return i + 1 < argc; };
        if      (arg == "--encoder"    && has_next()) encoder_path  = argv[++i];
        else if (arg == "--decoder"    && has_next()) decoder_path  = argv[++i];
        else if (arg == "--vocab"      && has_next()) vocab_path    = argv[++i];
        else if (arg == "--layout"     && has_next()) layout_path   = argv[++i];
        else if (arg == "--beam-width" && has_next()) beam_width    = std::atoi(argv[++i]);
        else if (arg == "--top-k"      && has_next()) top_k         = std::atoi(argv[++i]);
        else if (arg == "--warmup"     && has_next()) warmup        = std::atoi(argv[++i]);
        else if (arg == "--runs"       && has_next()) runs          = std::atoi(argv[++i]);
        else if (arg == "--threads"    && has_next()) threads       = std::atoi(argv[++i]);
        else if (arg == "--path-points"&& has_next()) path_points   = std::atoi(argv[++i]);
        else if (arg == "--paths"      && has_next()) paths_file    = argv[++i];
        else if (arg == "--lm-model"   && has_next()) lm_model_path = argv[++i];
        else if (arg == "--lm-vocab"   && has_next()) lm_vocab_path = argv[++i];
        else if (arg == "--affinity"   && has_next()) affinity      = argv[++i];
        else if (arg == "--json")        json_output = true;
        else if (arg == "--dump-times")  dump_times = true;
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
    }

    if (encoder_path.empty() || vocab_path.empty() || layout_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stderr, "Initializing pipeline\n");
    fprintf(stderr, "  Encoder: %s\n", encoder_path.c_str());
    if (!decoder_path.empty()) fprintf(stderr, "  Decoder: %s\n", decoder_path.c_str());
    fprintf(stderr, "  Vocab:   %s\n", vocab_path.c_str());
    fprintf(stderr, "  Layout:  %s\n", layout_path.c_str());

    auto init_start = Clock::now();
    SwipeEngine engine;
    SwipeEngine::Config config;

    auto dict = *load_trie_simple(vocab_path);
    config.dictionaries = { dict.itrie() };
    load_layout_json(layout_path, config.layout);

    config.encoder_path = encoder_path;
    config.decoder_path = decoder_path;
    config.num_threads  = threads;
    config.beam_width   = beam_width;
    config.top_k        = top_k;
    config.lm_model_path = lm_model_path;
    config.lm_vocab_path = lm_vocab_path.empty() ? vocab_path : lm_vocab_path;
    if (!engine.init(config)) {
        fprintf(stderr, "Error: engine.init() failed\n");
        return 1;
    }
    auto init_end = Clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(init_end - init_start).count();
    double rss_after_init = get_rss_mb();

#ifdef __linux__
    // Switch to the target affinity for warmup + benchmarking.
    if (!affinity.empty() && affinity != "none" && affinity != "big") {
        int n = pin_to_cores(affinity);
        if (n > 0) fprintf(stderr, "Bench: pinned to %d %s cores\n", n, affinity.c_str());
        else       fprintf(stderr, "Warning: failed to pin to %s cores\n", affinity.c_str());
    }
#endif

    fprintf(stderr, "  Threads:   %d\n", threads);
    fprintf(stderr, "  hasDecoder: %s\n", engine.hasDecoder() ? "yes" : "no");
    fprintf(stderr, "  Init:      %.1f ms\n", init_ms);
    fprintf(stderr, "  RSS:       %.1f MB\n", rss_after_init);

    bool have_lm = !lm_model_path.empty();

    // Load or generate swipe paths.
    struct SwipePath { std::vector<float> x, y, t; };
    std::vector<SwipePath> paths;
    std::string path_source;

    if (!paths_file.empty()) {
        std::ifstream f(paths_file, std::ios::binary);
        if (!f) { fprintf(stderr, "Error: cannot open paths file\n"); return 1; }
        uint32_t num_paths;
        f.read(reinterpret_cast<char*>(&num_paths), 4);
        paths.resize(num_paths);
        for (uint32_t i = 0; i < num_paths; ++i) {
            uint32_t n;
            f.read(reinterpret_cast<char*>(&n), 4);
            paths[i].x.resize(n); paths[i].y.resize(n); paths[i].t.resize(n);
            f.read(reinterpret_cast<char*>(paths[i].x.data()), n * 4);
            f.read(reinterpret_cast<char*>(paths[i].y.data()), n * 4);
            f.read(reinterpret_cast<char*>(paths[i].t.data()), n * 4);
        }
        path_source = "real (" + paths_file + ")";
        fprintf(stderr, "Loaded %u real swipe paths\n", num_paths);
        if (warmup + runs > static_cast<int>(paths.size())) {
            runs = std::max(1, static_cast<int>(paths.size()) - warmup);
            fprintf(stderr, "  Adjusted runs to %d\n", runs);
        }
    } else {
        int total = warmup + runs;
        fprintf(stderr, "Generating %d synthetic paths (%d points each)...\n", total, path_points);
        std::mt19937 rng(42);
        paths.resize(total);
        for (auto& p : paths) {
            std::uniform_int_distribution<int> len_dist(
                std::max(10, path_points - 20), path_points + 20);
            generate_swipe_path(rng, len_dist(rng), p.x, p.y, p.t);
        }
        path_source = "synthetic";
    }

    std::vector<std::string> lm_context = {"the", "quick", "brown"};

    std::vector<double> t_resample(runs), t_encoder(runs), t_decoder(runs);
    std::vector<double> t_beam(runs), t_lm(runs), t_total(runs);

    fprintf(stderr, "Warming up (%d runs)...\n", warmup);
    for (int i = 0; i < warmup; ++i) {
        auto& p = paths[i];
        engine.recognize(p.x.data(), p.y.data(), p.t.data(),
                         static_cast<int>(p.x.size()),
                         have_lm ? lm_context : std::vector<std::string>{}, top_k);
    }

    fprintf(stderr, "Benchmarking (%d runs)...\n", runs);
    for (int i = 0; i < runs; ++i) {
        auto& p = paths[warmup + i];
        engine.recognize(p.x.data(), p.y.data(), p.t.data(),
                         static_cast<int>(p.x.size()),
                         have_lm ? lm_context : std::vector<std::string>{}, top_k);
        const auto& tm = engine.last_timing();
        t_resample[i] = tm.resample_us;
        t_encoder[i]  = tm.encoder_us;
        t_decoder[i]  = tm.decoder_us;
        t_beam[i]     = tm.beam_us;
        t_lm[i]       = tm.lm_us;
        t_total[i]    = tm.total_us;
    }

    double rss_after_bench = get_rss_mb();

    if (dump_times) {
        printf("run,resample_us,encoder_us,decoder_us,beam_us,lm_us,total_us\n");
        for (int i = 0; i < runs; ++i) {
            printf("%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n", i,
                   t_resample[i], t_encoder[i], t_decoder[i], t_beam[i],
                   t_lm[i], t_total[i]);
        }
        return 0;
    }

    auto s_resample = compute_stats(t_resample);
    auto s_encoder  = compute_stats(t_encoder);
    auto s_decoder  = compute_stats(t_decoder);
    auto s_beam     = compute_stats(t_beam);
    auto s_lm       = compute_stats(t_lm);
    auto s_total    = compute_stats(t_total);

    if (json_output) {
        printf("{\n");
        printf("  \"encoder\": \"%s\",\n", encoder_path.c_str());
        if (!decoder_path.empty()) printf("  \"decoder\": \"%s\",\n", decoder_path.c_str());
        printf("  \"backend\": \"executorch+xnnpack\",\n");
        printf("  \"threads\": %d,\n", threads);
        printf("  \"beam_width\": %d,\n", beam_width);
        printf("  \"top_k\": %d,\n", top_k);
        printf("  \"warmup\": %d,\n", warmup);
        printf("  \"runs\": %d,\n", runs);
        printf("  \"has_decoder\": %s,\n", engine.hasDecoder() ? "true" : "false");
        printf("  \"has_lm\": %s,\n", have_lm ? "true" : "false");
        auto emit_stat = [](const char* key, const Stats& s, bool last) {
            printf("  \"%s\": { \"p50\": %.3f, \"p99\": %.3f, \"mean\": %.3f }%s\n",
                   key, s.p50, s.p99, s.mean, last ? "" : ",");
        };
        emit_stat("resample_us", s_resample, false);
        emit_stat("encoder_us",  s_encoder,  false);
        emit_stat("decoder_us",  s_decoder,  false);
        emit_stat("beam_us",     s_beam,     false);
        emit_stat("lm_us",       s_lm,       false);
        emit_stat("total_us",    s_total,    false);
        printf("  \"init_ms\": %.1f,\n", init_ms);
        printf("  \"rss_after_init_mb\": %.1f,\n", rss_after_init);
        printf("  \"rss_after_bench_mb\": %.1f\n", rss_after_bench);
        printf("}\n");
    } else {
        printf("End-to-End Inference Benchmark (ExecuTorch)\n");
        printf("============================================\n");
        printf("Encoder:      %s\n", encoder_path.c_str());
        if (!decoder_path.empty()) printf("Decoder:      %s\n", decoder_path.c_str());
        printf("Backend:      ExecuTorch + XNNPACK\n");
        printf("Threads:      %d\n", threads);
        printf("Beam width:   %d\n", beam_width);
        printf("Top-k:        %d\n", top_k);
        printf("Paths:        %s\n", path_source.c_str());
        printf("Warmup:       %d\n", warmup);
        printf("Runs:         %d\n", runs);
        if (have_lm) {
            printf("LM model:     %s\n", lm_model_path.c_str());
        }
        printf("\n");
        printf("  Stage              P50           P99\n");
        printf("  ──────────────  ──────────  ──────────\n");
        print_stage("Resample", s_resample, "us");
        print_stage("Encoder",  s_encoder,  "us");
        if (engine.hasDecoder())
            print_stage("Decoder", s_decoder, "us");
        print_stage("Beam search", s_beam, "us");
        if (have_lm)
            print_stage("LM scoring", s_lm, "us");
        printf("  ──────────────  ──────────  ──────────\n");
        printf("  %-14s  %8.3f ms  %8.3f ms\n", "TOTAL",
               s_total.p50 / 1000.0, s_total.p99 / 1000.0);
        printf("\n");
        printf("Memory & Init\n");
        printf("  Init time:      %8.1f ms\n", init_ms);
        printf("  RSS (init):     %8.1f MB\n", rss_after_init);
        printf("  RSS (+ bench):  %8.1f MB\n", rss_after_bench);
    }
    return 0;
}
