#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include "swipe_decoder/engine.hpp"
#include "swipe_decoder/layout.hpp"

using json = nlohmann::json;

std::string normalize_word(const std::string& word) {
    auto cps = swipe_decoder::utf8_to_codepoints(word);
    std::string result;
    result.reserve(word.size());
    for (int cp : cps) {
        int lower = swipe_decoder::unicode_tolower(cp);
        bool is_alpha = (lower >= 'a' && lower <= 'z')
                     || (lower >= 0x0430 && lower <= 0x044F)
                     || lower == 0x0451
                     || (lower > 0x7F && lower < 0xFFFD);
        if (is_alpha) {
            swipe_decoder::utf8_encode_one(lower, result);
        }
    }
    return result;
}

static void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " --encoder PATH --vocab PATH --layout PATH --jsonl PATH [OPTIONS]\n";
    std::cerr << "\nRequired:\n";
    std::cerr << "  --encoder PATH       Universal encoder .pte (honorable_sturgeon)\n";
    std::cerr << "  --vocab PATH         Combined vocab file\n";
    std::cerr << "  --layout PATH        Layout JSON file\n";
    std::cerr << "  --jsonl PATH         Input JSONL swipe paths\n";
    std::cerr << "\nOptional:\n";
    std::cerr << "  --decoder PATH       Paired decoder .pte (default: encoder-only)\n";
    std::cerr << "  --lm-model PATH      Context LM .pte\n";
    std::cerr << "  --lm-vocab PATH      Context LM vocab\n";
    std::cerr << "  --beam-width N       Beam width (default: 100)\n";
    std::cerr << "  --top-k N            Candidates per sample (default: 4)\n";
    std::cerr << "  --threads N          Encoder/decoder threads (default: 1)\n";
    std::cerr << "  --freq-key KEY       Frequency key in combined vocab (default: f)\n";
    std::cerr << "  --expansion          Enable surface-form expansion (default: on)\n";
    std::cerr << "  --no-expansion       Disable surface-form expansion\n";
    std::cerr << "  --skip-oov           Skip out of vocabulary words\n";
    std::cerr << "  --gamma F            Length-norm exponent override\n";
    std::cerr << "  --lambda F           Frequency bonus override\n";
    std::cerr << "  --beta F             Per-char length bonus override\n";
    std::cerr << "  --alpha F            Language model weight override\n";
}

int main(int argc, char* argv[]) {
    std::string encoder_path, decoder_path, vocab_path, layout_path, jsonl_path;
    std::string lm_model_path, lm_vocab_path;
    int beam_width = 100;
    int top_k = 4;
    int num_threads = 1;
    std::string freq_key = "f";
    bool use_expansion = true;
    bool skip_oov = false;
    std::optional<float> gamma_opt, lambda_opt, beta_opt, alpha_opt;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else if (arg == "--encoder" && i + 1 < argc) encoder_path = argv[++i];
        else if (arg == "--decoder" && i + 1 < argc) decoder_path = argv[++i];
        else if (arg == "--vocab" && i + 1 < argc) vocab_path = argv[++i];
        else if (arg == "--layout" && i + 1 < argc) layout_path = argv[++i];
        else if (arg == "--jsonl" && i + 1 < argc) jsonl_path = argv[++i];
        else if (arg == "--lm-model" && i + 1 < argc) lm_model_path = argv[++i];
        else if (arg == "--lm-vocab" && i + 1 < argc) lm_vocab_path = argv[++i];
        else if (arg == "--beam-width" && i + 1 < argc) beam_width = std::atoi(argv[++i]);
        else if (arg == "--top-k" && i + 1 < argc) top_k = std::atoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) num_threads = std::atoi(argv[++i]);
        else if (arg == "--freq-key" && i + 1 < argc) freq_key = argv[++i];
        else if (arg == "--expansion") use_expansion = true;
        else if (arg == "--no-expansion") use_expansion = false;
        else if (arg == "--skip-oov") skip_oov = true;
        else if (arg == "--gamma" && i + 1 < argc) gamma_opt = (float)std::atof(argv[++i]);
        else if (arg == "--lambda" && i + 1 < argc) lambda_opt = (float)std::atof(argv[++i]);
        else if (arg == "--beta" && i + 1 < argc) beta_opt = (float)std::atof(argv[++i]);
        else if (arg == "--alpha" && i + 1 < argc) alpha_opt = (float)std::atof(argv[++i]);
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    if (encoder_path.empty() || vocab_path.empty() || layout_path.empty() || jsonl_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    swipe_decoder::SwipeEngine engine;
    swipe_decoder::SwipeEngine::Config config;
    config.encoder_path = encoder_path;
    config.decoder_path = decoder_path;
    
    auto dict = *swipe_decoder::load_trie_simple(vocab_path);
    config.dictionaries = { dict.itrie() };
    swipe_decoder::load_layout_json(layout_path, config.layout);

    config.num_threads = num_threads;
    config.beam_width = beam_width;
    config.top_k = top_k;
    config.lm_model_path = lm_model_path;
    config.lm_vocab_path = lm_vocab_path;
    config.gamma = gamma_opt;
    config.lambda_ = lambda_opt;
    config.beta = beta_opt;
    config.alpha = alpha_opt;

    if (!engine.init(config)) {
        std::cerr << "Failed to initialize engine\n";
        return 1;
    }
    std::cerr << "Engine loaded: beam_width=" << beam_width
              << " top_k=" << top_k
              << " decoder=" << (engine.hasDecoder() ? "yes" : "no")
              << " lm=" << (!lm_model_path.empty() ? "yes" : "no") << "\n";

    std::ifstream file(jsonl_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << jsonl_path << "\n";
        return 1;
    }

    std::string line;
    int line_num = 0;
    int num_good = 0;
    int num_ok = 0;
    int num_bad = 0;

    double resample_us = 0.0;
    double model_us = 0.0;
    double beam_us = 0.0;
    double lm_us = 0.0;
    double total_us = 0.0;
    double num_iters = 0.0;

    while (std::getline(file, line)) {
        line_num++;
        try {
            auto j = json::parse(line);

            if (!j.contains("data")) {
                std::cerr << "Line " << line_num << ": missing 'data' field\n";
                continue;
            }

            const auto& data = j["data"];
            std::string tgt = normalize_word(j["word"].get<std::string>());
            if (tgt.length() <= 1) continue;

            if(skip_oov && !dict.contains(tgt)) {
                std::cout << "oov " << tgt << "\n";
                continue;
            }

            std::vector<swipe_decoder::SwipeSegOwned> left, right;            

            double t0;
            if(data.is_array()) {
                t0 = data[0]["t"].template get<double>();

                swipe_decoder::SwipeSegOwned s;
                for(const auto &point : data) {
                    if (!point.contains("x") || !point.contains("y") || !point.contains("t")) {
                        std::cerr << "Line " << line_num << ": invalid point format\n";
                        continue;
                    }
                    s.x.push_back(point["x"].template get<float>());
                    s.y.push_back(point["y"].template get<float>());

                    double t = point["t"].template get<double>();
                    s.t.push_back(t - t0);
                }
                if(s.x.size() <= 2) continue; 
                left.push_back(std::move(s));
            } else {     
                t0 = std::numeric_limits<double>::infinity();           
                if (data.contains("L") && data["L"].size() > 0)
                    t0 = data["L"][0][0]["t"].template get<double>();
                if (data.contains("R") && data["R"].size() > 0)
                    t0 = std::min(t0, data["R"][0][0]["t"].template get<double>());

                auto parse_segments = [&](const auto& arr, std::vector<swipe_decoder::SwipeSegOwned>& segs) {
                    for (const auto& seg : arr) {
                        swipe_decoder::SwipeSegOwned s;
                        for (const auto& point : seg) {
                            if (!point.contains("x") || !point.contains("y") || !point.contains("t")) {
                                std::cerr << "Line " << line_num << ": invalid point format\n";
                                continue;
                            }
                            s.x.push_back(point["x"].template get<float>());

                            // TODO: New jsonl files are scaled, maybe should fix this at jsonl-generation time.
                            // For now working around it in this target.
                            s.y.push_back(point["y"].template get<float>() * 1.401709402f);

                            double t = point["t"].template get<double>();
                            s.t.push_back(t - t0);
                        }
                        if (!s.x.empty()) segs.push_back(std::move(s));
                    }
                };

                if (data.contains("L")) parse_segments(data["L"], left);
                if (data.contains("R")) parse_segments(data["R"], right);
            }
            if (left.empty() && right.empty()) continue;

            std::vector<swipe_decoder::SwipeSeg> left_views, right_views;
            for(const auto &v : left)  left_views.emplace_back(v.view());
            for(const auto &v : right) right_views.emplace_back(v.view());

            // Run inference
            std::vector<std::string> context_words;

            if(j.contains("sentence") && j.contains("word_idx")) {
                std::string sentence = j["sentence"].get<std::string>();
                int word_idx = j["word_idx"].get<int>();

                std::istringstream iss(sentence);

                std::copy(std::istream_iterator<std::string>(iss),
                        std::istream_iterator<std::string>(),
                        std::back_inserter(context_words));
                context_words.resize(word_idx);
            }
            
            auto results = engine.recognize_multi(
                left_views, right_views,
                context_words,
                top_k, beam_width
            );

            auto timing = engine.last_timing();
            num_iters += 1;
            resample_us += timing.resample_us;
            model_us += timing.encoder_us;
            model_us += timing.decoder_us;
            beam_us += timing.beam_us;
            lm_us += timing.lm_us;
            total_us += timing.total_us;

            // Output results
            int idx_matched = std::numeric_limits<int>::max();
            int i = 0;
            for (const auto& word : results) {
                std::string w = normalize_word(word.word);
                if (w == tgt) {
                    idx_matched = i;
                    break;
                }
                i++;
            }

            if (idx_matched == 0) {
                std::cout << "good";
                num_good++;
            } else if (idx_matched <= 3) {
                std::cout << "ok";
                num_ok++;
            } else {
                std::cout << "bad";
                num_bad++;
            }
            std::cout << " ";
            std::cout << j["id"].get<int>();
            std::cout << " ";
            std::cout << tgt;
            std::cout << ":";
            if (!results.empty()) {
                std::cout << normalize_word(results[0].word);
            }
            std::cout << ":";
            if (idx_matched == 0) {
                std::cout << "[]";
            } else {
                std::cout << "[";
                int i = 0;
                for (const auto& word : results) {
                    if (i++ == 0) continue;
                    std::cout << "'" << normalize_word(word.word) << "'";
                    if (i < static_cast<int>(results.size())) {
                        std::cout << ", ";
                    } else {
                        break;
                    }
                }
                std::cout << "]";
            }
            std::cout << "\n";
        } catch (const json::parse_error& e) {
            std::cerr << "Line " << line_num << ": JSON parse error: " << e.what() << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Line " << line_num << ": error: " << e.what() << "\n";
        }
    }

    int num_total = num_good + num_ok + num_bad;
    if (num_total > 0) {
        float top1 = (num_good * 100.0f) / num_total;
        float top4 = ((num_good + num_ok) * 100.0f) / num_total;
        float fail = (num_bad * 100.0f) / num_total;

        std::cout << "\n";
        std::cout << "Results: " << num_total << " samples\n";
        std::cout << "  top-1:  " << num_good << " / " << num_total << " = " << top1 << "%\n";
        std::cout << "  top-4:  " << (num_good + num_ok) << " / " << num_total << " = " << top4 << "%\n";
        std::cout << "  miss:   " << num_bad << " / " << num_total << " = " << fail << "%\n";

        std::cout << "\n";
        std::cout << "Timings (avg µs per sample):\n";
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  resample: " << (num_iters ? resample_us / num_iters : 0.0) << "\n";
        std::cout << "  model:    " << (num_iters ? model_us    / num_iters : 0.0) << "\n";
        std::cout << "  beam:     " << (num_iters ? beam_us     / num_iters : 0.0) << "\n";
        std::cout << "  lm:       " << (num_iters ? lm_us       / num_iters : 0.0) << "\n";
        std::cout << "  total:    " << (num_iters ? total_us    / num_iters : 0.0) << "\n";
    }

    return 0;
}
