#include <sstream>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sys/stat.h>
#include <microhttpd.h>
#include <mutex>
#include <cstring>
#include <cmath>
#include "swipe_decoder/engine.hpp"
#include "swipe_decoder/layout.hpp"

using namespace swipe_decoder;

static SwipeEngine engine;
static int g_top_k = 4;
static int g_beam_width = 100;
static std::mutex engine_mutex;


static std::vector<std::string> split(const std::string& s) {
    std::istringstream iss(s);
    return std::vector<std::string>(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>());
}

struct ParseResult {
    std::vector<float> x_coords;
    std::vector<float> y_coords;
    std::vector<float> timestamps;
    std::vector<std::string> context;
    std::string error;
};

// Body is in this format:

// sentence context
// [x.xx; y.yy]t.tt
// [x.xx; y.yy]t.tt
// ... repeating

static ParseResult parse_swipe_data(const std::string& body) {
    ParseResult result;
    std::istringstream request(body);
    std::string context_line;
    
    if (!std::getline(request, context_line)) {
        result.error = "Empty request body";
        return result;
    }
    
    result.context = split(context_line);
    
    std::string line;
    float first_time = -1.0f;
    constexpr size_t MAX_POINTS = 2000;
    
    while (std::getline(request, line)) {
        if (line.empty()) continue;
        if (line.front() != '[') {
            result.error = "Malformed line: expected '['";
            return result;
        }
        
        size_t semi = line.find(';');
        size_t brack = line.find(']');
        if (semi == std::string::npos || brack == std::string::npos || brack <= semi) {
            result.error = "Malformed line: expected '[x; y]t' format";
            return result;
        }
        
        try {
            float x = std::stof(line.substr(1, semi - 1));
            float y = std::stof(line.substr(semi + 1, brack - semi - 1));
            float t = std::stof(line.substr(brack + 1));

            if (!std::isfinite(x) || !std::isfinite(y) || x < -1.0f || x > 2.0f || y < -1.0f || y > 2.0f) {
                result.error = "Expecting normalized coordinates";
                return result;
            }

            if (!std::isfinite(t)) {
                result.error = "Invalid timestamp";
                return result;
            }
            
            if (first_time == -1.0f) first_time = t;
            
            result.x_coords.push_back(x);
            result.y_coords.push_back(y);
            result.timestamps.push_back((t - first_time));
        } catch (const std::exception&) {
            result.error = "Invalid numeric value in line";
            return result;
        }
        
        if (result.x_coords.size() > MAX_POINTS) {
            result.error = "Too many points";
            return result;
        }
    }
    
    if (result.x_coords.size() < 2) {
        result.error = "Need at least 2 points";
        return result;
    }

    float duration = result.timestamps.back() - result.timestamps.front();
    if (!std::isfinite(duration) || duration > 30000.0f) {
        result.error = "Swipe length is more than 30 seconds";
        return result;
    }
    
    return result;
}


struct ConnData {
    std::string buffer;
};


static void request_completed(void* cls, MHD_Connection* conn,
                              void** con_cls,
                              enum MHD_RequestTerminationCode toe) {
    ConnData* data = static_cast<ConnData*>(*con_cls);
    delete data;
    *con_cls = nullptr;
}

static MHD_Result send_error(MHD_Connection* conn, unsigned int status, const char* msg) {
    MHD_Response* r = MHD_create_response_from_buffer(std::strlen(msg), (void*)msg, MHD_RESPMEM_MUST_COPY);
    if (!r) return MHD_NO;
    MHD_add_response_header(r, "Content-Type", "text/plain");
    MHD_add_response_header(r, "Access-Control-Allow-Origin", "*");
    MHD_add_response_header(r, "Access-Control-Allow-Methods", "*");
    MHD_add_response_header(r, "Access-Control-Allow-Headers", "*");
    MHD_Result ret = MHD_queue_response(conn, status, r);
    MHD_destroy_response(r);
    return ret;
}

static MHD_Result answer(void* cls, MHD_Connection* conn, const char* url, const char* method,
                  const char* version, const char* upload_data, size_t* upload_data_size, void** con_cls) {
    if (strcmp(method, "OPTIONS") == 0 && strcmp(url, "/recognize") == 0) {
        struct MHD_Response *response = MHD_create_response_from_buffer(0, nullptr, MHD_RESPMEM_PERSISTENT);
        MHD_add_response_header(response, "Access-Control-Allow-Origin", "*");
        MHD_add_response_header(response, "Access-Control-Allow-Methods", "*");
        MHD_add_response_header(response, "Access-Control-Allow-Headers", "*");
        MHD_add_response_header(response, "Access-Control-Max-Age", "86400");
        MHD_Result ret = MHD_queue_response(conn, MHD_HTTP_NO_CONTENT, response);
        MHD_destroy_response(response);
        return ret;
    }

    if (strcmp(method, "POST") != 0 || strcmp(url, "/recognize") != 0) {
        return send_error(conn, MHD_HTTP_NOT_FOUND, "Not Found");
    }

    ConnData* data = static_cast<ConnData*>(*con_cls);
    if (!data) {
        data = new ConnData();
        *con_cls = data;
        return MHD_YES;
    }

    if (*upload_data_size) {
        constexpr size_t MAX_SIZE = 32000;
        if ((*upload_data_size > MAX_SIZE) || (data->buffer.size() + *upload_data_size > MAX_SIZE)) {
            return send_error(conn, MHD_HTTP_BAD_REQUEST, "Payload Too Large");
        }
        data->buffer.append(upload_data, *upload_data_size);
        *upload_data_size = 0;
        return MHD_YES;
    }

    ParseResult parsed = parse_swipe_data(data->buffer);
    if (!parsed.error.empty()) {
        std::cerr << "Parse error: " << parsed.error << "\n";
        return send_error(conn, MHD_HTTP_BAD_REQUEST, "Invalid request format");
    }

    try {
        // SwipeEngine is not thread-safe: serialize access
        auto results = [&]() {
            std::lock_guard<std::mutex> lock(engine_mutex);
            return engine.recognize(parsed.x_coords.data(), parsed.y_coords.data(), parsed.timestamps.data(),
                                    parsed.x_coords.size(), parsed.context, g_top_k, g_beam_width);
        }();

        std::ostringstream out;
        for (size_t i = 0; i < results.size(); ++i) {
            if (i) out << ",";
            out << results[i].word;
        }

        const std::string& out_str = out.str();
        MHD_Response* response = MHD_create_response_from_buffer(out_str.size(),
                                                                (void*)out_str.c_str(), MHD_RESPMEM_MUST_COPY);
        MHD_add_response_header(response, "Content-Type", "text/plain");
        MHD_add_response_header(response, "Access-Control-Allow-Origin", "*");
        MHD_add_response_header(response, "Access-Control-Allow-Methods", "*");
        MHD_add_response_header(response, "Access-Control-Allow-Headers", "*");
        MHD_Result ret = MHD_queue_response(conn, MHD_HTTP_OK, response);
        MHD_destroy_response(response);
        return ret;
    } catch (const std::exception& e) {
        std::cerr << "Engine error: " << e.what() << "\n";
        return send_error(conn, MHD_HTTP_INTERNAL_SERVER_ERROR, "Engine error");
    } catch (...) {
        std::cerr << "Unknown engine error\n";
        return send_error(conn, MHD_HTTP_INTERNAL_SERVER_ERROR, "Internal Server Error");
    }
}

int main(int argc, char* argv[]) {
    std::string encoder_path, decoder_path, vocab_path, layout_path;
    std::string lm_model_path, lm_vocab_path;
    int beam_width = 100, top_k = 4, num_threads = 1;
    std::string freq_key = "f";
    bool use_expansion = true;
    std::optional<float> gamma_opt, lambda_opt, beta_opt;
    int port = 8012;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--encoder" && i + 1 < argc) encoder_path = argv[++i];
        else if (arg == "--decoder" && i + 1 < argc) decoder_path = argv[++i];
        else if (arg == "--layout" && i + 1 < argc) layout_path = argv[++i];
        else if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
            if (port <= 0 || port > 65535) {
                std::cerr << "Invalid port number\n";
                return 1;
            }
        }
        else if (arg == "--dict" && i + 1 < argc) vocab_path = argv[++i];
        else if (arg == "--lm-model" && i + 1 < argc) lm_model_path = argv[++i];
        else if (arg == "--lm-vocab" && i + 1 < argc) lm_vocab_path = argv[++i];
        else if (arg == "--beam-width" && i + 1 < argc) beam_width = std::atoi(argv[++i]);
        else if (arg == "--top-k" && i + 1 < argc) top_k = std::atoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) num_threads = std::atoi(argv[++i]);
        else if (arg == "--freq-key" && i + 1 < argc) freq_key = argv[++i];
        else if (arg == "--expansion") use_expansion = true;
        else if (arg == "--no-expansion") use_expansion = false;
        else if (arg == "--gamma" && i + 1 < argc) gamma_opt = (float)std::atof(argv[++i]);
        else if (arg == "--lambda" && i + 1 < argc) lambda_opt = (float)std::atof(argv[++i]);
        else if (arg == "--beta" && i + 1 < argc) beta_opt = (float)std::atof(argv[++i]);
    }

    if (encoder_path.empty() || vocab_path.empty()) {
        std::cerr << "Provide --encoder and --dict\n";
        return 1;
    }

    SwipeEngine::Config config;
    config.encoder_path = encoder_path;
    config.decoder_path = decoder_path;

    auto dict_opt = swipe_decoder::load_trie_simple(vocab_path);
    if (!dict_opt) {
        std::cerr << "Failed to load dictionary\n";
        return 1;
    }
    auto dict = *std::move(dict_opt);
    config.dictionaries = { dict.itrie() };

    if (layout_path.empty()) {
        config.layout = default_en_qwerty();
    } else {
        if (!swipe_decoder::load_layout_json(layout_path, config.layout)) {
            std::cerr << "Failed to load layout\n";
            return 1;
        }
    }

    config.num_threads = num_threads;
    config.beam_width = beam_width;
    config.top_k = top_k;
    config.lm_model_path = lm_model_path;
    config.lm_vocab_path = lm_vocab_path;
    config.gamma = gamma_opt;
    config.lambda_ = lambda_opt;
    config.beta = beta_opt;

    if (!engine.init(config)) {
        std::cerr << "Failed to initialize engine\n";
        return 1;
    }

    g_top_k = top_k;
    g_beam_width = beam_width;

    MHD_Daemon* daemon = MHD_start_daemon(MHD_USE_SELECT_INTERNALLY, port, nullptr, nullptr,
                                          &answer, nullptr,
                                          MHD_OPTION_NOTIFY_COMPLETED, &request_completed, nullptr,
                                          MHD_OPTION_END);
    if (!daemon) {
        std::cerr << "Failed to start server\n";
        return 1;
    }

    std::cout << "Listening on port " << port << "\n";
    pause();
    return 0;
}