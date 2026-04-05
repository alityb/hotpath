#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace hotpath {

struct ReplayRequest {
    std::string prompt;       // plain text (for /v1/completions)
    std::string messages_json; // raw JSON array string (for /v1/chat/completions)
    int max_tokens = 256;
    std::string model;
};

struct ReplayResult {
    std::string request_id;
    int64_t send_us = 0;
    int64_t first_token_us = 0;
    int64_t completion_us = 0;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    bool prompt_tokens_estimated = false;
    int tokens_generated = 0;
    bool success = false;
    std::string error;
};

struct ReplayConfig {
    std::string endpoint = "http://localhost:8000";
    int max_concurrency = 16;
    double rate_limit_rps = 0.0;  // 0 = unlimited
    std::string model;
    // Optional progress callback: called after every request completes.
    // Args: (done, total, ok_count, fail_count)
    std::function<void(int, int, int, int)> on_request_done;
};

std::vector<ReplayRequest> load_jsonl(const std::filesystem::path& path);
std::vector<ReplayRequest> load_sharegpt(const std::filesystem::path& path);

// Build the HTTP request body JSON for a given request
std::string build_request_body(const ReplayRequest& req, const std::string& model);

// Returns "/v1/chat/completions" for messages-format requests, "/v1/completions" otherwise
std::string api_path_for(const ReplayRequest& req);

// Send requests to the endpoint, return per-request timing results
std::vector<ReplayResult> replay_traffic(const std::vector<ReplayRequest>& requests,
                                         const ReplayConfig& config);

// Parse the first model id from an OpenAI-compatible /v1/models JSON response
std::string parse_model_from_models_response(const std::string& response);

// Try to detect model name from a running vLLM/SGLang endpoint
std::string detect_model(const std::string& endpoint);

}  // namespace hotpath
