#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "hotpath/request_trace.h"

namespace hotpath {

struct GpuInfo {
    int count = 0;
    std::string name;             // e.g. "NVIDIA A10G"
    int memory_mb = 0;            // per-GPU VRAM
};

GpuInfo detect_gpus();

struct ServeProfileOptions {
    std::string endpoint = "http://localhost:8000";
    int duration_seconds = 60;
    std::string traffic_path;
    std::string output = ".hotpath/serve_run";
    bool use_nsys = false;
    std::string engine = "vllm";  // vllm | sglang
    std::string model;            // auto-detected from endpoint if empty
    std::string server_log_path;  // optional external vLLM server log path
    std::int64_t server_pid = 0;  // optional external vLLM server PID for nsys attach
    bool launch_managed_server = false;
    int max_concurrency = 1;      // max concurrent in-flight requests during replay
};

enum class ServerTraceMatchMethod {
    NONE,
    ID,
    TIMESTAMP,
};

struct ServerTraceCorrelationResult {
    int matched_requests = 0;
    int total_requests = 0;
    ServerTraceMatchMethod method = ServerTraceMatchMethod::NONE;
    std::int64_t max_offset_us = 0;
};

ServerTraceCorrelationResult correlate_server_traces(
    std::vector<RequestTrace>& client_traces,
    const std::vector<RequestTrace>& server_traces,
    bool allow_timestamp_fallback = true,
    std::int64_t max_timestamp_offset_us = 50000);

int run_serve_profile(const ServeProfileOptions& opts);

}  // namespace hotpath
