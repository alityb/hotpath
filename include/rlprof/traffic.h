#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "rlprof/report.h"

namespace rlprof {

struct TrafficRequest {
  std::string prompt;
  std::int64_t output_len;
};

struct TrafficResult {
  bool ok;
  long http_status;
  std::int64_t completion_tokens;
  std::string body;
  std::string error;
};

struct TrafficRun {
  std::vector<TrafficResult> results;
  TrafficStats stats;
};

std::vector<TrafficRequest> generate_requests(
    std::int64_t num_prompts,
    std::int64_t rollouts_per_prompt,
    std::int64_t input_len,
    std::int64_t min_tokens,
    std::int64_t max_tokens,
    std::uint32_t seed = 0);

TrafficResult send_request(
    const std::string& server_url,
    const TrafficRequest& request);

TrafficStats summarize_traffic(
    const std::vector<TrafficResult>& results,
    const std::vector<TrafficRequest>& requests);

TrafficRun fire_rl_traffic(
    const std::string& server_url,
    std::int64_t num_prompts,
    std::int64_t rollouts_per_prompt,
    std::int64_t min_tokens,
    std::int64_t max_tokens,
    std::int64_t input_len,
    std::uint32_t seed = 0);

TrafficRun fire_rl_traffic(
    const std::vector<std::string>& server_urls,
    std::int64_t num_prompts,
    std::int64_t rollouts_per_prompt,
    std::int64_t min_tokens,
    std::int64_t max_tokens,
    std::int64_t input_len,
    std::uint32_t seed = 0);

}  // namespace rlprof
