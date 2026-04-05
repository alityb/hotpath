#include "hotpath/sglang_metrics.h"

#include <sstream>
#include <string>

namespace hotpath {
namespace {

bool starts_with(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

}  // namespace

std::vector<std::pair<std::string, double>> parse_sglang_metrics_text(const std::string& text) {
  std::vector<std::pair<std::string, double>> result;
  std::istringstream stream(text);
  std::string line;

  while (std::getline(stream, line)) {
    // Skip comments and empty lines
    if (line.empty() || line[0] == '#') continue;

    // Parse "metric_name{labels} value" or "metric_name value"
    std::string name;
    double value = 0.0;

    auto space_pos = line.find(' ');
    if (space_pos == std::string::npos) continue;

    // Handle labels in braces
    auto brace_pos = line.find('{');
    if (brace_pos != std::string::npos && brace_pos < space_pos) {
      name = line.substr(0, brace_pos);
      auto close_brace = line.find('}', brace_pos);
      if (close_brace != std::string::npos) {
        space_pos = line.find(' ', close_brace);
        if (space_pos == std::string::npos) continue;
      }
    } else {
      name = line.substr(0, space_pos);
    }

    try {
      value = std::stod(line.substr(space_pos + 1));
    } catch (...) {
      continue;
    }

    // Only include sglang metrics
    if (starts_with(name, "sglang:") || starts_with(name, "sglang_")) {
      result.emplace_back(name, value);
    }
  }

  return result;
}

SglangMetrics parse_sglang_metrics(const std::string& text) {
  SglangMetrics m;
  const auto pairs = parse_sglang_metrics_text(text);
  for (const auto& [name, value] : pairs) {
    if (name == "sglang:num_running_req" || name == "sglang_num_running_req" ||
        name == "sglang:num_running_reqs" || name == "sglang_num_running_reqs") {
      m.num_running_req = value;
    } else if (name == "sglang:num_waiting_req" || name == "sglang_num_waiting_req" ||
               name == "sglang:num_queue_reqs" || name == "sglang_num_queue_reqs") {
      m.num_waiting_req = value;
    } else if (name == "sglang:token_usage" || name == "sglang_token_usage") {
      m.token_usage = value;
    } else if (name == "sglang:cache_hit_rate" || name == "sglang_cache_hit_rate") {
      m.cache_hit_rate = value;
    } else if (name == "sglang:num_total_tokens" || name == "sglang_num_total_tokens") {
      m.num_total_tokens = value;
    }
  }
  return m;
}

MetricSnapshot sglang_to_snapshot(const SglangMetrics& m, int64_t timestamp_us) {
  MetricSnapshot snap;
  snap.timestamp_us = timestamp_us;
  snap.batch_size = m.num_running_req;
  snap.queue_depth = m.num_waiting_req;
  snap.preemption_total = 0;  // SGLang doesn't expose preemption counter
  snap.cache_usage = m.token_usage * 100.0;  // Normalize to percentage if needed
  return snap;
}

}  // namespace hotpath
