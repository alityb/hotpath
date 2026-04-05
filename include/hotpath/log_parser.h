#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "hotpath/request_trace.h"

namespace hotpath {

struct VllmLogParseResult {
  std::vector<RequestTrace> traces;
  std::optional<double> aggregate_cache_hit_rate;
};

VllmLogParseResult parse_vllm_log_details(const std::filesystem::path& log_path);
VllmLogParseResult parse_vllm_log_lines_detailed(const std::vector<std::string>& lines);
std::vector<RequestTrace> parse_vllm_log(const std::filesystem::path& log_path);
std::vector<RequestTrace> parse_vllm_log_lines(const std::vector<std::string>& lines);

}  // namespace hotpath
