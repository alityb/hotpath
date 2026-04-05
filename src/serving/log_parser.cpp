#include "hotpath/log_parser.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace hotpath {
namespace {

int64_t parse_timestamp(const std::string& line) {
  static const std::regex ts_re(R"((\d{2}):(\d{2}):(\d{2})\.(\d{1,6}))");
  std::smatch match;
  if (!std::regex_search(line, match, ts_re)) {
    return 0;
  }
  const int64_t hours = std::stoll(match[1].str());
  const int64_t minutes = std::stoll(match[2].str());
  const int64_t seconds = std::stoll(match[3].str());
  std::string frac = match[4].str();
  while (frac.size() < 6) {
    frac.push_back('0');
  }
  const int64_t micros = std::stoll(frac);
  return ((hours * 3600 + minutes * 60 + seconds) * 1000000) + micros;
}

std::string trim(std::string value) {
  while (!value.empty() &&
         (value.back() == '\n' || value.back() == '\r' || value.back() == ' ')) {
    value.pop_back();
  }
  std::size_t start = 0;
  while (start < value.size() && value[start] == ' ') {
    ++start;
  }
  return value.substr(start);
}

std::vector<std::string> split_ids(std::string text) {
  std::vector<std::string> ids;
  text = trim(std::move(text));
  if (!text.empty() && text.front() == '[' && text.back() == ']') {
    text = text.substr(1, text.size() - 2);
  }
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = trim(std::move(item));
    while (!item.empty() &&
           (item.back() == ']' || item.back() == ')' || item.back() == '.' || item.back() == ':')) {
      item.pop_back();
    }
    while (!item.empty() &&
           (item.front() == '[' || item.front() == '(' || item.front() == '"' || item.front() == '\'')) {
      item.erase(item.begin());
    }
    if (!item.empty()) {
      ids.push_back(item);
    }
  }
  return ids;
}

std::vector<std::string> extract_ids_from_brackets(const std::string& line) {
  static const std::regex bracket_re(R"(\[([^\]]+)\])");
  std::smatch match;
  if (!std::regex_search(line, match, bracket_re)) {
    return {};
  }
  return split_ids(match[1].str());
}

// Returns true if a string looks like a real vLLM/SGLang request ID.
// Filters out tqdm progress bar strings (00:00<00:01, 30.96it/s, ?it/s, etc.)
bool looks_like_request_id(const std::string& id) {
  if (id.empty() || id.size() < 4) return false;
  if (id.find('<') != std::string::npos || id.find('>') != std::string::npos) return false;
  if (id.find("it/s") != std::string::npos || id.find("?it") != std::string::npos) return false;
  if (id.back() == '%') return false;
  bool all_digits_or_colon = true;
  for (char c : id) {
    if (!std::isdigit(c) && c != ':' && c != '.') { all_digits_or_colon = false; break; }
  }
  if (all_digits_or_colon) return false;
  for (char c : id) {
    if (!std::isalnum(c) && c != '-' && c != '_') return false;
  }
  return true;
}

std::vector<std::string> extract_contextual_request_ids(const std::string& line) {
  std::smatch match;

  auto filter_ids = [](std::vector<std::string> ids) {
    ids.erase(std::remove_if(ids.begin(), ids.end(),
                             [](const std::string& id) { return !looks_like_request_id(id); }),
              ids.end());
    return ids;
  };

  static const std::regex added_re(
      R"((?:Added request|add_request)\s+([^\s,:\]]+))",
      std::regex::icase);
  if (std::regex_search(line, match, added_re)) {
    auto ids = filter_ids({match[1].str()});
    if (!ids.empty()) return ids;
  }

  static const std::regex finished_re(
      R"((?:Finished request|request_output|completed(?: request)?)\s+([^\s,:\]]+))",
      std::regex::icase);
  if (std::regex_search(line, match, finished_re)) {
    auto ids = filter_ids({match[1].str()});
    if (!ids.empty()) return ids;
  }

  if (line.find("Running:") != std::string::npos ||
      line.find("scheduled") != std::string::npos ||
      line.find("Scheduling") != std::string::npos) {
    const auto ids = filter_ids(extract_ids_from_brackets(line));
    if (!ids.empty()) return ids;
    static const std::regex running_inline_re(R"(Running:.*requests?,\s*(.+)$)",
                                              std::regex::icase);
    if (std::regex_search(line, match, running_inline_re)) {
      auto ids2 = filter_ids(split_ids(match[1].str()));
      if (!ids2.empty()) return ids2;
    }
  }

  // Only match "Prefill" in actual scheduler/request lines, not CUDA graph capture output
  if ((line.find("Prefill") != std::string::npos ||
       line.find("prompt processing") != std::string::npos) &&
      line.find("CUDA") == std::string::npos &&
      line.find("graph") == std::string::npos &&
      line.find("it/s") == std::string::npos) {
    const auto ids = filter_ids(extract_ids_from_brackets(line));
    if (!ids.empty()) return ids;
    static const std::regex request_re(R"((?:request|requests?)\s+([^\s,:\]]+))",
                                       std::regex::icase);
    if (std::regex_search(line, match, request_re)) {
      auto ids2 = filter_ids({match[1].str()});
      if (!ids2.empty()) return ids2;
    }
  }

  if (line.find("cache_hit") != std::string::npos ||
      line.find("prefix_cache") != std::string::npos) {
    const auto ids = filter_ids(extract_ids_from_brackets(line));
    if (!ids.empty()) return ids;
    static const std::regex cache_re(R"((?:cache_hit|prefix_cache)\s+([^\s,:\]]+))",
                                     std::regex::icase);
    if (std::regex_search(line, match, cache_re)) {
      auto ids2 = filter_ids({match[1].str()});
      if (!ids2.empty()) return ids2;
    }
  }

  if (line.find("preempt") != std::string::npos ||
      line.find("Preempt") != std::string::npos) {
    const auto ids = filter_ids(extract_ids_from_brackets(line));
    if (!ids.empty()) return ids;
    static const std::regex preempt_re(R"((?:request|requests?)\s+([^\s,:\]]+))",
                                       std::regex::icase);
    if (std::regex_search(line, match, preempt_re)) {
      auto ids2 = filter_ids({match[1].str()});
      if (!ids2.empty()) return ids2;
    }
  }

  return {};
}

int extract_token_count(const std::string& line, const std::string& keyword) {
  const std::regex re(keyword + R"([=:\s]+(\d+))", std::regex::icase);
  std::smatch match;
  if (std::regex_search(line, match, re)) {
    return std::stoi(match[1].str());
  }
  return 0;
}

std::optional<double> extract_aggregate_cache_hit_rate(const std::string& line) {
  static const std::regex rate_re(
      R"(Prefix cache hit rate:\s*([0-9]+(?:\.[0-9]+)?)%)",
      std::regex::icase);
  std::smatch match;
  if (!std::regex_search(line, match, rate_re)) {
    return std::nullopt;
  }
  return std::clamp(std::stod(match[1].str()) / 100.0, 0.0, 1.0);
}

bool timing_available_and_sane(const RequestTrace& trace) {
  if (trace.queue_start_us <= 0 || trace.prefill_start_us <= 0 ||
      trace.prefill_end_us <= 0) {
    return false;
  }
  return trace.queue_start_us <= trace.prefill_start_us &&
         trace.prefill_start_us <= trace.prefill_end_us;
}

RequestTrace& ensure_trace(std::map<std::string, RequestTrace>& traces,
                           const std::string& request_id) {
  auto& trace = traces[request_id];
  if (trace.request_id.empty()) {
    trace.request_id = request_id;
  }
  return trace;
}

void discard_invalid_timing(RequestTrace& trace) {
  std::cerr << "warning: discarding invalid server-side timing for request "
            << trace.request_id << " (queue_start=" << trace.queue_start_us
            << ", prefill_start=" << trace.prefill_start_us
            << ", prefill_end=" << trace.prefill_end_us << ")\n";
  trace.queue_start_us = 0;
  trace.prefill_start_us = 0;
  trace.prefill_end_us = 0;
  trace.server_last_token_us = 0;
  trace.server_timing_available = false;
}

}  // namespace

VllmLogParseResult parse_vllm_log_lines_detailed(const std::vector<std::string>& lines) {
  std::map<std::string, RequestTrace> traces;
  std::optional<double> aggregate_cache_hit_rate;

  for (const auto& line : lines) {
    if (const auto rate = extract_aggregate_cache_hit_rate(line); rate.has_value()) {
      aggregate_cache_hit_rate = rate;
    }

    const int64_t ts = parse_timestamp(line);
    const std::vector<std::string> request_ids = extract_contextual_request_ids(line);
    if (request_ids.empty()) {
      continue;
    }

    if (line.find("Added request") != std::string::npos ||
        line.find("add_request") != std::string::npos) {
      for (const auto& request_id : request_ids) {
        auto& trace = ensure_trace(traces, request_id);
        trace.arrival_us = ts;
        trace.queue_start_us = ts;
        trace.events.push_back(RequestEvent{
            .event_type = "queue",
            .timestamp_us = ts,
            .detail = "{}",
        });
        int prompt_tokens = extract_token_count(line, "prompt_token");
        if (prompt_tokens == 0) prompt_tokens = extract_token_count(line, "prompt_tokens");
        if (prompt_tokens == 0) prompt_tokens = extract_token_count(line, "num_prompt_token");
        if (prompt_tokens > 0) {
          trace.prompt_tokens = prompt_tokens;
        }
      }
      continue;
    }

    if (line.find("Running:") != std::string::npos ||
        line.find("scheduled") != std::string::npos ||
        line.find("Scheduling") != std::string::npos) {
      for (const auto& request_id : request_ids) {
        auto& trace = ensure_trace(traces, request_id);
        if (trace.prefill_start_us == 0) {
          trace.prefill_start_us = ts;
          trace.events.push_back(RequestEvent{
              .event_type = "schedule",
              .timestamp_us = ts,
              .detail = "{}",
          });
        }
      }
      continue;
    }

    if (line.find("Prefill") != std::string::npos ||
        line.find("prefill") != std::string::npos ||
        line.find("prompt processing") != std::string::npos) {
      for (const auto& request_id : request_ids) {
        auto& trace = ensure_trace(traces, request_id);
        if (trace.prefill_start_us == 0) {
          trace.prefill_start_us = ts;
        }
        if (trace.prefill_end_us == 0 ||
            line.find("done") != std::string::npos ||
            line.find("finished") != std::string::npos ||
            line.find("complete") != std::string::npos) {
          trace.prefill_end_us = ts;
        }
        trace.events.push_back(RequestEvent{
            .event_type = "prefill",
            .timestamp_us = ts,
            .detail = "{}",
        });
        int prompt_tokens = extract_token_count(line, "token");
        if (prompt_tokens > 0 && trace.prompt_tokens == 0) {
          trace.prompt_tokens = prompt_tokens;
        }
      }
      continue;
    }

    if (line.find("Finished request") != std::string::npos ||
        line.find("request_output") != std::string::npos ||
        line.find("completed") != std::string::npos) {
      for (const auto& request_id : request_ids) {
        auto& trace = ensure_trace(traces, request_id);
        trace.server_last_token_us = ts;
        if (trace.first_token_us == 0 && trace.prefill_end_us > 0) {
          trace.first_token_us = trace.prefill_end_us;
        }
        trace.server_timing_available = timing_available_and_sane(trace);
        trace.status = "ok";
        trace.events.push_back(RequestEvent{
            .event_type = "decode",
            .timestamp_us = ts,
            .detail = "{}",
        });
        int output_tokens = extract_token_count(line, "output_token");
        if (output_tokens == 0) output_tokens = extract_token_count(line, "output_tokens");
        if (output_tokens == 0) output_tokens = extract_token_count(line, "generated");
        if (output_tokens > 0) {
          trace.output_tokens = output_tokens;
        }
      }
      continue;
    }

    if (line.find("preempt") != std::string::npos ||
        line.find("Preempt") != std::string::npos) {
      for (const auto& request_id : request_ids) {
        auto& trace = ensure_trace(traces, request_id);
        trace.status = "preempted";
        trace.events.push_back(RequestEvent{
            .event_type = "preempt",
            .timestamp_us = ts,
            .detail = "{}",
        });
      }
      continue;
    }

    if (line.find("cache_hit") != std::string::npos ||
        line.find("prefix_cache") != std::string::npos) {
      for (const auto& request_id : request_ids) {
        auto& trace = ensure_trace(traces, request_id);
        trace.events.push_back(RequestEvent{
            .event_type = "cache_hit",
            .timestamp_us = ts,
            .detail = "{}",
        });
        const int cached_tokens = extract_token_count(line, "cached");
        if (cached_tokens > 0) {
          trace.cached_tokens = cached_tokens;
        }
      }
    }
  }

  std::vector<RequestTrace> result;
  result.reserve(traces.size());
  for (auto& [id, trace] : traces) {
    if ((trace.queue_start_us > 0 || trace.prefill_start_us > 0 || trace.prefill_end_us > 0) &&
        !timing_available_and_sane(trace)) {
      discard_invalid_timing(trace);
    }
    if (trace.status.empty()) {
      trace.status = "ok";
    }
    result.push_back(std::move(trace));
  }

  return VllmLogParseResult{
      .traces = std::move(result),
      .aggregate_cache_hit_rate = aggregate_cache_hit_rate,
  };
}

VllmLogParseResult parse_vllm_log_details(const std::filesystem::path& log_path) {
  std::ifstream file(log_path);
  if (!file.is_open()) {
    throw std::runtime_error("cannot open log file: " + log_path.string());
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    lines.push_back(std::move(line));
  }

  std::cerr << "vLLM server log preview (" << log_path.string() << "):\n";
  for (std::size_t i = 0; i < lines.size() && i < 20; ++i) {
    std::cerr << "  " << lines[i] << "\n";
  }

  return parse_vllm_log_lines_detailed(lines);
}

std::vector<RequestTrace> parse_vllm_log_lines(const std::vector<std::string>& lines) {
  return parse_vllm_log_lines_detailed(lines).traces;
}

std::vector<RequestTrace> parse_vllm_log(const std::filesystem::path& log_path) {
  return parse_vllm_log_details(log_path).traces;
}

}  // namespace hotpath
