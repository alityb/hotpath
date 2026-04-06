#include "hotpath/log_parser.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace hotpath {
namespace {

int64_t parse_timestamp(const std::string& line) {
  static const std::regex ts_re(R"((\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?)");
  std::smatch match;
  if (!std::regex_search(line, match, ts_re)) {
    return 0;
  }
  const int64_t hours = std::stoll(match[1].str());
  const int64_t minutes = std::stoll(match[2].str());
  const int64_t seconds = std::stoll(match[3].str());
  int64_t micros = 0;
  if (match[4].matched) {
    std::string frac = match[4].str();
    while (frac.size() < 6) {
      frac.push_back('0');
    }
    micros = std::stoll(frac);
  }
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

std::string clean_request_id_token(std::string token) {
  token = trim(std::move(token));
  while (!token.empty() &&
         (token.back() == ']' || token.back() == ')' || token.back() == '.' ||
          token.back() == ':' || token.back() == ',')) {
    token.pop_back();
  }
  while (!token.empty() &&
         (token.front() == '[' || token.front() == '(' || token.front() == '"' ||
          token.front() == '\'')) {
    token.erase(token.begin());
  }
  return token;
}

std::string canonicalize_vllm_request_id(const std::string& request_id) {
  if (request_id.rfind("cmpl-hotpath-req-", 0) == 0) {
    static const std::regex completion_internal_re(
        R"(^(cmpl-hotpath-req-\d+)(?:-\d+)?-[0-9a-fA-F]{8}$)");
    std::smatch match;
    if (std::regex_match(request_id, match, completion_internal_re)) {
      return match[1].str();
    }
  }
  if (request_id.rfind("chatcmpl-hotpath-req-", 0) == 0) {
    static const std::regex chat_internal_re(
        R"(^(chatcmpl-hotpath-req-\d+)(?:_\d+)?-[0-9a-fA-F]{8}$)");
    std::smatch match;
    if (std::regex_match(request_id, match, chat_internal_re)) {
      return match[1].str();
    }
  }
  if (request_id.rfind("responses-hotpath-req-", 0) == 0) {
    static const std::regex responses_internal_re(
        R"(^(responses-hotpath-req-\d+)(?:-\d+)?-[0-9a-fA-F]{8}$)");
    std::smatch match;
    if (std::regex_match(request_id, match, responses_internal_re)) {
      return match[1].str();
    }
  }
  return request_id;
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
    item = clean_request_id_token(std::move(item));
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
    for (auto& id : ids) {
      id = clean_request_id_token(std::move(id));
      id = canonicalize_vllm_request_id(id);
    }
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

bool line_has_v1_batch_descriptor(const std::string& line) {
  return line.find("BatchDescriptor(") != std::string::npos &&
         line.find("num_tokens=") != std::string::npos;
}

int extract_batch_num_tokens(const std::string& line) {
  static const std::regex num_tokens_re(R"(num_tokens=(\d+))");
  std::smatch match;
  if (std::regex_search(line, match, num_tokens_re)) {
    return std::stoi(match[1].str());
  }
  return 0;
}

bool line_has_v1_request_boundary(const std::string& line) {
  return line.find("EngineCore loop active.") != std::string::npos ||
         line.find("EngineCore waiting for work.") != std::string::npos ||
         line.find("\"POST /v1/completions HTTP/1.1\" 200 OK") != std::string::npos ||
         line.find("\"POST /v1/chat/completions HTTP/1.1\" 200 OK") != std::string::npos ||
         line.find("\"POST /v1/responses HTTP/1.1\" 200 OK") != std::string::npos;
}

std::vector<RequestTrace> parse_vllm_v1_anonymous_traces(const std::vector<std::string>& lines) {
  std::vector<RequestTrace> traces;
  RequestTrace current;
  bool current_open = false;
  int64_t last_batch_ts = 0;
  int next_trace_index = 1;

  auto open_trace = [&](int64_t ts) {
    current = RequestTrace{};
    std::ostringstream id;
    id << "v1-anon-" << std::setw(6) << std::setfill('0') << next_trace_index++;
    current.request_id = id.str();
    current.arrival_us = ts;
    current.queue_start_us = ts;
    current.status = "ok";
    current.events.push_back(RequestEvent{
        .event_type = "queue",
        .timestamp_us = ts,
        .detail = R"({"source":"vllm_v1"})",
    });
    current_open = true;
    last_batch_ts = 0;
  };

  auto close_trace = [&](int64_t ts) {
    if (!current_open) return;
    if (current.prefill_start_us == 0 && last_batch_ts > 0) {
      current.prefill_start_us = last_batch_ts;
    }
    if (current.prefill_end_us == 0 && last_batch_ts > 0) {
      current.prefill_end_us = last_batch_ts;
    }
    if (current.server_last_token_us == 0) {
      current.server_last_token_us = last_batch_ts > 0 ? last_batch_ts : ts;
    }
    if (current.first_token_us == 0 && current.prefill_end_us > 0) {
      current.first_token_us = current.prefill_end_us;
    }
    current.server_timing_available =
        current.prefill_start_us > 0 &&
        current.prefill_end_us >= current.prefill_start_us &&
        current.server_last_token_us >= current.prefill_end_us;
    if (current.server_timing_available) {
      traces.push_back(std::move(current));
    }
    current = RequestTrace{};
    current_open = false;
    last_batch_ts = 0;
  };

  for (const auto& line : lines) {
    const int64_t ts = parse_timestamp(line);
    if (line.find("EngineCore loop active.") != std::string::npos) {
      if (!current_open) {
        open_trace(ts);
      } else if (current.server_last_token_us > 0) {
        close_trace(ts);
        open_trace(ts);
      }
      current.events.push_back(RequestEvent{
          .event_type = "schedule",
          .timestamp_us = ts,
          .detail = R"({"source":"enginecore_loop_active"})",
      });
      continue;
    }

    if (line_has_v1_batch_descriptor(line)) {
      if (!current_open) {
        open_trace(ts);
      }
      const int num_tokens = extract_batch_num_tokens(line);
      last_batch_ts = ts;
      if (num_tokens > 1 && current.prefill_start_us == 0) {
        current.prefill_start_us = ts;
        current.prompt_tokens = num_tokens;
        current.events.push_back(RequestEvent{
            .event_type = "prefill",
            .timestamp_us = ts,
            .detail = "{\"num_tokens\":" + std::to_string(num_tokens) + "}",
        });
      } else if (num_tokens == 1) {
        if (current.prefill_start_us == 0) {
          current.prefill_start_us = ts;
        }
        if (current.prefill_end_us == 0) {
          current.prefill_end_us = ts;
          current.first_token_us = ts;
        }
        current.server_last_token_us = ts;
        current.events.push_back(RequestEvent{
            .event_type = "decode",
            .timestamp_us = ts,
            .detail = R"({"num_tokens":1})",
        });
      }
      continue;
    }

    if (line.find("EngineCore waiting for work.") != std::string::npos) {
      if (current_open) {
        current.events.push_back(RequestEvent{
            .event_type = "decode",
            .timestamp_us = ts,
            .detail = R"({"source":"enginecore_waiting"})",
        });
        close_trace(ts);
      }
      continue;
    }

    if ((line.find("\"POST /v1/completions HTTP/1.1\" 200 OK") != std::string::npos ||
         line.find("\"POST /v1/chat/completions HTTP/1.1\" 200 OK") != std::string::npos ||
         line.find("\"POST /v1/responses HTTP/1.1\" 200 OK") != std::string::npos) &&
        current_open) {
      if (current.prefill_end_us == 0 && current.prefill_start_us > 0) {
        current.prefill_end_us = ts;
        current.first_token_us = ts;
      }
      current.server_last_token_us = std::max(current.server_last_token_us, ts);
      current.events.push_back(RequestEvent{
          .event_type = "schedule",
          .timestamp_us = ts,
          .detail = R"({"source":"api_response"})",
      });
    }
  }

  if (current_open) {
    close_trace(last_batch_ts > 0 ? last_batch_ts : current.queue_start_us);
  }

  return traces;
}

void merge_v1_anonymous_timing(std::map<std::string, RequestTrace>& traces,
                               const std::vector<std::string>& request_order,
                               const std::vector<RequestTrace>& anonymous_traces) {
  if (request_order.empty() || anonymous_traces.empty()) {
    return;
  }

  std::size_t anonymous_index = 0;
  for (const auto& request_id : request_order) {
    auto it = traces.find(request_id);
    if (it == traces.end()) {
      continue;
    }
    auto& trace = it->second;
    if (trace.server_timing_available) {
      continue;
    }
    if (anonymous_index >= anonymous_traces.size()) {
      break;
    }

    const auto& anonymous = anonymous_traces[anonymous_index++];
    if (trace.queue_start_us <= 0) {
      trace.queue_start_us = anonymous.queue_start_us;
    }
    if (trace.prefill_start_us <= 0) {
      trace.prefill_start_us = anonymous.prefill_start_us;
    }
    if (trace.prefill_end_us <= 0) {
      trace.prefill_end_us = anonymous.prefill_end_us;
    }
    if (trace.first_token_us <= 0) {
      trace.first_token_us = anonymous.first_token_us;
    }
    if (trace.server_last_token_us <= 0) {
      trace.server_last_token_us = anonymous.server_last_token_us;
    }
    if (trace.prompt_tokens <= 0) {
      trace.prompt_tokens = anonymous.prompt_tokens;
    }
    if (trace.output_tokens <= 0) {
      trace.output_tokens = anonymous.output_tokens;
    }
    if (trace.events.empty()) {
      trace.events = anonymous.events;
    } else {
      trace.events.insert(trace.events.end(), anonymous.events.begin(), anonymous.events.end());
    }

    trace.server_timing_available =
        timing_available_and_sane(trace) &&
        trace.server_last_token_us >= trace.prefill_end_us;
  }
}

}  // namespace

VllmLogParseResult parse_vllm_log_lines_detailed(const std::vector<std::string>& lines) {
  std::map<std::string, RequestTrace> traces;
  std::optional<double> aggregate_cache_hit_rate;
  std::vector<std::string> request_order;

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
        if (std::find(request_order.begin(), request_order.end(), request_id) ==
            request_order.end()) {
          request_order.push_back(request_id);
        }
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

  const bool looks_like_v1 = std::any_of(lines.begin(), lines.end(), [](const std::string& line) {
    return line_has_v1_request_boundary(line) || line_has_v1_batch_descriptor(line);
  });
  std::vector<RequestTrace> anonymous_v1_traces;
  if (looks_like_v1) {
    anonymous_v1_traces = parse_vllm_v1_anonymous_traces(lines);
    if (!traces.empty()) {
      merge_v1_anonymous_timing(traces, request_order, anonymous_v1_traces);
    }
  }

  std::vector<RequestTrace> result;
  result.reserve(traces.size());
  for (auto& [id, trace] : traces) {
    if ((trace.queue_start_us > 0 || trace.prefill_start_us > 0 || trace.prefill_end_us > 0) &&
        !timing_available_and_sane(trace)) {
      if (looks_like_v1) {
        // vLLM v1 often gives us only exact request admission timestamps here.
        // Keep those partial anchors so the later correlation/refinement pass can
        // reconstruct queue/prefill/decode from client timing and Prometheus means.
        trace.server_timing_available = false;
        if (trace.prefill_start_us > 0 && trace.prefill_end_us > 0 &&
            trace.prefill_end_us < trace.prefill_start_us) {
          trace.prefill_start_us = 0;
          trace.prefill_end_us = 0;
          trace.server_last_token_us = 0;
        }
      } else {
        discard_invalid_timing(trace);
      }
    }
    if (trace.status.empty()) {
      trace.status = "ok";
    }
    result.push_back(std::move(trace));
  }

  if (result.empty() && looks_like_v1) {
    result = std::move(anonymous_v1_traces);
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
