#include "hotpath/traffic_replayer.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace hotpath {
namespace {

// Simple JSON string extraction (no dependency on JSON library)
std::string extract_json_string(const std::string& json, const std::string& key) {
  const std::string pattern = "\"" + key + "\"";
  auto pos = json.find(pattern);
  if (pos == std::string::npos) return "";

  // Skip past key and colon
  pos = json.find(':', pos + pattern.size());
  if (pos == std::string::npos) return "";
  pos++;

  // Skip whitespace
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

  if (pos >= json.size()) return "";

  if (json[pos] == '"') {
    // String value
    pos++;
    std::string result;
    while (pos < json.size() && json[pos] != '"') {
      if (json[pos] == '\\' && pos + 1 < json.size()) {
        pos++;
        if (json[pos] == '"') result += '"';
        else if (json[pos] == 'n') result += '\n';
        else if (json[pos] == '\\') result += '\\';
        else result += json[pos];
      } else {
        result += json[pos];
      }
      pos++;
    }
    return result;
  }

  return "";
}

int extract_json_int(const std::string& json, const std::string& key, int default_val) {
  const std::string pattern = "\"" + key + "\"";
  auto pos = json.find(pattern);
  if (pos == std::string::npos) return default_val;

  pos = json.find(':', pos + pattern.size());
  if (pos == std::string::npos) return default_val;
  pos++;

  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

  if (pos >= json.size() || !std::isdigit(json[pos])) return default_val;

  std::string num;
  while (pos < json.size() && std::isdigit(json[pos])) {
    num += json[pos++];
  }
  return std::stoi(num);
}

int extract_usage_int(const std::string& json, const std::string& key, int default_val) {
  auto usage_pos = json.rfind("\"usage\"");
  if (usage_pos == std::string::npos) return default_val;
  return extract_json_int(json.substr(usage_pos), key, default_val);
}

std::string extract_response_id(const std::string& body) {
  return extract_json_string(body, "id");
}

// Extract the raw "messages" JSON array from a line (returns empty string if absent)
std::string extract_messages_array(const std::string& json) {
  const auto pos = json.find("\"messages\"");
  if (pos == std::string::npos) return "";
  const auto colon = json.find(':', pos + 10);
  if (colon == std::string::npos) return "";
  auto start = colon + 1;
  while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) start++;
  if (start >= json.size() || json[start] != '[') return "";
  // Find matching closing bracket
  int depth = 0;
  size_t end = start;
  for (; end < json.size(); end++) {
    if (json[end] == '[' || json[end] == '{') depth++;
    else if (json[end] == ']' || json[end] == '}') {
      depth--;
      if (depth == 0) { end++; break; }
    }
  }
  return json.substr(start, end - start);
}

// Extract the last user role content from an OpenAI messages array string
std::string extract_user_content_from_messages(const std::string& messages_json) {
  std::string last_user_content;
  size_t pos = 0;
  while ((pos = messages_json.find("\"role\"", pos)) != std::string::npos) {
    const auto role = extract_json_string(messages_json.substr(pos), "role");
    if (role == "user") {
      // Find "content" after this role
      const auto content_pos = messages_json.find("\"content\"", pos);
      if (content_pos != std::string::npos) {
        last_user_content = extract_json_string(messages_json.substr(content_pos), "content");
      }
    }
    pos += 6;
  }
  return last_user_content;
}

// Extract first human message from ShareGPT conversations array
std::string extract_sharegpt_prompt(const std::string& json) {
  // Look for "from": "human" followed by "value": "..."
  auto pos = json.find("\"from\"");
  while (pos != std::string::npos) {
    auto from_val = extract_json_string(json.substr(pos), "from");
    if (from_val == "human") {
      auto value_pos = json.find("\"value\"", pos);
      if (value_pos != std::string::npos) {
        return extract_json_string(json.substr(value_pos), "value");
      }
    }
    pos = json.find("\"from\"", pos + 1);
  }
  return "";
}

}  // namespace

std::vector<ReplayRequest> load_jsonl(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("cannot open JSONL file: " + path.string());
  }

  std::vector<ReplayRequest> requests;
  std::string line;
  int line_num = 0;
  int skipped = 0;
  while (std::getline(file, line)) {
    ++line_num;
    if (line.empty() || line[0] == '#') continue;

    // Basic JSON sanity check: must start with '{' (after optional whitespace)
    const auto first_non_space = line.find_first_not_of(" \t\r");
    if (first_non_space == std::string::npos || line[first_non_space] != '{') {
      std::cerr << "warning: " << path.string() << ":" << line_num
                << ": skipping malformed JSONL entry (expected JSON object)\n";
      ++skipped;
      continue;
    }

    ReplayRequest req;
    req.max_tokens = extract_json_int(line, "max_tokens", 256);

    // Support both {"prompt": "..."} and {"messages": [...]} formats
    const std::string messages_arr = extract_messages_array(line);
    if (!messages_arr.empty()) {
      req.messages_json = messages_arr;
      // Extract user content as prompt_text for analysis (prefix sharing, etc.)
      req.prompt = extract_user_content_from_messages(messages_arr);
    } else {
      req.prompt = extract_json_string(line, "prompt");
    }

    if (!req.prompt.empty() || !req.messages_json.empty()) {
      requests.push_back(std::move(req));
    } else {
      std::cerr << "warning: " << path.string() << ":" << line_num
                << ": skipping entry with no 'prompt' or 'messages' field\n";
      ++skipped;
    }
  }
  if (skipped > 0) {
    std::cerr << "warning: skipped " << skipped << " malformed entries in " << path.string() << "\n";
  }

  return requests;
}

std::vector<ReplayRequest> load_sharegpt(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("cannot open ShareGPT file: " + path.string());
  }

  std::vector<ReplayRequest> requests;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    ReplayRequest req;
    req.prompt = extract_sharegpt_prompt(line);
    req.max_tokens = 256;

    if (!req.prompt.empty()) {
      requests.push_back(std::move(req));
    }
  }

  return requests;
}

static void json_escape(std::ostringstream& ss, const std::string& s) {
  for (char c : s) {
    if (c == '"') ss << "\\\"";
    else if (c == '\\') ss << "\\\\";
    else if (c == '\n') ss << "\\n";
    else if (c == '\r') ss << "\\r";
    else if (c == '\t') ss << "\\t";
    else ss << c;
  }
}

std::string build_request_body(const ReplayRequest& req, const std::string& model) {
  std::ostringstream ss;
  ss << "{";
  ss << "\"model\": \"";
  json_escape(ss, model);
  ss << "\", ";

  if (!req.messages_json.empty()) {
    // Chat completions format
    ss << "\"messages\": " << req.messages_json << ", ";
    ss << "\"max_tokens\": " << req.max_tokens << ", ";
    ss << "\"stream\": true, ";
    ss << "\"stream_options\": {\"include_usage\": true}";
  } else {
    // Plain completions format
    ss << "\"prompt\": \"";
    json_escape(ss, req.prompt);
    ss << "\", ";
    ss << "\"max_tokens\": " << req.max_tokens << ", ";
    ss << "\"stream\": true, ";
    ss << "\"stream_options\": {\"include_usage\": true}";
  }

  ss << "}";
  return ss.str();
}

// Returns the correct API path for a request (chat vs plain completions)
std::string api_path_for(const ReplayRequest& req) {
  return req.messages_json.empty() ? "/v1/completions" : "/v1/chat/completions";
}

namespace {

int64_t now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string run_cmd(const std::string& cmd) {
  std::array<char, 4096> buf{};
  std::string out;
  FILE* pipe = popen((cmd + " 2>/dev/null").c_str(), "r");
  if (!pipe) return "";
  while (fgets(buf.data(), static_cast<int>(buf.size()), pipe))
    out.append(buf.data());
  pclose(pipe);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r'))
    out.pop_back();
  return out;
}

std::string response_body_without_timing(const std::string& output) {
  auto timing_pos = output.rfind("__TIMING__:");
  return timing_pos == std::string::npos ? output : output.substr(0, timing_pos);
}

// Count "data:" SSE chunks in streaming response (each is one token)
int count_sse_tokens(const std::string& body) {
  int count = 0;
  size_t pos = 0;
  while ((pos = body.find("data:", pos)) != std::string::npos) {
    ++count;
    pos += 5;
  }
  // Subtract 1 for the final "data: [DONE]" marker
  return (count > 1) ? count - 1 : count;
}

}  // namespace

std::string parse_model_from_models_response(const std::string& response) {
  size_t pos = 0;
  while ((pos = response.find("\"id\"", pos)) != std::string::npos) {
    pos = response.find(':', pos + 4);
    if (pos == std::string::npos) return "";
    ++pos;
    while (pos < response.size() &&
           (response[pos] == ' ' || response[pos] == '\t' ||
            response[pos] == '\n' || response[pos] == '\r')) {
      ++pos;
    }
    if (pos >= response.size() || response[pos] != '"') {
      continue;
    }
    ++pos;
    std::string value;
    while (pos < response.size()) {
      const char ch = response[pos];
      if (ch == '\\' && pos + 1 < response.size()) {
        ++pos;
        value.push_back(response[pos]);
      } else if (ch == '"') {
        return value;
      } else {
        value.push_back(ch);
      }
      ++pos;
    }
    return "";
  }
  return "";
}

std::vector<ReplayResult> replay_traffic(const std::vector<ReplayRequest>& requests,
                                         const ReplayConfig& config) {
  std::vector<ReplayResult> results;
  results.reserve(requests.size());

  const double interval_us = (config.rate_limit_rps > 0)
      ? 1e6 / config.rate_limit_rps
      : 0.0;

  int req_ok = 0, req_fail = 0;
  const int total = static_cast<int>(requests.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];
    ReplayResult result;
    result.request_id = "req_" + std::to_string(i);

    const std::string body = build_request_body(req, config.model);

    // Write body to a temp file to avoid shell escaping issues
    const std::string tmp_path = "/tmp/hotpath_req_" + std::to_string(i) + ".json";
    {
      std::ofstream f(tmp_path);
      if (!f.is_open()) {
        result.success = false;
        result.error = "failed to open temp file: " + tmp_path;
        results.push_back(std::move(result));
        continue;
      }
      f << body;
      if (!f.good()) {
        result.success = false;
        result.error = "failed to write temp file: " + tmp_path + " (disk full?)";
        std::remove(tmp_path.c_str());
        results.push_back(std::move(result));
        continue;
      }
    }

    // Use curl with streaming (-N) — chat or plain completions based on request format
    const std::string api_path = api_path_for(req);
    const std::string curl_cmd =
        "curl -sS -N -X POST "
        "-H 'Content-Type: application/json' "
        "-d @" + tmp_path + " "
        "-w '\\n__TIMING__:%{time_starttransfer}:%{time_total}' "
        "'" + config.endpoint + api_path + "'";

    result.send_us = now_us();

    const std::string output = run_cmd(curl_cmd);

    const int64_t done_us = now_us();

    // Parse timing from curl -w output
    auto timing_pos = output.rfind("__TIMING__:");
    const std::string response_body = response_body_without_timing(output);
    const std::string response_id = extract_response_id(response_body);
    if (!response_id.empty()) {
      result.request_id = response_id;
    }
    result.prompt_tokens = extract_usage_int(response_body, "prompt_tokens", 0);
    result.completion_tokens = extract_usage_int(response_body, "completion_tokens", 0);
    result.prompt_tokens_estimated = result.prompt_tokens <= 0;
    if (timing_pos != std::string::npos) {
      const std::string timing = output.substr(timing_pos + 11);
      auto colon = timing.find(':');
      if (colon != std::string::npos) {
        try {
          double ttfb_s = std::stod(timing.substr(0, colon));
          double total_s = std::stod(timing.substr(colon + 1));
          result.first_token_us = result.send_us + static_cast<int64_t>(ttfb_s * 1e6);
          result.completion_us = result.send_us + static_cast<int64_t>(total_s * 1e6);
        } catch (...) {
          result.completion_us = done_us;
        }
      }
      // Use response usage if available; otherwise estimate from SSE data chunks.
      result.tokens_generated =
          result.completion_tokens > 0 ? result.completion_tokens : count_sse_tokens(response_body);
      result.success = !response_body.empty() && response_body.find("\"error\"") == std::string::npos;
    } else {
      result.completion_us = done_us;
      result.success = !output.empty() && output.find("\"error\"") == std::string::npos;
      result.tokens_generated =
          result.completion_tokens > 0 ? result.completion_tokens : count_sse_tokens(output);
    }

    if (result.success) ++req_ok; else ++req_fail;

    if (!result.success) {
      result.error = output.substr(0, 200);
    }

    results.push_back(std::move(result));

    if (config.on_request_done) {
      config.on_request_done(static_cast<int>(i) + 1, total, req_ok, req_fail);
    }

    // Remove temp file
    std::remove(tmp_path.c_str());

    // Rate limiting
    if (interval_us > 0 && i + 1 < requests.size()) {
      const int64_t elapsed = now_us() - result.send_us;
      const int64_t sleep_us = static_cast<int64_t>(interval_us) - elapsed;
      if (sleep_us > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
      }
    }
  }

  return results;
}

std::string detect_model(const std::string& endpoint) {
  // Query /v1/models to find the served model name
  const std::string output = run_cmd(
      "curl -fsS --max-time 5 '" + endpoint + "/v1/models'");
  if (output.empty()) return "";
  return parse_model_from_models_response(output);
}

}  // namespace hotpath
