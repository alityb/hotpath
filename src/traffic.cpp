#include "rlprof/traffic.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rlprof {
namespace {

std::string json_escape(const std::string& input) {
  std::string output;
  output.reserve(input.size());
  for (char ch : input) {
    switch (ch) {
      case '\\':
        output += "\\\\";
        break;
      case '"':
        output += "\\\"";
        break;
      case '\n':
        output += "\\n";
        break;
      default:
        output.push_back(ch);
        break;
    }
  }
  return output;
}

std::pair<long, std::string> run_curl_json(
    const std::string& server_url,
    const std::string& payload) {
  namespace fs = std::filesystem;
  const auto nonce =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path temp_path =
      fs::temp_directory_path() / ("rlprof_payload_" + std::to_string(nonce) + ".json");

  {
    std::ofstream out(temp_path);
    out << payload;
  }

  const std::string command =
      "curl -sS -o - -w '\\n%{http_code}' -H 'Content-Type: application/json' "
      "-X POST --data @" +
      temp_path.string() + " " + server_url + "/v1/completions";

  std::array<char, 4096> buffer{};
  std::string output;
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    fs::remove(temp_path);
    throw std::runtime_error("failed to run curl");
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output.append(buffer.data());
  }
  const int rc = pclose(pipe);
  fs::remove(temp_path);
  if (rc != 0) {
    throw std::runtime_error("curl request failed");
  }

  const std::size_t split = output.find_last_of('\n');
  if (split == std::string::npos) {
    return {0L, output};
  }
  const long status = std::stol(output.substr(split + 1));
  return {status, output.substr(0, split)};
}

std::int64_t parse_completion_tokens(const std::string& body, std::int64_t fallback) {
  const std::string needle = "\"completion_tokens\":";
  const std::size_t start = body.find(needle);
  if (start == std::string::npos) {
    return fallback;
  }

  std::size_t index = start + needle.size();
  while (index < body.size() && std::isspace(static_cast<unsigned char>(body[index]))) {
    ++index;
  }
  std::size_t end = index;
  while (end < body.size() && std::isdigit(static_cast<unsigned char>(body[end]))) {
    ++end;
  }
  if (end == index) {
    return fallback;
  }
  return std::stoll(body.substr(index, end - index));
}

double percentile(std::vector<std::int64_t> values, double quantile) {
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(
      std::llround((values.size() - 1) * quantile));
  return static_cast<double>(values[index]);
}

}  // namespace

std::vector<TrafficRequest> generate_requests(
    std::int64_t num_prompts,
    std::int64_t rollouts_per_prompt,
    std::int64_t input_len,
    std::int64_t min_tokens,
    std::int64_t max_tokens,
    std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<std::int64_t> lengths(min_tokens, max_tokens);
  std::vector<TrafficRequest> requests;

  for (std::int64_t prompt_index = 0; prompt_index < num_prompts; ++prompt_index) {
    std::ostringstream prompt;
    prompt << "Solve step by step: Problem " << prompt_index << ". ";
    for (std::int64_t token = 0; token < input_len; ++token) {
      prompt << "x ";
    }
    const std::string prompt_text = prompt.str();

    for (std::int64_t rollout = 0; rollout < rollouts_per_prompt; ++rollout) {
      requests.push_back(TrafficRequest{
          .prompt = prompt_text,
          .output_len = lengths(rng),
      });
    }
  }
  return requests;
}

TrafficResult send_request(
    const std::string& server_url,
    const TrafficRequest& request) {
  try {
    std::ostringstream payload;
    payload << "{\"prompt\":\"" << json_escape(request.prompt)
            << "\",\"max_tokens\":" << request.output_len
            << ",\"temperature\":1.0}";
    const auto [status, body] = run_curl_json(server_url, payload.str());
    return TrafficResult{
        .ok = status >= 200 && status < 300,
        .http_status = status,
        .completion_tokens = parse_completion_tokens(body, request.output_len),
        .body = body,
        .error = "",
    };
  } catch (const std::exception& exc) {
    return TrafficResult{
        .ok = false,
        .http_status = 0,
        .completion_tokens = 0,
        .body = "",
        .error = exc.what(),
    };
  }
}

TrafficStats summarize_traffic(
    const std::vector<TrafficResult>& results,
    const std::vector<TrafficRequest>& requests) {
  std::vector<std::int64_t> lengths;
  std::int64_t errors = 0;
  for (std::size_t index = 0; index < results.size(); ++index) {
    if (!results[index].ok) {
      ++errors;
      continue;
    }
    lengths.push_back(results[index].completion_tokens > 0
                          ? results[index].completion_tokens
                          : requests[index].output_len);
  }

  if (lengths.empty()) {
    return TrafficStats{
        .total_requests = static_cast<std::int64_t>(results.size()),
        .completion_length_mean = std::nullopt,
        .completion_length_p50 = std::nullopt,
        .completion_length_p99 = std::nullopt,
        .max_median_ratio = std::nullopt,
        .errors = errors,
    };
  }

  double sum = 0.0;
  for (std::int64_t length : lengths) {
    sum += static_cast<double>(length);
  }
  const double p50 = percentile(lengths, 0.50);
  const double p99 = percentile(lengths, 0.99);
  const auto max_it = std::max_element(lengths.begin(), lengths.end());

  return TrafficStats{
      .total_requests = static_cast<std::int64_t>(results.size()),
      .completion_length_mean = sum / static_cast<double>(lengths.size()),
      .completion_length_p50 = p50,
      .completion_length_p99 = p99,
      .max_median_ratio = p50 == 0.0 ? std::nullopt : std::optional<double>(*max_it / p50),
      .errors = errors,
  };
}

TrafficRun fire_rl_traffic(
    const std::string& server_url,
    std::int64_t num_prompts,
    std::int64_t rollouts_per_prompt,
    std::int64_t min_tokens,
    std::int64_t max_tokens,
    std::int64_t input_len,
    std::uint32_t seed) {
  return fire_rl_traffic(
      std::vector<std::string>{server_url},
      num_prompts,
      rollouts_per_prompt,
      min_tokens,
      max_tokens,
      input_len,
      seed);
}

TrafficRun fire_rl_traffic(
    const std::vector<std::string>& server_urls,
    std::int64_t num_prompts,
    std::int64_t rollouts_per_prompt,
    std::int64_t min_tokens,
    std::int64_t max_tokens,
    std::int64_t input_len,
    std::uint32_t seed) {
  if (server_urls.empty()) {
    throw std::runtime_error("at least one server url is required");
  }

  const std::vector<TrafficRequest> requests = generate_requests(
      num_prompts,
      rollouts_per_prompt,
      input_len,
      min_tokens,
      max_tokens,
      seed);

  std::vector<std::future<TrafficResult>> futures;
  futures.reserve(requests.size());
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const std::string& server_url =
        server_urls[i % server_urls.size()];
    futures.push_back(
        std::async(std::launch::async, send_request, server_url, requests[i]));
  }

  std::vector<TrafficResult> results;
  results.reserve(futures.size());
  for (auto& future : futures) {
    results.push_back(future.get());
  }

  return TrafficRun{
      .results = results,
      .stats = summarize_traffic(results, requests),
  };
}

}  // namespace rlprof
