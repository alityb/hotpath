#include <cstdlib>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "rlprof/traffic.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

void expect_throws(const std::function<void()>& fn, const std::string& message) {
  try {
    fn();
  } catch (const std::runtime_error&) {
    return;
  }
  std::cerr << message << "\n";
  std::exit(1);
}

}  // namespace

int main() {
  expect_throws(
      []() { static_cast<void>(rlprof::generate_requests(1, 1, 32, 9, 3)); },
      "expected invalid token bounds to throw");
  expect_throws(
      []() { static_cast<void>(rlprof::generate_requests(1, 1, 0, 1, 3)); },
      "expected invalid input length to throw");

  const std::vector<rlprof::TrafficRequest> requests = {{
      .prompt = "p",
      .output_len = 7,
  }};
  const std::vector<rlprof::TrafficResult> results = {{
      .ok = true,
      .http_status = 200,
      .completion_tokens = std::nullopt,
      .body = "{\"choices\":[{\"text\":\"hi\"}]}",
      .error = "",
  }};

  const auto stats = rlprof::summarize_traffic(results, requests);
  expect_true(stats.total_requests == 1, "expected total requests");
  expect_true(!stats.completion_length_mean.has_value(), "missing usage should not invent a mean");
  expect_true(!stats.completion_length_p50.has_value(), "missing usage should not invent p50");
  expect_true(!stats.completion_length_p99.has_value(), "missing usage should not invent p99");
  expect_true(!stats.max_median_ratio.has_value(), "missing usage should not invent ratio");
  expect_true(stats.errors == 0, "expected no errors");

  return 0;
}
