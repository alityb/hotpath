#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "hotpath/sglang_metrics.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  const std::string fixture = R"(
# HELP sglang:num_running_req Number of running requests
# TYPE sglang:num_running_req gauge
sglang:num_running_req 16
# HELP sglang:num_waiting_req Number of waiting requests
# TYPE sglang:num_waiting_req gauge
sglang:num_waiting_req 4
# HELP sglang:token_usage Token usage ratio
# TYPE sglang:token_usage gauge
sglang:token_usage 0.73
# HELP sglang:cache_hit_rate Cache hit rate
# TYPE sglang:cache_hit_rate gauge
sglang:cache_hit_rate 0.42
# HELP sglang:num_total_tokens Total tokens processed
# TYPE sglang:num_total_tokens counter
sglang:num_total_tokens 125000
)";

  // Test raw parsing
  const auto pairs = hotpath::parse_sglang_metrics_text(fixture);
  expect_true(pairs.size() == 5, "expected 5 metric pairs, got " + std::to_string(pairs.size()));

  // Test structured parsing
  const auto m = hotpath::parse_sglang_metrics(fixture);
  expect_true(std::abs(m.num_running_req - 16.0) < 0.01,
              "num_running_req: " + std::to_string(m.num_running_req));
  expect_true(std::abs(m.num_waiting_req - 4.0) < 0.01,
              "num_waiting_req: " + std::to_string(m.num_waiting_req));
  expect_true(std::abs(m.token_usage - 0.73) < 0.01,
              "token_usage: " + std::to_string(m.token_usage));
  expect_true(std::abs(m.cache_hit_rate - 0.42) < 0.01,
              "cache_hit_rate: " + std::to_string(m.cache_hit_rate));
  expect_true(std::abs(m.num_total_tokens - 125000.0) < 1.0,
              "num_total_tokens: " + std::to_string(m.num_total_tokens));

  // Test conversion to MetricSnapshot
  const auto snap = hotpath::sglang_to_snapshot(m, 1000000);
  expect_true(snap.timestamp_us == 1000000, "timestamp mismatch");
  expect_true(std::abs(snap.batch_size - 16.0) < 0.01, "batch_size mismatch");
  expect_true(std::abs(snap.queue_depth - 4.0) < 0.01, "queue_depth mismatch");

  // Test with underscore format (some SGLang versions use _ instead of :)
  const std::string underscore_fixture = R"(
sglang_num_running_req 8
sglang_num_waiting_req 2
sglang_token_usage 0.55
sglang_cache_hit_rate 0.30
sglang_num_total_tokens 50000
)";

  const auto m2 = hotpath::parse_sglang_metrics(underscore_fixture);
  expect_true(std::abs(m2.num_running_req - 8.0) < 0.01, "underscore num_running_req");
  expect_true(std::abs(m2.cache_hit_rate - 0.30) < 0.01, "underscore cache_hit_rate");

  const std::string plural_fixture = R"(
sglang:num_running_reqs 6
sglang:num_queue_reqs 3
sglang:token_usage 0.25
)";
  const auto m3 = hotpath::parse_sglang_metrics(plural_fixture);
  expect_true(std::abs(m3.num_running_req - 6.0) < 0.01, "plural num_running_reqs");
  expect_true(std::abs(m3.num_waiting_req - 3.0) < 0.01, "plural num_queue_reqs");

  // Test empty input
  const auto empty = hotpath::parse_sglang_metrics("");
  expect_true(empty.num_running_req == 0.0, "empty should be zero");

  std::cerr << "test_sglang_metrics: all tests passed\n";
  return 0;
}
