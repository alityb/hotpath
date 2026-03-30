#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "rlprof/profiler/vllm_metrics.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  const std::string text = R"TXT(
# HELP vllm:num_preemptions_total Cumulative number of preemptions
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{model_name="Qwen"} 47.0
vllm:gpu_cache_usage_perc{model_name="Qwen"} 0.723
vllm:num_requests_running{model_name="Qwen"} 84
ignored_metric 123
)TXT";

  const auto parsed = rlprof::profiler::parse_metrics_text(text);
  expect_true(parsed.size() == 3, "expected three parsed metrics");
  expect_true(parsed.at("vllm:num_preemptions_total") == 47.0, "bad preemption parse");
  expect_true(parsed.at("vllm:gpu_cache_usage_perc") == 0.723, "bad cache parse");
  expect_true(parsed.at("vllm:num_requests_running") == 84.0, "bad running parse");

  const std::vector<rlprof::MetricSample> samples = {
      {.sample_time = 0.0, .source = "cluster", .metric = "vllm:num_preemptions_total", .value = 3.0},
      {.sample_time = 1.0, .source = "cluster", .metric = "vllm:num_preemptions_total", .value = 5.0},
      {.sample_time = 0.0, .source = "cluster", .metric = "vllm:gpu_cache_usage_perc", .value = 0.5},
      {.sample_time = 1.0, .source = "cluster", .metric = "vllm:gpu_cache_usage_perc", .value = 0.75},
      {.sample_time = 0.0, .source = "peer1", .metric = "vllm:num_preemptions_total", .value = 30.0},
      {.sample_time = 1.0, .source = "peer1", .metric = "vllm:num_preemptions_total", .value = 50.0},
  };

  const auto summaries = rlprof::profiler::summarize_samples(samples);
  expect_true(summaries.size() == 2, "expected two summaries");
  expect_true(summaries[0].metric == "vllm:gpu_cache_usage_perc", "unexpected first summary metric");
  expect_true(summaries[0].avg.has_value() && *summaries[0].avg == 0.625, "unexpected first avg");
  expect_true(summaries[0].peak.has_value() && *summaries[0].peak == 0.75, "unexpected first peak");
  expect_true(summaries[0].min.has_value() && *summaries[0].min == 0.5, "unexpected first min");
  expect_true(summaries[1].metric == "vllm:num_preemptions_total", "unexpected second summary metric");
  expect_true(summaries[1].avg.has_value() && *summaries[1].avg == 4.0, "unexpected second avg");
  expect_true(summaries[1].peak.has_value() && *summaries[1].peak == 5.0, "unexpected second peak");
  expect_true(summaries[1].min.has_value() && *summaries[1].min == 3.0, "unexpected second min");

  return 0;
}
