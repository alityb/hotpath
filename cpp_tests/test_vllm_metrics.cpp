#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

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
vllm:num_preemptions_total{model_name="Qwen",worker="1"} 49.0
vllm:gpu_cache_usage_perc{model_name="Qwen"} 0.723
vllm:num_requests_running{model_name="Qwen"} 84
ignored_metric 123
)TXT";

  const auto parsed = rlprof::profiler::parse_metrics_text(text);
  expect_true(parsed.size() == 4, "expected all labeled series to be preserved");
  std::multimap<std::string, double> values(parsed.begin(), parsed.end());
  expect_true(values.count("vllm:num_preemptions_total") == 2, "expected two preemption series");
  expect_true(values.count("vllm:gpu_cache_usage_perc") == 1, "bad cache parse");
  expect_true(values.count("vllm:num_requests_running") == 1, "bad running parse");
  expect_true(values.find("vllm:gpu_cache_usage_perc")->second == 0.723, "bad cache value");
  expect_true(values.find("vllm:num_requests_running")->second == 84.0, "bad running value");

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

  namespace fs = std::filesystem;
  const fs::path temp_root = fs::temp_directory_path() / "rlprof_test_vllm_metrics";
  fs::remove_all(temp_root);
  fs::create_directories(temp_root);
  const fs::path curl_path = temp_root / "curl";
  {
    std::ofstream script(curl_path);
    script << "#!/bin/sh\n";
    script << "last=\"\"\n";
    script << "for arg in \"$@\"; do\n";
    script << "  last=\"$arg\"\n";
    script << "done\n";
    script << "case \"$last\" in\n";
    script << "  *node0*)\n";
    script << "    cat <<'EOF'\n";
    script << "vllm:num_requests_running 2\n";
    script << "vllm:time_to_first_token_seconds_p99 0.5\n";
    script << "EOF\n";
    script << "    ;;\n";
    script << "  *node1*)\n";
    script << "    cat <<'EOF'\n";
    script << "vllm:num_requests_running 3\n";
    script << "vllm:time_to_first_token_seconds_p99 1.5\n";
    script << "EOF\n";
    script << "    ;;\n";
    script << "esac\n";
  }
  chmod(curl_path.c_str(), 0755);

  const std::string original_path = std::getenv("PATH") == nullptr ? "" : std::getenv("PATH");
  setenv("PATH", (temp_root.string() + ":" + original_path).c_str(), 1);
  const auto fetched = rlprof::profiler::fetch_metrics_once({
      {.source = "local", .server_url = "http://node0"},
      {.source = "peer1", .server_url = "http://node1"},
  });
  setenv("PATH", original_path.c_str(), 1);

  bool saw_cluster_running = false;
  bool saw_cluster_p99 = false;
  for (const auto& sample : fetched) {
    if (sample.source == "cluster" && sample.metric == "vllm:num_requests_running" &&
        sample.value == 5.0) {
      saw_cluster_running = true;
    }
    if (sample.source == "cluster" &&
        sample.metric == "vllm:time_to_first_token_seconds_p99") {
      saw_cluster_p99 = true;
    }
  }
  expect_true(saw_cluster_running, "expected summed cluster running-request metric");
  expect_true(!saw_cluster_p99, "cluster percentile metrics should not be synthesized");

  fs::remove_all(temp_root);

  return 0;
}
