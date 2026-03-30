#include "rlprof/profiler/vllm_metrics.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>

namespace rlprof::profiler {
namespace {

const std::unordered_set<std::string> kKeyMetrics = {
    "vllm:num_preemptions_total",
    "vllm:gpu_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:time_to_first_token_seconds_p50",
    "vllm:time_to_first_token_seconds_p99",
    "vllm:time_per_output_token_seconds_p50",
    "vllm:time_per_output_token_seconds_p99",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:avg_generation_throughput_toks_per_s",
    "vllm:prefix_cache_hit_rate",
};

std::string run_command(const std::string& command) {
  std::array<char, 4096> buffer{};
  std::string output;
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    throw std::runtime_error("failed to run command: " + command);
  }

  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output.append(buffer.data());
  }

  const int rc = pclose(pipe);
  if (rc != 0) {
    throw std::runtime_error("command failed: " + command);
  }
  return output;
}

double now_seconds() {
  using Clock = std::chrono::system_clock;
  const auto now = Clock::now().time_since_epoch();
  return std::chrono::duration<double>(now).count();
}

bool aggregate_by_average(const std::string& metric) {
  return metric == "vllm:gpu_cache_usage_perc" ||
         metric == "vllm:prefix_cache_hit_rate" ||
         metric.find("_seconds_p50") != std::string::npos ||
         metric.find("_seconds_p99") != std::string::npos;
}

}  // namespace

std::unordered_map<std::string, double> parse_metrics_text(const std::string& text) {
  std::unordered_map<std::string, double> metrics;
  std::istringstream stream(text);
  std::string line;

  while (std::getline(stream, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    const std::size_t split = line.find_last_of(' ');
    if (split == std::string::npos) {
      continue;
    }
    const std::string name_with_labels = line.substr(0, split);
    const std::string value_text = line.substr(split + 1);
    const std::size_t brace = name_with_labels.find('{');
    const std::string metric_name =
        brace == std::string::npos ? name_with_labels : name_with_labels.substr(0, brace);

    if (!kKeyMetrics.contains(metric_name)) {
      continue;
    }
    metrics[metric_name] = std::stod(value_text);
  }

  return metrics;
}

std::vector<MetricSummary> summarize_samples(const std::vector<MetricSample>& samples) {
  std::unordered_map<std::string, std::vector<double>> grouped;
  bool saw_cluster_samples = false;
  for (const MetricSample& sample : samples) {
    if (sample.source == "cluster") {
      saw_cluster_samples = true;
      break;
    }
  }
  for (const MetricSample& sample : samples) {
    if (saw_cluster_samples && sample.source != "cluster") {
      continue;
    }
    grouped[sample.metric].push_back(sample.value);
  }

  std::vector<MetricSummary> summaries;
  for (auto& [metric, values] : grouped) {
    const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    double sum = 0.0;
    for (double value : values) {
      sum += value;
    }
    summaries.push_back(MetricSummary{
        .metric = metric,
        .avg = sum / static_cast<double>(values.size()),
        .peak = *max_it,
        .min = *min_it,
    });
  }

  std::sort(
      summaries.begin(),
      summaries.end(),
      [](const MetricSummary& lhs, const MetricSummary& rhs) {
        return lhs.metric < rhs.metric;
      });
  return summaries;
}

std::vector<MetricSample> fetch_metrics_once(const std::string& server_url) {
  return fetch_metrics_once(
      std::vector<MetricEndpoint>{{.source = "cluster", .server_url = server_url}});
}

std::vector<MetricSample> fetch_metrics_once(
    const std::vector<MetricEndpoint>& endpoints) {
  if (endpoints.empty()) {
    return {};
  }

  const double sample_time = now_seconds();
  std::vector<MetricSample> samples;
  std::unordered_map<std::string, std::vector<double>> aggregate_values;

  for (const auto& endpoint : endpoints) {
    try {
      const std::string command = "curl -fsS " + endpoint.server_url + "/metrics";
      const std::string body = run_command(command);
      for (const auto& [metric, value] : parse_metrics_text(body)) {
        if (endpoints.size() > 1) {
          samples.push_back(MetricSample{
              .sample_time = sample_time,
              .source = endpoint.source,
              .metric = metric,
              .value = value,
          });
        }
        aggregate_values[metric].push_back(value);
      }
    } catch (const std::exception&) {
    }
  }

  for (auto& [metric, values] : aggregate_values) {
    double aggregate = 0.0;
    if (aggregate_by_average(metric)) {
      for (double value : values) {
        aggregate += value;
      }
      aggregate /= static_cast<double>(values.size());
    } else {
      for (double value : values) {
        aggregate += value;
      }
    }
    samples.push_back(MetricSample{
        .sample_time = sample_time,
        .source = "cluster",
        .metric = metric,
        .value = aggregate,
    });
  }

  return samples;
}

std::vector<MetricSample> poll_metrics(
    const std::string& server_url,
    std::chrono::milliseconds interval,
    const std::atomic<bool>& stop_flag) {
  return poll_metrics(
      std::vector<MetricEndpoint>{{.source = "cluster", .server_url = server_url}},
      interval,
      stop_flag);
}

std::vector<MetricSample> poll_metrics(
    const std::vector<MetricEndpoint>& endpoints,
    std::chrono::milliseconds interval,
    const std::atomic<bool>& stop_flag) {
  std::vector<MetricSample> samples;

  while (!stop_flag.load()) {
    try {
      const std::vector<MetricSample> batch = fetch_metrics_once(endpoints);
      samples.insert(samples.end(), batch.begin(), batch.end());
    } catch (const std::exception&) {
    }

    auto slept = std::chrono::milliseconds::zero();
    while (!stop_flag.load() && slept < interval) {
      constexpr auto kSlice = std::chrono::milliseconds(50);
      std::this_thread::sleep_for(kSlice);
      slept += kSlice;
    }
  }

  return samples;
}

}  // namespace rlprof::profiler
