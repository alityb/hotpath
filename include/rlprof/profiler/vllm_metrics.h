#pragma once

#include <atomic>
#include <chrono>
#include <utility>
#include <string>
#include <vector>

#include "rlprof/report.h"
#include "rlprof/store.h"

namespace rlprof::profiler {

struct MetricEndpoint {
  std::string source;
  std::string server_url;
};

std::vector<std::pair<std::string, double>> parse_metrics_text(const std::string& text);

std::vector<MetricSummary> summarize_samples(const std::vector<MetricSample>& samples);

std::vector<MetricSample> fetch_metrics_once(const std::string& server_url);

std::vector<MetricSample> fetch_metrics_once(
    const std::vector<MetricEndpoint>& endpoints);

std::vector<MetricSample> poll_metrics(
    const std::string& server_url,
    std::chrono::milliseconds interval,
    const std::atomic<bool>& stop_flag);

std::vector<MetricSample> poll_metrics(
    const std::vector<MetricEndpoint>& endpoints,
    std::chrono::milliseconds interval,
    const std::atomic<bool>& stop_flag);

}  // namespace rlprof::profiler
