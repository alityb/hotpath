#include "rlprof/report.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <map>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rlprof {
namespace {

struct CategorySummary {
  std::int64_t total_ns = 0;
  std::int64_t calls = 0;
};

std::string repeat(char ch, std::size_t count) {
  return std::string(count, ch);
}

std::string format_fixed(double value, int precision) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string format_int(std::int64_t value) {
  std::string digits = std::to_string(value);
  std::string output;
  const std::size_t prefix = digits.size() % 3;
  if (prefix > 0) {
    output.append(digits.substr(0, prefix));
    if (digits.size() > prefix) {
      output.push_back(',');
    }
  }
  for (std::size_t i = prefix; i < digits.size(); i += 3) {
    output.append(digits.substr(i, 3));
    if (i + 3 < digits.size()) {
      output.push_back(',');
    }
  }
  return output.empty() ? "0" : output;
}

std::string format_optional_metric(
    std::string_view metric,
    const std::optional<double>& value) {
  if (!value.has_value()) {
    return "-";
  }

  if (metric == "vllm:gpu_cache_usage_perc" ||
      metric == "vllm:prefix_cache_hit_rate") {
    return format_fixed(*value, 2);
  }
  if (metric == "vllm:avg_generation_throughput_toks_per_s") {
    return format_int(static_cast<std::int64_t>(std::llround(*value)));
  }
  if (metric.find("_seconds_p50") != std::string_view::npos ||
      metric.find("_seconds_p99") != std::string_view::npos) {
    return format_fixed(*value * 1000.0, 1);
  }

  const double rounded = std::round(*value);
  if (std::fabs(*value - rounded) < 1e-9) {
    return format_int(static_cast<std::int64_t>(rounded));
  }
  return format_fixed(*value, 1);
}

std::string metric_label(std::string_view metric) {
  static const std::unordered_map<std::string, std::string> labels = {
      {"vllm:num_preemptions_total", "preemptions"},
      {"vllm:gpu_cache_usage_perc", "kv cache utilization"},
      {"vllm:num_requests_running", "requests running"},
      {"vllm:num_requests_waiting", "requests waiting"},
      {"vllm:avg_generation_throughput_toks_per_s", "generation throughput (tok/s)"},
      {"vllm:time_to_first_token_seconds_p50", "ttft p50 (ms)"},
      {"vllm:time_to_first_token_seconds_p99", "ttft p99 (ms)"},
      {"vllm:time_per_output_token_seconds_p50", "tpot p50 (ms)"},
      {"vllm:time_per_output_token_seconds_p99", "tpot p99 (ms)"},
      {"vllm:prefix_cache_hit_rate", "prefix cache hit rate"},
  };
  const auto it = labels.find(std::string(metric));
  if (it != labels.end()) {
    return it->second;
  }
  return std::string(metric);
}

std::optional<double> metadata_double(
    const std::map<std::string, std::string>& metadata,
    const std::string& key) {
  const auto it = metadata.find(key);
  if (it == metadata.end() || it->second.empty()) {
    return std::nullopt;
  }
  return std::stod(it->second);
}

std::string metadata_value(
    const std::map<std::string, std::string>& metadata,
    const std::string& key,
    const std::string& fallback = "-") {
  const auto it = metadata.find(key);
  return it == metadata.end() || it->second.empty() ? fallback : it->second;
}

std::vector<std::string> warning_messages(const std::map<std::string, std::string>& metadata) {
  std::vector<std::string> warnings;
  const auto append_if = [&](const std::string& key, const std::string& message) {
    const auto it = metadata.find(key);
    if (it != metadata.end() && it->second == "true") {
      warnings.push_back(message);
    }
  };

  append_if("warning_sm_clock_unstable", "sm clock varied materially during measurement");
  append_if("warning_power_capped", "power cap throttling observed");
  append_if("warning_thermal_slowdown", "thermal throttling observed");
  append_if("warning_any_clock_throttle", "clock throttling reasons were active");
  append_if("warning_temp_high", "gpu temperature reached high operating range");
  append_if(
      "warning_gpu_clocks_unlocked",
      "GPU clocks are not locked. Run `rlprof lock-clocks` for reproducible measurements. See: docs.nvidia.com/deploy/nvidia-smi/index.html");
  return warnings;
}

void append_row(
    std::ostringstream& output,
    const std::vector<std::pair<std::string, std::size_t>>& columns,
    const std::vector<std::string>& values) {
  for (std::size_t i = 0; i < columns.size(); ++i) {
    const bool left_align = (i == 0);
    output << std::setw(static_cast<int>(columns[i].second))
           << (left_align ? std::left : std::right)
           << values[i];
    if (i + 1 < columns.size()) {
      output << "  ";
    }
  }
  output << '\n';
}

}  // namespace

std::string render_report(
    const ReportMeta& meta,
    const std::map<std::string, std::string>& metadata,
    const std::vector<profiler::KernelRecord>& kernels,
    const std::vector<MetricSummary>& metrics_summary,
    const TrafficStats& traffic_stats) {
  std::ostringstream output;
  output << "rlprof | " << meta.model_name << " | " << meta.gpu_name
         << " | " << meta.vllm_version << "\n";
  output << "workload: " << meta.prompts << " prompts, " << meta.rollouts
         << " rollouts/prompt, " << meta.max_tokens << " max tokens\n\n";
  output << "category buckets use conservative substring matching; raw kernel names are authoritative\n\n";

  const bool cluster_mode = metadata_value(metadata, "cluster_mode", "false") == "true";

  const auto warnings = warning_messages(metadata);
  if (!warnings.empty()) {
    output << "MEASUREMENT WARNINGS\n";
    for (const auto& warning : warnings) {
      output << "- " << warning << "\n";
    }
    output << "\n";
  }

  const auto sm_min = metadata_double(metadata, "measurement_sm_clock_min_mhz");
  const auto sm_avg = metadata_double(metadata, "measurement_sm_clock_avg_mhz");
  const auto sm_max = metadata_double(metadata, "measurement_sm_clock_max_mhz");
  const auto mem_min = metadata_double(metadata, "measurement_mem_clock_min_mhz");
  const auto mem_avg = metadata_double(metadata, "measurement_mem_clock_avg_mhz");
  const auto mem_max = metadata_double(metadata, "measurement_mem_clock_max_mhz");
  const auto temp_min = metadata_double(metadata, "measurement_temp_min_c");
  const auto temp_max = metadata_double(metadata, "measurement_temp_max_c");
  const auto power_avg = metadata_double(metadata, "measurement_power_draw_avg_w");
  const auto power_peak = metadata_double(metadata, "measurement_power_draw_peak_w");
  const auto power_limit = metadata_double(metadata, "measurement_power_limit_w");

  if (sm_avg.has_value() || temp_max.has_value()) {
    output << "MEASUREMENT CONTEXT\n";
    const std::vector<std::pair<std::string, std::size_t>> context_columns = {
        {"metric", 30},
        {"value", 24},
    };
    append_row(output, context_columns, {"metric", "value"});
    output << repeat('-', 56) << '\n';
    append_row(output, context_columns, {"driver version", metadata_value(metadata, "measurement_driver_version")});
    append_row(output, context_columns, {"persistence mode", metadata_value(metadata, "measurement_persistence_mode")});
    append_row(output, context_columns, {"gpu clock policy", metadata_value(metadata, "measurement_gpu_clock_policy")});
    if (cluster_mode) {
      append_row(output, context_columns, {"cluster endpoints", metadata_value(metadata, "cluster_endpoint_count")});
      append_row(output, context_columns, {"peer endpoints", metadata_value(metadata, "cluster_peer_endpoint_count")});
      append_row(output, context_columns, {"trace scope", metadata_value(metadata, "cluster_trace_scope")});
    }
    if (metadata_value(metadata, "measurement_gpu_max_sm_clock_mhz", "").size() > 0) {
      append_row(
          output,
          context_columns,
          {"max supported sm clock mhz",
           metadata_value(metadata, "measurement_gpu_max_sm_clock_mhz")});
    }
    append_row(output, context_columns, {"observed pstate(s)", metadata_value(metadata, "measurement_pstates")});
    append_row(output, context_columns, {"gpu telemetry samples", metadata_value(metadata, "measurement_samples")});
    if (sm_avg.has_value()) {
      append_row(
          output,
          context_columns,
          {"sm clock (min/avg/max mhz)",
           format_fixed(*sm_min, 0) + " / " + format_fixed(*sm_avg, 0) + " / " +
               format_fixed(*sm_max, 0)});
    }
    if (mem_avg.has_value()) {
      append_row(
          output,
          context_columns,
          {"mem clock (min/avg/max mhz)",
           format_fixed(*mem_min, 0) + " / " + format_fixed(*mem_avg, 0) + " / " +
               format_fixed(*mem_max, 0)});
    }
    if (temp_max.has_value()) {
      append_row(
          output,
          context_columns,
          {"temperature (min/max c)",
           format_fixed(*temp_min, 0) + " / " + format_fixed(*temp_max, 0)});
    }
    if (power_avg.has_value()) {
      append_row(
          output,
          context_columns,
          {"power draw (avg/peak/limit w)",
           format_fixed(*power_avg, 1) + " / " + format_fixed(*power_peak, 1) + " / " +
               format_fixed(*power_limit, 1)});
    }
    output << '\n';
  }

  const std::int64_t total_ns = std::accumulate(
      kernels.begin(),
      kernels.end(),
      std::int64_t{0},
      [](std::int64_t total, const profiler::KernelRecord& record) {
        return total + record.total_ns;
      });

  std::map<std::string, CategorySummary> category_map;
  for (const profiler::KernelRecord& kernel : kernels) {
    CategorySummary& summary = category_map[kernel.category];
    summary.total_ns += kernel.total_ns;
    summary.calls += kernel.calls;
  }
  std::vector<std::pair<std::string, CategorySummary>> categories(
      category_map.begin(), category_map.end());
  std::sort(
      categories.begin(),
      categories.end(),
      [](const auto& lhs, const auto& rhs) {
        return lhs.second.total_ns > rhs.second.total_ns;
      });

  output << "KERNEL BREAKDOWN BY CATEGORY\n";
  const std::vector<std::pair<std::string, std::size_t>> category_columns = {
      {"category", 16},
      {"time (ms)", 9},
      {"%", 6},
      {"calls", 7},
      {"avg (us)", 9},
  };
  append_row(output, category_columns, {"category", "time (ms)", "%", "calls", "avg (us)"});
  output << repeat('-', 56) << '\n';
  for (const auto& [category, summary] : categories) {
    const double avg_ns =
        summary.calls == 0 ? 0.0
                           : static_cast<double>(summary.total_ns) / summary.calls;
    const double percent =
        total_ns == 0 ? 0.0
                      : (static_cast<double>(summary.total_ns) / total_ns) * 100.0;
    append_row(
        output,
        category_columns,
        {
            category,
            format_fixed(summary.total_ns / 1'000'000.0, 1),
            format_fixed(percent, 1),
            std::to_string(summary.calls),
            format_fixed(avg_ns / 1'000.0, 1),
        });
  }
  output << repeat('-', 56) << '\n';
  append_row(
      output,
      category_columns,
      {
          "total",
          format_fixed(total_ns / 1'000'000.0, 1),
          total_ns == 0 ? "0.0" : "100.0",
          std::to_string(std::accumulate(
              kernels.begin(),
              kernels.end(),
              std::int64_t{0},
              [](std::int64_t total, const profiler::KernelRecord& record) {
                return total + record.calls;
              })),
          "",
      });
  output << '\n';

  output << "TOP 10 KERNELS BY TOTAL TIME\n";
  const std::vector<std::pair<std::string, std::size_t>> kernel_columns = {
      {"kernel", 40},
      {"time (ms)", 9},
      {"%", 6},
      {"calls", 7},
      {"avg (us)", 9},
  };
  append_row(output, kernel_columns, {"kernel", "time (ms)", "%", "calls", "avg (us)"});
  output << repeat('-', 79) << '\n';
  std::vector<profiler::KernelRecord> sorted_kernels = kernels;
  std::sort(
      sorted_kernels.begin(),
      sorted_kernels.end(),
      [](const profiler::KernelRecord& lhs, const profiler::KernelRecord& rhs) {
        return lhs.total_ns > rhs.total_ns;
      });
  for (std::size_t i = 0; i < sorted_kernels.size() && i < 10; ++i) {
    const profiler::KernelRecord& kernel = sorted_kernels[i];
    const double percent =
        total_ns == 0 ? 0.0 : (static_cast<double>(kernel.total_ns) / total_ns) * 100.0;
    append_row(
        output,
        kernel_columns,
        {
            kernel.name,
            format_fixed(kernel.total_ns / 1'000'000.0, 1),
            format_fixed(percent, 1),
            std::to_string(kernel.calls),
            format_fixed(kernel.avg_ns / 1'000.0, 1),
        });
  }
  output << '\n';

  std::vector<profiler::KernelRecord> uncategorized_kernels;
  for (const profiler::KernelRecord& kernel : sorted_kernels) {
    if (kernel.category == "other") {
      uncategorized_kernels.push_back(kernel);
    }
  }
  if (!uncategorized_kernels.empty()) {
    output << "TOP UNCATEGORIZED KERNELS\n";
    append_row(output, kernel_columns, {"kernel", "time (ms)", "%", "calls", "avg (us)"});
    output << repeat('-', 79) << '\n';
    for (std::size_t i = 0; i < uncategorized_kernels.size() && i < 10; ++i) {
      const profiler::KernelRecord& kernel = uncategorized_kernels[i];
      const double percent =
          total_ns == 0 ? 0.0 : (static_cast<double>(kernel.total_ns) / total_ns) * 100.0;
      append_row(
          output,
          kernel_columns,
          {
              kernel.name,
              format_fixed(kernel.total_ns / 1'000'000.0, 1),
              format_fixed(percent, 1),
              std::to_string(kernel.calls),
              format_fixed(kernel.avg_ns / 1'000.0, 1),
          });
    }
    output << '\n';
  }

  output << (cluster_mode ? "CLUSTER VLLM METRICS\n" : "VLLM SERVER METRICS\n");
  const std::vector<std::pair<std::string, std::size_t>> metric_columns = {
      {"metric", 30},
      {"avg", 10},
      {"peak", 10},
  };
  append_row(output, metric_columns, {"metric", "avg", "peak"});
  output << repeat('-', 54) << '\n';
  for (const MetricSummary& summary : metrics_summary) {
    append_row(
        output,
        metric_columns,
        {
            metric_label(summary.metric),
            format_optional_metric(summary.metric, summary.avg),
            format_optional_metric(summary.metric, summary.peak),
        });
  }
  output << '\n';

  output << "TRAFFIC SHAPE\n";
  const std::vector<std::pair<std::string, std::size_t>> traffic_columns = {
      {"metric", 30},
      {"value", 12},
  };
  append_row(output, traffic_columns, {"metric", "value"});
  output << repeat('-', 44) << '\n';
  append_row(output, traffic_columns, {"total requests", format_int(traffic_stats.total_requests)});
  append_row(
      output,
      traffic_columns,
      {"completion length mean",
       traffic_stats.completion_length_mean.has_value()
           ? format_int(static_cast<std::int64_t>(std::llround(*traffic_stats.completion_length_mean)))
           : "-"});
  append_row(
      output,
      traffic_columns,
      {"completion length p50",
       traffic_stats.completion_length_p50.has_value()
           ? format_int(static_cast<std::int64_t>(std::llround(*traffic_stats.completion_length_p50)))
           : "-"});
  append_row(
      output,
      traffic_columns,
      {"completion length p99",
       traffic_stats.completion_length_p99.has_value()
           ? format_int(static_cast<std::int64_t>(std::llround(*traffic_stats.completion_length_p99)))
           : "-"});
  append_row(
      output,
      traffic_columns,
      {"max/median ratio",
       traffic_stats.max_median_ratio.has_value()
           ? format_fixed(*traffic_stats.max_median_ratio, 2) + "x"
           : "-"});
  append_row(output, traffic_columns, {"errors", format_int(traffic_stats.errors)});

  return output.str();
}

}  // namespace rlprof
