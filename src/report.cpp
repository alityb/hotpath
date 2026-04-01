#include "rlprof/report.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

#include <sys/ioctl.h>
#include <unistd.h>

namespace rlprof {
namespace {

// ── Layout constants ──

constexpr int kMinWidth = 80;
constexpr int kDefaultWidth = 80;
constexpr int kMinBarWidth = 12;
constexpr int kMaxBarWidth = 40;
constexpr int kCategoryTableFixedCols = 56;  // category(16) + time(9) + %(6) + calls(7) + avg(9) + gaps(9)

int detect_terminal_width() {
  struct winsize ws{};
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
    return std::max(static_cast<int>(ws.ws_col), kMinWidth);
  }
  return kDefaultWidth;
}

struct CategorySummary {
  std::int64_t total_ns = 0;
  std::int64_t calls = 0;
};

struct Colors {
  const char* reset;
  const char* bold;
  const char* dim;
  const char* red;
  const char* green;
  const char* yellow;
  const char* cyan;
  const char* white;
};

Colors make_colors(bool enabled) {
  if (!enabled) {
    return {"", "", "", "", "", "", "", ""};
  }
  return {
      "\033[0m",   // reset
      "\033[1m",   // bold
      "\033[2m",   // dim
      "\033[31m",  // red
      "\033[32m",  // green
      "\033[33m",  // yellow
      "\033[36m",  // cyan
      "\033[37m",  // white
  };
}

// ── Formatting helpers ──

std::string rule(int width) {
  std::string result;
  result.reserve(static_cast<std::size_t>(width) * 3);
  for (int i = 0; i < width; ++i) {
    result += "\xe2\x94\x80";  // ─
  }
  return result;
}

std::string heavy_rule(int width) {
  std::string result;
  result.reserve(static_cast<std::size_t>(width) * 3);
  for (int i = 0; i < width; ++i) {
    result += "\xe2\x95\x90";  // ═
  }
  return result;
}

std::string section_header(const Colors& c, const std::string& title, int total_width) {
  // "── TITLE ─────────" filling to total_width
  const int title_len = static_cast<int>(title.size());
  const int remaining = total_width - 3 - title_len - 1;  // "── " + title + " "
  return std::string(c.bold) + c.cyan + "\xe2\x94\x80\xe2\x94\x80 " + title + " "
         + c.reset + c.dim + rule(std::max(remaining, 4)) + c.reset + "\n";
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

// ── Bar rendering with gradient ──

// Returns a colored bar where each block character gets a color based on
// its position within the bar: early blocks are green/cyan, later blocks
// shift to yellow then red.  This creates a heat-map effect.
std::string render_gradient_bar(const Colors& c, double percent, int max_width) {
  const double filled = (percent / 100.0) * max_width;
  const int full_blocks = std::min(static_cast<int>(filled), max_width);
  const double remainder = filled - full_blocks;
  const int total_chars = full_blocks + (remainder >= 0.5 && full_blocks < max_width ? 1 : 0);

  if (total_chars == 0) return "";

  std::string bar;
  for (int i = 0; i < full_blocks; ++i) {
    const double pos = static_cast<double>(i) / std::max(total_chars - 1, 1);
    const char* block_color;
    if (pos < 0.5) block_color = c.green;
    else if (pos < 0.75) block_color = c.yellow;
    else block_color = c.red;
    bar += block_color;
    bar += "\xe2\x96\x88";  // █
  }
  if (remainder >= 0.5 && full_blocks < max_width) {
    const char* half_color = (total_chars <= 2) ? c.green : c.yellow;
    bar += half_color;
    bar += "\xe2\x96\x8c";  // ▌
  }
  bar += c.reset;
  return bar;
}

// Monochrome bar (single color) for simpler contexts
std::string render_bar(const char* color, double percent, int max_width) {
  const double filled = (percent / 100.0) * max_width;
  const int full_blocks = std::min(static_cast<int>(filled), max_width);
  const double remainder = filled - full_blocks;
  std::string bar;
  bar += color;
  for (int i = 0; i < full_blocks; ++i) {
    bar += "\xe2\x96\x88";  // █
  }
  if (remainder >= 0.5 && full_blocks < max_width) {
    bar += "\xe2\x96\x8c";  // ▌
  }
  return bar;
}

// ── Kernel name truncation ──
// GPU kernel names share long prefixes like "void cutlass::Kernel<...>"
// and differ at the end, so middle-truncation preserves the useful parts.
std::string truncate_kernel_name(const std::string& name, std::size_t max_len) {
  if (name.size() <= max_len) return name;
  if (max_len < 8) return name.substr(0, max_len);
  // Keep first ~40% and last ~60% to preserve the distinguishing suffix
  const std::size_t suffix_len = (max_len - 3) * 3 / 5;  // 60% for suffix
  const std::size_t prefix_len = max_len - 3 - suffix_len;
  return name.substr(0, prefix_len) + "\xe2\x80\xa6" + name.substr(name.size() - suffix_len);  // …
}

const char* percent_color(const Colors& c, double percent) {
  if (percent >= 40.0) return c.red;
  if (percent >= 20.0) return c.yellow;
  if (percent >= 8.0) return c.green;
  return c.cyan;
}

std::string format_optional_metric(
    std::string_view metric,
    const std::optional<double>& value) {
  if (!value.has_value()) {
    return "-";
  }

  if (metric == "vllm:gpu_cache_usage_perc" ||
      metric == "vllm:prefix_cache_hit_rate") {
    return format_fixed(*value * 100.0, 1) + "%";
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
  append_if("warning_no_kernel_trace", "no kernel trace was captured for this profile");
  append_if(
      "warning_aggregate_traffic_percentiles",
      "aggregate traffic p50/p99 are upper bounds from member runs; max/median is the max observed per-run ratio");
  append_if(
      "warning_gpu_clocks_unlocked",
      "GPU clocks are not locked. Run `rlprof lock-clocks` for reproducible measurements.");
  return warnings;
}

}  // namespace

std::string render_report(
    const ReportMeta& meta,
    const std::map<std::string, std::string>& metadata,
    const std::vector<profiler::KernelRecord>& kernels,
    const std::vector<MetricSummary>& metrics_summary,
    const TrafficStats& traffic_stats,
    bool color) {
  const Colors c = make_colors(color);
  const int term_width = color ? detect_terminal_width() : kDefaultWidth;

  // Adaptive bar width: use remaining space after fixed columns
  const int bar_width = std::clamp(term_width - kCategoryTableFixedCols - 2, kMinBarWidth, kMaxBarWidth);

  // Adaptive kernel name width
  const int kernel_name_width = std::clamp(term_width - 40, 30, 60);

  std::ostringstream output;

  // ── Header ──────────────────────────────────────────────────────────
  output << "\n"
         << c.bold << c.cyan << "  rlprof" << c.reset
         << c.dim << " \xe2\x94\x82 " << c.reset  // │
         << meta.model_name
         << c.dim << " \xe2\x94\x82 " << c.reset
         << meta.gpu_name
         << c.dim << " \xe2\x94\x82 " << c.reset
         << meta.vllm_version << "\n"
         << c.dim << "  workload: " << c.reset
         << meta.prompts << " prompts, "
         << meta.rollouts << " rollouts/prompt, "
         << meta.max_tokens << " max tokens\n"
         << c.dim << "  category buckets use conservative substring matching; "
         << "raw kernel names are authoritative" << c.reset << "\n\n";

  const bool cluster_mode = metadata_value(metadata, "cluster_mode", "false") == "true";

  // ── Warnings ────────────────────────────────────────────────────────
  const auto warnings = warning_messages(metadata);
  if (!warnings.empty()) {
    output << c.bold << c.yellow << "  \xe2\x9a\xa0  WARNINGS" << c.reset << "\n";
    for (std::size_t i = 0; i < warnings.size(); ++i) {
      const bool last = (i + 1 == warnings.size());
      output << c.dim << "  "
             << (last ? "\xe2\x94\x94\xe2\x94\x80 " : "\xe2\x94\x9c\xe2\x94\x80 ")
             << c.reset << c.yellow << warnings[i] << c.reset << "\n";
    }
    output << "\n";
  }

  // ── Measurement context ─────────────────────────────────────────────
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
    output << section_header(c, "MEASUREMENT CONTEXT", term_width);
    // Two-column key-value layout with dim labels, normal values
    const auto ctx_row = [&](const std::string& label, const std::string& value) {
      output << "  " << c.dim << std::left << std::setw(30) << label << c.reset
             << "  " << value << "\n";
    };
    ctx_row("driver version", metadata_value(metadata, "measurement_driver_version"));
    ctx_row("persistence mode", metadata_value(metadata, "measurement_persistence_mode"));
    ctx_row("gpu clock policy", metadata_value(metadata, "measurement_gpu_clock_policy"));
    if (cluster_mode) {
      ctx_row("cluster endpoints", metadata_value(metadata, "cluster_endpoint_count"));
      ctx_row("peer endpoints", metadata_value(metadata, "cluster_peer_endpoint_count"));
      ctx_row("trace scope", metadata_value(metadata, "cluster_trace_scope"));
    }
    if (metadata_value(metadata, "measurement_gpu_max_sm_clock_mhz", "").size() > 0) {
      ctx_row("max supported sm clock mhz", metadata_value(metadata, "measurement_gpu_max_sm_clock_mhz"));
    }
    ctx_row("observed pstate(s)", metadata_value(metadata, "measurement_pstates"));
    ctx_row("gpu telemetry samples", metadata_value(metadata, "measurement_samples"));
    if (sm_min.has_value() && sm_avg.has_value() && sm_max.has_value()) {
      ctx_row("sm clock (min/avg/max mhz)",
              format_fixed(*sm_min, 0) + " / " + format_fixed(*sm_avg, 0) + " / " +
                  format_fixed(*sm_max, 0));
    }
    if (mem_min.has_value() && mem_avg.has_value() && mem_max.has_value()) {
      ctx_row("mem clock (min/avg/max mhz)",
              format_fixed(*mem_min, 0) + " / " + format_fixed(*mem_avg, 0) + " / " +
                  format_fixed(*mem_max, 0));
    }
    if (temp_min.has_value() && temp_max.has_value()) {
      ctx_row("temperature (min/max c)",
              format_fixed(*temp_min, 0) + " / " + format_fixed(*temp_max, 0));
    }
    if (power_avg.has_value() && power_peak.has_value() && power_limit.has_value()) {
      ctx_row("power draw (avg/peak/limit w)",
              format_fixed(*power_avg, 1) + " / " + format_fixed(*power_peak, 1) + " / " +
                  format_fixed(*power_limit, 1));
    }
    output << '\n';
  }

  // ── Kernel breakdown by category ────────────────────────────────────
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

  output << section_header(c, "KERNEL BREAKDOWN BY CATEGORY", term_width);

  // Column header (dim chrome)
  output << c.dim
         << "  " << std::left << std::setw(16) << "category"
         << std::right << std::setw(10) << "time (ms)"
         << std::setw(7) << "%"
         << std::setw(8) << "calls"
         << std::setw(10) << "avg (us)"
         << "  " << c.reset << "\n";
  output << "  " << c.dim << rule(term_width - 4) << c.reset << '\n';

  for (const auto& [category, summary] : categories) {
    const double avg_ns =
        summary.calls == 0 ? 0.0
                           : static_cast<double>(summary.total_ns) / summary.calls;
    const double percent =
        total_ns == 0 ? 0.0
                      : (static_cast<double>(summary.total_ns) / total_ns) * 100.0;

    const std::string bar = render_gradient_bar(c, percent, bar_width);
    const char* pct_color = percent_color(c, percent);

    output << "  " << std::left << std::setw(16) << category
           << std::right << std::setw(10) << format_fixed(summary.total_ns / 1'000'000.0, 1)
           << pct_color << std::setw(6) << format_fixed(percent, 1) << "%" << c.reset
           << std::setw(8) << summary.calls
           << std::setw(10) << format_fixed(avg_ns / 1'000.0, 1)
           << "  " << bar << "\n";
  }

  // Total row: double-line separator for visual distinction
  output << "  " << c.dim << heavy_rule(term_width - 4) << c.reset << '\n';
  output << "  " << c.bold << std::left << std::setw(16) << "total" << c.reset
         << std::right << std::setw(10) << format_fixed(total_ns / 1'000'000.0, 1)
         << c.bold << std::setw(6) << (total_ns == 0 ? "0.0" : "100.0") << "%" << c.reset
         << std::setw(8) << std::accumulate(
                kernels.begin(),
                kernels.end(),
                std::int64_t{0},
                [](std::int64_t total, const profiler::KernelRecord& record) {
                  return total + record.calls;
                })
         << "\n\n";

  // ── Top 10 kernels ──────────────────────────────────────────────────
  output << section_header(c, "TOP 10 KERNELS BY TOTAL TIME", term_width);
  output << c.dim
         << "  " << std::left << std::setw(kernel_name_width) << "kernel"
         << std::right << std::setw(10) << "time (ms)"
         << std::setw(7) << "%"
         << std::setw(8) << "calls"
         << std::setw(10) << "avg (us)"
         << c.reset << "\n";
  output << "  " << c.dim << rule(term_width - 4) << c.reset << '\n';

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
    const char* pct_color = percent_color(c, percent);
    const std::string name = truncate_kernel_name(
        kernel.name, static_cast<std::size_t>(kernel_name_width));

    output << "  " << c.bold << std::left << std::setw(kernel_name_width) << name << c.reset
           << std::right << std::setw(10)
           << format_fixed(kernel.total_ns / 1'000'000.0, 1)
           << pct_color << std::setw(6)
           << format_fixed(percent, 1) << "%" << c.reset
           << std::setw(8) << kernel.calls
           << std::setw(10) << format_fixed(kernel.avg_ns / 1'000.0, 1) << "\n";
  }
  output << '\n';

  // ── Top uncategorized kernels ───────────────────────────────────────
  std::vector<profiler::KernelRecord> uncategorized_kernels;
  for (const profiler::KernelRecord& kernel : sorted_kernels) {
    if (kernel.category == "other") {
      uncategorized_kernels.push_back(kernel);
    }
  }
  if (!uncategorized_kernels.empty()) {
    output << section_header(c, "TOP UNCATEGORIZED KERNELS", term_width);
    output << c.dim
           << "  " << std::left << std::setw(kernel_name_width) << "kernel"
           << std::right << std::setw(10) << "time (ms)"
           << std::setw(7) << "%"
           << std::setw(8) << "calls"
           << std::setw(10) << "avg (us)"
           << c.reset << "\n";
    output << "  " << c.dim << rule(term_width - 4) << c.reset << '\n';
    for (std::size_t i = 0; i < uncategorized_kernels.size() && i < 10; ++i) {
      const profiler::KernelRecord& kernel = uncategorized_kernels[i];
      const double percent =
          total_ns == 0 ? 0.0 : (static_cast<double>(kernel.total_ns) / total_ns) * 100.0;
      const char* pct_color = percent_color(c, percent);
      const std::string name = truncate_kernel_name(
          kernel.name, static_cast<std::size_t>(kernel_name_width));

      output << "  " << c.bold << std::left << std::setw(kernel_name_width) << name << c.reset
             << std::right << std::setw(10)
             << format_fixed(kernel.total_ns / 1'000'000.0, 1)
             << pct_color << std::setw(6)
             << format_fixed(percent, 1) << "%" << c.reset
             << std::setw(8) << kernel.calls
             << std::setw(10) << format_fixed(kernel.avg_ns / 1'000.0, 1) << "\n";
    }
    output << '\n';
  }

  // ── vLLM metrics ────────────────────────────────────────────────────
  const std::string metrics_title =
      cluster_mode ? "CLUSTER VLLM METRICS" : "VLLM SERVER METRICS";
  output << section_header(c, metrics_title, term_width);

  output << c.dim
         << "  " << std::left << std::setw(30) << "metric"
         << std::right << std::setw(12) << "avg"
         << std::setw(12) << "peak"
         << c.reset << "\n";
  output << "  " << c.dim << rule(54) << c.reset << '\n';

  for (const MetricSummary& summary : metrics_summary) {
    const std::string label = metric_label(summary.metric);
    const std::string avg_val = format_optional_metric(summary.metric, summary.avg);
    const std::string peak_val = format_optional_metric(summary.metric, summary.peak);

    // Highlight preemptions >0 or cache usage >90%
    bool highlight = false;
    if (summary.metric == "vllm:num_preemptions_total" && summary.avg.has_value() && *summary.avg > 0) {
      highlight = true;
    }
    if (summary.metric == "vllm:gpu_cache_usage_perc" && summary.peak.has_value() && *summary.peak > 0.9) {
      highlight = true;
    }

    output << "  ";
    if (highlight) output << c.yellow;
    output << std::left << std::setw(30) << label
           << std::right << std::setw(12) << avg_val
           << std::setw(12) << peak_val;
    if (highlight) output << c.reset;
    output << "\n";
  }
  output << '\n';

  // ── Traffic shape ───────────────────────────────────────────────────
  output << section_header(c, "TRAFFIC SHAPE", term_width);

  output << c.dim
         << "  " << std::left << std::setw(30) << "metric"
         << std::right << std::setw(14) << "value"
         << c.reset << "\n";
  output << "  " << c.dim << rule(44) << c.reset << '\n';

  const auto traffic_row = [&](const std::string& label, const std::string& value) {
    output << "  " << std::left << std::setw(30) << label
           << std::right << std::setw(14) << value << "\n";
  };
  traffic_row("total requests", format_int(traffic_stats.total_requests));
  traffic_row("completion length mean",
              traffic_stats.completion_length_mean.has_value()
                  ? format_int(static_cast<std::int64_t>(std::llround(*traffic_stats.completion_length_mean)))
                  : "-");
  traffic_row(
      "completion length samples",
      traffic_stats.completion_length_samples > 0
          ? format_int(traffic_stats.completion_length_samples)
          : "-");
  traffic_row("completion length p50",
              traffic_stats.completion_length_p50.has_value()
                  ? format_int(static_cast<std::int64_t>(std::llround(*traffic_stats.completion_length_p50)))
                  : "-");
  traffic_row("completion length p99",
              traffic_stats.completion_length_p99.has_value()
                  ? format_int(static_cast<std::int64_t>(std::llround(*traffic_stats.completion_length_p99)))
                  : "-");
  traffic_row("max/median ratio",
              traffic_stats.max_median_ratio.has_value()
                  ? format_fixed(*traffic_stats.max_median_ratio, 2) + "x"
                  : "-");
  if (!traffic_stats.completion_length_p50.has_value()) {
    const auto upper = metadata_double(metadata, "aggregate_completion_length_p50_upper_bound");
    if (upper.has_value()) {
      traffic_row(
          "completion length p50 ub",
          format_int(static_cast<std::int64_t>(std::llround(*upper))));
    }
  }
  if (!traffic_stats.completion_length_p99.has_value()) {
    const auto upper = metadata_double(metadata, "aggregate_completion_length_p99_upper_bound");
    if (upper.has_value()) {
      traffic_row(
          "completion length p99 ub",
          format_int(static_cast<std::int64_t>(std::llround(*upper))));
    }
  }
  if (!traffic_stats.max_median_ratio.has_value()) {
    const auto observed = metadata_double(metadata, "aggregate_max_median_ratio_observed_max");
    if (observed.has_value()) {
      traffic_row("max/median ratio max", format_fixed(*observed, 2) + "x");
    }
  }
  traffic_row("errors", format_int(traffic_stats.errors));

  return output.str();
}

}  // namespace rlprof
