#include "hotpath/stability.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/ioctl.h>
#include <unistd.h>

namespace hotpath {
namespace {

constexpr double kCategoryWarnRatio = 1.10;
constexpr double kMetricWarnRatio = 1.10;
constexpr double kTotalWarnRatio = 1.05;

std::string format_fixed(double value, int precision) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string rule(int width) {
  std::string result;
  result.reserve(static_cast<std::size_t>(width) * 3);
  for (int i = 0; i < width; ++i) {
    result += "\xe2\x94\x80";  // ─
  }
  return result;
}

int detect_terminal_width() {
  struct winsize ws{};
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
    return std::max(static_cast<int>(ws.ws_col), 80);
  }
  return 80;
}

double population_stddev(const std::vector<double>& values, double mean) {
  if (values.empty()) {
    return 0.0;
  }
  double sum_sq = 0.0;
  for (double value : values) {
    const double delta = value - mean;
    sum_sq += delta * delta;
  }
  return std::sqrt(sum_sq / static_cast<double>(values.size()));
}

StabilityRow build_row(
    const std::string& label,
    const std::vector<double>& values,
    double warn_ratio_threshold) {
  if (values.empty()) {
    throw std::runtime_error("stability row requires at least one value");
  }

  const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
  const double min_value = *min_it;
  const double max_value = *max_it;
  const double mean =
      std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
  const double stddev = population_stddev(values, mean);
  const double cv_pct = mean == 0.0 ? 0.0 : (stddev / mean) * 100.0;
  const std::optional<double> ratio =
      min_value == 0.0 ? std::nullopt : std::optional<double>(max_value / min_value);
  const bool pass = !ratio.has_value() || *ratio <= warn_ratio_threshold;

  return StabilityRow{
      .label = label,
      .mean = mean,
      .min = min_value,
      .max = max_value,
      .max_min_ratio = ratio,
      .cv_pct = cv_pct,
      .pass = pass,
  };
}

std::optional<double> find_metric_value(
    const ProfileData& profile,
    const std::string& metric,
    bool use_peak) {
  for (const auto& summary : profile.metrics_summary) {
    if (summary.metric != metric) {
      continue;
    }
    return use_peak ? summary.peak : summary.avg;
  }
  return std::nullopt;
}

}  // namespace

StabilityReport compute_stability_report(const std::vector<ProfileData>& profiles) {
  if (profiles.size() < 2) {
    throw std::runtime_error("stability report requires at least two profiles");
  }

  std::vector<double> total_kernel_times;
  total_kernel_times.reserve(profiles.size());
  std::map<std::string, std::vector<double>> category_values;

  for (const ProfileData& profile : profiles) {
    double total_ns = 0.0;
    std::map<std::string, double> category_totals;
    for (const auto& kernel : profile.kernels) {
      total_ns += static_cast<double>(kernel.total_ns);
      category_totals[kernel.category] += static_cast<double>(kernel.total_ns);
    }
    total_kernel_times.push_back(total_ns / 1'000'000.0);

    for (auto& [category, values] : category_values) {
      if (!category_totals.contains(category)) {
        values.push_back(0.0);
      }
    }
    for (const auto& [category, total_ns_by_category] : category_totals) {
      auto& values = category_values[category];
      if (values.size() + 1 < profiles.size()) {
        values.resize(total_kernel_times.size() - 1, 0.0);
      }
      values.push_back(total_ns_by_category / 1'000'000.0);
    }
  }

  for (auto& [_, values] : category_values) {
    if (values.size() < profiles.size()) {
      values.resize(profiles.size(), 0.0);
    }
  }

  StabilityReport report;
  report.run_count = profiles.size();
  report.total_kernel_time =
      build_row("total kernel time", total_kernel_times, kTotalWarnRatio);

  for (const auto& [category, values] : category_values) {
    report.category_rows.push_back(build_row(category, values, kCategoryWarnRatio));
  }
  std::sort(
      report.category_rows.begin(),
      report.category_rows.end(),
      [](const StabilityRow& lhs, const StabilityRow& rhs) {
        return lhs.mean > rhs.mean;
      });

  struct MetricSpec {
    std::string metric;
    std::string label;
    bool use_peak = false;
  };
  const std::vector<MetricSpec> metric_specs = {
      {"vllm:num_preemptions_total", "preemptions (avg)", false},
      {"vllm:num_requests_waiting", "requests_waiting (peak)", true},
      {"vllm:gpu_cache_usage_perc", "kv_cache_usage (peak)", true},   // vLLM < 0.19
      {"vllm:kv_cache_usage_perc", "kv_cache_usage (peak)", true},    // vLLM 0.19+
  };

  for (const auto& spec : metric_specs) {
    std::vector<double> values;
    values.reserve(profiles.size());
    bool any_value = false;
    for (const ProfileData& profile : profiles) {
      const auto value = find_metric_value(profile, spec.metric, spec.use_peak);
      if (value.has_value()) {
        any_value = true;
        values.push_back(*value);
      } else {
        values.push_back(0.0);
      }
    }
    if (!any_value) {
      continue;
    }
    report.metric_rows.push_back(build_row(spec.label, values, kMetricWarnRatio));
  }

  return report;
}

std::string render_stability_report(const StabilityReport& report, bool color) {
  if (report.run_count == 0) {
    return {};
  }

  const char* RST = color ? "\033[0m" : "";
  const char* BLD = color ? "\033[1m" : "";
  const char* DIM = color ? "\033[2m" : "";
  const char* GRN = color ? "\033[32m" : "";
  const char* YLW = color ? "\033[33m" : "";
  const char* CYN = color ? "\033[36m" : "";

  const int term_width = color ? detect_terminal_width() : 80;
  const std::string title = "STABILITY REPORT (" + std::to_string(report.run_count) + " runs)";

  std::ostringstream output;
  output << BLD << CYN << "\xe2\x94\x80\xe2\x94\x80 " << title << " "
         << RST << DIM << rule(term_width - static_cast<int>(title.size()) - 4)
         << RST << "\n";

  // Column headers (dim chrome)
  output << DIM
         << "  " << std::left << std::setw(24) << "label"
         << std::right << std::setw(10) << "mean"
         << std::setw(10) << "min"
         << std::setw(10) << "max"
         << std::setw(10) << "max/min"
         << std::setw(8) << "CV%"
         << "  status"
         << RST << "\n";
  output << "  " << DIM << rule(term_width - 4) << RST << '\n';

  const auto append_stability_row = [&](const StabilityRow& row, int precision) {
    const char* status_color = row.pass ? GRN : YLW;

    output << "  " << std::left << std::setw(24) << row.label
           << std::right << std::setw(10) << format_fixed(row.mean, precision)
           << std::setw(10) << format_fixed(row.min, precision)
           << std::setw(10) << format_fixed(row.max, precision)
           << std::setw(10)
           << (row.max_min_ratio.has_value() ? format_fixed(*row.max_min_ratio, 3) : "-");

    // CV% — highlight if not passing
    if (!row.pass) output << YLW;
    output << std::setw(7) << format_fixed(row.cv_pct, 1) << "%";
    if (!row.pass) output << RST;

    // Status badge
    output << "  " << status_color << BLD
           << (row.pass ? "PASS" : "WARN")
           << RST << "\n";
  };

  append_stability_row(report.total_kernel_time, 1);
  for (const StabilityRow& row : report.category_rows) {
    append_stability_row(row, 1);
  }
  if (!report.metric_rows.empty()) {
    output << '\n';
    for (const StabilityRow& row : report.metric_rows) {
      append_stability_row(row, 1);
    }
  }

  return output.str();
}

}  // namespace hotpath
