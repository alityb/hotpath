#include "rlprof/stability.h"

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

namespace rlprof {
namespace {

constexpr double kCategoryWarnRatio = 1.10;
constexpr double kMetricWarnRatio = 1.10;
constexpr double kTotalWarnRatio = 1.05;

std::string repeat(char ch, std::size_t count) {
  return std::string(count, ch);
}

std::string format_fixed(double value, int precision) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
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

void append_row(
    std::ostringstream& output,
    const std::vector<std::pair<std::string, std::size_t>>& columns,
    const std::vector<std::string>& values) {
  for (std::size_t i = 0; i < columns.size(); ++i) {
    const bool left_align = (i == 0 || i + 1 == columns.size());
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
      {"vllm:gpu_cache_usage_perc", "kv_cache_usage (peak)", true},
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

std::string render_stability_report(const StabilityReport& report) {
  if (report.run_count == 0) {
    return {};
  }

  std::ostringstream output;
  output << "STABILITY REPORT (" << report.run_count << " runs)\n";
  const std::vector<std::pair<std::string, std::size_t>> columns = {
      {"label", 28},
      {"mean", 10},
      {"min", 10},
      {"max", 10},
      {"max/min", 10},
      {"CV%", 8},
      {"status", 6},
  };
  append_row(output, columns, {"label", "mean", "min", "max", "max/min", "CV%", "status"});
  output << repeat('-', 94) << '\n';

  const auto append_stability_row = [&](const StabilityRow& row, int precision) {
    append_row(
        output,
        columns,
        {
            row.label,
            format_fixed(row.mean, precision),
            format_fixed(row.min, precision),
            format_fixed(row.max, precision),
            row.max_min_ratio.has_value() ? format_fixed(*row.max_min_ratio, 3) : "-",
            format_fixed(row.cv_pct, 1) + "%",
            row.pass ? "PASS" : "WARN",
        });
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

}  // namespace rlprof
