#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "rlprof/store.h"

namespace rlprof {

struct StabilityRow {
  std::string label;
  double mean = 0.0;
  double min = 0.0;
  double max = 0.0;
  std::optional<double> max_min_ratio;
  double cv_pct = 0.0;
  bool pass = true;
};

struct StabilityReport {
  std::size_t run_count = 0;
  StabilityRow total_kernel_time;
  std::vector<StabilityRow> category_rows;
  std::vector<StabilityRow> metric_rows;
};

StabilityReport compute_stability_report(const std::vector<ProfileData>& profiles);

std::string render_stability_report(const StabilityReport& report);

}  // namespace rlprof
