#include "rlprof/validate.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rlprof/artifacts.h"
#include "rlprof/profiler/vllm_metrics.h"
#include "rlprof/store.h"

#include <sqlite3.h>

namespace rlprof {
namespace {

using SqliteDbPtr = std::unique_ptr<sqlite3, decltype(&sqlite3_close)>;

SqliteDbPtr open_db(const std::filesystem::path& path) {
  sqlite3* raw = nullptr;
  if (sqlite3_open_v2(path.c_str(), &raw, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
    const std::string message =
        raw == nullptr ? "failed to open sqlite db" : sqlite3_errmsg(raw);
    if (raw != nullptr) {
      sqlite3_close(raw);
    }
    throw std::runtime_error(message);
  }
  return SqliteDbPtr(raw, sqlite3_close);
}

std::string status_label(ValidationStatus status) {
  switch (status) {
    case ValidationStatus::kPass:
      return "PASS";
    case ValidationStatus::kWarn:
      return "WARN";
    case ValidationStatus::kFail:
      return "FAIL";
  }
  return "FAIL";
}

bool approx_equal(double a, double b, double tolerance = 1e-9) {
  return std::fabs(a - b) <= tolerance;
}

}  // namespace

std::vector<ValidationCheck> validate_profile(const std::filesystem::path& db_path) {
  const ProfileData profile = load_profile(db_path);
  std::vector<ValidationCheck> checks;

  const auto artifacts = profile_artifacts(db_path);
  const bool aggregated_profile =
      profile.meta.contains("aggregation_scope") &&
      !profile.meta.at("aggregation_scope").empty();
  const bool metrics_only_profile =
      profile.meta.contains("warning_no_kernel_trace") &&
      profile.meta.at("warning_no_kernel_trace") == "true";
  const bool remote_report_not_fetched =
      profile.meta.contains("warning_remote_nsys_report_not_fetched") &&
      profile.meta.at("warning_remote_nsys_report_not_fetched") == "true";
  std::size_t missing_required = 0;
  std::size_t missing_optional = 0;
  for (const auto& artifact : artifacts) {
    const bool optional_export =
        artifact.kind == "json export" || artifact.kind.ends_with("csv") ||
        artifact.kind == "server log";
    const bool optional_aggregate_trace =
        aggregated_profile &&
        (artifact.kind == "nsys report" || artifact.kind == "nsys sqlite" ||
         artifact.kind == "nvidia-smi xml");
    const bool optional_metrics_only_trace =
        metrics_only_profile &&
        (artifact.kind == "nsys report" || artifact.kind == "nsys sqlite" ||
         artifact.kind == "nvidia-smi xml");
    const bool optional_remote_report =
        remote_report_not_fetched && artifact.kind == "nsys report";
    if (!artifact.exists) {
      if (optional_export || optional_aggregate_trace || optional_metrics_only_trace ||
          optional_remote_report) {
        ++missing_optional;
      } else {
        ++missing_required;
      }
    }
  }
  checks.push_back(ValidationCheck{
      .name = "artifacts",
      .status = missing_required > 0 ? ValidationStatus::kFail
               : missing_optional > 0 ? ValidationStatus::kWarn
                                      : ValidationStatus::kPass,
      .detail = missing_required > 0
                    ? std::to_string(missing_required) + " required artifacts missing"
                    : missing_optional > 0
                          ? std::to_string(missing_optional) + " optional export artifacts missing"
                          : "all required artifacts present",
  });

  std::filesystem::path sqlite_path;
  const auto sqlite_it = profile.meta.find("artifact_nsys_sqlite_path");
  if (sqlite_it != profile.meta.end() && !sqlite_it->second.empty()) {
    sqlite_path = sqlite_it->second;
  } else {
    sqlite_path = db_path;
    sqlite_path.replace_extension(".sqlite");
  }

  if (std::filesystem::exists(sqlite_path)) {
    auto sqlite = open_db(sqlite_path);
    sqlite3_stmt* stmt = nullptr;
    constexpr const char* kSql =
        "SELECT COALESCE(SUM(end-start),0), COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL";
    if (sqlite3_prepare_v2(sqlite.get(), kSql, -1, &stmt, nullptr) != SQLITE_OK) {
      throw std::runtime_error(sqlite3_errmsg(sqlite.get()));
    }
    const int rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
      sqlite3_finalize(stmt);
      throw std::runtime_error("failed to read kernel totals from sqlite");
    }
    const auto raw_total_ns = sqlite3_column_int64(stmt, 0);
    const auto raw_calls = sqlite3_column_int64(stmt, 1);
    sqlite3_finalize(stmt);

    std::int64_t stored_total_ns = 0;
    std::int64_t stored_calls = 0;
    for (const auto& kernel : profile.kernels) {
      stored_total_ns += kernel.total_ns;
      stored_calls += kernel.calls;
    }
    checks.push_back(ValidationCheck{
        .name = "kernel totals",
        .status = (raw_total_ns == stored_total_ns && raw_calls == stored_calls)
                      ? ValidationStatus::kPass
                      : ValidationStatus::kFail,
        .detail = "raw total ns=" + std::to_string(raw_total_ns) +
                  ", stored total ns=" + std::to_string(stored_total_ns) +
                  ", raw calls=" + std::to_string(raw_calls) +
                  ", stored calls=" + std::to_string(stored_calls),
    });
  } else {
    checks.push_back(ValidationCheck{
        .name = "kernel totals",
        .status = ValidationStatus::kWarn,
        .detail = "sqlite artifact not available",
    });
  }

  const auto recomputed_summaries = profiler::summarize_samples(profile.metrics);
  std::map<std::string, MetricSummary> recomputed_by_metric;
  for (const auto& summary : recomputed_summaries) {
    recomputed_by_metric[summary.metric] = summary;
  }

  std::size_t metric_mismatches = 0;
  for (const auto& summary : profile.metrics_summary) {
    const auto it = recomputed_by_metric.find(summary.metric);
    if (it == recomputed_by_metric.end()) {
      continue;
    }
    const auto& recomputed = it->second;
    if ((summary.avg.has_value() &&
         (!recomputed.avg.has_value() || !approx_equal(*summary.avg, *recomputed.avg))) ||
        (summary.peak.has_value() &&
         (!recomputed.peak.has_value() || !approx_equal(*summary.peak, *recomputed.peak))) ||
        (summary.min.has_value() &&
         (!recomputed.min.has_value() || !approx_equal(*summary.min, *recomputed.min)))) {
      ++metric_mismatches;
    }
  }
  checks.push_back(ValidationCheck{
      .name = "metrics summary",
      .status = metric_mismatches == 0 ? ValidationStatus::kPass
                                       : ValidationStatus::kFail,
      .detail = metric_mismatches == 0 ? "stored summaries match raw metric samples"
                                       : std::to_string(metric_mismatches) + " metric summaries mismatched",
  });

  const bool has_remote_host = profile.meta.contains("remote_target_host") &&
                               !profile.meta.at("remote_target_host").empty();
  if (has_remote_host) {
    const bool has_remote_db = profile.meta.contains("remote_artifact_db_path");
    const bool has_remote_sqlite = profile.meta.contains("remote_artifact_nsys_sqlite_path");
    const bool has_remote_rep = profile.meta.contains("remote_artifact_nsys_rep_path");
    checks.push_back(ValidationCheck{
        .name = "remote provenance",
        .status = (has_remote_db && has_remote_sqlite && has_remote_rep)
                      ? ValidationStatus::kPass
                      : ValidationStatus::kWarn,
        .detail = (has_remote_db && has_remote_sqlite && has_remote_rep)
                      ? "remote artifact paths recorded"
                      : "remote target metadata incomplete",
    });
  }

  return checks;
}

std::string render_validation_report(
    const std::filesystem::path& db_path,
    const std::vector<ValidationCheck>& checks,
    bool color) {
  const char* reset = color ? "\033[0m" : "";
  const char* bold = color ? "\033[1m" : "";
  const char* green = color ? "\033[32m" : "";
  const char* yellow = color ? "\033[33m" : "";
  const char* red = color ? "\033[31m" : "";

  std::ostringstream out;
  out << "VALIDATE\n";
  out << "profile: " << db_path.string() << "\n\n";
  out << std::left << std::setw(18) << "check" << "  "
      << std::setw(6) << "status" << "  detail\n";
  out << std::string(88, '-') << "\n";
  for (const auto& check : checks) {
    const char* status_color = reset;
    if (check.status == ValidationStatus::kPass) {
      status_color = green;
    } else if (check.status == ValidationStatus::kWarn) {
      status_color = yellow;
    } else {
      status_color = red;
    }
    out << std::left << std::setw(18) << check.name << "  "
        << status_color << bold << std::setw(6) << status_label(check.status)
        << reset << "  " << check.detail << "\n";
  }
  return out.str();
}

}  // namespace rlprof
