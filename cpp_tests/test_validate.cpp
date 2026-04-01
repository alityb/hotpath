#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <map>

#include <sqlite3.h>

#include "rlprof/export.h"
#include "rlprof/profiler/vllm_metrics.h"
#include "rlprof/store.h"
#include "rlprof/validate.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

rlprof::ValidationStatus status_for(
    const std::vector<rlprof::ValidationCheck>& checks,
    const std::string& name) {
  for (const auto& check : checks) {
    if (check.name == name) {
      return check.status;
    }
  }
  std::cerr << "missing validation check: " << name << "\n";
  std::exit(1);
}

}  // namespace

int main() {
  namespace fs = std::filesystem;
  const fs::path temp_root = fs::temp_directory_path() / "rlprof_test_validate";
  fs::remove_all(temp_root);
  fs::create_directories(temp_root);

  const fs::path db_path = temp_root / "sample.db";
  const fs::path sqlite_path = temp_root / "sample.sqlite";
  const fs::path rep_path = temp_root / "sample.nsys-rep";
  const fs::path xml_path = temp_root / "sample_nvidia_smi.xml";
  const fs::path server_log_path = temp_root / "sample_server.log";

  rlprof::ProfileData profile = {
      .meta = {
          {"model_name", "Qwen/Qwen3-8B"},
          {"gpu_name", "NVIDIA A10G"},
          {"artifact_nsys_rep_path", rep_path.string()},
          {"artifact_nsys_sqlite_path", sqlite_path.string()},
          {"measurement_nvidia_smi_xml", xml_path.string()},
          {"artifact_server_log_path", server_log_path.string()},
      },
      .kernels = {
          {
              .name = "flash_fwd_splitkv_kernel",
              .category = "attention",
              .total_ns = 200,
              .calls = 2,
              .avg_ns = 100,
              .min_ns = 80,
              .max_ns = 120,
              .registers = 64,
              .shared_mem = 128,
          },
      },
      .metrics = {
          {
              .sample_time = 0.0,
              .source = "cluster",
              .metric = "vllm:num_preemptions_total",
              .value = 1.0,
          },
          {
              .sample_time = 1.0,
              .source = "cluster",
              .metric = "vllm:num_preemptions_total",
              .value = 3.0,
          },
      },
      .metrics_summary = {
          {
              .metric = "vllm:num_preemptions_total",
              .avg = 2.0,
              .peak = 3.0,
              .min = 1.0,
          },
      },
      .traffic_stats = {
          .total_requests = 4,
          .completion_length_mean = 32.0,
          .completion_length_p50 = 32.0,
          .completion_length_p99 = 48.0,
          .max_median_ratio = 1.5,
          .errors = 0,
      },
  };

  rlprof::save_profile(db_path, profile);
  std::ofstream(rep_path).put('\n');
  std::ofstream(xml_path).put('\n');
  std::ofstream(server_log_path).put('\n');

  sqlite3* sqlite = nullptr;
  expect_true(
      sqlite3_open(sqlite_path.c_str(), &sqlite) == SQLITE_OK,
      "failed to create sqlite artifact");
  expect_true(
      sqlite3_exec(
          sqlite,
          "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER);",
          nullptr,
          nullptr,
          nullptr) == SQLITE_OK,
      "failed to create kernel table");
  expect_true(
      sqlite3_exec(
          sqlite,
          "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 80), (100, 220);",
          nullptr,
          nullptr,
          nullptr) == SQLITE_OK,
      "failed to insert kernel rows");
  sqlite3_close(sqlite);

  for (const auto& exported : rlprof::export_profile(db_path, "csv")) {
    expect_true(fs::exists(exported), "expected exported csv artifact");
  }
  for (const auto& exported : rlprof::export_profile(db_path, "json")) {
    expect_true(fs::exists(exported), "expected exported json artifact");
  }

  const auto checks = rlprof::validate_profile(db_path);
  expect_true(
      status_for(checks, "artifacts") == rlprof::ValidationStatus::kPass,
      "expected artifacts validation pass");
  expect_true(
      status_for(checks, "kernel totals") == rlprof::ValidationStatus::kPass,
      "expected kernel totals validation pass");
  expect_true(
      status_for(checks, "metrics summary") == rlprof::ValidationStatus::kPass,
      "expected metrics summary validation pass");

  profile.metrics.push_back(
      {.sample_time = 0.0,
       .source = "peer1",
       .metric = "vllm:num_preemptions_total",
       .value = 100.0});
  profile.metrics_summary = rlprof::profiler::summarize_samples(profile.metrics);
  rlprof::save_profile(db_path, profile);
  const auto mixed_source_checks = rlprof::validate_profile(db_path);
  expect_true(
      status_for(mixed_source_checks, "metrics summary") == rlprof::ValidationStatus::kPass,
      "expected mixed-source metrics summary validation pass");

  const auto rendered =
      rlprof::render_validation_report(db_path, checks, false);
  expect_true(rendered.find("VALIDATE") != std::string::npos, "expected validate header");
  expect_true(rendered.find("kernel totals") != std::string::npos, "expected kernel totals row");

  profile.meta["artifact_nsys_rep_path"] = "";
  profile.meta["remote_artifact_nsys_rep_path"] = "/remote/sample.nsys-rep";
  profile.meta["warning_remote_nsys_report_not_fetched"] = "true";
  rlprof::save_profile(db_path, profile);
  fs::remove(rep_path);

  const auto remote_checks = rlprof::validate_profile(db_path);
  expect_true(
      status_for(remote_checks, "artifacts") == rlprof::ValidationStatus::kWarn,
      "expected remote no-report fetch artifacts validation warn");

  fs::remove_all(temp_root);
  return 0;
}
