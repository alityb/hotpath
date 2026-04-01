#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>

#include "rlprof/bench/runner.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  const std::string json = R"JSON(
{
  "gpu": {
    "name": "NVIDIA A10G",
    "driver_version": "580.126.16",
    "sm_clock_mhz": 1710.0,
    "mem_clock_mhz": 6251.0,
    "temp_c": 55.0,
    "power_draw_w": 205.1,
    "power_limit_w": 300.0
  },
  "results": [
    {
      "kernel": "silu_and_mul",
      "implementation": "vllm-cuda",
      "shape": "256x4096",
      "dtype": "bf16",
      "avg_us": 34.0,
      "stddev_us": 1.0,
      "cv_pct": 2.9,
      "min_us": 33.1,
      "p50_us": 33.9,
      "p99_us": 35.2,
      "bandwidth_gb_s": 184.1,
      "valid": true,
      "validation_max_abs_error": 0.0,
      "deterministic": true,
      "determinism_max_abs_error": 0.0,
      "timing_warning": true,
      "environment_warning": true,
      "unstable": false
    }
  ],
  "correctness_failures": [],
  "timing_warnings": [
    "silu_and_mul vllm-cuda 256x4096: repeat cv exceeded threshold (18.0%)"
  ],
  "environment_warnings": [
    "silu_and_mul vllm-cuda 256x4096: power cap throttling observed"
  ]
}
)JSON";

  const auto output = rlprof::bench::parse_bench_json(json);
  expect_true(output.gpu.has_value(), "gpu section should parse");
  expect_true(output.results.size() == 1, "one bench result should parse");
  expect_true(output.results[0].implementation == "vllm-cuda", "implementation mismatch");
  expect_true(output.results[0].shape[0] == 256 && output.results[0].shape[1] == 4096,
              "shape mismatch");
  expect_true(std::abs(output.results[0].avg_ms - 0.034) < 1e-9, "avg_us should convert to ms");
  expect_true(output.results[0].deterministic_passed, "deterministic field should parse");
  expect_true(output.timing_warnings.size() == 1, "timing warnings should parse");
  expect_true(output.environment_warnings.size() == 1, "environment warnings should parse");

  const std::string rendered = rlprof::bench::render_bench_output(output);
  expect_true(rendered.find("avg us") != std::string::npos, "render should use microsecond headings");
  expect_true(rendered.find("vllm-cuda") != std::string::npos, "render should contain implementation");
  expect_true(rendered.find("TIMING WARNINGS") != std::string::npos, "render should include timing warnings");
  expect_true(rendered.find("ENVIRONMENT WARNINGS") != std::string::npos, "render should include environment warnings");

  const std::string roundtrip = rlprof::bench::serialize_bench_output_json(output);
  expect_true(roundtrip.find("\"timing_warnings\"") != std::string::npos,
              "serialized output should include timing warnings");
  expect_true(roundtrip.find("\"environment_warnings\"") != std::string::npos,
              "serialized output should include environment warnings");

  rlprof::bench::BenchRunOutput left_only = output;
  rlprof::bench::BenchRunOutput right_only = output;
  right_only.results.clear();
  const std::string comparison = rlprof::bench::render_bench_comparison(left_only, right_only);
  expect_true(
      comparison.find("missing") != std::string::npos,
      "bench comparison should mark missing rows instead of using fake zeros");

  return 0;
}
