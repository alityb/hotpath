#include <cstdlib>
#include <iostream>
#include <string>

#include "hotpath/workload_classifier.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

hotpath::WorkloadClassifierInput make_base() {
  hotpath::WorkloadClassifierInput input;
  input.phase.prefill_fraction = 0.3;
  input.phase.decode_fraction = 0.5;
  input.batch.avg_batch_size = 16;
  input.cache.cache_hit_rate = 0.2;
  input.prefix.cacheable_token_fraction = 0.1;
  input.request_rate = 10.0;
  input.median_decode_latency_us = 5000;
  input.p99_decode_latency_us = 10000;
  return input;
}

}  // namespace

int main() {
  using hotpath::WorkloadClass;

  // 1. SHORT_CONTEXT: median prompt < 256
  {
    auto input = make_base();
    input.median_prompt_tokens = 128;
    input.median_output_tokens = 64;
    auto result = hotpath::classify_workload(input);
    expect_true(result.primary_class == WorkloadClass::SHORT_CONTEXT,
                "should be SHORT_CONTEXT");
  }

  // 2. CACHE_FRIENDLY: prefix sharing > 60%
  {
    auto input = make_base();
    input.median_prompt_tokens = 1024;
    input.median_output_tokens = 256;
    input.prefix.cacheable_token_fraction = 0.75;
    auto result = hotpath::classify_workload(input);
    expect_true(result.primary_class == WorkloadClass::CACHE_FRIENDLY,
                "should be CACHE_FRIENDLY");
  }

  // 3. PREFILL_HEAVY: prompt > 2048, output < prompt/4, prefill_fraction > 0.5
  {
    auto input = make_base();
    input.median_prompt_tokens = 4096;
    input.median_output_tokens = 256;
    input.phase.prefill_fraction = 0.65;
    auto result = hotpath::classify_workload(input);
    expect_true(result.primary_class == WorkloadClass::PREFILL_HEAVY,
                "should be PREFILL_HEAVY");
  }

  // 4. DECODE_HEAVY: output > prompt * 2
  {
    auto input = make_base();
    input.median_prompt_tokens = 512;
    input.median_output_tokens = 2048;
    auto result = hotpath::classify_workload(input);
    expect_true(result.primary_class == WorkloadClass::DECODE_HEAVY,
                "should be DECODE_HEAVY");
  }

  // 5. BALANCED: neither dominates
  {
    auto input = make_base();
    input.median_prompt_tokens = 1024;
    input.median_output_tokens = 512;
    auto result = hotpath::classify_workload(input);
    expect_true(result.primary_class == WorkloadClass::BALANCED,
                "should be BALANCED");
  }

  // Verify contention estimation
  {
    auto input = make_base();
    input.median_prompt_tokens = 1024;
    input.median_output_tokens = 512;
    input.median_decode_latency_us = 5000;
    input.p99_decode_latency_us = 15000;
    auto result = hotpath::classify_workload(input);
    // contention = (15000 - 5000) / 5000 = 2.0
    expect_true(result.prefill_contention > 1.9 && result.prefill_contention < 2.1,
                "contention: " + std::to_string(result.prefill_contention));
  }

  std::cerr << "test_workload_classifier: all tests passed\n";
  return 0;
}
