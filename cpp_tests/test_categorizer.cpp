#include <cstdlib>
#include <iostream>
#include <string_view>

#include "rlprof/profiler/categorizer.h"

namespace {

void expect_equal(std::string_view actual, std::string_view expected) {
  if (actual != expected) {
    std::cerr << "expected [" << expected << "] but got [" << actual << "]\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  using rlprof::profiler::categorize;

  expect_equal(
      categorize("sm80_xmma_gemm_bf16_bf16_bf16f32_f32_tn_128x128"),
      "gemm");
  expect_equal(
      categorize("ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_tn"),
      "gemm");
  expect_equal(
      categorize("vllm::act_and_mul_kernel<bf16,silu,true>"),
      "activation");
  expect_equal(
      categorize("_selective_scan_update_kernel"),
      "mamba");
  expect_equal(
      categorize("triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_2"),
      "other");
  expect_equal(categorize("_topk_topp_kernel"), "sampling");
  expect_equal(categorize("flash_fwd_splitkv_bf16_sm80"), "attention");
  expect_equal(categorize("vectorized_elementwise_kernel"), "other");
  expect_equal(categorize("triton_poi_fused_4"), "other");
  expect_equal(categorize("some_unknown_kernel"), "other");

  return 0;
}
