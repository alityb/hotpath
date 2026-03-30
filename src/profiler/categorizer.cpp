#include "rlprof/profiler/categorizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <string_view>

namespace rlprof::profiler {
namespace {

struct PatternGroup {
  std::string_view category;
  std::array<std::string_view, 10> patterns;
  std::size_t count;
};

constexpr std::array<PatternGroup, 11> kPatternGroups = {{
    {"gemm", {"xmma_gemm", "cutlass", "cublas", "nvjet", "wgmma", "ampere_bf16_s168", "gemv2t_kernel", "hmma", "", ""}, 8},
    {"attention", {"flash", "fmha", "paged_attention", "flashattnfwd", "splitkv", "", "", "", "", ""}, 5},
    {"moe", {"fused_moe", "expert", "topk_softmax", "moe_align", "", "", "", "", "", ""}, 4},
    {"mamba", {"mamba", "selective_scan", "ssm", "_chunk_scan", "_chunk_state", "_state_passing", "_causal_conv1d", "_chunk_cumsum", "_bmm_chunk", ""}, 9},
    {"activation", {"silu", "gelu", "relu", "act_and_mul", "swiglu", "", "", "", "", ""}, 5},
    {"norm", {"rms_norm", "layer_norm", "fused_add_rms", "", "", "", "", "", "", ""}, 3},
    {"position", {"rotary", "rope", "", "", "", "", "", "", "", ""}, 2},
    {"cache", {"reshape_and_cache", "cache_kernel", "kv_cache", "", "", "", "", "", "", ""}, 3},
    {"sampling", {"top_k", "top_p", "sample", "argmax", "topk_topp", "repetition_penalt", "", "", "", ""}, 6},
    {"comm", {"allreduce", "allgather", "nccl", "reduce_scatter", "", "", "", "", "", ""}, 4},
    {"memory", {"memcpy", "memset", "fill_kernel", "", "", "", "", "", "", ""}, 3},
}};

std::string to_lower(std::string_view input) {
  std::string lowered;
  lowered.reserve(input.size());
  for (unsigned char ch : input) {
    lowered.push_back(static_cast<char>(std::tolower(ch)));
  }
  return lowered;
}

}  // namespace

std::string_view categorize(std::string_view kernel_name) {
  const std::string lowered_name = to_lower(kernel_name);

  for (const PatternGroup& group : kPatternGroups) {
    for (std::size_t i = 0; i < group.count; ++i) {
      const std::string lowered_pattern = to_lower(group.patterns[i]);
      if (lowered_name.find(lowered_pattern) != std::string::npos) {
        return group.category;
      }
    }
  }

  return "other";
}

}  // namespace rlprof::profiler
