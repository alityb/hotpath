#include "hotpath/workload_classifier.h"

#include <algorithm>

namespace hotpath {

WorkloadProfile classify_workload(const WorkloadClassifierInput& input) {
  WorkloadProfile profile;
  profile.prefill_fraction = input.phase.prefill_fraction;
  profile.median_prompt_tokens = input.median_prompt_tokens;
  profile.median_output_tokens = input.median_output_tokens;
  profile.request_rate = input.request_rate;
  profile.prefix_sharing_rate = input.prefix.cacheable_token_fraction;
  profile.cache_hit_rate = input.cache.cache_hit_rate;

  // Contention estimation: (p99 - median) / median.
  // Clamped to >= 0 because measurement noise can make p99 < median on small samples,
  // which would produce negative contention and cause the disagg model to overestimate
  // throughput beyond the physical maximum.
  if (input.median_decode_latency_us > 0) {
    profile.prefill_contention = std::max(0.0,
        (input.p99_decode_latency_us - input.median_decode_latency_us) /
        input.median_decode_latency_us);
  }

  // Classification rules (priority order)
  if (input.median_prompt_tokens < 256) {
    profile.primary_class = WorkloadClass::SHORT_CONTEXT;
  } else if (input.prefix.cacheable_token_fraction > 0.6) {
    profile.primary_class = WorkloadClass::CACHE_FRIENDLY;
  } else if (input.median_prompt_tokens > 2048 &&
             input.median_output_tokens < input.median_prompt_tokens / 4.0 &&
             input.phase.prefill_fraction > 0.5) {
    profile.primary_class = WorkloadClass::PREFILL_HEAVY;
  } else if (input.median_output_tokens > input.median_prompt_tokens * 2.0) {
    profile.primary_class = WorkloadClass::DECODE_HEAVY;
  } else {
    profile.primary_class = WorkloadClass::BALANCED;
  }

  return profile;
}

}  // namespace hotpath
