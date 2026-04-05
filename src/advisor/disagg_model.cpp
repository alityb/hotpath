#include "hotpath/disagg_model.h"

#include <algorithm>
#include <cmath>

namespace hotpath {
namespace {

// Simplified M/G/1 queue approximation
// Returns (throughput_rps, p99_wait_ms) for a queue with given params
struct QueueResult {
    double throughput_rps;
    double p99_wait_ms;
};

QueueResult mg1_estimate(double service_time_ms, int num_servers, double arrival_rate) {
  if (service_time_ms <= 0.0 || num_servers <= 0) {
    // Instantaneous service → infinite capacity, zero latency.
    return {1e12, 0.0};
  }
  const double mu = static_cast<double>(num_servers) / service_time_ms * 1000.0;
  const double rho = arrival_rate / mu;

  if (rho >= 0.95) {
    // Near saturation
    return {mu * 0.95, service_time_ms * 20.0};
  }

  const double max_throughput = mu;
  // M/G/1 P99 approximation: service_time * (1 + rho/(1-rho)) * 2.33
  const double p99_wait = service_time_ms * (1.0 + rho / (1.0 - rho + 0.001)) * 2.33;

  return {max_throughput, p99_wait};
}

}  // namespace

DisaggEstimate estimate_disaggregation(const DisaggModelInput& input) {
  DisaggEstimate est;
  const auto& prof = input.profile;

  // Estimate service times from workload profile
  // Prefill time scales with prompt tokens
  const double prefill_time_ms = prof.median_prompt_tokens * 0.01;  // ~10us per token
  // Decode time scales with output tokens
  const double decode_step_ms = 5.0;  // ~5ms per decode step
  const double total_decode_ms = prof.median_output_tokens * decode_step_ms;

  // Monolithic baseline: service_time = prefill + decode
  // Head-of-line blocking: prefill stalls decode for all in-flight requests.
  // Contention reduces effective monolithic throughput.
  const double mono_service_ms = prefill_time_ms + total_decode_ms;
  const int gpus = std::max(input.total_gpus, 1);
  const auto mono = mg1_estimate(mono_service_ms, gpus, prof.request_rate);
  const double blocking_factor = 1.0 / (1.0 + prof.prefill_contention * 0.3);
  est.mono_throughput_rps = mono.throughput_rps * blocking_factor;
  // Use measured server prefill p99 when available — it already includes real
  // scheduling contention and is more accurate than the token-count model.
  // The queue throughput model still uses prefill_time_ms (a median proxy),
  // since p99 is too pessimistic as a service-time input to M/G/1.
  // Use measured p99 when >= 0 (sentinel is -1.0; 0.0 is a valid measurement
  // e.g. fully cached workloads where prefill is near-instant).
  est.mono_p99_ttft_ms = (input.measured_prefill_p99_ms >= 0.0)
      ? input.measured_prefill_p99_ms
      : prefill_time_ms * (1.0 + prof.prefill_contention);
  est.mono_p99_itl_ms = decode_step_ms * (1.0 + prof.prefill_contention * 0.5);

  // KV transfer overhead — use cross-node bandwidth if multi-node
  const double bytes_per_gbps = 1e9 / 8.0;  // bytes per second at 1 Gbps
  double kv_bytes = input.avg_kv_transfer_bytes;
  if (kv_bytes <= 0.0) {
    // Estimate: ~256 bytes per token per layer, 32 layers typical
    kv_bytes = prof.median_prompt_tokens * 256.0 * 32.0;
  }
  double effective_bandwidth = input.network_bandwidth_gbps;
  if (input.num_nodes > 1 && input.cross_node_bandwidth_gbps > 0.0) {
    effective_bandwidth = input.cross_node_bandwidth_gbps;
  }
  if (effective_bandwidth <= 0.0) effective_bandwidth = 1.0;  // prevent div-by-zero
  est.kv_transfer_overhead_ms =
      (kv_bytes / (effective_bandwidth * bytes_per_gbps)) * 1000.0;

  // Minimum bandwidth: transfer must complete in < prefill_time to be worthwhile
  if (prefill_time_ms > 0) {
    est.min_bandwidth_gbps =
        (kv_bytes / (prefill_time_ms / 1000.0)) / bytes_per_gbps;
  }

  // Disaggregation requires at least 2 GPUs (1 prefill + 1 decode).
  if (input.total_gpus < 2) {
    est.optimal_prefill_gpus = 0;
    est.optimal_decode_gpus = 0;
    est.disagg_throughput_rps = 0.0;
    est.disagg_p99_ttft_ms = 0.0;
    est.disagg_p99_itl_ms = 0.0;
    est.should_disaggregate = false;
    est.throughput_improvement = 1.0;
    est.reason = "Disaggregation requires at least 2 GPUs (have " +
                 std::to_string(input.total_gpus) + ").";
    return est;
  }

  // Search for optimal P:D split
  double best_throughput = 0.0;
  int best_p = 1;
  int best_d = input.total_gpus - 1;

  for (int p = 1; p < input.total_gpus; ++p) {
    const int d = input.total_gpus - p;
    const auto prefill_q = mg1_estimate(prefill_time_ms, p, prof.request_rate);
    const auto decode_q = mg1_estimate(total_decode_ms, d, prof.request_rate);
    const double throughput = std::min(prefill_q.throughput_rps, decode_q.throughput_rps);

    if (throughput > best_throughput) {
      best_throughput = throughput;
      best_p = p;
      best_d = d;
    }
  }

  est.optimal_prefill_gpus = best_p;
  est.optimal_decode_gpus = best_d;
  est.disagg_throughput_rps = best_throughput;
  // disagg prefill = measured mono prefill when available (conservative upper bound;
  // disagg removes head-of-line blocking so real value could be lower, but we
  // don't have a measurement of that).  Falls back to the token-count model.
  const double disagg_prefill_base = (input.measured_prefill_p99_ms >= 0.0)
      ? input.measured_prefill_p99_ms
      : prefill_time_ms;
  est.disagg_p99_ttft_ms = disagg_prefill_base + est.kv_transfer_overhead_ms;
  est.disagg_p99_itl_ms = decode_step_ms;  // No prefill contention in disagg

  // Decision logic
  est.throughput_improvement = est.mono_throughput_rps > 0
      ? est.disagg_throughput_rps / est.mono_throughput_rps
      : 1.0;

  const bool kv_transfer_acceptable = est.kv_transfer_overhead_ms < prefill_time_ms * 0.5;
  const bool throughput_gain = est.throughput_improvement > 1.1;
  const bool significant_contention = prof.prefill_contention > 0.5;
  const bool needs_disagg = prof.median_prompt_tokens >= 1024;
  const bool disagg_not_worse = est.disagg_throughput_rps > est.mono_throughput_rps;

  est.should_disaggregate =
      needs_disagg && kv_transfer_acceptable && disagg_not_worse &&
      (throughput_gain || significant_contention);

  if (est.should_disaggregate) {
    est.reason = "Workload has long prefill (" +
                 std::to_string(static_cast<int>(prof.median_prompt_tokens)) +
                 " tokens) with " +
                 std::to_string(static_cast<int>(std::lround((est.throughput_improvement - 1.0) * 100))) +
                 "% throughput improvement. Optimal split: " +
                 std::to_string(best_p) + ":" + std::to_string(best_d) + " (P:D).";
  } else {
    if (!needs_disagg) {
      est.reason = "Short prompts (" +
                   std::to_string(static_cast<int>(prof.median_prompt_tokens)) +
                   " tokens) — prefill is not a bottleneck.";
    } else if (!kv_transfer_acceptable) {
      est.reason = "KV transfer overhead (" +
                   std::to_string(static_cast<int>(est.kv_transfer_overhead_ms)) +
                   "ms) too high. Need >= " +
                   std::to_string(static_cast<int>(std::ceil(est.min_bandwidth_gbps))) +
                   " Gbps bandwidth.";
    } else {
      est.reason = "Throughput improvement insufficient (" +
                   std::to_string(static_cast<int>(std::lround((est.throughput_improvement - 1.0) * 100))) +
                   "% gain).";
    }
  }

  return est;
}

}  // namespace hotpath
