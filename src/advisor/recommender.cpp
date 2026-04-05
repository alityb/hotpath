#include "hotpath/recommender.h"

#include <cmath>
#include <sstream>

namespace hotpath {

std::string generate_vllm_config(const DisaggEstimate& estimate, const std::string& model) {
  std::ostringstream ss;

  if (!estimate.should_disaggregate) {
    const int tp = std::max(1, estimate.optimal_prefill_gpus + estimate.optimal_decode_gpus);
    ss << "#!/bin/bash\n";
    ss << "# Monolithic deployment (disaggregation not recommended)\n";
    ss << "# Reason: " << estimate.reason << "\n\n";
    ss << "vllm serve " << model << " \\\n";
    ss << "  --tensor-parallel-size " << tp << "\n";
    return ss.str();
  }

  ss << "#!/bin/bash\n";
  ss << "# Disaggregated vLLM deployment\n";
  ss << "# P:D ratio = " << estimate.optimal_prefill_gpus << ":"
     << estimate.optimal_decode_gpus << "\n";
  ss << "# Expected throughput improvement: "
     << static_cast<int>(std::lround((estimate.throughput_improvement - 1.0) * 100)) << "%\n\n";

  ss << "# Prefill instance\n";
  ss << "vllm serve " << model << " \\\n";
  ss << "  --tensor-parallel-size " << estimate.optimal_prefill_gpus << " \\\n";
  ss << "  --kv-connector PyNcclConnector \\\n";
  ss << "  --kv-role kv_producer \\\n";
  ss << "  --kv-parallel-size 1 \\\n";
  ss << "  --kv-buffer-size 1e9 \\\n";
  ss << "  --port 8100 &\n\n";

  ss << "# Decode instance\n";
  ss << "vllm serve " << model << " \\\n";
  ss << "  --tensor-parallel-size " << estimate.optimal_decode_gpus << " \\\n";
  ss << "  --kv-connector PyNcclConnector \\\n";
  ss << "  --kv-role kv_consumer \\\n";
  ss << "  --kv-parallel-size 1 \\\n";
  ss << "  --kv-buffer-size 1e9 \\\n";
  ss << "  --port 8200 &\n\n";

  ss << "# KV connector config\n";
  ss << "cat > kv_connector.json << 'EOF'\n";
  ss << "{\n";
  ss << "  \"connector\": \"PyNcclConnector\",\n";
  ss << "  \"prefill_port\": 8100,\n";
  ss << "  \"decode_port\": 8200,\n";
  ss << "  \"buffer_size\": 1000000000\n";
  ss << "}\n";
  ss << "EOF\n";

  return ss.str();
}

std::string generate_llmd_config(const DisaggEstimate& estimate, const std::string& model) {
  std::ostringstream ss;

  if (!estimate.should_disaggregate) {
    ss << "# llm-d Helm values (monolithic)\n";
    ss << "model:\n";
    ss << "  name: " << model << "\n";
    ss << "replicas: " << std::max(1, estimate.optimal_prefill_gpus + estimate.optimal_decode_gpus) << "\n";
    ss << "mode: monolithic\n";
    return ss.str();
  }

  ss << "# llm-d Helm values (disaggregated)\n";
  ss << "model:\n";
  ss << "  name: " << model << "\n";
  ss << "prefill:\n";
  ss << "  replicas: " << estimate.optimal_prefill_gpus << "\n";
  ss << "  resources:\n";
  ss << "    gpu: 1\n";
  ss << "decode:\n";
  ss << "  replicas: " << estimate.optimal_decode_gpus << "\n";
  ss << "  resources:\n";
  ss << "    gpu: 1\n";
  ss << "kvTransfer:\n";
  ss << "  enabled: true\n";
  ss << "  minBandwidthGbps: " << static_cast<int>(std::ceil(estimate.min_bandwidth_gbps)) << "\n";

  return ss.str();
}

std::string generate_dynamo_config(const DisaggEstimate& estimate, const std::string& model) {
  std::ostringstream ss;

  if (!estimate.should_disaggregate) {
    ss << "# Dynamo deployment spec (monolithic)\n";
    ss << "apiVersion: dynamo/v1\n";
    ss << "kind: InferenceService\n";
    ss << "spec:\n";
    ss << "  model: " << model << "\n";
    ss << "  mode: monolithic\n";
    ss << "  replicas: " << std::max(1, estimate.optimal_prefill_gpus + estimate.optimal_decode_gpus) << "\n";
    return ss.str();
  }

  ss << "# Dynamo deployment spec (disaggregated)\n";
  ss << "apiVersion: dynamo/v1\n";
  ss << "kind: InferenceService\n";
  ss << "spec:\n";
  ss << "  model: " << model << "\n";
  ss << "  mode: disaggregated\n";
  ss << "  prefill:\n";
  ss << "    replicas: " << estimate.optimal_prefill_gpus << "\n";
  ss << "    gpu: 1\n";
  ss << "  decode:\n";
  ss << "    replicas: " << estimate.optimal_decode_gpus << "\n";
  ss << "    gpu: 1\n";
  ss << "  kvTransfer:\n";
  ss << "    minBandwidthGbps: " << static_cast<int>(std::ceil(estimate.min_bandwidth_gbps)) << "\n";

  return ss.str();
}

std::string generate_summary(const DisaggEstimate& estimate, const std::string& model) {
  std::ostringstream ss;

  ss << "Disaggregation Analysis for " << model << "\n";
  ss << "=========================================\n\n";

  if (estimate.should_disaggregate) {
    ss << "Recommendation: DISAGGREGATE\n";
    ss << "Optimal P:D ratio: " << estimate.optimal_prefill_gpus << ":"
       << estimate.optimal_decode_gpus << "\n";
    ss << "Projected throughput: +"
       << static_cast<int>(std::lround((estimate.throughput_improvement - 1.0) * 100))
       << "% (" << estimate.disagg_throughput_rps << " req/s)\n";
    ss << "Projected p99 TTFT (est.): " << estimate.mono_p99_ttft_ms << "ms -> "
       << estimate.disagg_p99_ttft_ms << "ms\n";
    ss << "Min network bandwidth: "
       << static_cast<int>(std::ceil(estimate.min_bandwidth_gbps)) << " Gbps\n";
  } else {
    ss << "Recommendation: MONOLITHIC\n";
    ss << "Reason: " << estimate.reason << "\n";
  }

  return ss.str();
}

}  // namespace hotpath
