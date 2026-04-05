#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "hotpath/log_parser.h"
#include "hotpath/serve_profiler.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  namespace fs = std::filesystem;

  fs::path fixture_path = fs::path(__FILE__).parent_path() / "fixtures" / "vllm_debug.log";
  if (!fs::exists(fixture_path)) {
    fixture_path = "cpp_tests/fixtures/vllm_debug.log";
  }
  expect_true(fs::exists(fixture_path), "fixture file not found: " + fixture_path.string());

  const auto traces = hotpath::parse_vllm_log(fixture_path);
  expect_true(traces.size() == 5, "expected 5 traces, got " + std::to_string(traces.size()));

  std::vector<std::string> ids;
  for (const auto& trace : traces) {
    ids.push_back(trace.request_id);
  }
  std::sort(ids.begin(), ids.end());
  expect_true(std::find(ids.begin(), ids.end(), "cmpl-019001") != ids.end(),
              "missing cmpl-019001");
  expect_true(std::find(ids.begin(), ids.end(), "cmpl-019002") != ids.end(),
              "missing cmpl-019002");
  expect_true(std::find(ids.begin(), ids.end(), "cmpl-019003") != ids.end(),
              "missing cmpl-019003");
  expect_true(std::find(ids.begin(), ids.end(), "cmpl-019004") != ids.end(),
              "missing cmpl-019004");
  expect_true(std::find(ids.begin(), ids.end(), "chatcmpl-019005") != ids.end(),
              "missing chatcmpl-019005");

  for (const auto& trace : traces) {
    expect_true(!trace.events.empty(), trace.request_id + ": should have parsed events");
    expect_true(trace.queue_start_us <= trace.prefill_start_us,
                trace.request_id + ": queue_start must be <= prefill_start");
    expect_true(trace.prefill_start_us <= trace.prefill_end_us,
                trace.request_id + ": prefill_start must be <= prefill_end");
    if (trace.server_last_token_us > 0) {
      expect_true(trace.prefill_end_us <= trace.server_last_token_us,
                  trace.request_id + ": prefill_end must be <= server_last_token");
    }
  }

  const std::vector<std::string> lines = {
      "INFO 04-04 10:00:00.000000 api_server.py:100 Added request cmpl-test001 prompt_tokens=64",
      "DEBUG 04-04 10:00:00.010000 scheduler.py:200 Running: [cmpl-test001, chatcmpl-test002]",
      "DEBUG 04-04 10:00:00.020000 worker.py:300 Prefill done for requests [cmpl-test001] token=64",
      "INFO 04-04 10:00:00.030000 scheduler.py:210 Prefix cache hit rate: 25.0%",
      "INFO 04-04 10:00:00.500000 engine.py:400 Finished request cmpl-test001 output_tokens=16",
      "INFO 04-04 10:00:00.700000 api_server.py:100 Added request chatcmpl-test002 prompt_tokens=32",
      "DEBUG 04-04 10:00:00.710000 worker.py:300 Prefill complete for requests [chatcmpl-test002] token=32",
      "INFO 04-04 10:00:01.000000 engine.py:400 Finished request chatcmpl-test002 output_tokens=8",
  };
  const auto simple = hotpath::parse_vllm_log_lines(lines);
  const auto detailed = hotpath::parse_vllm_log_lines_detailed(lines);
  expect_true(simple.size() == 2, "expected 2 traces from mixed vLLM 0.19 input");
  expect_true(detailed.aggregate_cache_hit_rate.has_value(),
              "aggregate cache hit rate should be parsed");
  expect_true(std::abs(*detailed.aggregate_cache_hit_rate - 0.25) < 1e-9,
              "aggregate cache hit rate should equal 0.25");

  std::vector<std::string> false_positive_lines = {
      "DEBUG 04-04 11:00:00.000000 config.py:100 loaded hash e6e225a1e5d19659baa2744480c86579",
      "DEBUG 04-04 11:00:00.100000 kernels.py:200 tensor checksum 2fafe0cf81a062ddc6183d1f61f97652",
      "INFO 04-04 11:00:01.000000 scheduler.py:300 Prefix cache hit rate: 53.0%",
  };
  const auto false_positive = hotpath::parse_vllm_log_lines_detailed(false_positive_lines);
  expect_true(false_positive.traces.empty(),
              "bare hex hashes must not be treated as request IDs");
  expect_true(false_positive.aggregate_cache_hit_rate.has_value(),
              "aggregate cache hit rate should still parse without request IDs");
  expect_true(*false_positive.aggregate_cache_hit_rate > 0.5 &&
                  *false_positive.aggregate_cache_hit_rate < 0.6,
              "aggregate cache hit rate should reflect the log value");

  const std::vector<std::string> internal_id_lines = {
      "INFO 04-04 12:00:00.000000 api_server.py:100 Added request request-0 prompt_tokens=64",
      "DEBUG 04-04 12:00:00.005000 scheduler.py:200 Running: [request-0]",
      "DEBUG 04-04 12:00:00.010000 model_runner.py:300 prompt processing complete for request request-0 token=64",
      "INFO 04-04 12:00:00.300000 engine.py:400 Finished request request-0 output_tokens=12",
      "INFO 04-04 12:00:01.000000 api_server.py:100 Added request 7b1d9f4e-req prompt_tokens=32",
      "DEBUG 04-04 12:00:01.006000 scheduler.py:200 Running: [7b1d9f4e-req]",
      "DEBUG 04-04 12:00:01.012000 model_runner.py:300 Prefill done for requests [7b1d9f4e-req] token=32",
      "INFO 04-04 12:00:01.200000 engine.py:400 Finished request 7b1d9f4e-req output_tokens=8",
  };
  const auto internal = hotpath::parse_vllm_log_lines(internal_id_lines);
  expect_true(internal.size() == 2, "expected 2 traces from internal-id logs");
  std::vector<std::string> internal_ids;
  for (const auto& trace : internal) internal_ids.push_back(trace.request_id);
  expect_true(std::find(internal_ids.begin(), internal_ids.end(), "request-0") != internal_ids.end(),
              "missing internal request-0 id");
  expect_true(std::find(internal_ids.begin(), internal_ids.end(), "7b1d9f4e-req") != internal_ids.end(),
              "missing internal 7b1d9f4e-req id");

  std::vector<hotpath::RequestTrace> client_traces;
  std::vector<hotpath::RequestTrace> server_traces;
  for (int i = 0; i < 5; ++i) {
    client_traces.push_back(hotpath::RequestTrace{
        .request_id = "cmpl-" + std::to_string(i),
        .arrival_us = 1000000 + i * 100000,
    });
    server_traces.push_back(hotpath::RequestTrace{
        .request_id = "request-" + std::to_string(i),
        .queue_start_us = 1000000 + i * 100000 + (i + 2) * 1000,
        .prefill_start_us = 1000000 + i * 100000 + (i + 4) * 1000,
        .prefill_end_us = 1000000 + i * 100000 + (i + 9) * 1000,
        .server_last_token_us = 1000000 + i * 100000 + (i + 40) * 1000,
        .status = "ok",
        .server_timing_available = true,
    });
  }
  const auto correlated = hotpath::correlate_server_traces(client_traces, server_traces);
  expect_true(correlated.method == hotpath::ServerTraceMatchMethod::TIMESTAMP,
              "timestamp fallback should be used when IDs differ");
  expect_true(correlated.matched_requests == 5, "timestamp fallback should match all 5 requests");
  expect_true(correlated.max_offset_us >= 2000 && correlated.max_offset_us <= 10000,
              "timestamp fallback max offset should be within the synthetic range");
  for (const auto& trace : client_traces) {
    expect_true(trace.server_timing_available,
                "timestamp-correlated client trace should gain server timing");
  }

  std::vector<hotpath::RequestTrace> far_client = {
      hotpath::RequestTrace{.request_id = "cmpl-far", .arrival_us = 1000000},
  };
  std::vector<hotpath::RequestTrace> far_server = {
      hotpath::RequestTrace{
          .request_id = "request-far",
          .queue_start_us = 1105000,
          .prefill_start_us = 1107000,
          .prefill_end_us = 1110000,
          .status = "ok",
          .server_timing_available = true,
      },
  };
  const auto far_result = hotpath::correlate_server_traces(far_client, far_server);
  expect_true(far_result.method == hotpath::ServerTraceMatchMethod::NONE,
              "timestamp fallback should reject offsets beyond 50ms");
  expect_true(far_result.matched_requests == 0,
              "timestamp fallback should not false-match distant requests");

  // ── CUDA graph capture lines must not produce traces ──
  // These appeared in real vLLM logs and were previously misidentified as request IDs.
  {
    const std::vector<std::string> cuda_graph_lines = {
        "INFO 04-04 10:00:00.000000 model_runner.py:123 Capturing CUDA graphs #batch_size=1 ...",
        "INFO 04-04 10:00:00.100000 model_runner.py:124 Capturing CUDA graphs (mixed prefill-decode): [00:00<00:01, 30.96it/s]",
        "INFO 04-04 10:00:00.200000 model_runner.py:125 Capturing CUDA graphs (prefill): 100%|██████████| 20/20 [00:00<00:00, 50.12it/s]",
        "INFO 04-04 10:00:01.000000 model_runner.py:126 CUDA graph capture complete.",
    };
    const auto cuda_result = hotpath::parse_vllm_log_lines_detailed(cuda_graph_lines);
    expect_true(cuda_result.traces.empty(),
                "CUDA graph progress lines must not produce request traces (got " +
                    std::to_string(cuda_result.traces.size()) + ")");
  }

  // ── tqdm / progress bar strings must be filtered out ──
  {
    const std::vector<std::string> tqdm_lines = {
        "Loading weights: 100%|████| 308/308 [00:08<00:00, 36.78it/s]",
        "Prefill:  80%|████    | 4/5 [00:01<00:00, 3.96it/s]",
        "DEBUG 04-04 10:00:01.000000 engine.py:400 Running:  [00:00<00:01, ?it/s]",
    };
    const auto tqdm_result = hotpath::parse_vllm_log_lines(tqdm_lines);
    expect_true(tqdm_result.empty(),
                "tqdm/progress bar lines must not produce request traces (got " +
                    std::to_string(tqdm_result.size()) + ")");
  }

  // ── Preempted requests should get preempted status, not be silently dropped ──
  {
    const std::vector<std::string> preempt_lines = {
        "INFO 04-04 10:00:00.000000 api_server.py:100 Added request preempt-req-1",
        "DEBUG 04-04 10:00:00.005000 scheduler.py:200 Running: [preempt-req-1]",
        "DEBUG 04-04 10:00:00.010000 scheduler.py:300 Preempting request preempt-req-1 (swapping out)",
    };
    const auto preempt_result = hotpath::parse_vllm_log_lines(preempt_lines);
    expect_true(!preempt_result.empty(),
                "preempted request should produce a trace");
    bool found_preempted = false;
    for (const auto& t : preempt_result) {
      if (t.request_id == "preempt-req-1") {
        expect_true(t.status == "preempted",
                    "preempted request should have status='preempted', got: " + t.status);
        found_preempted = true;
      }
    }
    expect_true(found_preempted, "preempt-req-1 trace should be found");
  }

  // ── Both cmpl- (completions) and chatcmpl- (chat completions) IDs should parse ──
  {
    const std::vector<std::string> mixed_id_lines = {
        "INFO 04-04 10:00:00.000000 api_server.py:100 Added request cmpl-abc123",
        "INFO 04-04 10:00:00.001000 api_server.py:100 Added request chatcmpl-xyz789",
        "INFO 04-04 10:00:00.500000 engine.py:400 Finished request cmpl-abc123 output_tokens=10",
        "INFO 04-04 10:00:00.600000 engine.py:400 Finished request chatcmpl-xyz789 output_tokens=20",
    };
    const auto mixed_ids = hotpath::parse_vllm_log_lines(mixed_id_lines);
    expect_true(mixed_ids.size() == 2,
                "should parse both cmpl- and chatcmpl- IDs, got " +
                    std::to_string(mixed_ids.size()));
    std::vector<std::string> mixed_req_ids;
    for (const auto& t : mixed_ids) mixed_req_ids.push_back(t.request_id);
    expect_true(std::find(mixed_req_ids.begin(), mixed_req_ids.end(), "cmpl-abc123") != mixed_req_ids.end(),
                "cmpl-abc123 should be found");
    expect_true(std::find(mixed_req_ids.begin(), mixed_req_ids.end(), "chatcmpl-xyz789") != mixed_req_ids.end(),
                "chatcmpl-xyz789 should be found");
  }

  std::cerr << "test_log_parser: all tests passed\n";
  return 0;
}
