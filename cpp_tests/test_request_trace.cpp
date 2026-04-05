#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "hotpath/request_trace.h"
#include "hotpath/store.h"

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
  const fs::path temp_dir = fs::temp_directory_path() / "hotpath_test_request_trace";
  fs::create_directories(temp_dir);
  const fs::path db_path = temp_dir / "traces.db";
  fs::remove(db_path);

  hotpath::init_db(db_path);

  const int64_t profile_id = 1;

  // Insert 100 synthetic traces
  for (int i = 0; i < 100; ++i) {
    const int64_t base = static_cast<int64_t>(i) * 1000000;  // 1s apart
    hotpath::RequestTrace trace;
    trace.request_id = "req_" + std::to_string(i);
    trace.arrival_us = base;
    trace.queue_start_us = base + 100;
    trace.prefill_start_us = base + 200;
    // Vary prefill duration: even requests get 60ms, odd get 30ms
    trace.prefill_end_us = base + 200 + ((i % 2 == 0) ? 60000 : 30000);
    trace.first_token_us = trace.prefill_end_us + 500;
    trace.last_token_us = trace.first_token_us + 200000;
    trace.completion_us = trace.last_token_us + 100;
    trace.prompt_tokens = 512 + i;
    trace.output_tokens = 64 + i;
    // First 40 requests have cached tokens > 0
    trace.cached_tokens = (i < 40) ? (100 + i) : 0;
    trace.status = "ok";
    trace.prompt_tokens_estimated = (i % 3 == 0);
    trace.prompt_text = "prompt_" + std::to_string(i);

    // Add a couple of events per trace
    trace.events.push_back(hotpath::RequestEvent{
        .event_type = "queue",
        .timestamp_us = base + 100,
        .detail = "{\"position\": " + std::to_string(i) + "}",
    });
    trace.events.push_back(hotpath::RequestEvent{
        .event_type = "prefill",
        .timestamp_us = base + 200,
        .detail = "{}",
    });

    hotpath::insert_request_trace(db_path, profile_id, trace);
  }

  // Load all traces back
  const auto loaded = hotpath::load_request_traces(db_path, profile_id);
  expect_true(loaded.size() == 100, "expected 100 traces, got " + std::to_string(loaded.size()));

  // Verify roundtrip for first trace
  expect_true(loaded[0].request_id == "req_0", "first trace request_id mismatch");
  expect_true(loaded[0].arrival_us == 0, "first trace arrival_us mismatch");
  expect_true(loaded[0].prompt_tokens == 512, "first trace prompt_tokens mismatch");
  expect_true(loaded[0].cached_tokens == 100, "first trace cached_tokens mismatch");
  expect_true(loaded[0].status == "ok", "first trace status mismatch");
  expect_true(loaded[0].prompt_tokens_estimated, "first trace estimated flag mismatch");
  expect_true(loaded[0].prompt_text == "prompt_0", "first trace prompt_text mismatch");
  expect_true(loaded[0].events.size() == 2, "first trace should have 2 events");
  expect_true(loaded[0].events[0].event_type == "queue", "first event type mismatch");

  // Verify ordering
  for (size_t i = 1; i < loaded.size(); ++i) {
    expect_true(loaded[i].arrival_us >= loaded[i - 1].arrival_us,
                "traces not ordered by arrival_us");
  }

  // Query traces where prefill > 50ms (50000 us)
  const auto slow_prefill = hotpath::query_traces_prefill_gt(db_path, profile_id, 50000);
  // Even-indexed traces have 60ms prefill, so 50 traces
  expect_true(slow_prefill.size() == 50,
              "expected 50 slow prefill traces, got " + std::to_string(slow_prefill.size()));
  for (const auto& t : slow_prefill) {
    const int64_t prefill_dur = t.prefill_end_us - t.prefill_start_us;
    expect_true(prefill_dur > 50000,
                "prefill duration should be > 50ms: " + std::to_string(prefill_dur));
  }

  // Query traces with cached_tokens > 0
  const auto cached = hotpath::query_traces_cached_gt(db_path, profile_id, 0);
  expect_true(cached.size() == 40,
              "expected 40 cached traces, got " + std::to_string(cached.size()));
  for (const auto& t : cached) {
    expect_true(t.cached_tokens > 0, "cached_tokens should be > 0");
  }

  // Query for different profile_id returns empty
  const auto empty = hotpath::load_request_traces(db_path, 999);
  expect_true(empty.empty(), "expected no traces for non-existent profile_id");

  // Cleanup
  fs::remove_all(temp_dir);

  std::cerr << "test_request_trace: all tests passed\n";
  return 0;
}
