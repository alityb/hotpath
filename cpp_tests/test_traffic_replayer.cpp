#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "hotpath/traffic_replayer.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

bool contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

}  // namespace

int main() {
  namespace fs = std::filesystem;

  // Find fixture
  fs::path fixture_path = fs::path(__FILE__).parent_path() / "fixtures" / "traffic.jsonl";
  if (!fs::exists(fixture_path)) {
    fixture_path = "cpp_tests/fixtures/traffic.jsonl";
  }
  expect_true(fs::exists(fixture_path), "fixture not found: " + fixture_path.string());

  const auto requests = hotpath::load_jsonl(fixture_path);
  expect_true(requests.size() == 5, "expected 5 requests, got " + std::to_string(requests.size()));

  // Verify first request
  expect_true(requests[0].prompt == "What is the capital of France?",
              "prompt mismatch: " + requests[0].prompt);
  expect_true(requests[0].max_tokens == 64,
              "max_tokens mismatch: " + std::to_string(requests[0].max_tokens));

  // Verify third request
  expect_true(requests[2].prompt == "Write a haiku about programming.",
              "prompt 3 mismatch");
  expect_true(requests[2].max_tokens == 32, "max_tokens 3 mismatch");

  // Test request body building
  const auto body = hotpath::build_request_body(requests[0], "llama-3-70b");
  expect_true(contains(body, "\"model\": \"llama-3-70b\""), "body should contain model");
  expect_true(contains(body, "\"prompt\":"), "body should contain prompt");
  expect_true(contains(body, "\"max_tokens\": 64"), "body should contain max_tokens");
  expect_true(contains(body, "\"stream\": true"), "body should contain stream");

  // Test prompt with special characters
  hotpath::ReplayRequest special;
  special.prompt = "He said \"hello\"\nNew line\\back";
  special.max_tokens = 10;
  const auto special_body = hotpath::build_request_body(special, "test");
  expect_true(contains(special_body, "\\\"hello\\\""), "should escape quotes");
  expect_true(contains(special_body, "\\n"), "should escape newlines");

  const std::string models_json =
      R"({"object":"list","data":[{"id":"meta-llama/Llama-3-8B-Instruct","object":"model"}]})";
  expect_true(
      hotpath::parse_model_from_models_response(models_json) ==
          "meta-llama/Llama-3-8B-Instruct",
      "should parse model id from compact /v1/models response");

  const std::string pretty_models_json =
      "{\n"
      "  \"object\": \"list\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"id\": \"Qwen/Qwen2.5-72B-Instruct\",\n"
      "      \"object\": \"model\"\n"
      "    }\n"
      "  ]\n"
      "}\n";
  expect_true(
      hotpath::parse_model_from_models_response(pretty_models_json) ==
          "Qwen/Qwen2.5-72B-Instruct",
      "should parse model id from pretty-printed /v1/models response");

  // ── Chat completions format ({"messages": [...]}) ──
  {
    // Write a temp JSONL file with messages format
    const fs::path tmp_dir = fs::temp_directory_path() / "hotpath_test_replayer";
    fs::create_directories(tmp_dir);
    const fs::path chat_path = tmp_dir / "chat.jsonl";
    {
      std::ofstream f(chat_path);
      // Chat format: messages array
      f << R"({"messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"What is 2+2?"}],"max_tokens":32})" << "\n";
      // Plain format: still works
      f << R"({"prompt":"Tell me a joke.","max_tokens":16})" << "\n";
      // Another chat format
      f << R"({"messages":[{"role":"user","content":"Summarize the Iliad."}],"max_tokens":128})" << "\n";
    }
    const auto mixed = hotpath::load_jsonl(chat_path);
    expect_true(mixed.size() == 3, "expected 3 mixed-format requests, got " + std::to_string(mixed.size()));

    // First is chat format: messages_json should be set, prompt should be the user content
    expect_true(!mixed[0].messages_json.empty(),
                "first request should have messages_json set");
    expect_true(mixed[0].prompt == "What is 2+2?",
                "first request prompt should be last user content, got: " + mixed[0].prompt);
    expect_true(mixed[0].max_tokens == 32, "first request max_tokens should be 32");

    // Second is plain format
    expect_true(mixed[1].messages_json.empty(),
                "second request (plain) should have empty messages_json");
    expect_true(mixed[1].prompt == "Tell me a joke.",
                "second request prompt mismatch: " + mixed[1].prompt);

    // Third is chat format with only user role
    expect_true(!mixed[2].messages_json.empty(),
                "third request should have messages_json set");
    expect_true(mixed[2].prompt == "Summarize the Iliad.",
                "third request prompt should be user content");
    expect_true(mixed[2].max_tokens == 128, "third request max_tokens should be 128");

    // build_request_body for chat format should use /v1/chat/completions path
    const auto chat_body = hotpath::build_request_body(mixed[0], "gpt-test");
    expect_true(contains(chat_body, "\"messages\":"),
                "chat body should contain messages field, not prompt");
    expect_true(!contains(chat_body, "\"prompt\":"),
                "chat body must NOT contain prompt field");
    expect_true(contains(chat_body, "\"model\": \"gpt-test\""),
                "chat body should contain model");
    expect_true(contains(chat_body, "\"max_tokens\": 32"),
                "chat body should contain max_tokens");
    expect_true(contains(chat_body, "\"stream\": true"),
                "chat body should enable streaming");

    // api_path_for: chat format goes to /v1/chat/completions, plain to /v1/completions
    expect_true(hotpath::api_path_for(mixed[0]) == "/v1/chat/completions",
                "chat request should route to /v1/chat/completions");
    expect_true(hotpath::api_path_for(mixed[1]) == "/v1/completions",
                "plain request should route to /v1/completions");

    // build_request_body for plain format still uses prompt
    const auto plain_body = hotpath::build_request_body(mixed[1], "gpt-test");
    expect_true(contains(plain_body, "\"prompt\":"),
                "plain body should contain prompt field");
    expect_true(!contains(plain_body, "\"messages\":"),
                "plain body must NOT contain messages field");

    fs::remove_all(tmp_dir);
  }

  // ── Edge cases: empty / comment lines in JSONL ──
  {
    const fs::path tmp_dir2 = fs::temp_directory_path() / "hotpath_test_replayer2";
    fs::create_directories(tmp_dir2);
    const fs::path edge_path = tmp_dir2 / "edge.jsonl";
    {
      std::ofstream f(edge_path);
      f << "\n";  // blank line
      f << "# this is a comment\n";
      f << R"({"prompt":"valid","max_tokens":8})" << "\n";
    }
    const auto edge = hotpath::load_jsonl(edge_path);
    expect_true(edge.size() == 1,
                "blank lines and comments should be skipped, got " + std::to_string(edge.size()));
    fs::remove_all(tmp_dir2);
  }

  std::cerr << "test_traffic_replayer: all tests passed\n";
  return 0;
}
