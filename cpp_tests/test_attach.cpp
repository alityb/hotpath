#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "rlprof/profiler/attach.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  {
    const std::vector<std::string> argv = {
        "/home/ubuntu/rlprof/.venv/bin/vllm",
        "serve",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--port",
        "8050",
        "--tensor-parallel-size",
        "2",
        "--max-model-len",
        "4096",
        "--trust-remote-code",
    };
    const auto parsed = rlprof::profiler::parse_vllm_serve_argv(argv);
    expect_true(parsed.has_value(), "expected vllm serve argv to parse");
    expect_true(parsed->model == "Qwen/Qwen2.5-0.5B-Instruct", "expected model");
    expect_true(parsed->port == 8050, "expected port");
    expect_true(parsed->tp == 2, "expected tp");
    expect_true(parsed->max_model_len == 4096, "expected max model len");
    expect_true(parsed->trust_remote_code, "expected trust_remote_code");
  }

  {
    const std::vector<std::string> argv = {
        "/usr/bin/python3",
        "-m",
        "http.server",
    };
    const auto parsed = rlprof::profiler::parse_vllm_serve_argv(argv);
    expect_true(!parsed.has_value(), "expected non-vllm argv to be rejected");
  }

  expect_true(
      rlprof::profiler::attach_server_is_local("http://127.0.0.1:8050"),
      "expected loopback attach server to be local");
  expect_true(
      rlprof::profiler::attach_server_is_local("http://localhost:8050"),
      "expected localhost attach server to be local");
  expect_true(
      !rlprof::profiler::attach_server_is_local("http://example.invalid:8050"),
      "expected unrelated host to be non-local");

  return 0;
}
