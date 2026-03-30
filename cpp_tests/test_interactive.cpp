#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "interactive.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

bool contains_sequence(
    const std::vector<std::string>& values,
    const std::vector<std::string>& sequence) {
  if (sequence.empty() || values.size() < sequence.size()) {
    return false;
  }
  for (std::size_t i = 0; i + sequence.size() <= values.size(); ++i) {
    bool matches = true;
    for (std::size_t j = 0; j < sequence.size(); ++j) {
      if (values[i + j] != sequence[j]) {
        matches = false;
        break;
      }
    }
    if (matches) {
      return true;
    }
  }
  return false;
}

}  // namespace

int main() {
  namespace fs = std::filesystem;

  const auto profile_args = rlprof::interactive::build_profile_args({
      .model = "Qwen/Qwen3-8B",
      .prompts = 64,
      .rollouts = 4,
      .min_tokens = 256,
      .max_tokens = 1024,
      .input_len = 512,
      .port = 8000,
      .tp = 1,
      .trust_remote_code = true,
      .repeat = 3,
      .output = "auto",
  });
  expect_true(!profile_args.empty() && profile_args.front() == "profile", "expected profile command");
  expect_true(contains_sequence(profile_args, {"--model", "Qwen/Qwen3-8B"}), "expected model args");
  expect_true(contains_sequence(profile_args, {"--repeat", "3"}), "expected repeat args");
  expect_true(contains_sequence(profile_args, {"--trust-remote-code"}), "expected trust flag");
  expect_true(!contains_sequence(profile_args, {"--output", "auto"}), "auto output should be omitted");

  const auto bench_args = rlprof::interactive::build_bench_args({
      .kernel = "silu_and_mul",
      .shapes = "1x4096,64x4096,256x4096",
      .dtype = "bf16",
      .warmup = 20,
      .n_iter = 200,
      .repeats = 3,
  });
  expect_true(!bench_args.empty() && bench_args.front() == "bench", "expected bench command");
  expect_true(contains_sequence(bench_args, {"--kernel", "silu_and_mul"}), "expected kernel arg");

  const std::string gpu_name = rlprof::interactive::detect_gpu_name();
  expect_true(!gpu_name.empty(), "detect_gpu_name should return a non-empty string");

  const fs::path previous_cwd = fs::current_path();
  const fs::path temp_root = fs::temp_directory_path() / "rlprof_interactive_test";
  fs::remove_all(temp_root);
  fs::create_directories(temp_root / ".rlprof");
  fs::current_path(temp_root);

  std::ofstream(temp_root / ".rlprof" / "old.db").put('\n');
  std::ofstream(temp_root / ".rlprof" / "mid.db").put('\n');
  std::ofstream(temp_root / ".rlprof" / "new.db").put('\n');

  fs::last_write_time(temp_root / ".rlprof" / "old.db", fs::file_time_type::clock::now() - std::chrono::hours(3));
  fs::last_write_time(temp_root / ".rlprof" / "mid.db", fs::file_time_type::clock::now() - std::chrono::hours(2));
  fs::last_write_time(temp_root / ".rlprof" / "new.db", fs::file_time_type::clock::now() - std::chrono::hours(1));

  const auto recent = rlprof::interactive::list_recent_profiles(2);
  expect_true(recent.size() == 2, "expected two recent profiles");
  expect_true(recent[0].find("new.db") != std::string::npos, "expected newest profile first");
  expect_true(recent[1].find("mid.db") != std::string::npos, "expected second newest profile second");

  fs::current_path(previous_cwd);
  fs::remove_all(temp_root);
  return 0;
}
