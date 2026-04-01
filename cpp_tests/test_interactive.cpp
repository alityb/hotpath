#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "rlprof/bench/runner.h"
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
      .target = "selfboot",
      .target_workdir = "/tmp/rlprof_bootstrap_test",
      .prompts = 64,
      .rollouts = 4,
      .min_tokens = 256,
      .max_tokens = 1024,
      .input_len = 512,
      .port = 8000,
      .tp = 1,
      .peer_servers = "http://10.0.0.2:8000,http://10.0.0.3:8000",
      .trust_remote_code = true,
      .discard_first_run = true,
      .repeat = 3,
      .output = "auto",
  });
  expect_true(!profile_args.empty() && profile_args.front() == "profile", "expected profile command");
  expect_true(contains_sequence(profile_args, {"--model", "Qwen/Qwen3-8B"}), "expected model args");
  expect_true(contains_sequence(profile_args, {"--target", "selfboot"}), "expected target args");
  expect_true(
      contains_sequence(profile_args, {"--target-workdir", "/tmp/rlprof_bootstrap_test"}),
      "expected target workdir args");
  expect_true(contains_sequence(profile_args, {"--repeat", "3"}), "expected repeat args");
  expect_true(
      contains_sequence(
          profile_args,
          {"--peer-servers", "http://10.0.0.2:8000,http://10.0.0.3:8000"}),
      "expected peer server args");
  expect_true(contains_sequence(profile_args, {"--trust-remote-code"}), "expected trust flag");
  expect_true(contains_sequence(profile_args, {"--discard-first-run"}), "expected discard first run flag");
  expect_true(!contains_sequence(profile_args, {"--output", "auto"}), "auto output should be omitted");

  const auto bench_args = rlprof::interactive::build_bench_args({
      .kernel = "silu_and_mul",
      .target = "selfboot",
      .target_workdir = "/tmp/rlprof_bootstrap_test",
      .shapes = "1x4096,64x4096,256x4096",
      .dtype = "bf16",
      .warmup = 20,
      .n_iter = 200,
      .repeats = 3,
  });
  expect_true(!bench_args.empty() && bench_args.front() == "bench", "expected bench command");
  expect_true(contains_sequence(bench_args, {"--kernel", "silu_and_mul"}), "expected kernel arg");
  expect_true(contains_sequence(bench_args, {"--target", "selfboot"}), "expected bench target arg");

  const std::string gpu_name = rlprof::interactive::detect_gpu_name();
  expect_true(!gpu_name.empty(), "detect_gpu_name should return a non-empty string");

  const fs::path previous_cwd = fs::current_path();
  const fs::path temp_root = fs::temp_directory_path() / "rlprof_interactive_test";
  fs::remove_all(temp_root);
  fs::create_directories(temp_root / ".rlprof");
  fs::current_path(temp_root);
  const fs::path defaults_file = temp_root / "interactive_defaults.cfg";
  setenv("RLPROF_DEFAULTS_PATH", defaults_file.c_str(), 1);

  std::ofstream(temp_root / ".rlprof" / "old.db").put('\n');
  std::ofstream(temp_root / ".rlprof" / "mid.db").put('\n');
  std::ofstream(temp_root / ".rlprof" / "new.db").put('\n');
  {
    rlprof::bench::BenchRunOutput output;
    output.results.push_back(rlprof::bench::BenchResult{
        .kernel = "silu_and_mul",
        .implementation = "torch-eager",
        .shape = rlprof::bench::Shape{64, 4096},
        .dtype = "bf16",
        .avg_ms = 0.1,
        .min_ms = 0.09,
        .p50_ms = 0.1,
        .p99_ms = 0.12,
        .bandwidth_gb_s = 50.0,
    });
    std::ofstream(temp_root / ".rlprof" / "old.json")
        << rlprof::bench::serialize_bench_output_json(output);
    std::ofstream(temp_root / ".rlprof" / "new.json")
        << rlprof::bench::serialize_bench_output_json(output);
  }
  std::ofstream(temp_root / ".rlprof" / "profile_export.json") << "{\"meta\":{}}\n";

  fs::last_write_time(temp_root / ".rlprof" / "old.db", fs::file_time_type::clock::now() - std::chrono::hours(3));
  fs::last_write_time(temp_root / ".rlprof" / "mid.db", fs::file_time_type::clock::now() - std::chrono::hours(2));
  fs::last_write_time(temp_root / ".rlprof" / "new.db", fs::file_time_type::clock::now() - std::chrono::hours(1));
  fs::last_write_time(temp_root / ".rlprof" / "old.json", fs::file_time_type::clock::now() - std::chrono::hours(4));
  fs::last_write_time(temp_root / ".rlprof" / "new.json", fs::file_time_type::clock::now() - std::chrono::minutes(30));
  fs::last_write_time(temp_root / ".rlprof" / "profile_export.json", fs::file_time_type::clock::now() - std::chrono::minutes(10));

  const auto recent = rlprof::interactive::list_recent_profiles(2);
  expect_true(recent.size() == 2, "expected two recent profiles");
  expect_true(recent[0].find("new.db") != std::string::npos, "expected newest profile first");
  expect_true(recent[1].find("mid.db") != std::string::npos, "expected second newest profile second");

  const auto recent_bench = rlprof::interactive::list_recent_bench_results(2);
  expect_true(recent_bench.size() == 2, "expected two recent bench results");
  expect_true(
      recent_bench[0].find("new.json") != std::string::npos,
      "expected newest bench result first");
  expect_true(
      recent_bench[1].find("old.json") != std::string::npos,
      "expected older bench result second");
  expect_true(
      recent_bench[0].find("profile_export.json") == std::string::npos &&
          recent_bench[1].find("profile_export.json") == std::string::npos,
      "expected non-bench json to be excluded");

  const rlprof::interactive::ProfileConfig saved_profile = {
      .model = "Qwen/Qwen3-8B",
      .target = "selfboot",
      .target_workdir = "/tmp/rlprof_bootstrap_test",
      .prompts = 32,
      .rollouts = 2,
      .min_tokens = 128,
      .max_tokens = 512,
      .input_len = 256,
      .port = 9000,
      .tp = 2,
      .peer_servers = "http://10.0.0.2:8000",
      .trust_remote_code = true,
      .discard_first_run = true,
      .repeat = 4,
      .output = ".rlprof/saved_profile",
  };
  rlprof::interactive::save_profile_defaults(saved_profile);
  const auto loaded_profile = rlprof::interactive::load_profile_defaults();
  expect_true(loaded_profile.model == saved_profile.model, "expected saved profile model");
  expect_true(loaded_profile.target == saved_profile.target, "expected saved profile target");
  expect_true(
      loaded_profile.target_workdir == saved_profile.target_workdir,
      "expected saved profile target workdir");
  expect_true(loaded_profile.prompts == saved_profile.prompts, "expected saved profile prompts");
  expect_true(loaded_profile.rollouts == saved_profile.rollouts, "expected saved profile rollouts");
  expect_true(loaded_profile.min_tokens == saved_profile.min_tokens, "expected saved profile min tokens");
  expect_true(loaded_profile.max_tokens == saved_profile.max_tokens, "expected saved profile max tokens");
  expect_true(loaded_profile.input_len == saved_profile.input_len, "expected saved profile input len");
  expect_true(loaded_profile.port == saved_profile.port, "expected saved profile port");
  expect_true(loaded_profile.tp == saved_profile.tp, "expected saved profile tp");
  expect_true(loaded_profile.peer_servers == saved_profile.peer_servers, "expected saved profile peers");
  expect_true(loaded_profile.trust_remote_code == saved_profile.trust_remote_code, "expected saved profile trust");
  expect_true(loaded_profile.discard_first_run == saved_profile.discard_first_run, "expected saved profile discard");
  expect_true(loaded_profile.repeat == saved_profile.repeat, "expected saved profile repeat");
  expect_true(loaded_profile.output == saved_profile.output, "expected saved profile output");

  const rlprof::interactive::BenchConfig saved_bench = {
      .kernel = "rotary_embedding",
      .target = "selfboot",
      .target_workdir = "/tmp/rlprof_bootstrap_test",
      .shapes = "1x1024,64x1024",
      .dtype = "fp16",
      .warmup = 11,
      .n_iter = 22,
      .repeats = 3,
  };
  rlprof::interactive::save_bench_defaults(saved_bench);
  const auto loaded_bench = rlprof::interactive::load_bench_defaults();
  expect_true(loaded_bench.kernel == saved_bench.kernel, "expected saved bench kernel");
  expect_true(loaded_bench.target == saved_bench.target, "expected saved bench target");
  expect_true(
      loaded_bench.target_workdir == saved_bench.target_workdir,
      "expected saved bench target workdir");
  expect_true(loaded_bench.shapes == saved_bench.shapes, "expected saved bench shapes");
  expect_true(loaded_bench.dtype == saved_bench.dtype, "expected saved bench dtype");
  expect_true(loaded_bench.warmup == saved_bench.warmup, "expected saved bench warmup");
  expect_true(loaded_bench.n_iter == saved_bench.n_iter, "expected saved bench n_iter");
  expect_true(loaded_bench.repeats == saved_bench.repeats, "expected saved bench repeats");

  rlprof::interactive::clear_saved_defaults();
  const auto cleared_profile = rlprof::interactive::load_profile_defaults();
  expect_true(cleared_profile.model.empty(), "expected cleared profile defaults");
  const auto cleared_bench = rlprof::interactive::load_bench_defaults();
  expect_true(cleared_bench.kernel == "silu_and_mul", "expected cleared bench kernel default");
  expect_true(cleared_bench.shapes == "64x4096,256x4096", "expected cleared bench shapes default");

  fs::current_path(previous_cwd);
  unsetenv("RLPROF_DEFAULTS_PATH");
  fs::remove_all(temp_root);
  return 0;
}
