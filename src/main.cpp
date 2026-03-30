#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "rlprof/diff.h"
#include "rlprof/export.h"
#include "rlprof/bench/registry.h"
#include "rlprof/bench/runner.h"
#include "rlprof/clock_control.h"
#include "rlprof/profiler/runner.h"
#include "rlprof/report.h"
#include "rlprof/stability.h"
#include "rlprof/store.h"
#include "rlprof/traffic.h"

namespace {

using Args = std::vector<std::string>;

std::string require_value(
    const Args& args,
    std::size_t& index,
    const std::string& flag) {
  if (index + 1 >= args.size()) {
    throw std::runtime_error("missing value for " + flag);
  }
  ++index;
  return args[index];
}

std::filesystem::path latest_profile_path() {
  namespace fs = std::filesystem;
  const fs::path dir = ".rlprof";
  fs::path latest;
  std::filesystem::file_time_type latest_time;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.path().extension() != ".db") {
      continue;
    }
    if (latest.empty() || entry.last_write_time() > latest_time) {
      latest = entry.path();
      latest_time = entry.last_write_time();
    }
  }
  if (latest.empty()) {
    throw std::runtime_error("No profile database found in .rlprof/");
  }
  return latest;
}

std::string sanitize_model_name(std::string value) {
  std::replace(value.begin(), value.end(), '/', '_');
  return value;
}

std::filesystem::path repeat_output_base(const rlprof::profiler::ProfileConfig& config) {
  if (!config.output.empty()) {
    return config.output;
  }
  const auto now = std::chrono::system_clock::now().time_since_epoch().count();
  return std::filesystem::path(".rlprof") /
         (sanitize_model_name(config.model) + "_" + std::to_string(now));
}

std::filesystem::path append_repeat_suffix(
    const std::filesystem::path& base,
    std::int64_t repeat_index) {
  const std::string suffix = "_r" + std::to_string(repeat_index);
  if (base.extension().empty()) {
    return std::filesystem::path(base.string() + suffix);
  }
  return base.parent_path() /
         (base.stem().string() + suffix + base.extension().string());
}

rlprof::ReportMeta to_report_meta(const std::map<std::string, std::string>& meta) {
  const auto get = [&](const std::string& key, const std::string& fallback = "") {
    const auto it = meta.find(key);
    return it == meta.end() ? fallback : it->second;
  };
  const auto get_i64 = [&](const std::string& key, std::int64_t fallback) {
    const auto it = meta.find(key);
    return it == meta.end() ? fallback : std::stoll(it->second);
  };

  return rlprof::ReportMeta{
      .model_name = get("model_name", "unknown-model"),
      .gpu_name = get("gpu_name", "unknown-gpu"),
      .vllm_version = get("vllm_version", "unknown-vllm"),
      .prompts = get_i64("prompts", 0),
      .rollouts = get_i64("rollouts", 0),
      .max_tokens = get_i64("max_tokens", 0),
  };
}

std::string shell_escape(const std::string& value) {
  std::string escaped = "'";
  for (char ch : value) {
    if (ch == '\'') {
      escaped += "'\\''";
    } else {
      escaped.push_back(ch);
    }
  }
  escaped += "'";
  return escaped;
}

std::string run_command_capture(const std::string& command) {
  std::array<char, 4096> buffer{};
  std::string output;
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    throw std::runtime_error("failed to run command: " + command);
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output.append(buffer.data());
  }
  const int rc = pclose(pipe);
  if (rc != 0) {
    throw std::runtime_error(output.empty() ? "bench helper failed" : output);
  }
  return output;
}

std::string optional_json(const std::optional<double>& value) {
  return value.has_value() ? std::to_string(*value) : "null";
}

std::string render_traffic_json(const rlprof::TrafficStats& stats) {
  return "{"
         "\"total_requests\":" + std::to_string(stats.total_requests) +
         ",\"completion_length_mean\":" + optional_json(stats.completion_length_mean) +
         ",\"completion_length_p50\":" + optional_json(stats.completion_length_p50) +
         ",\"completion_length_p99\":" + optional_json(stats.completion_length_p99) +
         ",\"max_median_ratio\":" + optional_json(stats.max_median_ratio) +
         ",\"errors\":" + std::to_string(stats.errors) +
         "}\n";
}

int handle_profile(const Args& args) {
  rlprof::profiler::ProfileConfig config;
  std::int64_t repeats = 1;
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--model") {
      config.model = require_value(args, i, "--model");
    } else if (args[i] == "--prompts") {
      config.prompts = std::stoll(require_value(args, i, "--prompts"));
    } else if (args[i] == "--rollouts") {
      config.rollouts = std::stoll(require_value(args, i, "--rollouts"));
    } else if (args[i] == "--max-tokens") {
      config.max_tokens = std::stoll(require_value(args, i, "--max-tokens"));
    } else if (args[i] == "--min-tokens") {
      config.min_tokens = std::stoll(require_value(args, i, "--min-tokens"));
    } else if (args[i] == "--input-len") {
      config.input_len = std::stoll(require_value(args, i, "--input-len"));
    } else if (args[i] == "--port") {
      config.port = std::stoll(require_value(args, i, "--port"));
    } else if (args[i] == "--tp") {
      config.tp = std::stoll(require_value(args, i, "--tp"));
    } else if (args[i] == "--trust-remote-code") {
      config.trust_remote_code = true;
    } else if (args[i] == "--output") {
      config.output = require_value(args, i, "--output");
    } else if (args[i] == "--repeat") {
      repeats = std::stoll(require_value(args, i, "--repeat"));
    } else if (args[i] == "--help") {
      std::cout << "Usage: rlprof profile --model MODEL [options] [--repeat N]\n";
      return 0;
    }
  }

  if (config.model.empty()) {
    throw std::runtime_error("--model is required");
  }

  if (repeats <= 0) {
    throw std::runtime_error("--repeat must be > 0");
  }

  if (repeats == 1) {
    const auto result = rlprof::profiler::run_profile(config);
    std::cout << result.db_path << "\n";
    return 0;
  }

  const std::filesystem::path output_base = repeat_output_base(config);
  std::vector<std::filesystem::path> db_paths;
  std::vector<rlprof::ProfileData> profiles;
  db_paths.reserve(static_cast<std::size_t>(repeats));
  profiles.reserve(static_cast<std::size_t>(repeats));

  for (std::int64_t run_index = 1; run_index <= repeats; ++run_index) {
    auto run_config = config;
    run_config.output = append_repeat_suffix(output_base, run_index);
    const auto result = rlprof::profiler::run_profile(run_config);
    db_paths.push_back(result.db_path);
    profiles.push_back(rlprof::load_profile(result.db_path));
    std::cout << result.db_path << "\n";
  }

  const auto& final_profile = profiles.back();
  std::cout << "\n";
  std::cout << rlprof::render_report(
      to_report_meta(final_profile.meta),
      final_profile.meta,
      final_profile.kernels,
      final_profile.metrics_summary,
      final_profile.traffic_stats);
  std::cout << "\n";
  std::cout << rlprof::render_stability_report(
      rlprof::compute_stability_report(profiles));
  return 0;
}

int handle_report(const Args& args) {
  const std::filesystem::path path =
      args.size() >= 2 ? std::filesystem::path(args[1]) : latest_profile_path();
  const auto profile = rlprof::load_profile(path);
  std::cout << rlprof::render_report(
      to_report_meta(profile.meta),
      profile.meta,
      profile.kernels,
      profile.metrics_summary,
      profile.traffic_stats);
  return 0;
}

int handle_export(const Args& args) {
  std::filesystem::path path;
  std::string format;
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--format") {
      format = require_value(args, i, "--format");
    } else if (!args[i].starts_with("--")) {
      path = args[i];
    }
  }

  if (format.empty()) {
    throw std::runtime_error("--format is required");
  }
  if (path.empty()) {
    path = latest_profile_path();
  }

  for (const auto& output : rlprof::export_profile(path, format)) {
    std::cout << output << "\n";
  }
  return 0;
}

int handle_diff(const Args& args) {
  if (args.size() < 3) {
    throw std::runtime_error("diff requires two database paths");
  }
  std::cout << rlprof::render_diff(args[1], args[2]);
  return 0;
}

int handle_traffic(const Args& args) {
  std::string server;
  std::int64_t prompts = 128;
  std::int64_t rollouts_per_prompt = 8;
  std::int64_t max_tokens = 4096;
  std::int64_t min_tokens = 256;
  std::int64_t input_len = 512;

  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--server") {
      server = require_value(args, i, "--server");
    } else if (args[i] == "--prompts") {
      prompts = std::stoll(require_value(args, i, "--prompts"));
    } else if (args[i] == "--rollouts-per-prompt") {
      rollouts_per_prompt = std::stoll(require_value(args, i, "--rollouts-per-prompt"));
    } else if (args[i] == "--max-tokens") {
      max_tokens = std::stoll(require_value(args, i, "--max-tokens"));
    } else if (args[i] == "--min-tokens") {
      min_tokens = std::stoll(require_value(args, i, "--min-tokens"));
    } else if (args[i] == "--input-len") {
      input_len = std::stoll(require_value(args, i, "--input-len"));
    }
  }

  if (server.empty()) {
    throw std::runtime_error("--server is required");
  }

  const auto run = rlprof::fire_rl_traffic(
      server, prompts, rollouts_per_prompt, min_tokens, max_tokens, input_len);
  std::cout << render_traffic_json(run.stats);
  return 0;
}

std::string bench_helper_command(
    const std::string& kernel,
    const std::string& shapes,
    const std::string& dtype,
    std::int64_t warmup,
    std::int64_t n_iter,
    std::int64_t repeats) {
  return shell_escape(".venv/bin/python") + " " + shell_escape("tools/bench_cuda.py") +
        " --kernel " + shell_escape(kernel) +
        " --shapes " + shell_escape(shapes) +
        " --dtype " + shell_escape(dtype) +
        " --warmup " + shell_escape(std::to_string(warmup)) +
        " --n-iter " + shell_escape(std::to_string(n_iter)) +
         " --repeats " + shell_escape(std::to_string(repeats)) + " 2>&1";
}

bool gpu_bench_available() {
  if (!std::filesystem::exists(".venv/bin/python") ||
      !std::filesystem::exists("tools/bench_cuda.py")) {
    return false;
  }
  const std::string probe =
      ".venv/bin/python -c " +
      shell_escape(
          "import torch, vllm; raise SystemExit(0 if torch.cuda.is_available() else 1)") +
      " > /dev/null 2>&1";
  return std::system(probe.c_str()) == 0;
}

int handle_bench(const Args& args) {
  std::string kernel;
  std::string shapes = "1x4096,64x4096,256x4096";
  std::string dtype = "bf16";
  std::int64_t warmup = 20;
  std::int64_t n_iter = 200;
  std::int64_t repeats = 5;

  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--kernel") {
      kernel = require_value(args, i, "--kernel");
    } else if (args[i] == "--shapes") {
      shapes = require_value(args, i, "--shapes");
    } else if (args[i] == "--dtype") {
      dtype = require_value(args, i, "--dtype");
    } else if (args[i] == "--warmup") {
      warmup = std::stoll(require_value(args, i, "--warmup"));
    } else if (args[i] == "--n-iter") {
      n_iter = std::stoll(require_value(args, i, "--n-iter"));
    } else if (args[i] == "--repeats") {
      repeats = std::stoll(require_value(args, i, "--repeats"));
    }
  }

  if (kernel.empty()) {
    throw std::runtime_error("--kernel is required");
  }

  if (gpu_bench_available()) {
    const auto output = rlprof::bench::parse_bench_json(
        run_command_capture(
            bench_helper_command(kernel, shapes, dtype, warmup, n_iter, repeats)));
    std::cout << rlprof::bench::render_bench_output(output);
    return 0;
  }

  std::cerr << "warning: torch/vllm CUDA bench helper unavailable, falling back to native CPU stubs\n";
  rlprof::bench::register_builtin_kernels();
  const auto results = rlprof::bench::benchmark_category(
      kernel,
      rlprof::bench::parse_shapes(shapes),
      dtype,
      warmup,
      n_iter);
  std::cout << rlprof::bench::render_bench_results(results);
  return 0;
}

int handle_lock_clocks(const Args& args) {
  std::optional<std::int64_t> freq_mhz;
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--freq") {
      freq_mhz = std::stoll(require_value(args, i, "--freq"));
    } else if (args[i] == "--help") {
      std::cout << "Usage: rlprof lock-clocks [--freq MHZ]\n";
      return 0;
    }
  }

  rlprof::lock_gpu_clocks(freq_mhz);
  const std::int64_t effective_freq =
      freq_mhz.value_or(rlprof::query_max_sm_clock_mhz());
  std::cout << "GPU clocks locked to " << effective_freq << " MHz\n";
  return 0;
}

int handle_unlock_clocks(const Args& args) {
  if (args.size() > 1 && args[1] == "--help") {
    std::cout << "Usage: rlprof unlock-clocks\n";
    return 0;
  }
  rlprof::unlock_gpu_clocks();
  std::cout << "GPU clocks unlocked\n";
  return 0;
}

void print_help() {
  std::cout << "Usage: rlprof <command> [options]\n\n"
            << "Commands:\n"
            << "  profile --model MODEL [options] [--repeat N]\n"
            << "  lock-clocks [--freq MHZ]\n"
            << "  unlock-clocks\n"
            << "  report [path]\n"
            << "  export [path] --format csv|json\n"
            << "  diff <a.db> <b.db>\n"
            << "  bench --kernel NAME --shapes SPEC [options]\n"
            << "  traffic --server URL [options]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      print_help();
      return 0;
    }

    const Args args(argv + 1, argv + argc);
    const std::string command = args[0];

    if (command == "profile") {
      return handle_profile(args);
    }

    if (command == "report") {
      return handle_report(args);
    }

    if (command == "lock-clocks") {
      return handle_lock_clocks(args);
    }

    if (command == "unlock-clocks") {
      return handle_unlock_clocks(args);
    }

    if (command == "export") {
      return handle_export(args);
    }

    if (command == "diff") {
      return handle_diff(args);
    }

    if (command == "traffic") {
      return handle_traffic(args);
    }

    if (command == "bench") {
      return handle_bench(args);
    }

    if (command == "--help" || command == "help") {
      print_help();
      return 0;
    }

    throw std::runtime_error("unknown command: " + command);
  } catch (const std::exception& exc) {
    std::cerr << exc.what() << "\n";
    return 1;
  }
}
