#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
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
#include "interactive.h"

namespace {

using Args = std::vector<std::string>;

struct ProfileCommandOptions {
  rlprof::profiler::ProfileConfig config;
  std::int64_t repeats = 1;
  bool assume_yes = false;
  bool show_help = false;
};

struct BenchCommandOptions {
  std::string kernel;
  std::string shapes = "64x4096,256x4096";
  std::string dtype = "bf16";
  std::int64_t warmup = 20;
  std::int64_t n_iter = 200;
  std::int64_t repeats = 5;
  bool assume_yes = false;
  bool show_help = false;
};

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

bool has_flag(const Args& args, const std::string& flag) {
  return std::find(args.begin(), args.end(), flag) != args.end();
}

bool has_positional_arg(const Args& args) {
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (!args[i].starts_with("--")) {
      return true;
    }
  }
  return false;
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

void remove_profile_artifacts(const std::filesystem::path& base) {
  std::error_code ec;
  const auto remove_if_exists = [&](const std::filesystem::path& path) {
    std::filesystem::remove(path, ec);
    ec.clear();
  };
  remove_if_exists(base);
  remove_if_exists(base.string() + ".db");
  remove_if_exists(base.string() + ".sqlite");
  remove_if_exists(base.string() + ".nsys-rep");
  remove_if_exists(base.string() + "_nvidia_smi.xml");
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

std::string trim(std::string value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back()))) {
    value.pop_back();
  }
  std::size_t start = 0;
  while (start < value.size() &&
         std::isspace(static_cast<unsigned char>(value[start]))) {
    ++start;
  }
  return value.substr(start);
}

std::vector<std::string> split_csv_list(const std::string& value) {
  std::vector<std::string> items;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    item = trim(item);
    if (!item.empty()) {
      items.push_back(item);
    }
  }
  return items;
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

ProfileCommandOptions parse_profile_args(const Args& args) {
  ProfileCommandOptions options;
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--model") {
      options.config.model = require_value(args, i, "--model");
    } else if (args[i] == "--prompts") {
      options.config.prompts = std::stoll(require_value(args, i, "--prompts"));
    } else if (args[i] == "--rollouts") {
      options.config.rollouts = std::stoll(require_value(args, i, "--rollouts"));
    } else if (args[i] == "--max-tokens") {
      options.config.max_tokens = std::stoll(require_value(args, i, "--max-tokens"));
    } else if (args[i] == "--min-tokens") {
      options.config.min_tokens = std::stoll(require_value(args, i, "--min-tokens"));
    } else if (args[i] == "--input-len") {
      options.config.input_len = std::stoll(require_value(args, i, "--input-len"));
    } else if (args[i] == "--port") {
      options.config.port = std::stoll(require_value(args, i, "--port"));
    } else if (args[i] == "--tp") {
      options.config.tp = std::stoll(require_value(args, i, "--tp"));
    } else if (args[i] == "--peer-servers") {
      options.config.peer_servers = split_csv_list(require_value(args, i, "--peer-servers"));
    } else if (args[i] == "--trust-remote-code") {
      options.config.trust_remote_code = true;
    } else if (args[i] == "--output") {
      options.config.output = require_value(args, i, "--output");
    } else if (args[i] == "--repeat") {
      options.repeats = std::stoll(require_value(args, i, "--repeat"));
    } else if (args[i] == "--discard-first-run") {
      options.config.discard_first_run = true;
    } else if (args[i] == "--yes") {
      options.assume_yes = true;
    } else if (args[i] == "--help") {
      options.show_help = true;
    }
  }
  return options;
}

std::string run_profile_command(
    const ProfileCommandOptions& options,
    const rlprof::profiler::ProgressCallback& progress = {}) {
  if (options.config.model.empty()) {
    throw std::runtime_error("--model is required");
  }

  if (options.repeats <= 0) {
    throw std::runtime_error("--repeat must be > 0");
  }

  std::ostringstream output;
  if (options.config.discard_first_run) {
    auto warmup_config = options.config;
    const auto output_base = repeat_output_base(options.config);
    warmup_config.output = append_repeat_suffix(output_base, 0);
    const auto warmup_progress = [&](const std::string& status) {
      if (progress) {
        progress("[warmup] " + status);
      }
    };
    rlprof::profiler::run_profile(warmup_config, warmup_progress);
    remove_profile_artifacts(warmup_config.output);
  }

  if (options.repeats == 1) {
    const auto result = rlprof::profiler::run_profile(options.config, progress);
    output << result.db_path << "\n";
    return output.str();
  }

  const std::filesystem::path output_base = repeat_output_base(options.config);
  std::vector<std::filesystem::path> db_paths;
  std::vector<rlprof::ProfileData> profiles;
  db_paths.reserve(static_cast<std::size_t>(options.repeats));
  profiles.reserve(static_cast<std::size_t>(options.repeats));

  for (std::int64_t run_index = 1; run_index <= options.repeats; ++run_index) {
    auto run_config = options.config;
    run_config.output = append_repeat_suffix(output_base, run_index);
    const auto run_progress = [&](const std::string& status) {
      if (progress) {
        progress(
            "[run " + std::to_string(run_index) + "/" + std::to_string(options.repeats) +
            "] " + status);
      }
    };
    const auto result = rlprof::profiler::run_profile(run_config, run_progress);
    db_paths.push_back(result.db_path);
    profiles.push_back(rlprof::load_profile(result.db_path));
    output << result.db_path << "\n";
  }

  const auto& final_profile = profiles.back();
  output << "\n";
  output << rlprof::render_report(
      to_report_meta(final_profile.meta),
      final_profile.meta,
      final_profile.kernels,
      final_profile.metrics_summary,
      final_profile.traffic_stats);
  output << "\n";
  output << rlprof::render_stability_report(
      rlprof::compute_stability_report(profiles));
  return output.str();
}

int handle_profile(const Args& args) {
  const auto options = parse_profile_args(args);
  if (options.show_help) {
    std::cout << "Usage: rlprof profile --model MODEL [options] [--repeat N]\n"
              << "       optional: --peer-servers URL1,URL2,...\n"
              << "       flags: --discard-first-run --yes\n";
    return 0;
  }
  std::cout << run_profile_command(options);
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
  std::vector<std::string> servers;
  std::int64_t prompts = 128;
  std::int64_t rollouts_per_prompt = 8;
  std::int64_t max_tokens = 4096;
  std::int64_t min_tokens = 256;
  std::int64_t input_len = 512;

  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--server") {
      server = require_value(args, i, "--server");
    } else if (args[i] == "--servers") {
      servers = split_csv_list(require_value(args, i, "--servers"));
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

  if (servers.empty() && !server.empty()) {
    servers.push_back(server);
  }

  if (servers.empty()) {
    throw std::runtime_error("--server or --servers is required");
  }

  const auto run = rlprof::fire_rl_traffic(
      servers, prompts, rollouts_per_prompt, min_tokens, max_tokens, input_len);
  std::cout << render_traffic_json(run.stats);
  return 0;
}

rlprof::interactive::ProfileConfig profile_interactive_defaults(const Args& args) {
  auto config = rlprof::interactive::load_profile_defaults();
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--model") {
      config.model = require_value(args, i, "--model");
    } else if (args[i] == "--prompts") {
      config.prompts = std::stoi(require_value(args, i, "--prompts"));
    } else if (args[i] == "--rollouts") {
      config.rollouts = std::stoi(require_value(args, i, "--rollouts"));
    } else if (args[i] == "--max-tokens") {
      config.max_tokens = std::stoi(require_value(args, i, "--max-tokens"));
    } else if (args[i] == "--min-tokens") {
      config.min_tokens = std::stoi(require_value(args, i, "--min-tokens"));
    } else if (args[i] == "--input-len") {
      config.input_len = std::stoi(require_value(args, i, "--input-len"));
    } else if (args[i] == "--port") {
      config.port = std::stoi(require_value(args, i, "--port"));
    } else if (args[i] == "--tp") {
      config.tp = std::stoi(require_value(args, i, "--tp"));
    } else if (args[i] == "--peer-servers") {
      config.peer_servers = require_value(args, i, "--peer-servers");
    } else if (args[i] == "--trust-remote-code") {
      config.trust_remote_code = true;
    } else if (args[i] == "--discard-first-run") {
      config.discard_first_run = true;
    } else if (args[i] == "--repeat") {
      config.repeat = std::stoi(require_value(args, i, "--repeat"));
    } else if (args[i] == "--output") {
      config.output = require_value(args, i, "--output");
    }
  }
  return config;
}

std::string bench_helper_command(
    const std::string& kernel,
    const std::string& shapes,
    const std::string& dtype,
    std::int64_t warmup,
    std::int64_t n_iter,
    std::int64_t repeats) {
  const char* configured_python = std::getenv("RLPROF_PYTHON_EXECUTABLE");
  const std::string python =
      configured_python != nullptr && std::string(configured_python).size() > 0
          ? std::string(configured_python)
          : (std::filesystem::exists(".venv/bin/python") ? ".venv/bin/python" : "python3");
  return shell_escape(python) + " -m " + shell_escape("rlprof_py.bench_cuda") +
        " --kernel " + shell_escape(kernel) +
        " --shapes " + shell_escape(shapes) +
        " --dtype " + shell_escape(dtype) +
        " --warmup " + shell_escape(std::to_string(warmup)) +
        " --n-iter " + shell_escape(std::to_string(n_iter)) +
         " --repeats " + shell_escape(std::to_string(repeats)) + " 2>&1";
}

bool gpu_bench_available() {
  const char* configured_python = std::getenv("RLPROF_PYTHON_EXECUTABLE");
  const std::string python =
      configured_python != nullptr && std::string(configured_python).size() > 0
          ? std::string(configured_python)
          : (std::filesystem::exists(".venv/bin/python") ? ".venv/bin/python" : "python3");
  const std::string probe =
      shell_escape(python) + " -c " +
      shell_escape(
          "import torch, vllm, rlprof_py.bench_cuda; raise SystemExit(0 if torch.cuda.is_available() else 1)") +
      " > /dev/null 2>&1";
  return std::system(probe.c_str()) == 0;
}

BenchCommandOptions parse_bench_args(const Args& args) {
  BenchCommandOptions options;

  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--kernel") {
      options.kernel = require_value(args, i, "--kernel");
    } else if (args[i] == "--shapes") {
      options.shapes = require_value(args, i, "--shapes");
    } else if (args[i] == "--dtype") {
      options.dtype = require_value(args, i, "--dtype");
    } else if (args[i] == "--warmup") {
      options.warmup = std::stoll(require_value(args, i, "--warmup"));
    } else if (args[i] == "--n-iter") {
      options.n_iter = std::stoll(require_value(args, i, "--n-iter"));
    } else if (args[i] == "--repeats") {
      options.repeats = std::stoll(require_value(args, i, "--repeats"));
    } else if (args[i] == "--yes") {
      options.assume_yes = true;
    } else if (args[i] == "--help") {
      options.show_help = true;
    }
  }
  return options;
}

std::string run_bench_command(const BenchCommandOptions& options) {
  if (options.kernel.empty()) {
    throw std::runtime_error("--kernel is required");
  }

  if (gpu_bench_available()) {
    const auto output = rlprof::bench::parse_bench_json(
        run_command_capture(
            bench_helper_command(
                options.kernel,
                options.shapes,
                options.dtype,
                options.warmup,
                options.n_iter,
                options.repeats)));
    return rlprof::bench::render_bench_output(output);
  }

  std::ostringstream output;
  output << "warning: torch/vllm CUDA bench helper unavailable, falling back to native CPU stubs\n";
  rlprof::bench::register_builtin_kernels();
  const auto results = rlprof::bench::benchmark_category(
      options.kernel,
      rlprof::bench::parse_shapes(options.shapes),
      options.dtype,
      options.warmup,
      options.n_iter);
  output << rlprof::bench::render_bench_results(results);
  return output.str();
}

int handle_bench(const Args& args) {
  const auto options = parse_bench_args(args);
  if (options.show_help) {
    std::cout << "Usage: rlprof bench --kernel NAME --shapes SPEC [options]\n"
              << "       flags: --yes\n";
    return 0;
  }
  std::cout << run_bench_command(options);
  return 0;
}

rlprof::interactive::BenchConfig bench_interactive_defaults(const Args& args) {
  auto config = rlprof::interactive::load_bench_defaults();
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--kernel") {
      config.kernel = require_value(args, i, "--kernel");
    } else if (args[i] == "--shapes") {
      config.shapes = require_value(args, i, "--shapes");
    } else if (args[i] == "--dtype") {
      config.dtype = require_value(args, i, "--dtype");
    } else if (args[i] == "--warmup") {
      config.warmup = std::stoi(require_value(args, i, "--warmup"));
    } else if (args[i] == "--n-iter") {
      config.n_iter = std::stoi(require_value(args, i, "--n-iter"));
    } else if (args[i] == "--repeats") {
      config.repeats = std::stoi(require_value(args, i, "--repeats"));
    }
  }
  return config;
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

int handle_reset_defaults(const Args& args) {
  if (args.size() > 1 && args[1] == "--help") {
    std::cout << "Usage: rlprof reset-defaults\n";
    return 0;
  }
  rlprof::interactive::clear_saved_defaults();
  std::cout << "Interactive defaults cleared\n";
  return 0;
}

std::string format_recent_profile_option(const std::string& path, bool include_timestamp) {
  const auto filename = std::filesystem::path(path).filename().string();
  if (!include_timestamp) {
    return filename;
  }
  try {
    const auto file_time = std::filesystem::last_write_time(path);
    const auto system_time =
        std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            file_time - std::filesystem::file_time_type::clock::now() +
            std::chrono::system_clock::now());
    const std::time_t time = std::chrono::system_clock::to_time_t(system_time);
    std::tm tm{};
    gmtime_r(&time, &tm);
    std::ostringstream stream;
    stream << filename << "  "
           << std::put_time(&tm, "%Y-%m-%d %H:%M");
    return stream.str();
  } catch (const std::exception&) {
    return filename;
  }
}

std::optional<std::string> choose_recent_profile(
    const std::string& header,
    bool include_timestamp) {
  rlprof::interactive::print_header(header);
  const auto profiles = rlprof::interactive::list_recent_profiles(10);
  if (profiles.empty()) {
    rlprof::interactive::print_warning("No profiles found in .rlprof/");
    return std::nullopt;
  }
  std::vector<std::string> options;
  options.reserve(profiles.size());
  for (const auto& path : profiles) {
    options.push_back(format_recent_profile_option(path, include_timestamp));
  }
  const auto choice =
      rlprof::interactive::prompt_choice("Recent profiles", options, 0);
  if (!choice.has_value()) {
    return std::nullopt;
  }
  return profiles[static_cast<std::size_t>(*choice)];
}

int execute_profile_config(
    const rlprof::interactive::ProfileConfig& config,
    bool confirm_start,
    bool persist_defaults) {
  if (config.model.empty()) {
    throw std::runtime_error("profile requires a model");
  }

  const std::string gpu_name = rlprof::interactive::detect_gpu_name();
  const std::string clock_status = rlprof::interactive::clock_status_label();
  const bool clocks_locked = rlprof::interactive::are_clocks_locked();

  std::cout << "\n";
  rlprof::interactive::print_info(
      "-> " + config.model + " · " + std::to_string(config.prompts) + " prompts x " +
          std::to_string(config.rollouts) + " rollouts · " +
          std::to_string(config.min_tokens) + "-" + std::to_string(config.max_tokens) +
          " tokens",
      "");
  rlprof::interactive::print_info(
      "-> " + gpu_name + " · clocks: " + clock_status,
      "");
  if (!trim(config.peer_servers).empty()) {
    const auto peers = split_csv_list(config.peer_servers);
    rlprof::interactive::print_info(
        "-> cluster endpoints: " + std::to_string(peers.size() + 1),
        "");
  }
  if (config.discard_first_run) {
    rlprof::interactive::print_info("-> discard first run", "enabled");
  }
  if (!clocks_locked) {
    rlprof::interactive::print_warning(
        "GPU clocks unlocked - run `rlprof lock-clocks` for reproducibility");
  }

  if (confirm_start) {
    const auto confirm = rlprof::interactive::prompt_bool("Start profiling", true);
    if (!confirm.has_value() || !*confirm) {
      return 0;
    }
  }

  std::string output_text;
  rlprof::interactive::run_with_progress(
      "Starting vLLM server...",
      [&](const rlprof::interactive::ProgressCallback& progress) {
        output_text = run_profile_command(
            parse_profile_args(rlprof::interactive::build_profile_args(config)),
            progress);
      });

  if (persist_defaults) {
    rlprof::interactive::save_profile_defaults(config);
  }

  const std::string trimmed_output = trim(output_text);
  if (config.repeat == 1) {
    const auto newline = trimmed_output.find('\n');
    const std::string saved_path =
        newline == std::string::npos ? trimmed_output : trimmed_output.substr(0, newline);
    std::cout << "  Saved: " << saved_path << "\n";
    std::cout << "  Run `rlprof report " << saved_path << "` to view results\n";
    if (newline != std::string::npos) {
      std::cout << trim(trimmed_output.substr(newline + 1)) << "\n";
    }
  } else {
    std::cout << trimmed_output << "\n";
  }
  return 0;
}

int interactive_profile_flow(const rlprof::interactive::ProfileConfig& initial) {
  rlprof::interactive::print_header("rlprof · profile your rl environment");

  std::optional<std::string> model;
  while (!model.has_value() || model->empty()) {
    model = rlprof::interactive::prompt_string("Model", initial.model);
    if (!model.has_value()) {
      return 0;
    }
    if (model->empty()) {
      rlprof::interactive::print_warning("Model is required");
    }
  }

  auto prompts = rlprof::interactive::prompt_int("Prompts per batch", initial.prompts);
  if (!prompts.has_value()) {
    return 0;
  }
  auto rollouts = rlprof::interactive::prompt_int("Rollouts per prompt", initial.rollouts);
  if (!rollouts.has_value()) {
    return 0;
  }
  auto min_tokens = rlprof::interactive::prompt_int("Min output tokens", initial.min_tokens);
  if (!min_tokens.has_value()) {
    return 0;
  }
  auto max_tokens = rlprof::interactive::prompt_int("Max output tokens", initial.max_tokens);
  if (!max_tokens.has_value()) {
    return 0;
  }
  if (*min_tokens > *max_tokens) {
    rlprof::interactive::print_warning("Min output tokens must be <= max output tokens");
    return 0;
  }
  auto input_len = rlprof::interactive::prompt_int("Input length", initial.input_len);
  if (!input_len.has_value()) {
    return 0;
  }
  auto port = rlprof::interactive::prompt_int("Server port", initial.port);
  if (!port.has_value()) {
    return 0;
  }
  auto tp = rlprof::interactive::prompt_int("Tensor parallel size", initial.tp);
  if (!tp.has_value()) {
    return 0;
  }
  auto peer_servers = rlprof::interactive::prompt_string("Peer servers csv", initial.peer_servers);
  if (!peer_servers.has_value()) {
    return 0;
  }
  auto trust_remote_code =
      rlprof::interactive::prompt_bool("Trust remote code", initial.trust_remote_code);
  if (!trust_remote_code.has_value()) {
    return 0;
  }
  auto discard_first_run =
      rlprof::interactive::prompt_bool("Discard first run", initial.discard_first_run);
  if (!discard_first_run.has_value()) {
    return 0;
  }
  auto repeat = rlprof::interactive::prompt_int("Repeat runs", initial.repeat);
  if (!repeat.has_value()) {
    return 0;
  }
  auto output = rlprof::interactive::prompt_string("Output path", initial.output);
  if (!output.has_value()) {
    return 0;
  }

  const rlprof::interactive::ProfileConfig config = {
      .model = *model,
      .prompts = *prompts,
      .rollouts = *rollouts,
      .min_tokens = *min_tokens,
      .max_tokens = *max_tokens,
      .input_len = *input_len,
      .port = *port,
      .tp = *tp,
      .peer_servers = *peer_servers,
      .trust_remote_code = *trust_remote_code,
      .discard_first_run = *discard_first_run,
      .repeat = *repeat,
      .output = output->empty() ? "auto" : *output,
  };
  return execute_profile_config(config, true, true);
}

int execute_bench_config(
    const rlprof::interactive::BenchConfig& config,
    bool persist_defaults) {
  std::cout << "\n  Benchmarking " << config.kernel
            << " on " << rlprof::interactive::detect_gpu_name() << "...\n";

  std::string output_text;
  rlprof::interactive::run_with_progress(
      "Benchmarking...",
      [&](const rlprof::interactive::ProgressCallback&) {
        output_text = run_bench_command(
            parse_bench_args(rlprof::interactive::build_bench_args(config)));
      });

  if (persist_defaults) {
    rlprof::interactive::save_bench_defaults(config);
  }

  std::cout << output_text;
  return 0;
}

int interactive_bench_flow(const rlprof::interactive::BenchConfig& initial) {
  rlprof::interactive::print_header("rlprof · benchmark kernel implementations");
  const std::vector<std::string> kernels = {
      "silu_and_mul",
      "fused_add_rms_norm",
      "rotary_embedding",
  };
  int default_kernel = 0;
  for (std::size_t i = 0; i < kernels.size(); ++i) {
    if (kernels[i] == initial.kernel) {
      default_kernel = static_cast<int>(i);
      break;
    }
  }
  const auto kernel_choice =
      rlprof::interactive::prompt_choice("Kernel", kernels, default_kernel);
  if (!kernel_choice.has_value()) {
    return 0;
  }
  const auto shapes = rlprof::interactive::prompt_string("Shapes", initial.shapes);
  if (!shapes.has_value()) {
    return 0;
  }
  const auto dtype = rlprof::interactive::prompt_string("Dtype", initial.dtype);
  if (!dtype.has_value()) {
    return 0;
  }
  const auto warmup = rlprof::interactive::prompt_int("Warmup iterations", initial.warmup);
  if (!warmup.has_value()) {
    return 0;
  }
  const auto n_iter = rlprof::interactive::prompt_int("Timed iterations", initial.n_iter);
  if (!n_iter.has_value()) {
    return 0;
  }
  const auto repeats = rlprof::interactive::prompt_int("Repeat runs", initial.repeats);
  if (!repeats.has_value()) {
    return 0;
  }

  const rlprof::interactive::BenchConfig config = {
      .kernel = kernels[static_cast<std::size_t>(*kernel_choice)],
      .shapes = *shapes,
      .dtype = *dtype,
      .warmup = *warmup,
      .n_iter = *n_iter,
      .repeats = *repeats,
  };
  return execute_bench_config(config, true);
}

int interactive_report_flow() {
  const auto path = choose_recent_profile("rlprof · view a saved profile", true);
  if (!path.has_value()) {
    return 0;
  }
  return handle_report({"report", *path});
}

int interactive_diff_flow() {
  rlprof::interactive::print_header("rlprof · compare two profiles");
  const auto profiles = rlprof::interactive::list_recent_profiles(10);
  if (profiles.size() < 2) {
    rlprof::interactive::print_warning("Need at least two profiles in .rlprof/");
    return 0;
  }
  std::vector<std::string> options;
  options.reserve(profiles.size());
  for (const auto& path : profiles) {
    options.push_back(std::filesystem::path(path).filename().string());
  }
  const auto baseline =
      rlprof::interactive::prompt_choice("Baseline", options, 0);
  if (!baseline.has_value()) {
    return 0;
  }
  const auto candidate =
      rlprof::interactive::prompt_choice("Candidate", options, std::min<int>(1, options.size() - 1));
  if (!candidate.has_value()) {
    return 0;
  }
  return handle_diff(
      {"diff", profiles[static_cast<std::size_t>(*baseline)], profiles[static_cast<std::size_t>(*candidate)]});
}

int interactive_export_flow(const std::string& default_format = "csv") {
  const auto path = choose_recent_profile("rlprof · export profile data", false);
  if (!path.has_value()) {
    return 0;
  }
  const std::vector<std::string> formats = {"csv", "json"};
  const int default_index = default_format == "json" ? 1 : 0;
  const auto format_choice =
      rlprof::interactive::prompt_choice("Format", formats, default_index);
  if (!format_choice.has_value()) {
    return 0;
  }
  return handle_export(
      {"export", *path, "--format", formats[static_cast<std::size_t>(*format_choice)]});
}

std::string completion_script(const std::string& shell) {
  if (shell == "bash") {
    return R"(# bash completion for rlprof
_rlprof_complete() {
  local cur prev
  COMPREPLY=()
  cur="${COMP_WORDS[COMP_CWORD]}"
  prev="${COMP_WORDS[COMP_CWORD-1]}"
  local commands="profile report bench diff export traffic lock-clocks unlock-clocks reset-defaults completion help"
  if [[ $COMP_CWORD -eq 1 ]]; then
    COMPREPLY=( $(compgen -W "$commands" -- "$cur") )
    return
  fi
  case "${COMP_WORDS[1]}" in
    profile)
      COMPREPLY=( $(compgen -W "--model --prompts --rollouts --min-tokens --max-tokens --input-len --port --tp --peer-servers --trust-remote-code --discard-first-run --repeat --output --yes --help" -- "$cur") )
      ;;
    bench)
      COMPREPLY=( $(compgen -W "--kernel --shapes --dtype --warmup --n-iter --repeats --yes --help" -- "$cur") )
      ;;
    report)
      COMPREPLY=( $(compgen -f -- "$cur") )
      ;;
    diff)
      COMPREPLY=( $(compgen -f -- "$cur") )
      ;;
    export)
      COMPREPLY=( $(compgen -W "--format csv json" -- "$cur") $(compgen -f -- "$cur") )
      ;;
    traffic)
      COMPREPLY=( $(compgen -W "--server --servers --prompts --rollouts-per-prompt --min-tokens --max-tokens --input-len" -- "$cur") )
      ;;
    lock-clocks)
      COMPREPLY=( $(compgen -W "--freq --help" -- "$cur") )
      ;;
    unlock-clocks|reset-defaults|completion|help)
      COMPREPLY=()
      ;;
  esac
}
complete -F _rlprof_complete rlprof
)";
  }
  if (shell == "zsh") {
    return R"(#compdef rlprof
_rlprof() {
  local -a commands
  commands=(
    'profile:run GPU profiling under RL traffic'
    'report:view a saved profile'
    'bench:benchmark kernel implementations'
    'diff:compare two profiles'
    'export:export profile data'
    'traffic:run traffic generator'
    'lock-clocks:lock GPU clocks'
    'unlock-clocks:unlock GPU clocks'
    'reset-defaults:clear saved interactive defaults'
    'completion:print shell completion'
    'help:show help'
  )
  _arguments \
    '1:command:->command' \
    '*::arg:->args'
  case $state in
    command)
      _describe 'command' commands
      ;;
    args)
      case $words[2] in
        profile)
          _arguments '--model[model name]' '--prompts[prompt count]' '--rollouts[rollouts per prompt]' '--min-tokens[min output tokens]' '--max-tokens[max output tokens]' '--input-len[input length]' '--port[server port]' '--tp[tensor parallel size]' '--peer-servers[peer endpoints]' '--trust-remote-code[trust remote code]' '--discard-first-run[run and discard a warmup pass]' '--repeat[repeat count]' '--output[output path]' '--yes[accept saved defaults]' '--help[show help]'
          ;;
        bench)
          _arguments '--kernel[kernel name]' '--shapes[shape list]' '--dtype[data type]' '--warmup[warmup iterations]' '--n-iter[timed iterations]' '--repeats[repeat count]' '--yes[accept saved defaults]' '--help[show help]'
          ;;
        export)
          _arguments '--format[export format]:format:(csv json)' '*:path:_files'
          ;;
        report|diff)
          _files
          ;;
      esac
      ;;
  esac
}
_rlprof "$@"
)";
  }
  if (shell == "fish") {
    return R"(complete -c rlprof -f
complete -c rlprof -n '__fish_use_subcommand' -a 'profile' -d 'run GPU profiling under RL traffic'
complete -c rlprof -n '__fish_use_subcommand' -a 'report' -d 'view a saved profile'
complete -c rlprof -n '__fish_use_subcommand' -a 'bench' -d 'benchmark kernel implementations'
complete -c rlprof -n '__fish_use_subcommand' -a 'diff' -d 'compare two profiles'
complete -c rlprof -n '__fish_use_subcommand' -a 'export' -d 'export profile data'
complete -c rlprof -n '__fish_use_subcommand' -a 'traffic' -d 'run traffic generator'
complete -c rlprof -n '__fish_use_subcommand' -a 'lock-clocks' -d 'lock GPU clocks'
complete -c rlprof -n '__fish_use_subcommand' -a 'unlock-clocks' -d 'unlock GPU clocks'
complete -c rlprof -n '__fish_use_subcommand' -a 'reset-defaults' -d 'clear saved interactive defaults'
complete -c rlprof -n '__fish_use_subcommand' -a 'completion' -d 'print shell completion'
complete -c rlprof -n '__fish_use_subcommand' -a 'help' -d 'show help'
complete -c rlprof -n '__fish_seen_subcommand_from profile' -l model
complete -c rlprof -n '__fish_seen_subcommand_from profile' -l prompts
complete -c rlprof -n '__fish_seen_subcommand_from profile' -l rollouts
complete -c rlprof -n '__fish_seen_subcommand_from profile' -l min-tokens
complete -c rlprof -n '__fish_seen_subcommand_from profile' -l max-tokens
complete -c rlprof -n '__fish_seen_subcommand_from profile' -l input-len
complete -c rlprof -n '__fish_seen_subcommand_from profile' -l port
complete -c rlprof -n '__fish_seen_subcommand_from profile' -l tp
complete -c rlprof -n '__fish_seen_subcommand_from profile' -l peer-servers
complete -c rlprof -n '__fish_seen_subcommand_from profile' -l trust-remote-code
complete -c rlprof -n '__fish_seen_subcommand_from profile' -l discard-first-run
complete -c rlprof -n '__fish_seen_subcommand_from profile' -l repeat
complete -c rlprof -n '__fish_seen_subcommand_from profile' -l output
complete -c rlprof -n '__fish_seen_subcommand_from profile' -l yes
complete -c rlprof -n '__fish_seen_subcommand_from bench' -l kernel -a 'silu_and_mul fused_add_rms_norm rotary_embedding'
complete -c rlprof -n '__fish_seen_subcommand_from bench' -l shapes
complete -c rlprof -n '__fish_seen_subcommand_from bench' -l dtype -a 'bf16 fp16'
complete -c rlprof -n '__fish_seen_subcommand_from bench' -l warmup
complete -c rlprof -n '__fish_seen_subcommand_from bench' -l n-iter
complete -c rlprof -n '__fish_seen_subcommand_from bench' -l repeats
complete -c rlprof -n '__fish_seen_subcommand_from bench' -l yes
complete -c rlprof -n '__fish_seen_subcommand_from export' -l format -a 'csv json'
)";
  }
  throw std::runtime_error("unsupported shell: " + shell);
}

int handle_completion(const Args& args) {
  std::string shell = "bash";
  if (args.size() >= 2 && args[1] != "--help") {
    shell = args[1];
  }
  if (args.size() >= 2 && args[1] == "--help") {
    std::cout << "Usage: rlprof completion [bash|zsh|fish]\n"
              << "Examples:\n"
              << "  rlprof completion bash > ~/.local/share/bash-completion/completions/rlprof\n"
              << "  rlprof completion zsh > ~/.zfunc/_rlprof\n"
              << "  rlprof completion fish > ~/.config/fish/completions/rlprof.fish\n";
    return 0;
  }
  std::cout << completion_script(shell);
  return 0;
}

void print_help() {
  std::cout << "Usage: rlprof <command> [options]\n\n"
            << "Commands:\n"
            << "  profile --model MODEL [options] [--repeat N]\n"
            << "    optional: --peer-servers URL1,URL2,... --discard-first-run --yes\n"
            << "  lock-clocks [--freq MHZ]\n"
            << "  unlock-clocks\n"
            << "  reset-defaults\n"
            << "  report [path]\n"
            << "  export [path] --format csv|json\n"
            << "  diff <a.db> <b.db>\n"
            << "  bench --kernel NAME --shapes SPEC [options]\n"
            << "    optional: --yes\n"
            << "  traffic --server URL | --servers URL1,URL2,... [options]\n"
            << "  completion [bash|zsh|fish]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      rlprof::interactive::print_header("rlprof · profile your rl environment");
      const auto choice = rlprof::interactive::prompt_choice(
          "What would you like to do?",
          {
              "Profile - run GPU profiling under RL traffic",
              "Report - view a saved profile",
              "Bench - benchmark kernel implementations",
              "Diff - compare two profiles",
              "Export - export profile data",
              "Lock clocks - lock GPU clocks for reproducibility",
              "Unlock clocks",
              "Reset defaults",
          },
          0);
      if (!choice.has_value()) {
        return 0;
      }
      if (*choice == 0) {
        return interactive_profile_flow(rlprof::interactive::load_profile_defaults());
      }
      if (*choice == 1) {
        return interactive_report_flow();
      }
      if (*choice == 2) {
        return interactive_bench_flow(rlprof::interactive::load_bench_defaults());
      }
      if (*choice == 3) {
        return interactive_diff_flow();
      }
      if (*choice == 4) {
        return interactive_export_flow();
      }
      if (*choice == 5) {
        return handle_lock_clocks({"lock-clocks"});
      }
      if (*choice == 6) {
        return handle_unlock_clocks({"unlock-clocks"});
      }
      if (*choice == 7) {
        return handle_reset_defaults({"reset-defaults"});
      }
      return 0;
    }

    const Args args(argv + 1, argv + argc);
    const std::string command = args[0];

    if (command == "profile" && !has_flag(args, "--model") && !has_flag(args, "--help")) {
      const auto config = profile_interactive_defaults(args);
      if (has_flag(args, "--yes")) {
        if (config.model.empty()) {
          throw std::runtime_error(
              "--yes requires a saved profile default model or explicit --model");
        }
        return execute_profile_config(config, false, true);
      }
      return interactive_profile_flow(config);
    }

    if (command == "profile") {
      return handle_profile(args);
    }

    if (command == "report" && args.size() == 1) {
      return interactive_report_flow();
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

    if (command == "reset-defaults") {
      return handle_reset_defaults(args);
    }

    if (command == "export" && !has_positional_arg(args) && !has_flag(args, "--help")) {
      std::string default_format = "csv";
      for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--format") {
          default_format = require_value(args, i, "--format");
        }
      }
      return interactive_export_flow(default_format);
    }

    if (command == "export") {
      return handle_export(args);
    }

    if (command == "diff" && args.size() == 1) {
      return interactive_diff_flow();
    }

    if (command == "diff") {
      return handle_diff(args);
    }

    if (command == "traffic") {
      return handle_traffic(args);
    }

    if (command == "bench" && !has_flag(args, "--kernel") && !has_flag(args, "--help")) {
      const auto config = bench_interactive_defaults(args);
      if (has_flag(args, "--yes")) {
        return execute_bench_config(config, true);
      }
      return interactive_bench_flow(config);
    }

    if (command == "bench") {
      return handle_bench(args);
    }

    if (command == "--help" || command == "help") {
      print_help();
      return 0;
    }

    if (command == "completion") {
      return handle_completion(args);
    }

    throw std::runtime_error("unknown command: " + command);
  } catch (const std::exception& exc) {
    std::cerr << exc.what() << "\n";
    return 1;
  }
}
