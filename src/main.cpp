#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "hotpath/aggregate.h"
#include "hotpath/diff.h"
#include "hotpath/doctor.h"
#include "hotpath/export.h"
#include "hotpath/artifacts.h"
#include "hotpath/ops.h"
#include "hotpath/bench/registry.h"
#include "hotpath/bench/runner.h"
#include "hotpath/clock_control.h"
#include "hotpath/profiler/runner.h"
#include "hotpath/profiler/server.h"
#include "hotpath/remote.h"
#include "hotpath/report.h"
#include "hotpath/stability.h"
#include "hotpath/store.h"
#include "hotpath/targets.h"
#include "hotpath/traffic.h"
#include "hotpath/validate.h"
#include "hotpath/serve_profiler.h"
#include "hotpath/recommender.h"
#include "hotpath/disagg_model.h"
#include "interactive.h"

namespace {

bool stdout_supports_color() {
  if (std::getenv("NO_COLOR") != nullptr) return false;
  const char* term = std::getenv("TERM");
  if (term != nullptr && std::string(term) == "dumb") return false;
  return isatty(STDOUT_FILENO);
}

using Args = std::vector<std::string>;

struct ProfileCommandOptions {
  hotpath::profiler::ProfileConfig config;
  hotpath::RemoteTarget target;
  std::int64_t repeats = 1;
  bool fetch_nsys_rep = false;
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
  double batch_ms_target = 10.0;
  std::string cuda_graph_replay = "off";
  bool flush_l2 = false;
  std::string output = "auto";
  hotpath::RemoteTarget target;
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
  const fs::path dir = ".hotpath";
  if (!fs::exists(dir) || !fs::is_directory(dir)) {
    throw std::runtime_error("No profile database found in .hotpath/");
  }
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
    throw std::runtime_error("No profile database found in .hotpath/");
  }
  return latest;
}

std::string sanitize_model_name(std::string value) {
  std::replace(value.begin(), value.end(), '/', '_');
  return value;
}

std::string sanitize_label(std::string value) {
  for (char& ch : value) {
    if (!std::isalnum(static_cast<unsigned char>(ch))) {
      ch = '_';
    }
  }
  return value;
}

std::filesystem::path repeat_output_base(const hotpath::profiler::ProfileConfig& config) {
  if (!config.output.empty()) {
    return config.output;
  }
  const auto now = std::chrono::system_clock::now().time_since_epoch().count();
  return std::filesystem::path(".hotpath") /
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

std::filesystem::path append_iteration_suffix(
    const std::filesystem::path& base,
    std::int64_t iteration_index) {
  const std::string suffix = "_i" + std::to_string(iteration_index);
  if (base.extension().empty()) {
    return std::filesystem::path(base.string() + suffix);
  }
  return base.parent_path() /
         (base.stem().string() + suffix + base.extension().string());
}

hotpath::ReportMeta to_report_meta(const std::map<std::string, std::string>& meta) {
  const auto get = [&](const std::string& key, const std::string& fallback = "") {
    const auto it = meta.find(key);
    return it == meta.end() ? fallback : it->second;
  };
  const auto get_i64 = [&](const std::string& key, std::int64_t fallback) {
    const auto it = meta.find(key);
    return it == meta.end() ? fallback : std::stoll(it->second);
  };

  return hotpath::ReportMeta{
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

bool command_succeeds(const std::string& command) {
  return std::system(command.c_str()) == 0;
}

std::string trim(std::string value);
std::string run_command_capture(const std::string& command);

std::string sha256_command(const std::string& escaped_path) {
  return "(if command -v sha256sum >/dev/null 2>&1; then sha256sum " + escaped_path +
         "; elif command -v shasum >/dev/null 2>&1; then shasum -a 256 " + escaped_path +
         "; else openssl dgst -sha256 " + escaped_path +
         "; fi) | awk 'NF {print $NF}'";
}

std::string local_sha256(const std::filesystem::path& path) {
  return trim(run_command_capture(sha256_command(shell_escape(path.string()))));
}

std::string with_extension(const std::filesystem::path& path, const std::string& extension) {
  std::filesystem::path sibling = path;
  sibling.replace_extension(extension);
  return sibling.string();
}

std::filesystem::path with_extension_local(
    const std::filesystem::path& path,
    const std::string& extension) {
  std::filesystem::path sibling = path;
  sibling.replace_extension(extension);
  return sibling;
}

std::filesystem::path with_suffix_local(
    const std::filesystem::path& path,
    const std::string& suffix) {
  return path.parent_path() / (path.stem().string() + suffix);
}

std::string with_suffix_remote(
    const std::string& remote_db_path,
    const std::string& suffix) {
  const std::filesystem::path remote_path(remote_db_path);
  return (remote_path.parent_path() /
          (remote_path.stem().string() + suffix))
      .string();
}

void copy_remote_file_if_present(
    const hotpath::RemoteTarget& target,
    const std::string& remote_path,
    const std::filesystem::path& local_path,
    bool required) {
  if (!command_succeeds(hotpath::remote_file_exists_command(target, remote_path))) {
    if (required) {
      throw std::runtime_error("missing remote artifact: " + remote_path);
    }
    return;
  }
  if (!local_path.parent_path().empty()) {
    std::filesystem::create_directories(local_path.parent_path());
  }
  const std::string remote_checksum =
      trim(run_command_capture(hotpath::remote_checksum_command(target, remote_path)));
  for (int attempt = 0; attempt < 3; ++attempt) {
    const std::string copy = hotpath::remote_copy_from_command(target, remote_path, local_path);
    if (!command_succeeds(copy)) {
      continue;
    }
    if (!remote_checksum.empty()) {
      if (local_sha256(local_path) == remote_checksum) {
        return;
      }
      continue;
    }
    return;
  }
  throw std::runtime_error("failed to copy remote artifact: " + remote_path);
}

void rewrite_profile_artifact_paths(
    const std::filesystem::path& local_db_path,
    const hotpath::RemoteTarget& target,
    const std::string& remote_db_path,
    bool fetched_nsys_rep) {
  auto profile = hotpath::load_profile(local_db_path);
  profile.meta["remote_target_host"] = target.host;
  profile.meta["remote_target_workdir"] = target.workdir;
  profile.meta["artifact_db_path"] = local_db_path.string();
  profile.meta["artifact_nsys_sqlite_path"] = with_extension_local(local_db_path, ".sqlite").string();
  profile.meta["artifact_nsys_rep_path"] =
      fetched_nsys_rep ? with_extension_local(local_db_path, ".nsys-rep").string() : "";
  profile.meta["measurement_nvidia_smi_xml"] = with_suffix_local(local_db_path, "_nvidia_smi.xml").string();
  profile.meta["artifact_server_log_path"] = with_suffix_local(local_db_path, "_remote_server.log").string();
  profile.meta["remote_artifact_db_path"] = remote_db_path;
  profile.meta["remote_artifact_nsys_sqlite_path"] = with_extension(remote_db_path, ".sqlite");
  profile.meta["remote_artifact_nsys_rep_path"] = with_extension(remote_db_path, ".nsys-rep");
  profile.meta["remote_measurement_nvidia_smi_xml"] = with_suffix_remote(remote_db_path, "_nvidia_smi.xml");
  profile.meta["remote_artifact_server_log_path"] = with_suffix_remote(remote_db_path, "_server.log");
  profile.meta["warning_remote_nsys_report_not_fetched"] = fetched_nsys_rep ? "false" : "true";
  hotpath::save_profile(local_db_path, profile);
}

void fetch_remote_profile_artifacts(
    const hotpath::RemoteTarget& target,
    const std::filesystem::path& local_db_path,
    const std::string& remote_db_path,
    bool fetch_nsys_rep) {
  copy_remote_file_if_present(target, remote_db_path, local_db_path, true);
  copy_remote_file_if_present(
      target,
      with_extension(remote_db_path, ".sqlite"),
      with_extension_local(local_db_path, ".sqlite"),
      true);
  if (fetch_nsys_rep) {
    copy_remote_file_if_present(
        target,
        with_extension(remote_db_path, ".nsys-rep"),
        with_extension_local(local_db_path, ".nsys-rep"),
        true);
  }
  copy_remote_file_if_present(
      target,
      with_suffix_remote(remote_db_path, "_nvidia_smi.xml"),
      with_suffix_local(local_db_path, "_nvidia_smi.xml"),
      false);
  copy_remote_file_if_present(
      target,
      with_suffix_remote(remote_db_path, "_server.log"),
      with_suffix_local(local_db_path, "_remote_server.log"),
      false);
  rewrite_profile_artifact_paths(local_db_path, target, remote_db_path, fetch_nsys_rep);
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

void notify_progress(
    const hotpath::profiler::ProgressCallback& progress,
    const std::string& status) {
  if (progress) {
    progress(status);
  }
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

hotpath::RemoteTarget resolve_cli_target(const hotpath::RemoteTarget& target) {
  if (!hotpath::has_remote_target(target)) {
    return target;
  }
  const std::string override =
      target.workdir == hotpath::RemoteTarget{}.workdir ? "" : target.workdir;
  hotpath::RemoteTarget resolved;
  try {
    resolved = hotpath::resolve_target(target.host, override);
  } catch (const std::runtime_error&) {
    if (override.empty() && target.python_executable.empty() &&
        target.vllm_executable.empty()) {
      throw;
    }
    resolved.host = target.host;
    resolved.workdir =
        override.empty() ? hotpath::RemoteTarget{}.workdir : override;
  }
  if (!target.python_executable.empty()) {
    resolved.python_executable = target.python_executable;
  }
  if (!target.vllm_executable.empty()) {
    resolved.vllm_executable = target.vllm_executable;
  }
  return resolved;
}

std::string run_command_capture(const std::string& command) {
  std::array<char, 4096> buffer{};
  std::string output;
  FILE* pipe = popen((command + " 2>&1").c_str(), "r");
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

std::string try_run_command_capture(const std::string& command) {
  std::array<char, 4096> buffer{};
  std::string output;
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    return "";
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output.append(buffer.data());
  }
  pclose(pipe);
  return trim(output);
}

std::string remote_bootstrap_check_script() {
  return R"BOOT(
print_row() {
  printf "%-18s  %-6s  %s\n" "$1" "$2" "$3"
}

tool_python() {
  _py_exec="${HOTPATH_PYTHON_EXECUTABLE:-${RLPROF_PYTHON_EXECUTABLE:-}}"
  if [ -n "$_py_exec" ]; then
    if [ -x "$_py_exec" ]; then
      print_row python PASS "$_py_exec ($("$_py_exec" --version 2>&1 | head -n 1))"
    else
      print_row python FAIL "missing configured executable: $_py_exec"
      FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    return
  fi
  py3=$(command -v python3 2>/dev/null || true)
  py=$(command -v python 2>/dev/null || true)
  chosen="$py3"
  if [ -z "$chosen" ]; then
    chosen="$py"
  fi
  if [ -n "$chosen" ]; then
    print_row python PASS "$chosen ($("$chosen" --version 2>&1 | head -n 1))"
  else
    print_row python FAIL "not found"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
}

tool_vllm() {
  _vllm_exec="${HOTPATH_VLLM_EXECUTABLE:-${RLPROF_VLLM_EXECUTABLE:-}}"
  if [ -n "$_vllm_exec" ]; then
    if [ -x "$_vllm_exec" ]; then
      print_row vllm PASS "$_vllm_exec ($("$_vllm_exec" --version 2>&1 | head -n 1))"
    else
      print_row vllm FAIL "missing configured executable: $_vllm_exec"
      FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    return
  fi
  chosen=$(command -v vllm 2>/dev/null || true)
  if [ -n "$chosen" ]; then
    print_row vllm PASS "$chosen ($("$chosen" --version 2>&1 | head -n 1))"
  else
    print_row vllm FAIL "not found"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
}

tool_simple() {
  label="$1"
  version_arg="$2"
  chosen=$(command -v "$label" 2>/dev/null || true)
  if [ -n "$chosen" ]; then
    if [ -n "$version_arg" ]; then
      print_row "$label" PASS "$chosen ($("$chosen" "$version_arg" 2>&1 | head -n 1))"
    else
      print_row "$label" PASS "$chosen"
    fi
  else
    print_row "$label" FAIL "not found"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
}

tool_cuda() {
  gpu_line=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)
  gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l | tr -d " " || true)
  if [ -n "$gpu_line" ] && [ "${gpu_count:-0}" -gt 0 ]; then
    print_row "cuda visibility" PASS "$gpu_line | visible GPUs=$gpu_count"
  else
    print_row "cuda visibility" FAIL "no visible NVIDIA GPUs"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
}

FAIL_COUNT=0
printf "BOOTSTRAP\n\n"
printf "%-18s  %-6s  %s\n" "check" "status" "detail"
printf "%s\n" "------------------------------------------------------------------------"
tool_simple cmake --version
tool_python
tool_vllm
tool_simple nsys --version
tool_cuda
exit "$FAIL_COUNT"
)BOOT";
}

std::int64_t current_unix_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string optional_json(const std::optional<double>& value) {
  return value.has_value() ? std::to_string(*value) : "null";
}

std::string render_traffic_json(const hotpath::TrafficStats& stats) {
  return "{"
         "\"total_requests\":" + std::to_string(stats.total_requests) +
         ",\"completion_length_samples\":" + std::to_string(stats.completion_length_samples) +
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
    } else if (args[i] == "--server") {
      options.config.managed_server_name = require_value(args, i, "--server");
    } else if (args[i] == "--attach") {
      options.config.attach_server = require_value(args, i, "--attach");
    } else if (args[i] == "--attach-pid") {
      options.config.attach_pid = std::stoll(require_value(args, i, "--attach-pid"));
    } else if (args[i] == "--target") {
      options.target.host = require_value(args, i, "--target");
    } else if (args[i] == "--target-workdir") {
      options.target.workdir = require_value(args, i, "--target-workdir");
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
    } else if (args[i] == "--fetch-nsys-rep") {
      options.fetch_nsys_rep = true;
    } else if (args[i] == "--start-at-ms") {
      options.config.start_at_unix_ms = std::stoll(require_value(args, i, "--start-at-ms"));
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

void append_profile_invocation_args(
    Args& args,
    const ProfileCommandOptions& options,
    bool include_repeat,
    const std::string& output_override = "") {
  if (!options.config.managed_server_name.empty()) {
    args.insert(args.end(), {"--server", options.config.managed_server_name});
  } else if (!options.config.attach_server.empty()) {
    args.insert(args.end(), {"--attach", options.config.attach_server});
  } else {
    args.insert(args.end(), {"--model", options.config.model});
  }
  if (options.config.attach_pid > 0) {
    args.insert(
        args.end(),
        {"--attach-pid", std::to_string(options.config.attach_pid)});
  }
  args.insert(args.end(), {"--prompts", std::to_string(options.config.prompts)});
  args.insert(args.end(), {"--rollouts", std::to_string(options.config.rollouts)});
  args.insert(
      args.end(), {"--min-tokens", std::to_string(options.config.min_tokens)});
  args.insert(
      args.end(), {"--max-tokens", std::to_string(options.config.max_tokens)});
  args.insert(
      args.end(), {"--input-len", std::to_string(options.config.input_len)});
  args.insert(args.end(), {"--port", std::to_string(options.config.port)});
  args.insert(args.end(), {"--tp", std::to_string(options.config.tp)});
  if (!options.config.peer_servers.empty()) {
    std::ostringstream peers;
    for (std::size_t i = 0; i < options.config.peer_servers.size(); ++i) {
      if (i > 0) {
        peers << ",";
      }
      peers << options.config.peer_servers[i];
    }
    args.insert(args.end(), {"--peer-servers", peers.str()});
  }
  if (options.config.trust_remote_code) {
    args.push_back("--trust-remote-code");
  }
  if (options.config.discard_first_run) {
    args.push_back("--discard-first-run");
  }
  if (include_repeat) {
    args.insert(args.end(), {"--repeat", std::to_string(options.repeats)});
  }
  if (options.config.start_at_unix_ms > 0) {
    args.insert(
        args.end(),
        {"--start-at-ms", std::to_string(options.config.start_at_unix_ms)});
  }
  args.insert(
      args.end(),
      {"--output",
       output_override.empty() ? options.config.output.string() : output_override});
}

std::pair<std::vector<std::filesystem::path>, std::vector<hotpath::ProfileData>>
fetch_remote_profile_runs(
    const hotpath::RemoteTarget& target,
    const std::filesystem::path& local_output_base,
    const std::string& remote_output_base,
    std::int64_t repeats,
    bool remote_iteration_suffix,
    bool fetch_nsys_rep,
    const hotpath::profiler::ProgressCallback& progress) {
  std::vector<std::future<hotpath::ProfileData>> fetch_futures;
  fetch_futures.reserve(static_cast<std::size_t>(repeats));
  for (std::int64_t run_index = 1; run_index <= repeats; ++run_index) {
    const std::filesystem::path local_db_path =
        repeats == 1
            ? with_extension_local(local_output_base, ".db")
            : with_extension_local(append_repeat_suffix(local_output_base, run_index), ".db");
    const std::string remote_db_path =
        repeats == 1
            ? with_extension(remote_output_base, ".db")
            : with_extension(
                  remote_iteration_suffix
                      ? append_iteration_suffix(remote_output_base, run_index)
                      : append_repeat_suffix(remote_output_base, run_index),
                  ".db");
    fetch_futures.push_back(std::async(
        std::launch::async,
        [target, local_db_path, remote_db_path, fetch_nsys_rep]() {
          fetch_remote_profile_artifacts(
              target, local_db_path, remote_db_path, fetch_nsys_rep);
          return hotpath::load_profile(local_db_path);
        }));
  }

  std::vector<std::filesystem::path> db_paths;
  std::vector<hotpath::ProfileData> profiles;
  db_paths.reserve(static_cast<std::size_t>(repeats));
  profiles.reserve(static_cast<std::size_t>(repeats));
  for (std::int64_t run_index = 1; run_index <= repeats; ++run_index) {
    const std::filesystem::path local_db_path =
        repeats == 1
            ? with_extension_local(local_output_base, ".db")
            : with_extension_local(append_repeat_suffix(local_output_base, run_index), ".db");
    notify_progress(
        progress, "Fetching remote artifacts for run " + std::to_string(run_index) + "...");
    profiles.push_back(fetch_futures[static_cast<std::size_t>(run_index - 1)].get());
    db_paths.push_back(local_db_path);
  }
  return {db_paths, profiles};
}

std::string run_profile_command(
    const ProfileCommandOptions& options,
    const hotpath::profiler::ProgressCallback& progress = {}) {
  auto effective_options = options;
  effective_options.target = resolve_cli_target(effective_options.target);
  if (effective_options.config.model.empty()) {
    if (effective_options.config.attach_server.empty() &&
        effective_options.config.managed_server_name.empty()) {
      throw std::runtime_error("--model is required");
    }
    if (!effective_options.config.managed_server_name.empty()) {
      effective_options.config.model =
          hotpath::profiler::load_managed_server(
              effective_options.config.managed_server_name)
              .model;
    } else {
      effective_options.config.model = "attached-server";
    }
  }

  if (effective_options.repeats <= 0) {
    throw std::runtime_error("--repeat must be > 0");
  }
  if (!effective_options.config.managed_server_name.empty() &&
      hotpath::has_remote_target(effective_options.target)) {
    throw std::runtime_error("--server is currently local-only and cannot be combined with --target");
  }
  if (hotpath::has_remote_target(effective_options.target)) {
    if (!effective_options.config.attach_server.empty() &&
        effective_options.config.attach_pid <= 0) {
      auto attach_result = hotpath::profiler::run_profile(effective_options.config, progress);
      auto attach_profile = hotpath::load_profile(attach_result.db_path);
      attach_profile.meta["remote_target_host"] = effective_options.target.host;
      attach_profile.meta["remote_target_workdir"] = effective_options.target.workdir;
      attach_profile.meta["warning_remote_attach_metrics_only"] = "true";
      hotpath::save_profile(attach_result.db_path, attach_profile);
      std::ostringstream output;
      output << attach_result.db_path.string() << "\n";
      return output.str();
    }

    const std::filesystem::path local_output_base = repeat_output_base(effective_options.config);
    const std::string remote_output_base =
        hotpath::remote_join(effective_options.target, local_output_base);

    const bool use_fast_remote_repeat =
        effective_options.repeats > 1 &&
        (effective_options.config.attach_server.empty() ||
         effective_options.config.attach_pid > 0) &&
        effective_options.config.start_at_unix_ms == 0;
    Args remote_args = {use_fast_remote_repeat ? "soak-profile" : "profile"};
    append_profile_invocation_args(
        remote_args,
        effective_options,
        !use_fast_remote_repeat,
        remote_output_base);
    if (use_fast_remote_repeat) {
      remote_args.insert(
          remote_args.end(),
          {"--iterations", std::to_string(effective_options.repeats)});
    }

    notify_progress(progress, "Running remote profile on " + effective_options.target.host + "...");
    try {
      run_command_capture(hotpath::remote_cli_command(effective_options.target, remote_args));
    } catch (const std::exception& exc) {
      const std::string tail = try_run_command_capture(
          hotpath::remote_tail_command(
              effective_options.target,
              with_suffix_remote(with_extension(remote_output_base, ".db"), "_server.log"),
              120));
      if (tail.empty()) {
        throw;
      }
      throw std::runtime_error(
          std::string(exc.what()) + "\n\nremote vLLM log tail:\n" + tail);
    }

    std::ostringstream output;
    const auto [db_paths, profiles] = fetch_remote_profile_runs(
        effective_options.target,
        local_output_base,
        remote_output_base,
        effective_options.repeats,
        use_fast_remote_repeat,
        effective_options.fetch_nsys_rep,
        progress);
    for (const auto& db_path : db_paths) {
      output << db_path.string() << "\n";
    }

    const auto& final_profile = profiles.back();
    const bool use_color = stdout_supports_color();
    output << "\n";
    output << hotpath::render_report(
        to_report_meta(final_profile.meta),
        final_profile.meta,
        final_profile.kernels,
        final_profile.metrics_summary,
        final_profile.traffic_stats,
        use_color);
    if (effective_options.repeats > 1) {
      output << "\n";
      output << hotpath::render_stability_report(
          hotpath::compute_stability_report(profiles), use_color);
    }
    return output.str();
  }

  std::ostringstream output;
  if (effective_options.repeats == 1) {
    const auto result = hotpath::profiler::run_profile(effective_options.config, progress);
    output << result.db_path << "\n";
    return output.str();
  }

  if (!effective_options.config.attach_server.empty() &&
      effective_options.config.attach_pid <= 0) {
    throw std::runtime_error(
        "profile --repeat with attach mode still uses single-run profiling only");
  }
  const auto results = hotpath::profiler::run_soak_profile(
      effective_options.config, effective_options.repeats, 0, progress);
  std::vector<hotpath::ProfileData> profiles;
  profiles.reserve(results.size());
  for (const auto& result : results) {
    output << result.db_path << "\n";
    profiles.push_back(hotpath::load_profile(result.db_path));
  }

  const auto& final_profile = profiles.back();
  const bool use_color = stdout_supports_color();
  output << "\n";
  output << hotpath::render_report(
      to_report_meta(final_profile.meta),
      final_profile.meta,
      final_profile.kernels,
      final_profile.metrics_summary,
      final_profile.traffic_stats,
      use_color);
  output << "\n";
  output << hotpath::render_stability_report(
      hotpath::compute_stability_report(profiles), use_color);
  return output.str();
}

int handle_profile(const Args& args) {
  auto options = parse_profile_args(args);
  options.target = resolve_cli_target(options.target);
  if (options.show_help) {
    std::cout << "Usage: hotpath profile (--model MODEL | --server NAME | --attach URL) [options] [--repeat N]\n"
              << "       optional: --server NAME --attach URL --attach-pid PID --peer-servers URL1,URL2,...\n"
              << "                 --target HOST --target-workdir DIR\n"
              << "       flags: --discard-first-run --fetch-nsys-rep --yes\n";
    return 0;
  }
  std::cout << run_profile_command(options);
  return 0;
}

int handle_report(const Args& args) {
  const std::filesystem::path path =
      args.size() >= 2 ? std::filesystem::path(args[1]) : latest_profile_path();
  const auto profile = hotpath::load_profile(path);
  std::cout << hotpath::render_report(
      to_report_meta(profile.meta),
      profile.meta,
      profile.kernels,
      profile.metrics_summary,
      profile.traffic_stats,
      stdout_supports_color());
  return 0;
}

int handle_aggregate(const Args& args) {
  if (args.size() < 3 || (args.size() > 1 && args[1] == "--help")) {
    std::cout << "Usage: hotpath aggregate A.db B.db [more.db ...] --output OUT.db\n";
    return 0;
  }
  std::vector<std::filesystem::path> inputs;
  std::filesystem::path output;
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--output") {
      output = require_value(args, i, "--output");
    } else if (!args[i].starts_with("--")) {
      inputs.push_back(args[i]);
    }
  }
  if (inputs.size() < 2 || output.empty()) {
    throw std::runtime_error("aggregate requires at least two input dbs and --output");
  }
  auto aggregate = hotpath::aggregate_profiles(inputs);
  aggregate.meta["artifact_db_path"] = output.string();
  hotpath::save_profile(output, aggregate);
  std::cout << output.string() << "\n";
  return 0;
}

int handle_export(const Args& args) {
  if (args.size() > 1 && args[1] == "--help") {
    std::cout << "Usage: hotpath export [path] --format csv|json\n";
    return 0;
  }
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

  for (const auto& output : hotpath::export_profile(path, format)) {
    std::cout << output.string() << "\n";
  }
  return 0;
}

int handle_validate(const Args& args) {
  if (args.size() > 1 && args[1] == "--help") {
    std::cout << "Usage: hotpath validate [path]\n";
    return 0;
  }
  const std::filesystem::path path =
      args.size() >= 2 ? std::filesystem::path(args[1]) : latest_profile_path();
  const auto checks = hotpath::validate_profile(path);
  std::cout << hotpath::render_validation_report(path, checks, stdout_supports_color());
  return 0;
}

int handle_artifacts(const Args& args) {
  if (args.size() > 1 && args[1] == "--help") {
    std::cout << "Usage: hotpath artifacts [path]\n";
    return 0;
  }
  const std::filesystem::path path =
      args.size() >= 2 ? std::filesystem::path(args[1]) : latest_profile_path();
  std::cout << hotpath::render_artifacts(path, hotpath::profile_artifacts(path));
  return 0;
}

int handle_trace(const Args& args) {
  if (args.size() > 1 && args[1] == "--help") {
    std::cout << "Usage: hotpath trace [path] [--open]\n";
    return 0;
  }
  std::filesystem::path path;
  bool open = false;
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--open") {
      open = true;
    } else if (!args[i].starts_with("--")) {
      path = args[i];
    }
  }
  if (path.empty()) {
    path = latest_profile_path();
  }

  const auto profile = hotpath::load_profile(path);
  const bool metrics_only =
      profile.meta.contains("warning_no_kernel_trace") &&
      profile.meta.at("warning_no_kernel_trace") == "true";
  const auto artifacts = hotpath::trace_artifacts(path);
  if (open) {
    for (const auto& artifact : artifacts) {
      if (artifact.kind == "nsys report" && artifact.exists) {
        const int rc = std::system(("nsys-ui " + shell_escape(artifact.path.string()) +
                                    " > /dev/null 2>&1 &")
                                       .c_str());
        if (rc == 0) {
          std::cout << "Opened: " << artifact.path << "\n";
          return 0;
        }
        throw std::runtime_error("failed to open nsys-ui for " + artifact.path.string());
      }
    }
    if (profile.meta.contains("remote_artifact_nsys_rep_path") &&
        !profile.meta.at("remote_artifact_nsys_rep_path").empty()) {
      throw std::runtime_error(
          "no local nsys report artifact available; rerun with --fetch-nsys-rep or use recover "
          "to copy " + profile.meta.at("remote_artifact_nsys_rep_path"));
    }
    throw std::runtime_error("no nsys report artifact available to open");
  }

  std::cout << hotpath::render_trace_artifacts(path, artifacts, metrics_only);
  return 0;
}

int handle_manifest(const Args& args) {
  if (args.size() > 1 && args[1] == "--help") {
    std::cout << "Usage: hotpath manifest [path] [--output PATH|auto]\n";
    return 0;
  }
  std::filesystem::path path;
  std::filesystem::path output;
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--output") {
      const auto value = require_value(args, i, "--output");
      if (value != "auto") {
        output = value;
      }
    } else if (!args[i].starts_with("--")) {
      path = args[i];
    }
  }
  if (path.empty()) {
    path = latest_profile_path();
  }
  std::cout << hotpath::write_manifest(path, output).string() << "\n";
  return 0;
}

int handle_cleanup(const Args& args) {
  std::filesystem::path dir = ".hotpath";
  int keep = 10;
  bool compress = false;
  bool apply = false;
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--dir") {
      dir = require_value(args, i, "--dir");
    } else if (args[i] == "--keep") {
      keep = std::stoi(require_value(args, i, "--keep"));
    } else if (args[i] == "--compress") {
      compress = true;
    } else if (args[i] == "--apply") {
      apply = true;
    } else if (args[i] == "--help") {
      std::cout << "Usage: hotpath cleanup [--dir DIR] [--keep N] [--compress] [--apply]\n";
      return 0;
    }
  }
  std::cout << hotpath::cleanup_artifacts(dir, keep, compress, apply);
  return 0;
}

int handle_cluster_profile(const Args& args) {
  if (args.size() > 1 && args[1] == "--help") {
    std::cout << "Usage: hotpath cluster-profile --targets T1,T2 [profile options] --output BASE [--allow-duplicate-hosts]\n";
    return 0;
  }
  std::string targets_csv;
  bool allow_duplicate_hosts = false;
  Args profile_args = {"profile"};
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--targets") {
      targets_csv = require_value(args, i, "--targets");
    } else if (args[i] == "--allow-duplicate-hosts") {
      allow_duplicate_hosts = true;
    } else {
      profile_args.push_back(args[i]);
    }
  }
  if (targets_csv.empty()) {
    throw std::runtime_error("cluster-profile requires --targets");
  }
  auto options = parse_profile_args(profile_args);
  if (options.repeats != 1) {
    throw std::runtime_error("cluster-profile requires --repeat 1");
  }
  if (options.config.output.empty()) {
    throw std::runtime_error("cluster-profile requires --output BASE");
  }
  const auto targets = split_csv_list(targets_csv);
  if (targets.size() < 2) {
    throw std::runtime_error("cluster-profile requires at least two targets");
  }

  std::vector<ProfileCommandOptions> run_options_list;
  std::vector<std::int64_t> target_now_ms;
  std::map<std::string, int> host_occurrences;
  std::set<std::string> distinct_hosts;
  run_options_list.reserve(targets.size());
  target_now_ms.reserve(targets.size());
  for (const auto& target_spec : targets) {
    auto run_options = options;
    run_options.target.host = target_spec;
    run_options.target = resolve_cli_target(run_options.target);
    const int host_index = host_occurrences[run_options.target.host]++;
    distinct_hosts.insert(run_options.target.host);
    run_options.config.port = options.config.port + host_index;
    run_options.config.output =
        options.config.output.string() + "_" + sanitize_label(target_spec);
    target_now_ms.push_back(std::stoll(trim(
        run_command_capture(hotpath::remote_epoch_ms_command(run_options.target)))));
    run_options_list.push_back(run_options);
  }
  if (!allow_duplicate_hosts && distinct_hosts.size() != targets.size()) {
    throw std::runtime_error(
        "cluster-profile requires distinct target hosts. Use --allow-duplicate-hosts only for loopback testing.");
  }

  const auto [min_it, max_it] =
      std::minmax_element(target_now_ms.begin(), target_now_ms.end());
  const std::int64_t start_at_ms =
      std::max(current_unix_ms(), *max_it) + 15000;

  std::vector<std::future<std::string>> futures;
  std::vector<std::filesystem::path> outputs;
  futures.reserve(run_options_list.size());
  outputs.reserve(run_options_list.size());
  for (auto& run_options : run_options_list) {
    run_options.config.start_at_unix_ms = start_at_ms;
    std::filesystem::path db_path = run_options.config.output;
    db_path.replace_extension(".db");
    outputs.push_back(db_path);
    futures.push_back(std::async(std::launch::async, [run_options]() {
      return run_profile_command(run_options);
    }));
  }

  for (auto& future : futures) {
    std::cout << future.get();
  }

  auto aggregate = hotpath::aggregate_profiles(outputs);
  aggregate.meta["aggregate_profile_count"] = std::to_string(outputs.size());
  aggregate.meta["aggregation_scope"] = "cluster_controller";
  aggregate.meta["cluster_synchronized_start"] = "true";
  aggregate.meta["cluster_start_at_unix_ms"] = std::to_string(start_at_ms);
  aggregate.meta["cluster_clock_skew_ms"] = std::to_string(*max_it - *min_it);
  aggregate.meta["cluster_distinct_host_count"] = std::to_string(distinct_hosts.size());
  aggregate.meta["cluster_duplicate_hosts_allowed"] = allow_duplicate_hosts ? "true" : "false";
  const std::filesystem::path aggregate_path =
      options.config.output.string() + "_aggregate.db";
  aggregate.meta["artifact_db_path"] = aggregate_path.string();
  hotpath::save_profile(aggregate_path, aggregate);
  std::cout << aggregate_path.string() << "\n";
  return 0;
}

int handle_soak_profile(const Args& args) {
  if (args.size() > 1 && args[1] == "--help") {
    std::cout << "Usage: hotpath soak-profile [profile options] --iterations N --output BASE [--pause-sec N] [--validate-each]\n";
    return 0;
  }
  int iterations = 5;
  int pause_sec = 0;
  bool validate_each = false;
  Args profile_args = {"profile"};
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--iterations") {
      iterations = std::stoi(require_value(args, i, "--iterations"));
    } else if (args[i] == "--pause-sec") {
      pause_sec = std::stoi(require_value(args, i, "--pause-sec"));
    } else if (args[i] == "--validate-each") {
      validate_each = true;
    } else {
      profile_args.push_back(args[i]);
    }
  }
  auto options = parse_profile_args(profile_args);
  if (iterations <= 0) {
    throw std::runtime_error("--iterations must be > 0");
  }
  if (options.repeats != 1) {
    throw std::runtime_error("soak-profile requires --repeat 1");
  }
  if (options.config.output.empty()) {
    throw std::runtime_error("soak-profile requires --output BASE");
  }

  options.target = resolve_cli_target(options.target);
  if (hotpath::has_remote_target(options.target)) {
    const std::filesystem::path local_output_base = repeat_output_base(options.config);
    const std::string remote_output_base =
        hotpath::remote_join(options.target, local_output_base);
    Args remote_args = {"soak-profile"};
    append_profile_invocation_args(remote_args, options, false, remote_output_base);
    remote_args.insert(remote_args.end(), {"--iterations", std::to_string(iterations)});
    if (pause_sec > 0) {
      remote_args.insert(remote_args.end(), {"--pause-sec", std::to_string(pause_sec)});
    }
    if (validate_each) {
      remote_args.push_back("--validate-each");
    }
    notify_progress(
        {},
        "Running remote soak profile on " + options.target.host + "...");
    try {
      run_command_capture(hotpath::remote_cli_command(options.target, remote_args));
    } catch (const std::exception& exc) {
      const std::string tail = try_run_command_capture(
          hotpath::remote_tail_command(
              options.target,
              with_suffix_remote(with_extension(remote_output_base, ".db"), "_server.log"),
              120));
      if (tail.empty()) {
        throw;
      }
      throw std::runtime_error(
          std::string(exc.what()) + "\n\nremote vLLM log tail:\n" + tail);
    }
    const auto [db_paths, _profiles] = fetch_remote_profile_runs(
        options.target,
        local_output_base,
        remote_output_base,
        iterations,
        true,
        options.fetch_nsys_rep,
        {});
    for (const auto& db_path : db_paths) {
      std::cout << db_path.string() << "\n";
      if (validate_each) {
        const auto checks = hotpath::validate_profile(db_path);
        for (const auto& check : checks) {
          if (check.status == hotpath::ValidationStatus::kFail) {
            throw std::runtime_error(
                "soak validation failed for " + db_path.string() + ": " + check.name);
          }
        }
      }
    }
    return 0;
  }

  if (!options.config.attach_server.empty() && options.config.attach_pid <= 0) {
    for (int i = 1; i <= iterations; ++i) {
      auto run_options = options;
      run_options.config.output =
          options.config.output.string() + "_i" + std::to_string(i);
      std::cout << run_profile_command(run_options);
      std::filesystem::path db_path = run_options.config.output;
      db_path.replace_extension(".db");
      if (validate_each) {
        const auto checks = hotpath::validate_profile(db_path);
        for (const auto& check : checks) {
          if (check.status == hotpath::ValidationStatus::kFail) {
            throw std::runtime_error(
                "soak validation failed for " + db_path.string() + ": " + check.name);
          }
        }
      }
      if (pause_sec > 0 && i < iterations) {
        std::this_thread::sleep_for(std::chrono::seconds(pause_sec));
      }
    }
    return 0;
  }

  const auto results =
      hotpath::profiler::run_soak_profile(options.config, iterations, pause_sec, {});
  for (const auto& result : results) {
    std::cout << result.db_path.string() << "\n";
    if (validate_each) {
      const auto checks = hotpath::validate_profile(result.db_path);
      for (const auto& check : checks) {
        if (check.status == hotpath::ValidationStatus::kFail) {
          throw std::runtime_error(
              "soak validation failed for " + result.db_path.string() + ": " + check.name);
        }
      }
    }
  }
  return 0;
}

int handle_diff(const Args& args) {
  if (args.size() < 3) {
    throw std::runtime_error("diff requires two database paths");
  }
  std::cout << hotpath::render_diff(args[1], args[2], stdout_supports_color());
  return 0;
}

int handle_doctor(const Args& args) {
  hotpath::RemoteTarget target;
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--target") {
      target.host = require_value(args, i, "--target");
    } else if (args[i] == "--target-workdir") {
      target.workdir = require_value(args, i, "--target-workdir");
    } else if (args[i] == "--help") {
      std::cout << "Usage: hotpath doctor [--target HOST] [--target-workdir DIR]\n";
      return 0;
    }
  }
  target = resolve_cli_target(target);
  if (hotpath::has_remote_target(target)) {
    std::cout << run_command_capture(
        hotpath::remote_cli_command(target, {"doctor"}));
    return 0;
  }
  std::cout << hotpath::render_doctor_report(hotpath::run_doctor(), stdout_supports_color());
  return 0;
}

int handle_server(const Args& args) {
  if (args.size() == 1 || (args.size() > 1 && args[1] == "--help")) {
    std::cout << "Usage: hotpath server <start|stop|list|show|prune> ...\n";
    return 0;
  }

  const std::string action = args[1];
  if (action == "list") {
    std::cout << hotpath::profiler::render_managed_servers(
        hotpath::profiler::list_managed_servers());
    return 0;
  }

  if (action == "show") {
    if (args.size() < 3) {
      throw std::runtime_error("server show requires NAME");
    }
    const auto state = hotpath::profiler::find_managed_server(args[2]);
    if (!state.has_value()) {
      throw std::runtime_error("unknown managed server: " + args[2]);
    }
    std::cout << "name: " << state->name << "\n";
    std::cout << "model: " << state->model << "\n";
    std::cout << "url: " << state->server_url << "\n";
    std::cout << "port: " << state->port << "\n";
    std::cout << "tp: " << state->tp << "\n";
    std::cout << "max_model_len: " << state->max_model_len << "\n";
    std::cout << "session: " << state->session_name << "\n";
    std::cout << "pid: " << state->pid << "\n";
    std::cout << "ready: " << (hotpath::profiler::managed_server_ready(*state) ? "yes" : "no") << "\n";
    std::cout << "log: " << state->log_path.string() << "\n";
    return 0;
  }

  if (action == "start") {
    hotpath::profiler::ManagedServerConfig config;
    for (std::size_t i = 2; i < args.size(); ++i) {
      if (args[i] == "--name") {
        config.name = require_value(args, i, "--name");
      } else if (args[i] == "--model") {
        config.model = require_value(args, i, "--model");
      } else if (args[i] == "--port") {
        config.port = std::stoll(require_value(args, i, "--port"));
      } else if (args[i] == "--tp") {
        config.tp = std::stoll(require_value(args, i, "--tp"));
      } else if (args[i] == "--max-model-len") {
        config.max_model_len = std::stoll(require_value(args, i, "--max-model-len"));
      } else if (args[i] == "--trust-remote-code") {
        config.trust_remote_code = true;
      } else if (args[i] == "--startup-timeout-s") {
        config.startup_timeout_s = std::stoll(require_value(args, i, "--startup-timeout-s"));
      }
    }
    const auto state = hotpath::profiler::start_managed_server(config);
    std::cout << state.name << "\n";
    std::cout << state.server_url << "\n";
    return 0;
  }

  if (action == "stop") {
    if (args.size() < 3) {
      throw std::runtime_error("server stop requires NAME");
    }
    const auto state = hotpath::profiler::find_managed_server(args[2]);
    if (!state.has_value()) {
      throw std::runtime_error("unknown managed server: " + args[2]);
    }
    hotpath::profiler::stop_managed_server(*state);
    std::cout << "Stopped " << state->name << "\n";
    return 0;
  }

  if (action == "prune") {
    const auto removed = hotpath::profiler::prune_stale_managed_servers();
    std::cout << "Pruned " << removed << " stale managed server"
              << (removed == 1 ? "" : "s") << "\n";
    return 0;
  }

  throw std::runtime_error("unknown server action: " + action);
}

int handle_target(const Args& args) {
  if (args.size() == 1 || (args.size() > 1 && args[1] == "--help")) {
    std::cout << "Usage: hotpath target <add|list|remove|show|bootstrap> ...\n";
    return 0;
  }

  const std::string action = args[1];
  if (action == "list") {
    std::cout << hotpath::render_targets(hotpath::list_targets());
    return 0;
  }

  if (action == "add") {
    if (args.size() < 3) {
      throw std::runtime_error("target add requires NAME");
    }
    hotpath::SavedTarget target{
        .name = args[2],
    };
    for (std::size_t i = 3; i < args.size(); ++i) {
      if (args[i] == "--host") {
        target.host = require_value(args, i, "--host");
      } else if (args[i] == "--workdir") {
        target.workdir = require_value(args, i, "--workdir");
      } else if (args[i] == "--python") {
        target.python_executable = require_value(args, i, "--python");
      } else if (args[i] == "--vllm") {
        target.vllm_executable = require_value(args, i, "--vllm");
      }
    }
    if (target.host.empty()) {
      throw std::runtime_error("target add requires --host");
    }
    hotpath::save_target(target);
    std::cout << "Saved target " << target.name << " -> " << target.host << "\n";
    return 0;
  }

  if (action == "remove") {
    if (args.size() < 3) {
      throw std::runtime_error("target remove requires NAME");
    }
    if (!hotpath::remove_target(args[2])) {
      throw std::runtime_error("unknown target: " + args[2]);
    }
    std::cout << "Removed target " << args[2] << "\n";
    return 0;
  }

  if (action == "show") {
    if (args.size() < 3) {
      throw std::runtime_error("target show requires NAME");
    }
    const auto target = hotpath::resolve_target(args[2]);
    std::cout << "name: " << args[2] << "\n";
    std::cout << "host: " << target.host << "\n";
    std::cout << "workdir: " << target.workdir << "\n";
    if (!target.python_executable.empty()) {
      std::cout << "python: " << target.python_executable << "\n";
    }
    if (!target.vllm_executable.empty()) {
      std::cout << "vllm: " << target.vllm_executable << "\n";
    }
    return 0;
  }

  if (action == "bootstrap") {
    if (args.size() < 3) {
      throw std::runtime_error("target bootstrap requires NAME or HOST");
    }
    std::string workdir;
    std::string python_executable;
    std::string vllm_executable;
    for (std::size_t i = 3; i < args.size(); ++i) {
      if (args[i] == "--workdir") {
        workdir = require_value(args, i, "--workdir");
      } else if (args[i] == "--python") {
        python_executable = require_value(args, i, "--python");
      } else if (args[i] == "--vllm") {
        vllm_executable = require_value(args, i, "--vllm");
      }
    }
    hotpath::RemoteTarget target{
        .host = args[2],
        .workdir = workdir.empty() ? hotpath::RemoteTarget{}.workdir : workdir,
        .python_executable = python_executable,
        .vllm_executable = vllm_executable,
    };
    target = resolve_cli_target(target);
    try {
      std::cout << run_command_capture(
          hotpath::remote_shell_command(target, remote_bootstrap_check_script()));
    } catch (const std::exception& exc) {
      throw std::runtime_error(
          "remote bootstrap preflight failed for " + target.host + "\n\n" + exc.what());
    }
    const auto command =
        hotpath::bootstrap_target_command(target, std::filesystem::current_path().string());
    const int rc = std::system(command.c_str());
    if (rc != 0) {
      throw std::runtime_error("remote bootstrap failed for " + target.host);
    }
    const auto doctor_output = run_command_capture(
        hotpath::remote_cli_command(target, {"doctor"}));
    std::cout << doctor_output;
    if (doctor_output.find("FAIL") != std::string::npos) {
      throw std::runtime_error("remote bootstrap validation failed for " + target.host);
    }
    std::cout << "Bootstrapped " << target.host << " at " << target.workdir << "\n";
    return 0;
  }

  throw std::runtime_error("unknown target action: " + action);
}

int handle_recover(const Args& args) {
  hotpath::RemoteTarget target;
  std::string remote_db_path;
  std::filesystem::path local_output;
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--target") {
      target.host = require_value(args, i, "--target");
    } else if (args[i] == "--target-workdir") {
      target.workdir = require_value(args, i, "--target-workdir");
    } else if (args[i] == "--remote-db") {
      remote_db_path = require_value(args, i, "--remote-db");
    } else if (args[i] == "--output") {
      local_output = require_value(args, i, "--output");
    } else if (args[i] == "--help") {
      std::cout << "Usage: hotpath recover --target HOST --remote-db REMOTE.db --output LOCAL.db [--target-workdir DIR]\n";
      return 0;
    }
  }
  target = resolve_cli_target(target);
  if (!hotpath::has_remote_target(target) || remote_db_path.empty() || local_output.empty()) {
    throw std::runtime_error("recover requires --target, --remote-db, and --output");
  }
  fetch_remote_profile_artifacts(target, local_output, remote_db_path, true);
  std::cout << local_output.string() << "\n";
  return 0;
}

int handle_bench_compare(const Args& args) {
  if (args.size() < 3 || (args.size() > 1 && args[1] == "--help")) {
    std::cout << "Usage: hotpath bench-compare A.json B.json\n";
    return 0;
  }
  const auto left = hotpath::bench::parse_bench_json(
      run_command_capture("cat " + shell_escape(args[1])));
  const auto right = hotpath::bench::parse_bench_json(
      run_command_capture("cat " + shell_escape(args[2])));
  std::cout << hotpath::bench::render_bench_comparison(left, right);
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
  if (prompts <= 0) {
    throw std::runtime_error("--prompts must be > 0");
  }
  if (rollouts_per_prompt <= 0) {
    throw std::runtime_error("--rollouts-per-prompt must be > 0");
  }
  if (input_len <= 0) {
    throw std::runtime_error("--input-len must be > 0");
  }
  if (min_tokens <= 0) {
    throw std::runtime_error("--min-tokens must be > 0");
  }
  if (max_tokens <= 0) {
    throw std::runtime_error("--max-tokens must be > 0");
  }
  if (min_tokens > max_tokens) {
    throw std::runtime_error("--min-tokens must be <= --max-tokens");
  }

  const auto run = hotpath::fire_rl_traffic(
      servers, prompts, rollouts_per_prompt, min_tokens, max_tokens, input_len);
  std::cout << render_traffic_json(run.stats);
  return 0;
}

hotpath::interactive::ProfileConfig profile_interactive_defaults(const Args& args) {
  auto config = hotpath::interactive::load_profile_defaults();
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
    std::int64_t repeats,
    double batch_ms_target,
    const std::string& cuda_graph_replay,
    bool flush_l2 = false) {
  const char* _py1 = std::getenv("HOTPATH_PYTHON_EXECUTABLE");
  if (!_py1 || std::string(_py1).empty()) _py1 = std::getenv("RLPROF_PYTHON_EXECUTABLE");
  const std::string python =
      _py1 != nullptr && std::string(_py1).size() > 0
          ? std::string(_py1)
          : (std::filesystem::exists(".venv/bin/python") ? ".venv/bin/python" : "python3");
  std::string cmd = shell_escape(python) + " -m " + shell_escape("hotpath_py.bench_cuda") +
        " --kernel " + shell_escape(kernel) +
        " --shapes " + shell_escape(shapes) +
        " --dtype " + shell_escape(dtype) +
        " --warmup " + shell_escape(std::to_string(warmup)) +
        " --n-iter " + shell_escape(std::to_string(n_iter)) +
         " --repeats " + shell_escape(std::to_string(repeats)) +
         " --batch-ms-target " + shell_escape(std::to_string(batch_ms_target)) +
         " --cuda-graph-replay " + shell_escape(cuda_graph_replay);
  if (flush_l2) cmd += " --flush-l2";
  cmd += " 2>&1";
  return cmd;
}

bool gpu_bench_available() {
  const char* _py2 = std::getenv("HOTPATH_PYTHON_EXECUTABLE");
  if (!_py2 || std::string(_py2).empty()) _py2 = std::getenv("RLPROF_PYTHON_EXECUTABLE");
  const std::string python =
      _py2 != nullptr && std::string(_py2).size() > 0
          ? std::string(_py2)
          : (std::filesystem::exists(".venv/bin/python") ? ".venv/bin/python" : "python3");
  const std::string probe =
      shell_escape(python) + " -c " +
      shell_escape(
          "import torch, vllm, hotpath_py.bench_cuda; raise SystemExit(0 if torch.cuda.is_available() else 1)") +
      " > /dev/null 2>&1";
  return std::system(probe.c_str()) == 0;
}

BenchCommandOptions parse_bench_args(const Args& args) {
  BenchCommandOptions options;

  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--kernel") {
      options.kernel = require_value(args, i, "--kernel");
    } else if (args[i] == "--target") {
      options.target.host = require_value(args, i, "--target");
    } else if (args[i] == "--target-workdir") {
      options.target.workdir = require_value(args, i, "--target-workdir");
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
    } else if (args[i] == "--batch-ms-target") {
      options.batch_ms_target = std::stod(require_value(args, i, "--batch-ms-target"));
    } else if (args[i] == "--cuda-graph-replay") {
      options.cuda_graph_replay = require_value(args, i, "--cuda-graph-replay");
    } else if (args[i] == "--flush-l2") {
      options.flush_l2 = true;
    } else if (args[i] == "--output") {
      options.output = require_value(args, i, "--output");
    } else if (args[i] == "--yes") {
      options.assume_yes = true;
    } else if (args[i] == "--help") {
      options.show_help = true;
    }
  }
  return options;
}

std::string run_bench_command(const BenchCommandOptions& options) {
  auto effective_options = options;
  effective_options.target = resolve_cli_target(effective_options.target);
  if (effective_options.kernel.empty()) {
    throw std::runtime_error("--kernel is required");
  }

  if (hotpath::has_remote_target(effective_options.target)) {
    std::filesystem::path local_output_path =
        hotpath::bench::resolve_bench_output_path(effective_options.kernel, effective_options.output);
    if (local_output_path.empty()) {
      local_output_path =
          hotpath::bench::resolve_bench_output_path(effective_options.kernel, "auto");
    }
    const std::string remote_output_path =
        local_output_path.empty() ? "none"
                                  : hotpath::remote_join(effective_options.target, local_output_path);
    Args remote_args = {
        "bench",
        "--kernel",
        effective_options.kernel,
        "--shapes",
        effective_options.shapes,
        "--dtype",
        effective_options.dtype,
        "--warmup",
        std::to_string(effective_options.warmup),
        "--n-iter",
        std::to_string(effective_options.n_iter),
        "--repeats",
        std::to_string(effective_options.repeats),
        "--batch-ms-target",
        std::to_string(effective_options.batch_ms_target),
        "--cuda-graph-replay",
        effective_options.cuda_graph_replay,
        "--output",
        remote_output_path,
    };
    run_command_capture(hotpath::remote_cli_command(effective_options.target, remote_args));
    copy_remote_file_if_present(effective_options.target, remote_output_path, local_output_path, true);
    const auto output =
        hotpath::bench::parse_bench_json(run_command_capture(
            "cat " + shell_escape(local_output_path.string())));
    std::ostringstream rendered;
    rendered << "saved: " << local_output_path.string() << "\n";
    rendered << hotpath::bench::render_bench_output(output);
    return rendered.str();
  }

  if (gpu_bench_available()) {
    const std::string raw_json = run_command_capture(
        bench_helper_command(
            effective_options.kernel,
            effective_options.shapes,
            effective_options.dtype,
            effective_options.warmup,
            effective_options.n_iter,
            effective_options.repeats,
            effective_options.batch_ms_target,
            effective_options.cuda_graph_replay,
            effective_options.flush_l2));
    const auto output = hotpath::bench::parse_bench_json(raw_json);
    const auto output_path =
        hotpath::bench::resolve_bench_output_path(effective_options.kernel, effective_options.output);
    std::ostringstream rendered;
    if (!output_path.empty()) {
      std::filesystem::create_directories(output_path.parent_path());
      std::ofstream out(output_path);
      out << raw_json;
      rendered << "saved: " << output_path.string() << "\n";
    }
    rendered << hotpath::bench::render_bench_output(output);
    return rendered.str();
  }

  std::ostringstream output;
  output << "warning: torch/vllm CUDA bench helper unavailable, falling back to native CPU stubs\n";
  hotpath::bench::register_builtin_kernels();
  const auto results = hotpath::bench::benchmark_category(
      options.kernel,
      hotpath::bench::parse_shapes(effective_options.shapes),
      effective_options.dtype,
      effective_options.warmup,
      effective_options.n_iter);
  const hotpath::bench::BenchRunOutput run_output{
      .gpu = std::nullopt,
      .results = results,
      .correctness_failures = {},
      .timing_warnings = {},
      .environment_warnings = {},
  };
  const auto output_path =
      hotpath::bench::resolve_bench_output_path(effective_options.kernel, effective_options.output);
  if (!output_path.empty()) {
    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream out(output_path);
    out << hotpath::bench::serialize_bench_output_json(run_output);
    output << "saved: " << output_path.string() << "\n";
  }
  output << hotpath::bench::render_bench_output(run_output);
  return output.str();
}

int handle_bench(const Args& args) {
  auto options = parse_bench_args(args);
  options.target = resolve_cli_target(options.target);
  if (options.show_help) {
    std::cout << "Usage: hotpath bench --kernel NAME --shapes SPEC [options]\n"
              << "       flags: --yes\n"
              << "       timing: --batch-ms-target FLOAT --cuda-graph-replay off|on --flush-l2\n"
              << "       remote: --target HOST --target-workdir DIR\n"
              << "       output: --output PATH|auto|none\n";
    return 0;
  }
  std::cout << run_bench_command(options);
  return 0;
}

hotpath::interactive::BenchConfig bench_interactive_defaults(const Args& args) {
  auto config = hotpath::interactive::load_bench_defaults();
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
      std::cout << "Usage: hotpath lock-clocks [--freq MHZ]\n";
      return 0;
    }
  }

  hotpath::lock_gpu_clocks(freq_mhz);
  const std::int64_t effective_freq =
      freq_mhz.value_or(hotpath::query_max_sm_clock_mhz());
  std::cout << "GPU clocks locked to " << effective_freq << " MHz\n";
  return 0;
}

int handle_unlock_clocks(const Args& args) {
  if (args.size() > 1 && args[1] == "--help") {
    std::cout << "Usage: hotpath unlock-clocks\n";
    return 0;
  }
  hotpath::unlock_gpu_clocks();
  std::cout << "GPU clocks unlocked\n";
  return 0;
}

int handle_reset_defaults(const Args& args) {
  if (args.size() > 1 && args[1] == "--help") {
    std::cout << "Usage: hotpath reset-defaults\n";
    return 0;
  }
  hotpath::interactive::clear_saved_defaults();
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
  hotpath::interactive::print_header(header);
  const auto profiles = hotpath::interactive::list_recent_profiles(10);
  if (profiles.empty()) {
    hotpath::interactive::print_warning("No profiles found in .hotpath/");
    return std::nullopt;
  }
  std::vector<std::string> options;
  options.reserve(profiles.size());
  for (const auto& path : profiles) {
    options.push_back(format_recent_profile_option(path, include_timestamp));
  }
  const auto choice =
      hotpath::interactive::prompt_choice("Recent profiles", options, 0);
  if (!choice.has_value()) {
    return std::nullopt;
  }
  return profiles[static_cast<std::size_t>(*choice)];
}

std::optional<std::string> choose_recent_bench_result(
    const std::string& header,
    bool include_timestamp) {
  hotpath::interactive::print_header(header);
  const auto results = hotpath::interactive::list_recent_bench_results(10);
  if (results.empty()) {
    hotpath::interactive::print_warning("No bench results found in .hotpath/");
    return std::nullopt;
  }
  std::vector<std::string> options;
  options.reserve(results.size());
  for (const auto& path : results) {
    options.push_back(format_recent_profile_option(path, include_timestamp));
  }
  const auto choice =
      hotpath::interactive::prompt_choice("Recent bench results", options, 0);
  if (!choice.has_value()) {
    return std::nullopt;
  }
  return results[static_cast<std::size_t>(*choice)];
}

int execute_profile_config(
    const hotpath::interactive::ProfileConfig& config,
    bool confirm_start,
    bool persist_defaults) {
  if (config.model.empty()) {
    throw std::runtime_error("profile requires a model");
  }

  const std::string gpu_name = hotpath::interactive::detect_gpu_name();
  const std::string clock_status = hotpath::interactive::clock_status_label();
  const bool clocks_locked = hotpath::interactive::are_clocks_locked();

  std::cout << "\n";
  hotpath::interactive::print_info(
      "-> " + config.model + " · " + std::to_string(config.prompts) + " prompts x " +
          std::to_string(config.rollouts) + " rollouts · " +
          std::to_string(config.min_tokens) + "-" + std::to_string(config.max_tokens) +
          " tokens",
      "");
  hotpath::interactive::print_info(
      "-> " + gpu_name + " · clocks: " + clock_status,
      "");
  if (!trim(config.peer_servers).empty()) {
    const auto peers = split_csv_list(config.peer_servers);
    hotpath::interactive::print_info(
        "-> cluster endpoints: " + std::to_string(peers.size() + 1),
        "");
  }
  if (config.discard_first_run) {
    hotpath::interactive::print_info("-> discard first run", "enabled");
  }
  if (!clocks_locked) {
    hotpath::interactive::print_warning(
        "GPU clocks unlocked - run `hotpath lock-clocks` for reproducibility");
  }

  if (confirm_start) {
    const auto confirm = hotpath::interactive::prompt_bool("Start profiling", true);
    if (!confirm.has_value() || !*confirm) {
      return 0;
    }
  }

  std::string output_text;
  hotpath::interactive::run_with_progress(
      "Starting vLLM server...",
      [&](const hotpath::interactive::ProgressCallback& progress) {
        output_text = run_profile_command(
            parse_profile_args(hotpath::interactive::build_profile_args(config)),
            progress);
      });

  if (persist_defaults) {
    hotpath::interactive::save_profile_defaults(config);
  }

  const std::string trimmed_output = trim(output_text);
  if (config.repeat == 1) {
    const auto newline = trimmed_output.find('\n');
    const std::string saved_path =
        newline == std::string::npos ? trimmed_output : trimmed_output.substr(0, newline);
    std::cout << "  Saved: " << saved_path << "\n";
    std::cout << "  Run `hotpath report " << saved_path << "` to view results\n";
    if (newline != std::string::npos) {
      std::cout << trim(trimmed_output.substr(newline + 1)) << "\n";
    }
  } else {
    std::cout << trimmed_output << "\n";
  }
  return 0;
}

int interactive_profile_flow(const hotpath::interactive::ProfileConfig& initial) {
  hotpath::interactive::print_header("hotpath · profile your rl environment");

  std::optional<std::string> model;
  while (!model.has_value() || model->empty()) {
    model = hotpath::interactive::prompt_string("Model", initial.model);
    if (!model.has_value()) {
      return 0;
    }
    if (model->empty()) {
      hotpath::interactive::print_warning("Model is required");
    }
  }

  auto prompts = hotpath::interactive::prompt_int("Prompts per batch", initial.prompts);
  if (!prompts.has_value()) {
    return 0;
  }
  auto target = hotpath::interactive::prompt_string("Target host", initial.target);
  if (!target.has_value()) {
    return 0;
  }
  std::string target_workdir = initial.target_workdir;
  if (!target->empty()) {
    auto prompted_workdir =
        hotpath::interactive::prompt_string("Target workdir", initial.target_workdir);
    if (!prompted_workdir.has_value()) {
      return 0;
    }
    target_workdir = *prompted_workdir;
  }
  auto rollouts = hotpath::interactive::prompt_int("Rollouts per prompt", initial.rollouts);
  if (!rollouts.has_value()) {
    return 0;
  }
  auto min_tokens = hotpath::interactive::prompt_int("Min output tokens", initial.min_tokens);
  if (!min_tokens.has_value()) {
    return 0;
  }
  auto max_tokens = hotpath::interactive::prompt_int("Max output tokens", initial.max_tokens);
  if (!max_tokens.has_value()) {
    return 0;
  }
  if (*min_tokens > *max_tokens) {
    hotpath::interactive::print_warning("Min output tokens must be <= max output tokens");
    return 0;
  }
  auto input_len = hotpath::interactive::prompt_int("Input length", initial.input_len);
  if (!input_len.has_value()) {
    return 0;
  }
  auto port = hotpath::interactive::prompt_int("Server port", initial.port);
  if (!port.has_value()) {
    return 0;
  }
  auto tp = hotpath::interactive::prompt_int("Tensor parallel size", initial.tp);
  if (!tp.has_value()) {
    return 0;
  }
  auto peer_servers = hotpath::interactive::prompt_string("Peer servers csv", initial.peer_servers);
  if (!peer_servers.has_value()) {
    return 0;
  }
  auto trust_remote_code =
      hotpath::interactive::prompt_bool("Trust remote code", initial.trust_remote_code);
  if (!trust_remote_code.has_value()) {
    return 0;
  }
  auto discard_first_run =
      hotpath::interactive::prompt_bool("Discard first run", initial.discard_first_run);
  if (!discard_first_run.has_value()) {
    return 0;
  }
  auto repeat = hotpath::interactive::prompt_int("Repeat runs", initial.repeat);
  if (!repeat.has_value()) {
    return 0;
  }
  auto output = hotpath::interactive::prompt_string("Output path", initial.output);
  if (!output.has_value()) {
    return 0;
  }

  const hotpath::interactive::ProfileConfig config = {
      .model = *model,
      .target = *target,
      .target_workdir = target_workdir,
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
    const hotpath::interactive::BenchConfig& config,
    bool persist_defaults) {
  std::cout << "\n  Benchmarking " << config.kernel
            << " on " << hotpath::interactive::detect_gpu_name() << "...\n";

  std::string output_text;
  hotpath::interactive::run_with_progress(
      "Benchmarking...",
      [&](const hotpath::interactive::ProgressCallback&) {
        output_text = run_bench_command(
            parse_bench_args(hotpath::interactive::build_bench_args(config)));
      });

  if (persist_defaults) {
    hotpath::interactive::save_bench_defaults(config);
  }

  std::cout << output_text;
  return 0;
}

int interactive_bench_flow(const hotpath::interactive::BenchConfig& initial) {
  hotpath::interactive::print_header("hotpath · benchmark kernel implementations");
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
      hotpath::interactive::prompt_choice("Kernel", kernels, default_kernel);
  if (!kernel_choice.has_value()) {
    return 0;
  }
  const auto shapes = hotpath::interactive::prompt_string("Shapes", initial.shapes);
  if (!shapes.has_value()) {
    return 0;
  }
  const auto target = hotpath::interactive::prompt_string("Target host", initial.target);
  if (!target.has_value()) {
    return 0;
  }
  std::string target_workdir = initial.target_workdir;
  if (!target->empty()) {
    const auto prompted_workdir =
        hotpath::interactive::prompt_string("Target workdir", initial.target_workdir);
    if (!prompted_workdir.has_value()) {
      return 0;
    }
    target_workdir = *prompted_workdir;
  }
  const auto dtype = hotpath::interactive::prompt_string("Dtype", initial.dtype);
  if (!dtype.has_value()) {
    return 0;
  }
  const auto warmup = hotpath::interactive::prompt_int("Warmup iterations", initial.warmup);
  if (!warmup.has_value()) {
    return 0;
  }
  const auto n_iter = hotpath::interactive::prompt_int("Timed iterations", initial.n_iter);
  if (!n_iter.has_value()) {
    return 0;
  }
  const auto repeats = hotpath::interactive::prompt_int("Repeat runs", initial.repeats);
  if (!repeats.has_value()) {
    return 0;
  }

  const hotpath::interactive::BenchConfig config = {
      .kernel = kernels[static_cast<std::size_t>(*kernel_choice)],
      .target = *target,
      .target_workdir = target_workdir,
      .shapes = *shapes,
      .dtype = *dtype,
      .warmup = *warmup,
      .n_iter = *n_iter,
      .repeats = *repeats,
  };
  return execute_bench_config(config, true);
}

int interactive_report_flow() {
  const auto path = choose_recent_profile("hotpath · view a saved profile", true);
  if (!path.has_value()) {
    return 0;
  }
  return handle_report({"report", *path});
}

int interactive_diff_flow() {
  hotpath::interactive::print_header("hotpath · compare two profiles");
  const auto profiles = hotpath::interactive::list_recent_profiles(10);
  if (profiles.size() < 2) {
    hotpath::interactive::print_warning("Need at least two profiles in .hotpath/");
    return 0;
  }
  std::vector<std::string> options;
  options.reserve(profiles.size());
  for (const auto& path : profiles) {
    options.push_back(std::filesystem::path(path).filename().string());
  }
  const auto baseline =
      hotpath::interactive::prompt_choice("Baseline", options, 0);
  if (!baseline.has_value()) {
    return 0;
  }
  const auto candidate =
      hotpath::interactive::prompt_choice("Candidate", options, std::min<int>(1, options.size() - 1));
  if (!candidate.has_value()) {
    return 0;
  }
  return handle_diff(
      {"diff", profiles[static_cast<std::size_t>(*baseline)], profiles[static_cast<std::size_t>(*candidate)]});
}

int interactive_export_flow(const std::string& default_format = "csv") {
  const auto path = choose_recent_profile("hotpath · export profile data", false);
  if (!path.has_value()) {
    return 0;
  }
  const std::vector<std::string> formats = {"csv", "json"};
  const int default_index = default_format == "json" ? 1 : 0;
  const auto format_choice =
      hotpath::interactive::prompt_choice("Format", formats, default_index);
  if (!format_choice.has_value()) {
    return 0;
  }
  return handle_export(
      {"export", *path, "--format", formats[static_cast<std::size_t>(*format_choice)]});
}

int interactive_bench_compare_flow() {
  hotpath::interactive::print_header("hotpath · compare archived bench runs");
  const auto left = choose_recent_bench_result("Left bench result", true);
  if (!left.has_value()) {
    return 0;
  }
  const auto right = choose_recent_bench_result("Right bench result", true);
  if (!right.has_value()) {
    return 0;
  }
  return handle_bench_compare({"bench-compare", *left, *right});
}

std::string completion_script(const std::string& shell) {
  if (shell == "bash") {
    return R"(# bash completion for hotpath
_hotpath_complete() {
  local cur prev
  COMPREPLY=()
  cur="${COMP_WORDS[COMP_CWORD]}"
  prev="${COMP_WORDS[COMP_CWORD-1]}"
  local commands="version profile serve-profile serve-report disagg-config report aggregate bench bench-compare diff export trace artifacts manifest validate cleanup traffic doctor server target recover soak-profile cluster-profile lock-clocks unlock-clocks reset-defaults completion help"
  if [[ $COMP_CWORD -eq 1 ]]; then
    COMPREPLY=( $(compgen -W "$commands" -- "$cur") )
    return
  fi
  case "${COMP_WORDS[1]}" in
        profile)
          COMPREPLY=( $(compgen -W "--model --server --attach --attach-pid --target --target-workdir --prompts --rollouts --min-tokens --max-tokens --input-len --port --tp --peer-servers --trust-remote-code --discard-first-run --repeat --fetch-nsys-rep --output --yes --help" -- "$cur") )
          ;;
        serve-profile)
          COMPREPLY=( $(compgen -W "--endpoint --duration --traffic --output --model --nsys --server-pid --server-log --engine --help" -- "$cur") )
          ;;
        serve-report|disagg-config)
          COMPREPLY=( $(compgen -f -- "$cur") )
          ;;
        server)
          COMPREPLY=( $(compgen -W "start stop list show prune --name --model --port --tp --max-model-len --trust-remote-code --startup-timeout-s --help" -- "$cur") )
          ;;
    bench)
      COMPREPLY=( $(compgen -W "--kernel --target --target-workdir --shapes --dtype --warmup --n-iter --repeats --batch-ms-target --cuda-graph-replay --output --yes --help" -- "$cur") )
      ;;
        aggregate|report|artifacts|trace|validate|bench-compare)
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
        doctor)
          COMPREPLY=( $(compgen -W "--target --target-workdir --help" -- "$cur") )
          ;;
        target)
          COMPREPLY=( $(compgen -W "add list remove show bootstrap --host --workdir" -- "$cur") )
          ;;
        recover)
          COMPREPLY=( $(compgen -W "--target --target-workdir --remote-db --output --help" -- "$cur") )
          ;;
        bench-compare)
          COMPREPLY=( $(compgen -f -- "$cur") )
          ;;
        unlock-clocks|reset-defaults|completion|help)
          COMPREPLY=()
          ;;
  esac
}
complete -F _hotpath_complete hotpath
)";
  }
  if (shell == "zsh") {
    return R"(#compdef hotpath
_hotpath() {
  local -a commands
  commands=(
    'version:print hotpath version'
    'profile:run GPU profiling under RL traffic'
    'serve-profile:profile a live vLLM or SGLang server'
    'serve-report:view a serving analysis report'
    'disagg-config:print serving deployment configs'
    'soak-profile:run repeated serving profiles with a shared server'
    'cluster-profile:profile across multiple remote targets'
    'report:view a saved profile'
    'aggregate:combine multiple profiles'
    'bench:benchmark kernel implementations'
    'bench-compare:compare two archived bench runs'
    'diff:compare two profiles'
    'export:export profile data'
    'trace:view raw trace artifact paths'
    'artifacts:view stored artifact paths'
    'manifest:write artifact manifest'
    'validate:validate stored profile against raw artifacts'
    'cleanup:remove old hotpath artifacts'
    'traffic:run traffic generator'
    'doctor:check local profiling environment'
    'server:manage local warm vllm servers'
    'target:manage saved ssh targets'
    'recover:recover remote profile artifacts'
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
          _arguments '--model[model name]' '--server[managed warm server name]' '--attach[existing server url]' '--attach-pid[remote process id; requires nsys PID attach support]' '--target[ssh target host]' '--target-workdir[remote hotpath workdir]' '--prompts[prompt count]' '--rollouts[rollouts per prompt]' '--min-tokens[min output tokens]' '--max-tokens[max output tokens]' '--input-len[input length]' '--port[server port]' '--tp[tensor parallel size]' '--peer-servers[peer endpoints]' '--trust-remote-code[trust remote code]' '--discard-first-run[run and discard a warmup pass]' '--repeat[repeat count]' '--fetch-nsys-rep[copy the remote nsys report locally]' '--output[output path]' '--yes[accept saved defaults]' '--help[show help]'
          ;;
        serve-profile)
          _arguments '--endpoint[serving endpoint url]' '--duration[profiling duration seconds]' '--traffic[jsonl or sharegpt traffic file]' '--output[output directory]' '--model[model name]' '--nsys[capture kernel trace via nsys]' '--server-pid[external vLLM server pid for nsys attach]' '--server-log[path to vLLM DEBUG log for server-side timing]' '--engine[serving engine]:engine:(vllm sglang)' '--help[show help]'
          ;;
        serve-report)
          _files
          ;;
        disagg-config)
          _arguments '--format[config format]:format:(vllm llmd dynamo all)' '*:path:_files'
          ;;
        server)
          _arguments '1:action:(start stop list show prune)' '--name[managed server name]' '--model[model name]' '--port[server port]' '--tp[tensor parallel size]' '--max-model-len[managed server max model length]' '--trust-remote-code[trust remote code]' '--startup-timeout-s[startup timeout seconds]' '--help[show help]'
          ;;
        bench)
          _arguments '--kernel[kernel name]' '--target[ssh target host]' '--target-workdir[remote hotpath workdir]' '--shapes[shape list]' '--dtype[data type]' '--warmup[warmup iterations]' '--n-iter[timed iterations]' '--repeats[repeat count]' '--batch-ms-target[target milliseconds per timed block]' '--cuda-graph-replay[use CUDA Graph replay]:mode:(off on)' '--output[result output path]' '--yes[accept saved defaults]' '--help[show help]'
          ;;
        aggregate|bench-compare)
          _files
          ;;
        export)
          _arguments '--format[export format]:format:(csv json)' '*:path:_files'
          ;;
        doctor)
          _arguments '--target[ssh target host]' '--target-workdir[remote hotpath workdir]' '--help[show help]'
          ;;
        target)
          _arguments '1:action:(add list remove show bootstrap)' '--host[ssh target host]' '--workdir[remote hotpath workdir]'
          ;;
        recover)
          _arguments '--target[ssh target host]' '--target-workdir[remote hotpath workdir]' '--remote-db[remote db path]' '--output[local db path]'
          ;;
        report|diff|artifacts|trace|validate)
          _files
          ;;
      esac
      ;;
  esac
}
_hotpath "$@"
)";
  }
  if (shell == "fish") {
    return R"(complete -c hotpath -f
complete -c hotpath -n '__fish_use_subcommand' -a 'version' -d 'print hotpath version'
complete -c hotpath -n '__fish_use_subcommand' -a 'profile' -d 'run GPU profiling under RL traffic'
complete -c hotpath -n '__fish_use_subcommand' -a 'serve-profile' -d 'profile a live vllm or sglang server'
complete -c hotpath -n '__fish_use_subcommand' -a 'serve-report' -d 'view a serving analysis report'
complete -c hotpath -n '__fish_use_subcommand' -a 'disagg-config' -d 'print serving deployment configs'
complete -c hotpath -n '__fish_use_subcommand' -a 'soak-profile' -d 'run repeated serving profiles with a shared server'
complete -c hotpath -n '__fish_use_subcommand' -a 'cluster-profile' -d 'profile across multiple remote targets'
complete -c hotpath -n '__fish_use_subcommand' -a 'report' -d 'view a saved profile'
complete -c hotpath -n '__fish_use_subcommand' -a 'aggregate' -d 'combine multiple profiles'
complete -c hotpath -n '__fish_use_subcommand' -a 'bench' -d 'benchmark kernel implementations'
complete -c hotpath -n '__fish_use_subcommand' -a 'bench-compare' -d 'compare archived bench runs'
complete -c hotpath -n '__fish_use_subcommand' -a 'diff' -d 'compare two profiles'
complete -c hotpath -n '__fish_use_subcommand' -a 'export' -d 'export profile data'
complete -c hotpath -n '__fish_use_subcommand' -a 'trace' -d 'view raw trace artifact paths'
complete -c hotpath -n '__fish_use_subcommand' -a 'artifacts' -d 'view stored artifact paths'
complete -c hotpath -n '__fish_use_subcommand' -a 'manifest' -d 'write artifact manifest'
complete -c hotpath -n '__fish_use_subcommand' -a 'validate' -d 'validate stored profile against raw artifacts'
complete -c hotpath -n '__fish_use_subcommand' -a 'cleanup' -d 'remove old hotpath artifacts'
complete -c hotpath -n '__fish_use_subcommand' -a 'traffic' -d 'run traffic generator'
complete -c hotpath -n '__fish_use_subcommand' -a 'doctor' -d 'check local profiling environment'
complete -c hotpath -n '__fish_use_subcommand' -a 'server' -d 'manage local warm vllm servers'
complete -c hotpath -n '__fish_use_subcommand' -a 'target' -d 'manage saved ssh targets'
complete -c hotpath -n '__fish_use_subcommand' -a 'recover' -d 'recover remote profile artifacts'
complete -c hotpath -n '__fish_use_subcommand' -a 'lock-clocks' -d 'lock GPU clocks'
complete -c hotpath -n '__fish_use_subcommand' -a 'unlock-clocks' -d 'unlock GPU clocks'
complete -c hotpath -n '__fish_use_subcommand' -a 'reset-defaults' -d 'clear saved interactive defaults'
complete -c hotpath -n '__fish_use_subcommand' -a 'completion' -d 'print shell completion'
complete -c hotpath -n '__fish_use_subcommand' -a 'help' -d 'show help'
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l model
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l server
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l attach
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l attach-pid
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l target
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l target-workdir
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l prompts
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l rollouts
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l min-tokens
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l max-tokens
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l input-len
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l port
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l tp
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l peer-servers
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l trust-remote-code
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l discard-first-run
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l repeat
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l fetch-nsys-rep
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l output
complete -c hotpath -n '__fish_seen_subcommand_from profile' -l yes
complete -c hotpath -n '__fish_seen_subcommand_from serve-profile' -l endpoint
complete -c hotpath -n '__fish_seen_subcommand_from serve-profile' -l duration
complete -c hotpath -n '__fish_seen_subcommand_from serve-profile' -l traffic
complete -c hotpath -n '__fish_seen_subcommand_from serve-profile' -l output
complete -c hotpath -n '__fish_seen_subcommand_from serve-profile' -l model
complete -c hotpath -n '__fish_seen_subcommand_from serve-profile' -l nsys
complete -c hotpath -n '__fish_seen_subcommand_from serve-profile' -l server-pid
complete -c hotpath -n '__fish_seen_subcommand_from serve-profile' -l server-log
complete -c hotpath -n '__fish_seen_subcommand_from serve-profile' -l engine -a 'vllm sglang'
complete -c hotpath -n '__fish_seen_subcommand_from disagg-config' -l format -a 'vllm llmd dynamo all'
complete -c hotpath -n '__fish_seen_subcommand_from bench' -l kernel -a 'silu_and_mul fused_add_rms_norm rotary_embedding'
complete -c hotpath -n '__fish_seen_subcommand_from bench' -l target
complete -c hotpath -n '__fish_seen_subcommand_from bench' -l target-workdir
complete -c hotpath -n '__fish_seen_subcommand_from bench' -l shapes
complete -c hotpath -n '__fish_seen_subcommand_from bench' -l dtype -a 'bf16 fp16'
complete -c hotpath -n '__fish_seen_subcommand_from bench' -l warmup
complete -c hotpath -n '__fish_seen_subcommand_from bench' -l n-iter
complete -c hotpath -n '__fish_seen_subcommand_from bench' -l repeats
complete -c hotpath -n '__fish_seen_subcommand_from bench' -l batch-ms-target
complete -c hotpath -n '__fish_seen_subcommand_from bench' -l cuda-graph-replay -a 'off on'
complete -c hotpath -n '__fish_seen_subcommand_from bench' -l output
complete -c hotpath -n '__fish_seen_subcommand_from bench' -l yes
complete -c hotpath -n '__fish_seen_subcommand_from export' -l format -a 'csv json'
complete -c hotpath -n '__fish_seen_subcommand_from doctor' -l target
complete -c hotpath -n '__fish_seen_subcommand_from doctor' -l target-workdir
complete -c hotpath -n '__fish_seen_subcommand_from server' -a 'start stop list show prune'
complete -c hotpath -n '__fish_seen_subcommand_from server' -l name
complete -c hotpath -n '__fish_seen_subcommand_from server' -l model
complete -c hotpath -n '__fish_seen_subcommand_from server' -l port
complete -c hotpath -n '__fish_seen_subcommand_from server' -l tp
complete -c hotpath -n '__fish_seen_subcommand_from server' -l max-model-len
complete -c hotpath -n '__fish_seen_subcommand_from server' -l trust-remote-code
complete -c hotpath -n '__fish_seen_subcommand_from server' -l startup-timeout-s
complete -c hotpath -n '__fish_seen_subcommand_from target' -a 'add list remove show bootstrap'
complete -c hotpath -n '__fish_seen_subcommand_from target' -l host
complete -c hotpath -n '__fish_seen_subcommand_from target' -l workdir
complete -c hotpath -n '__fish_seen_subcommand_from recover' -l target
complete -c hotpath -n '__fish_seen_subcommand_from recover' -l target-workdir
complete -c hotpath -n '__fish_seen_subcommand_from recover' -l remote-db
complete -c hotpath -n '__fish_seen_subcommand_from recover' -l output
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
    std::cout << "Usage: hotpath completion [bash|zsh|fish]\n"
              << "Examples:\n"
              << "  hotpath completion bash > ~/.local/share/bash-completion/completions/hotpath\n"
              << "  hotpath completion zsh > ~/.zfunc/_hotpath\n"
              << "  hotpath completion fish > ~/.config/fish/completions/hotpath.fish\n";
    return 0;
  }
  std::cout << completion_script(shell);
  return 0;
}

void print_help() {
  std::cout << "Usage: hotpath <command> [options]\n\n"
            << "Commands:\n"
            << "  version\n"
            << "  profile (--model MODEL | --server NAME | --attach URL) [options] [--repeat N]\n"
            << "    optional: --server NAME --attach URL --attach-pid PID --target HOST --target-workdir DIR\n"
            << "              --peer-servers URL1,URL2,... --discard-first-run --yes\n"
            << "  serve-profile [options]\n"
            << "  serve-report <serve_profile.db>\n"
            << "  disagg-config <serve_profile.db> [--format vllm|llmd|dynamo|all]\n"
            << "  server <start|stop|list|show|prune> ...\n"
            << "  cluster-profile --targets T1,T2 [profile options] --output base [--allow-duplicate-hosts]\n"
            << "  soak-profile [profile options] --iterations N --output base [--pause-sec N] [--validate-each]\n"
            << "  lock-clocks [--freq MHZ]\n"
            << "  unlock-clocks\n"
            << "  reset-defaults\n"
            << "  report [path]\n"
            << "  aggregate <a.db> <b.db> [more.db ...] --output out.db\n"
            << "  artifacts [path]\n"
            << "  manifest [path] [--output PATH|auto]\n"
            << "  cleanup [--dir DIR] [--keep N] [--compress] [--apply]\n"
            << "  validate [path]\n"
            << "  trace [path] [--open]\n"
            << "  export [path] --format csv|json\n"
            << "  target <add|list|remove|show|bootstrap> ...\n"
            << "  recover --target HOST --remote-db REMOTE.db --output LOCAL.db [--target-workdir DIR]\n"
            << "  diff <a.db> <b.db>\n"
            << "  bench --kernel NAME --shapes SPEC [options]\n"
            << "    optional: --output PATH|auto|none --yes\n"
            << "  bench-compare <a.json> <b.json>\n"
            << "  traffic --server URL | --servers URL1,URL2,... [options]\n"
            << "  doctor [--target HOST] [--target-workdir DIR]\n"
            << "  completion [bash|zsh|fish]\n";
}

int handle_serve_profile(const Args& args) {
  hotpath::ServeProfileOptions opts;
  bool saw_endpoint = false;
  bool saw_model = false;
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--endpoint") { opts.endpoint = require_value(args, i, "--endpoint"); saw_endpoint = true; }
    else if (args[i] == "--duration") { opts.duration_seconds = std::stoi(require_value(args, i, "--duration")); }
    else if (args[i] == "--traffic") { opts.traffic_path = require_value(args, i, "--traffic"); }
    else if (args[i] == "--output") { opts.output = require_value(args, i, "--output"); }
    else if (args[i] == "--nsys") { opts.use_nsys = true; }
    else if (args[i] == "--server-pid") { opts.server_pid = std::stoll(require_value(args, i, "--server-pid")); }
    else if (args[i] == "--server-log") { opts.server_log_path = require_value(args, i, "--server-log"); }
    else if (args[i] == "--engine") { opts.engine = require_value(args, i, "--engine"); }
    else if (args[i] == "--model") { opts.model = require_value(args, i, "--model"); saw_model = true; }
    else if (args[i] == "--concurrency") { opts.max_concurrency = std::stoi(require_value(args, i, "--concurrency")); }
    else if (args[i] == "--help") {
      std::cout << "Usage: hotpath serve-profile [options]\n"
                << "  --endpoint URL     profile an existing vLLM/SGLang endpoint (default: http://localhost:8000)\n"
                << "  --duration N       profiling duration in seconds (default: 60)\n"
                << "  --traffic PATH     JSONL or ShareGPT traffic file\n"
                << "  --concurrency N    max concurrent in-flight requests (default: 1)\n"
                << "  --output PATH      output directory (default: .hotpath/serve_run)\n"
                << "  --model NAME       launch and profile a managed vLLM server for this model\n"
                << "  --nsys             also capture kernel trace via nsys\n"
                << "  --server-pid PID   external vLLM server PID for nsys attach\n"
                << "  --server-log PATH  vLLM DEBUG log for server-side timing/cache analysis\n"
                << "  --engine TYPE      vllm or sglang (default: vllm)\n";
      return 0;
    }
  }
  if (saw_endpoint && saw_model) {
    std::cerr << "error: --endpoint and --model are mutually exclusive\n";
    return 1;
  }
  if (saw_model) {
    opts.launch_managed_server = true;
    opts.endpoint.clear();
  }
  return hotpath::run_serve_profile(opts);
}

int handle_serve_report(const Args& args) {
  std::string db_path;
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (!args[i].starts_with("--")) {
      db_path = args[i];
      break;
    }
  }
  if (db_path.empty()) {
    std::cerr << "Usage: hotpath serve-report <serve_profile.db>\n";
    return 1;
  }
  if (!std::filesystem::exists(db_path)) {
    std::cerr << "error: file not found: " << db_path << "\n";
    return 1;
  }
  const auto kv = hotpath::load_serve_analysis(db_path);
  if (kv.empty()) {
    std::cerr << "error: no serve analysis data in " << db_path << "\n";
    std::cerr << "Run 'hotpath serve-profile' first.\n";
    return 1;
  }
  auto get = [&](const std::string& k, const std::string& def = "") -> std::string {
    auto it = kv.find(k);
    return (it != kv.end()) ? it->second : def;
  };
  auto get_d = [&](const std::string& k) -> double {
    try { return std::stod(get(k, "0")); } catch (...) { return 0.0; }
  };
  auto get_i = [&](const std::string& k) -> int {
    try { return std::stoi(get(k, "0")); } catch (...) { return 0; }
  };

  hotpath::ServeReportData d;
  d.model_name = get("meta.model");
  d.engine = get("meta.engine");
  d.gpu_info = std::to_string(get_i("meta.gpu_count")) + "x " + get("meta.gpu_name");
  d.total_requests = get_i("meta.total_requests");
  d.duration_seconds = get_d("meta.duration_seconds");
  d.throughput_rps = get_d("meta.throughput_rps");

  d.queue_wait_available = get("latency.queue_available") == "true";
  d.server_timing_available = get("latency.server_timing_available") == "true";
  d.server_timing_match_method = get("latency.server_timing_match_method");
  d.server_timing_max_offset_ms = get_d("latency.server_timing_match_max_offset_ms");
  d.server_timing_remote_correlation =
      get("latency.server_timing_remote_correlation") == "true";
  d.queue_p50 = get_d("latency.queue_p50");
  d.queue_p90 = get_d("latency.queue_p90");
  d.queue_p99 = get_d("latency.queue_p99");
  d.server_prefill_p50 = get_d("latency.server_prefill_p50");
  d.server_prefill_p90 = get_d("latency.server_prefill_p90");
  d.server_prefill_p99 = get_d("latency.server_prefill_p99");
  d.server_decode_p50 = get_d("latency.server_decode_p50");
  d.server_decode_p90 = get_d("latency.server_decode_p90");
  d.server_decode_p99 = get_d("latency.server_decode_p99");
  d.server_ttft_mean_ms = get_d("latency.server_ttft_mean_ms");
  d.prefill_p50 = get_d("latency.prefill_p50");
  d.prefill_p90 = get_d("latency.prefill_p90");
  d.prefill_p99 = get_d("latency.prefill_p99");
  d.decode_total_p50 = get_d("latency.decode_total_p50");
  d.decode_total_p90 = get_d("latency.decode_total_p90");
  d.decode_total_p99 = get_d("latency.decode_total_p99");
  d.decode_per_token_p50 = get_d("latency.decode_per_token_p50");
  d.decode_per_token_p90 = get_d("latency.decode_per_token_p90");
  d.decode_per_token_p99 = get_d("latency.decode_per_token_p99");
  d.e2e_p50 = get_d("latency.e2e_p50");
  d.e2e_p90 = get_d("latency.e2e_p90");
  d.e2e_p99 = get_d("latency.e2e_p99");

  d.gpu_phase_available = get("phase.available") == "true";
  d.prefill_compute_pct = get_d("phase.prefill_pct");
  d.decode_compute_pct = get_d("phase.decode_pct");
  d.other_idle_pct = get_d("phase.other_pct");

  d.cache_hit_rate_available = get("cache.hit_rate_available") == "true";
  d.cache_usage_available = get("cache.usage_available") == "true";
  d.cache_hit_rate_aggregate_only = get("cache.hit_rate_aggregate_only") == "true";
  d.cache_histogram_available = get("cache.histogram_available") == "true";
  d.cache_hit_rate = get_d("cache.hit_rate");
  d.avg_cache_usage = get_d("cache.avg_usage");
  d.peak_cache_usage = get_d("cache.peak_usage");
  d.evictions = get_i("cache.evictions");
  for (int i = 0; i < 5; ++i) {
    d.cache_hit_rate_histogram[static_cast<size_t>(i)] =
        get_i("cache.histogram_" + std::to_string(i));
  }

  d.prefix_sharing_available = get("prefix.available") == "true";
  d.unique_prefixes = get_i("prefix.unique_prefixes");
  d.cacheable_tokens_pct = get_d("prefix.cacheable_tokens_pct");

  d.should_disaggregate = get("disagg.should") == "true";
  d.optimal_p = get_i("disagg.optimal_p");
  d.optimal_d = get_i("disagg.optimal_d");
  // throughput_improvement defaults to 0.0 via get_d when missing; clamp to
  // avoid displaying negative throughput if DB is missing this key.
  d.projected_throughput_pct = std::max(0.0, (get_d("disagg.throughput_improvement") - 1.0) * 100.0);
  d.projected_throughput_rps = get_d("disagg.disagg_throughput");
  d.mono_p99_ttft = get_d("disagg.mono_p99_ttft");
  d.disagg_p99_ttft = get_d("disagg.disagg_p99_ttft");
  d.min_bandwidth_gbps = get_d("disagg.min_bandwidth");
  d.advisor_caveat = get("disagg.caveat");

  std::cout << hotpath::render_serve_report(d);
  return 0;
}

int handle_disagg_config(const Args& args) {
  std::string db_path;
  std::string format = "all";
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--format") { format = require_value(args, i, "--format"); }
    else if (args[i] == "--help") {
      std::cout << "Usage: hotpath disagg-config <serve_profile.db> [--format vllm|llmd|dynamo|all]\n";
      return 0;
    }
    else if (!args[i].starts_with("--")) { db_path = args[i]; }
  }
  if (db_path.empty()) {
    std::cerr << "Usage: hotpath disagg-config <serve_profile.db> [--format vllm|llmd|dynamo|all]\n";
    return 1;
  }
  if (!std::filesystem::exists(db_path)) {
    std::cerr << "error: file not found: " << db_path << "\n";
    return 1;
  }
  const auto kv = hotpath::load_serve_analysis(db_path);
  if (kv.empty()) {
    std::cerr << "error: no serve analysis data in " << db_path << "\n";
    return 1;
  }
  auto get = [&](const std::string& k, const std::string& def = "") -> std::string {
    auto it = kv.find(k);
    return (it != kv.end()) ? it->second : def;
  };
  auto get_d = [&](const std::string& k) -> double {
    try { return std::stod(get(k, "0")); } catch (...) { return 0.0; }
  };
  auto get_i = [&](const std::string& k) -> int {
    try { return std::stoi(get(k, "0")); } catch (...) { return 0; }
  };

  hotpath::DisaggEstimate est;
  est.should_disaggregate = get("disagg.should") == "true";
  est.optimal_prefill_gpus = get_i("disagg.optimal_p");
  est.optimal_decode_gpus = get_i("disagg.optimal_d");
  est.mono_throughput_rps = get_d("disagg.mono_throughput");
  est.disagg_throughput_rps = get_d("disagg.disagg_throughput");
  est.throughput_improvement = get_d("disagg.throughput_improvement");
  est.mono_p99_ttft_ms = get_d("disagg.mono_p99_ttft");
  est.disagg_p99_ttft_ms = get_d("disagg.disagg_p99_ttft");
  est.kv_transfer_overhead_ms = get_d("disagg.kv_transfer_overhead");
  est.min_bandwidth_gbps = get_d("disagg.min_bandwidth");
  est.reason = get("disagg.reason");

  const std::string model_name = get("meta.model", "model");

  if (format == "vllm" || format == "all") {
    std::cout << hotpath::generate_vllm_config(est, model_name);
    if (format == "all") std::cout << "\n---\n\n";
  }
  if (format == "llmd" || format == "all") {
    std::cout << hotpath::generate_llmd_config(est, model_name);
    if (format == "all") std::cout << "\n---\n\n";
  }
  if (format == "dynamo" || format == "all") {
    std::cout << hotpath::generate_dynamo_config(est, model_name);
  }
  if (format != "vllm" && format != "llmd" && format != "dynamo" && format != "all") {
    std::cerr << "unknown format: " << format << " (use vllm, llmd, dynamo, or all)\n";
    return 1;
  }

  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
  if (argc < 2) {
      hotpath::interactive::print_header("hotpath · the profiler for LLM inference");
      const auto choice = hotpath::interactive::prompt_choice(
          "What would you like to do?",
          {
              "serve-profile   profile a live vLLM/SGLang server",
              "serve-report    view a serving analysis report",
              "profile         GPU profiling under RL traffic",
              "report          view a saved profile",
              "aggregate       combine multiple profiles",
              "bench           benchmark kernel implementations",
              "diff            compare two profiles",
              "export          export profile data",
              "doctor          check local environment",
              "targets         manage SSH targets",
              "lock-clocks     lock GPU clocks",
              "unlock-clocks",
              "reset-defaults",
          },
          0);
      if (!choice.has_value()) {
        return 0;
      }
      if (*choice == 0) {
        // Interactive serve-profile flow
        hotpath::interactive::print_header("serve-profile — profile a live inference server");
        const auto endpoint = hotpath::interactive::prompt_string(
            "Endpoint URL", "http://localhost:8000");
        if (!endpoint.has_value() || endpoint->empty()) return 0;

        const auto engine_choice = hotpath::interactive::prompt_choice(
            "Serving engine", {"vllm", "sglang"}, 0);
        if (!engine_choice.has_value()) return 0;
        const std::string engine = (*engine_choice == 0) ? "vllm" : "sglang";

        const auto duration = hotpath::interactive::prompt_int("Duration (seconds)", 60);
        if (!duration.has_value()) return 0;

        const auto traffic = hotpath::interactive::prompt_string(
            "Traffic file (JSONL path, or empty to just collect metrics)", "");

        Args serve_args = {"serve-profile",
                           "--endpoint", *endpoint,
                           "--engine", engine,
                           "--duration", std::to_string(*duration)};
        if (traffic.has_value() && !traffic->empty()) {
          serve_args.push_back("--traffic");
          serve_args.push_back(*traffic);
        }
        return handle_serve_profile(serve_args);
      }
      if (*choice == 1) {
        // List .hotpath serve_profile.db files
        hotpath::interactive::print_header("serve-report — view a serving analysis");
        namespace fs = std::filesystem;
        std::vector<std::string> dbs;
        const fs::path hotpath_dir = ".hotpath";
        if (fs::exists(hotpath_dir)) {
          for (auto& entry : fs::recursive_directory_iterator(hotpath_dir)) {
            if (entry.path().filename() == "serve_profile.db") {
              dbs.push_back(entry.path().string());
            }
          }
        }
        if (dbs.empty()) {
          std::cerr << "No serve-profile databases found in .hotpath/\n";
          std::cerr << "Run 'hotpath serve-profile' first.\n";
          return 0;
        }
        if (dbs.size() == 1) {
          return handle_serve_report({"serve-report", dbs[0]});
        }
        const auto pick = hotpath::interactive::prompt_choice("Select a profile", dbs, 0);
        if (!pick.has_value()) return 0;
        return handle_serve_report({"serve-report", dbs[static_cast<size_t>(*pick)]});
      }
      if (*choice == 2) {
        return interactive_profile_flow(hotpath::interactive::load_profile_defaults());
      }
      if (*choice == 3) {
        return interactive_report_flow();
      }
      if (*choice == 4) {
        std::cout << "Usage: hotpath aggregate <a.db> <b.db> [more.db ...] --output out.db\n";
        return 0;
      }
      if (*choice == 5) {
        return interactive_bench_flow(hotpath::interactive::load_bench_defaults());
      }
      if (*choice == 6) {
        return interactive_diff_flow();
      }
      if (*choice == 7) {
        return interactive_export_flow();
      }
      if (*choice == 8) {
        return handle_doctor({"doctor"});
      }
      if (*choice == 9) {
        return handle_target({"target", "list"});
      }
      if (*choice == 10) {
        return handle_lock_clocks({"lock-clocks"});
      }
      if (*choice == 11) {
        return handle_unlock_clocks({"unlock-clocks"});
      }
      if (*choice == 12) {
        return handle_reset_defaults({"reset-defaults"});
      }
      return 0;
    }

    const Args args(argv + 1, argv + argc);
    const std::string command = args[0];

    if (command == "profile" &&
        !has_flag(args, "--model") &&
        !has_flag(args, "--server") &&
        !has_flag(args, "--attach") &&
        !has_flag(args, "--help")) {
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

    if (command == "cluster-profile") {
      return handle_cluster_profile(args);
    }

    if (command == "soak-profile") {
      return handle_soak_profile(args);
    }

    if (command == "report" && args.size() == 1) {
      return interactive_report_flow();
    }

    if (command == "report") {
      return handle_report(args);
    }

    if (command == "aggregate") {
      return handle_aggregate(args);
    }

    if (command == "artifacts") {
      return handle_artifacts(args);
    }

    if (command == "manifest") {
      return handle_manifest(args);
    }

    if (command == "cleanup") {
      return handle_cleanup(args);
    }

    if (command == "validate") {
      return handle_validate(args);
    }

    if (command == "trace") {
      return handle_trace(args);
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

    if (command == "export" && !has_flag(args, "--help")) {
      bool export_has_path = false;
      for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--format") {
          require_value(args, i, "--format");
          continue;
        }
        if (!args[i].starts_with("--")) {
          export_has_path = true;
          break;
        }
      }
      if (export_has_path) {
        return handle_export(args);
      }
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

    if (command == "doctor") {
      return handle_doctor(args);
    }

    if (command == "server") {
      return handle_server(args);
    }

    if (command == "target") {
      return handle_target(args);
    }

    if (command == "recover") {
      return handle_recover(args);
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

    if (command == "bench-compare") {
      return handle_bench_compare(args);
    }

    if (command == "serve-profile") {
      return handle_serve_profile(args);
    }

    if (command == "serve-report") {
      return handle_serve_report(args);
    }

    if (command == "disagg-config") {
      return handle_disagg_config(args);
    }

    if (command == "--help" || command == "help") {
      print_help();
      return 0;
    }

    if (command == "version") {
      std::cout << "hotpath 0.2.1\n";
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
