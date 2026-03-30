#include "rlprof/profiler/runner.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <sqlite3.h>
#include <errno.h>
#include <sys/types.h>

#include "rlprof/clock_control.h"
#include "rlprof/profiler/parser.h"
#include "rlprof/profiler/vllm_metrics.h"
#include "rlprof/store.h"

namespace rlprof::profiler {
namespace {

struct GpuTelemetrySample {
  double sample_time = 0.0;
  std::string driver_version;
  std::string pstate;
  std::string persistence_mode;
  double sm_clock_mhz = 0.0;
  double mem_clock_mhz = 0.0;
  double temp_c = 0.0;
  double power_draw_w = 0.0;
  double power_limit_w = 0.0;
  std::string throttle_active;
  std::string sw_power_cap;
  std::string sw_thermal_slowdown;
  std::string hw_thermal_slowdown;
};

std::string run_command(const std::string& command) {
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
    throw std::runtime_error("command failed: " + command);
  }
  return output;
}

pid_t start_background_command(const std::string& command) {
  const std::string wrapped = "bash -lc '" + command + " > /tmp/rlprof_server.log 2>&1 & echo $!'";
  const std::string output = run_command(wrapped);
  return static_cast<pid_t>(std::stol(output));
}

bool sqlite_kernel_table_ready(const std::filesystem::path& path) {
  sqlite3* db = nullptr;
  if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
    if (db != nullptr) {
      sqlite3_close(db);
    }
    return false;
  }

  sqlite3_stmt* stmt = nullptr;
  constexpr const char* kSql =
      "SELECT 1 FROM sqlite_master "
      "WHERE type = 'table' AND name = 'CUPTI_ACTIVITY_KIND_KERNEL' LIMIT 1";
  const int prepare_rc = sqlite3_prepare_v2(db, kSql, -1, &stmt, nullptr);
  if (prepare_rc != SQLITE_OK) {
    sqlite3_close(db);
    return false;
  }

  const int step_rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return step_rc == SQLITE_ROW;
}

bool wait_for_kernel_table(const std::filesystem::path& path, std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (std::filesystem::exists(path) && sqlite_kernel_table_ready(path)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  return std::filesystem::exists(path) && sqlite_kernel_table_ready(path);
}

void stop_process(pid_t pid) {
  if (pid > 0) {
    kill(pid, SIGTERM);
  }
}

bool process_alive(pid_t pid) {
  if (pid <= 0) {
    return false;
  }
  const int rc = kill(pid, 0);
  if (rc == 0) {
    return true;
  }
  return errno != ESRCH;
}

bool server_ready(const std::string& server_url) {
  const std::string command =
      "curl -fsS " + server_url + "/metrics > /dev/null 2>&1";
  return std::system(command.c_str()) == 0;
}

std::string trim(std::string value) {
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) {
    value.pop_back();
  }
  return value;
}

std::string server_log_tail(std::size_t max_lines = 120) {
  std::ifstream stream("/tmp/rlprof_server.log");
  if (!stream.good()) {
    return "";
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(stream, line)) {
    lines.push_back(line);
  }

  if (lines.empty()) {
    return "";
  }

  const std::size_t start =
      lines.size() > max_lines ? lines.size() - max_lines : 0;
  std::ostringstream tail;
  for (std::size_t i = start; i < lines.size(); ++i) {
    tail << lines[i] << "\n";
  }
  return trim(tail.str());
}

std::runtime_error server_startup_error(const std::string& message) {
  const std::string tail = server_log_tail();
  if (tail.empty()) {
    return std::runtime_error(message);
  }
  return std::runtime_error(message + "\n\nvLLM log tail:\n" + tail);
}

std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string part;
  while (std::getline(stream, part, ',')) {
    fields.push_back(trim(part));
  }
  return fields;
}

std::vector<std::string> split_csv_list(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string part;
  while (std::getline(stream, part, ',')) {
    part = trim(part);
    if (!part.empty()) {
      fields.push_back(part);
    }
  }
  return fields;
}

double now_seconds() {
  using Clock = std::chrono::system_clock;
  const auto now = Clock::now().time_since_epoch();
  return std::chrono::duration<double>(now).count();
}

std::optional<double> parse_double_field(const std::string& field) {
  if (field.empty() || field == "[N/A]" || field == "N/A") {
    return std::nullopt;
  }
  return std::stod(field);
}

std::optional<GpuTelemetrySample> query_gpu_telemetry() {
  try {
    const std::string output = trim(run_command(
        "nvidia-smi --query-gpu="
        "timestamp,driver_version,pstate,persistence_mode,"
        "clocks.current.sm,clocks.current.memory,temperature.gpu,"
        "power.draw,power.limit,"
        "clocks_throttle_reasons.active,"
        "clocks_throttle_reasons.sw_power_cap,"
        "clocks_throttle_reasons.sw_thermal_slowdown,"
        "clocks_throttle_reasons.hw_thermal_slowdown "
        "--format=csv,noheader,nounits | head -n 1"));
    const auto fields = split_csv_line(output);
    if (fields.size() < 13) {
      return std::nullopt;
    }
    const auto sm_clock = parse_double_field(fields[4]);
    const auto mem_clock = parse_double_field(fields[5]);
    const auto temp = parse_double_field(fields[6]);
    const auto power_draw = parse_double_field(fields[7]);
    const auto power_limit = parse_double_field(fields[8]);
    if (!sm_clock.has_value() || !mem_clock.has_value() || !temp.has_value() ||
        !power_draw.has_value() || !power_limit.has_value()) {
      return std::nullopt;
    }
    return GpuTelemetrySample{
        .sample_time = now_seconds(),
        .driver_version = fields[1],
        .pstate = fields[2],
        .persistence_mode = fields[3],
        .sm_clock_mhz = *sm_clock,
        .mem_clock_mhz = *mem_clock,
        .temp_c = *temp,
        .power_draw_w = *power_draw,
        .power_limit_w = *power_limit,
        .throttle_active = fields[9],
        .sw_power_cap = fields[10],
        .sw_thermal_slowdown = fields[11],
        .hw_thermal_slowdown = fields[12],
    };
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

void write_nvidia_smi_snapshot(const std::filesystem::path& output_prefix) {
  std::filesystem::path snapshot_path = output_prefix;
  snapshot_path += "_nvidia_smi.xml";
  const std::string command =
      "nvidia-smi -q -x > " + snapshot_path.string() + " 2>/dev/null";
  std::system(command.c_str());
}

std::vector<GpuTelemetrySample> poll_gpu_telemetry(
    std::chrono::milliseconds interval,
    const std::atomic<bool>& stop_flag) {
  std::vector<GpuTelemetrySample> samples;
  while (!stop_flag.load()) {
    if (const auto sample = query_gpu_telemetry(); sample.has_value()) {
      samples.push_back(*sample);
    }

    auto slept = std::chrono::milliseconds::zero();
    while (!stop_flag.load() && slept < interval) {
      constexpr auto kSlice = std::chrono::milliseconds(50);
      std::this_thread::sleep_for(kSlice);
      slept += kSlice;
    }
  }
  return samples;
}

template <typename Getter>
std::string summary_triplet(
    const std::vector<GpuTelemetrySample>& samples,
    Getter getter,
    int precision = 0) {
  if (samples.empty()) {
    return "";
  }
  double min_value = std::numeric_limits<double>::max();
  double max_value = std::numeric_limits<double>::lowest();
  double sum = 0.0;
  for (const auto& sample : samples) {
    const double value = getter(sample);
    min_value = std::min(min_value, value);
    max_value = std::max(max_value, value);
    sum += value;
  }
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision)
         << min_value << "," << (sum / samples.size()) << "," << max_value;
  return stream.str();
}

template <typename Getter>
double max_value(const std::vector<GpuTelemetrySample>& samples, Getter getter) {
  double value = std::numeric_limits<double>::lowest();
  for (const auto& sample : samples) {
    value = std::max(value, getter(sample));
  }
  return value;
}

template <typename Getter>
double min_value(const std::vector<GpuTelemetrySample>& samples, Getter getter) {
  double value = std::numeric_limits<double>::max();
  for (const auto& sample : samples) {
    value = std::min(value, getter(sample));
  }
  return value;
}

std::map<std::string, std::string> measurement_metadata(
    const std::vector<GpuTelemetrySample>& samples) {
  std::map<std::string, std::string> metadata;
  if (samples.empty()) {
    return metadata;
  }

  std::set<std::string> pstates;
  std::size_t throttle_active_samples = 0;
  std::size_t power_cap_samples = 0;
  std::size_t thermal_samples = 0;
  for (const auto& sample : samples) {
    pstates.insert(sample.pstate);
    if (sample.throttle_active != "Not Active") {
      ++throttle_active_samples;
    }
    if (sample.sw_power_cap == "Active") {
      ++power_cap_samples;
    }
    if (sample.sw_thermal_slowdown == "Active" ||
        sample.hw_thermal_slowdown == "Active") {
      ++thermal_samples;
    }
  }

  std::ostringstream pstates_joined;
  for (auto it = pstates.begin(); it != pstates.end(); ++it) {
    if (it != pstates.begin()) {
      pstates_joined << ",";
    }
    pstates_joined << *it;
  }

  const double sm_min = min_value(samples, [](const auto& sample) { return sample.sm_clock_mhz; });
  const double sm_max = max_value(samples, [](const auto& sample) { return sample.sm_clock_mhz; });
  double sm_sum = 0.0;
  double mem_sum = 0.0;
  double power_sum = 0.0;
  double temp_min = std::numeric_limits<double>::max();
  double temp_max = std::numeric_limits<double>::lowest();
  double mem_min = std::numeric_limits<double>::max();
  double mem_max = std::numeric_limits<double>::lowest();
  double power_peak = std::numeric_limits<double>::lowest();
  double power_limit = samples.front().power_limit_w;
  for (const auto& sample : samples) {
    sm_sum += sample.sm_clock_mhz;
    mem_sum += sample.mem_clock_mhz;
    power_sum += sample.power_draw_w;
    temp_min = std::min(temp_min, sample.temp_c);
    temp_max = std::max(temp_max, sample.temp_c);
    mem_min = std::min(mem_min, sample.mem_clock_mhz);
    mem_max = std::max(mem_max, sample.mem_clock_mhz);
    power_peak = std::max(power_peak, sample.power_draw_w);
  }
  const double sm_avg = sm_sum / samples.size();
  const double mem_avg = mem_sum / samples.size();
  const double power_avg = power_sum / samples.size();
  const bool sm_clock_unstable = sm_avg > 0.0 && ((sm_max - sm_min) / sm_avg) > 0.10;

  metadata["measurement_driver_version"] = samples.front().driver_version;
  metadata["measurement_persistence_mode"] = samples.front().persistence_mode;
  metadata["measurement_pstates"] = pstates_joined.str();
  metadata["measurement_samples"] = std::to_string(samples.size());
  metadata["measurement_sm_clock_min_mhz"] = std::to_string(sm_min);
  metadata["measurement_sm_clock_avg_mhz"] = std::to_string(sm_avg);
  metadata["measurement_sm_clock_max_mhz"] = std::to_string(sm_max);
  metadata["measurement_mem_clock_min_mhz"] = std::to_string(mem_min);
  metadata["measurement_mem_clock_avg_mhz"] = std::to_string(mem_avg);
  metadata["measurement_mem_clock_max_mhz"] = std::to_string(mem_max);
  metadata["measurement_temp_min_c"] = std::to_string(temp_min);
  metadata["measurement_temp_max_c"] = std::to_string(temp_max);
  metadata["measurement_power_draw_avg_w"] = std::to_string(power_avg);
  metadata["measurement_power_draw_peak_w"] = std::to_string(power_peak);
  metadata["measurement_power_limit_w"] = std::to_string(power_limit);
  metadata["measurement_throttle_active_samples"] = std::to_string(throttle_active_samples);
  metadata["measurement_power_capped_samples"] = std::to_string(power_cap_samples);
  metadata["measurement_thermal_slowdown_samples"] = std::to_string(thermal_samples);
  metadata["warning_sm_clock_unstable"] = sm_clock_unstable ? "true" : "false";
  metadata["warning_any_clock_throttle"] = throttle_active_samples > 0 ? "true" : "false";
  metadata["warning_power_capped"] = power_cap_samples > 0 ? "true" : "false";
  metadata["warning_thermal_slowdown"] = thermal_samples > 0 ? "true" : "false";
  metadata["warning_temp_high"] = temp_max >= 80.0 ? "true" : "false";
  return metadata;
}

std::filesystem::path default_output_prefix(const ProfileConfig& config) {
  if (!config.output.empty()) {
    return config.output;
  }
  const auto now = std::chrono::system_clock::now().time_since_epoch().count();
  std::string model = config.model;
  std::replace(model.begin(), model.end(), '/', '_');
  return std::filesystem::path(".rlprof") / (model + "_" + std::to_string(now));
}

std::int64_t inferred_max_model_len(const ProfileConfig& config) {
  return std::max<std::int64_t>(2048, config.input_len + config.max_tokens);
}

std::string vllm_binary() {
  if (std::filesystem::exists(".venv/bin/vllm")) {
    return ".venv/bin/vllm";
  }
  return "vllm";
}

std::string make_session_name(const std::filesystem::path& output_prefix) {
  std::string session = "rlprof_";
  const std::string stem = output_prefix.filename().string();
  for (unsigned char ch : stem) {
    session.push_back(std::isalnum(ch) ? static_cast<char>(ch) : '_');
  }
  return session;
}

void notify_progress(const ProgressCallback& progress, const std::string& status) {
  if (progress) {
    progress(status);
  }
}

std::vector<std::string> cluster_server_urls(
    const ProfileConfig& config,
    const std::string& local_server_url) {
  std::vector<std::string> servers;
  servers.push_back(local_server_url);
  for (const auto& peer : config.peer_servers) {
    if (!peer.empty()) {
      servers.push_back(peer);
    }
  }
  return servers;
}

std::vector<MetricEndpoint> metric_endpoints(
    const std::vector<std::string>& server_urls) {
  std::vector<MetricEndpoint> endpoints;
  endpoints.reserve(server_urls.size());
  for (std::size_t i = 0; i < server_urls.size(); ++i) {
    endpoints.push_back(MetricEndpoint{
        .source = i == 0 ? "local" : "peer" + std::to_string(i),
        .server_url = server_urls[i],
    });
  }
  return endpoints;
}

}  // namespace

ProfileRunResult run_profile(const ProfileConfig& config, ProgressCallback progress) {
  const std::filesystem::path output_prefix = default_output_prefix(config);
  std::filesystem::path nvidia_smi_snapshot_path = output_prefix;
  nvidia_smi_snapshot_path += "_nvidia_smi.xml";
  if (!output_prefix.parent_path().empty()) {
    std::filesystem::create_directories(output_prefix.parent_path());
  }

  const std::string server_url = "http://127.0.0.1:" + std::to_string(config.port);
  const std::vector<std::string> traffic_servers =
      cluster_server_urls(config, server_url);
  const std::vector<MetricEndpoint> metrics_endpoints =
      metric_endpoints(traffic_servers);
  const std::string gpu_name = trim(run_command("nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1"));
  const rlprof::ClockPolicyInfo clock_policy = rlprof::query_clock_policy();
  const std::string vllm_path = vllm_binary();
  const std::string vllm_version = trim(run_command(vllm_path + " --version"));
  const std::int64_t max_model_len = inferred_max_model_len(config);
  const std::string session_name = make_session_name(output_prefix);
  notify_progress(progress, "Starting vLLM server...");
  write_nvidia_smi_snapshot(output_prefix);

  const std::string serve_command =
      "VLLM_WORKER_MULTIPROC_METHOD=spawn nsys profile --trace=cuda,nvtx,osrt "
      "--sample=none --cpuctxsw=none --export=sqlite "
      "--force-overwrite=true --session-new " + session_name +
      " --start-later=true -o " + output_prefix.string() +
      " " + vllm_path + " serve " + config.model +
      " --port " + std::to_string(config.port) +
      " --tensor-parallel-size " + std::to_string(config.tp) +
      " --max-model-len " + std::to_string(max_model_len) +
      (config.trust_remote_code ? " --trust-remote-code" : "");
  const pid_t server_pid = start_background_command(serve_command);

  std::atomic<bool> stop_flag{false};
  std::vector<MetricSample> metrics;
  std::vector<GpuTelemetrySample> gpu_samples;
  std::thread metrics_thread;
  std::thread gpu_thread;

  try {
    notify_progress(progress, "Waiting for server ready...");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(config.startup_timeout_s);
    bool ready = false;
    while (std::chrono::steady_clock::now() < deadline) {
      if (!process_alive(server_pid)) {
        throw server_startup_error("vLLM server exited before becoming ready");
      }
      if (server_ready(server_url)) {
        ready = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (!ready) {
      stop_process(server_pid);
      throw server_startup_error(
          "vLLM server did not become ready within " +
          std::to_string(config.startup_timeout_s) + " seconds");
    }

    notify_progress(progress, "Collecting nsys trace...");
    run_command("nsys start --session=" + session_name);

    metrics_thread = std::thread([&]() {
      metrics = poll_metrics(
          metrics_endpoints,
          std::chrono::milliseconds(config.metrics_interval_ms),
          stop_flag);
    });
    gpu_thread = std::thread([&]() {
      gpu_samples = poll_gpu_telemetry(
          std::chrono::milliseconds(config.metrics_interval_ms),
          stop_flag);
    });

    notify_progress(
        progress,
        "Firing RL traffic (" +
            std::to_string(config.prompts * config.rollouts) + " requests)...");
    const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe");
    std::string traffic_command =
        self.string() + " traffic " +
        (traffic_servers.size() == 1
             ? "--server " + server_url
             : "--servers " + [&]() {
                 std::ostringstream joined;
                 for (std::size_t i = 0; i < traffic_servers.size(); ++i) {
                   if (i > 0) {
                     joined << ",";
                   }
                   joined << traffic_servers[i];
                 }
                 return joined.str();
               }()) +
        " --prompts " + std::to_string(config.prompts) +
        " --rollouts-per-prompt " + std::to_string(config.rollouts) +
        " --max-tokens " + std::to_string(config.max_tokens) +
        " --min-tokens " + std::to_string(config.min_tokens) +
        " --input-len " + std::to_string(config.input_len);
    const std::string traffic_output = trim(run_command(traffic_command));

    stop_flag.store(true);
    if (metrics_thread.joinable()) {
      metrics_thread.join();
    }
    if (gpu_thread.joinable()) {
      gpu_thread.join();
    }
    run_command("nsys stop --session=" + session_name);
    stop_process(server_pid);

    TrafficStats traffic_stats{
        .total_requests = 0,
        .completion_length_mean = std::nullopt,
        .completion_length_p50 = std::nullopt,
        .completion_length_p99 = std::nullopt,
        .max_median_ratio = std::nullopt,
        .errors = 0,
    };
    {
      const auto extract = [&](const std::string& key) -> std::optional<double> {
        const std::string needle = "\"" + key + "\":";
        const std::size_t start = traffic_output.find(needle);
        if (start == std::string::npos) {
          return std::nullopt;
        }
        std::size_t index = start + needle.size();
        while (index < traffic_output.size() && std::isspace(static_cast<unsigned char>(traffic_output[index]))) {
          ++index;
        }
        std::size_t end = index;
        while (end < traffic_output.size() &&
               (std::isdigit(static_cast<unsigned char>(traffic_output[end])) ||
                traffic_output[end] == '.' || traffic_output[end] == '-')) {
          ++end;
        }
        return std::stod(traffic_output.substr(index, end - index));
      };
      traffic_stats.total_requests = static_cast<std::int64_t>(extract("total_requests").value_or(0.0));
      traffic_stats.completion_length_mean = extract("completion_length_mean");
      traffic_stats.completion_length_p50 = extract("completion_length_p50");
      traffic_stats.completion_length_p99 = extract("completion_length_p99");
      traffic_stats.max_median_ratio = extract("max_median_ratio");
      traffic_stats.errors = static_cast<std::int64_t>(extract("errors").value_or(0.0));
    }

    std::filesystem::path sqlite_path = output_prefix;
    sqlite_path.replace_extension(".sqlite");
    if (!wait_for_kernel_table(sqlite_path, std::chrono::seconds(30))) {
      throw std::runtime_error("timed out waiting for nsys sqlite export readiness: " + sqlite_path.string());
    }
    notify_progress(progress, "Parsing kernel data...");
    const std::vector<KernelRecord> kernels = parse_nsys_sqlite(sqlite_path);
    const std::vector<MetricSummary> summaries = summarize_samples(metrics);

    ProfileData profile;
    profile.meta = {
        {"model_name", config.model},
        {"gpu_name", gpu_name},
        {"vllm_version", vllm_version},
        {"prompts", std::to_string(config.prompts)},
        {"rollouts", std::to_string(config.rollouts)},
        {"max_tokens", std::to_string(config.max_tokens)},
        {"min_tokens", std::to_string(config.min_tokens)},
        {"input_len", std::to_string(config.input_len)},
        {"cluster_mode", traffic_servers.size() > 1 ? "true" : "false"},
        {"cluster_endpoint_count", std::to_string(traffic_servers.size())},
        {"cluster_peer_endpoint_count", std::to_string(config.peer_servers.size())},
        {"cluster_trace_scope", traffic_servers.size() > 1 ? "local_server_plus_peers" : "local_server"},
        {"cluster_endpoints", [&]() {
           std::ostringstream joined;
           for (std::size_t i = 0; i < traffic_servers.size(); ++i) {
             if (i > 0) {
               joined << ",";
             }
             joined << traffic_servers[i];
           }
           return joined.str();
         }()},
        {"cluster_local_trace_endpoint", server_url},
        {"max_model_len", std::to_string(max_model_len)},
        {"trust_remote_code", config.trust_remote_code ? "true" : "false"},
        {"discard_first_run", config.discard_first_run ? "true" : "false"},
        {"port", std::to_string(config.port)},
        {"tp", std::to_string(config.tp)},
        {"measurement_nvidia_smi_xml", nvidia_smi_snapshot_path.string()},
        {"measurement_gpu_clock_policy", rlprof::render_clock_policy(clock_policy)},
        {"measurement_gpu_clocks_locked", clock_policy.gpu_clocks_locked ? "true" : "false"},
        {"measurement_gpu_clock_policy_query_ok", clock_policy.query_ok ? "true" : "false"},
    };
    if (clock_policy.max_sm_clock_mhz.has_value()) {
      profile.meta["measurement_gpu_max_sm_clock_mhz"] =
          std::to_string(*clock_policy.max_sm_clock_mhz);
    }
    if (clock_policy.locked_sm_clock_mhz.has_value()) {
      profile.meta["measurement_gpu_locked_sm_clock_mhz"] =
          std::to_string(*clock_policy.locked_sm_clock_mhz);
    }
    if (clock_policy.query_ok && !clock_policy.gpu_clocks_locked) {
      profile.meta["warning_gpu_clocks_unlocked"] = "true";
    }
    for (const auto& [key, value] : measurement_metadata(gpu_samples)) {
      profile.meta[key] = value;
    }
    profile.kernels = kernels;
    profile.metrics = metrics;
    profile.metrics_summary = summaries;
    profile.traffic_stats = traffic_stats;

    std::filesystem::path db_path = output_prefix;
    db_path.replace_extension(".db");
    notify_progress(progress, "Saving profile...");
    save_profile(db_path, profile);

    return ProfileRunResult{
        .db_path = db_path,
        .nsys_sqlite_path = sqlite_path,
        .meta = profile.meta,
        .kernels = kernels,
        .metrics = metrics,
        .traffic_stats = traffic_stats,
    };
  } catch (...) {
    stop_flag.store(true);
    if (metrics_thread.joinable()) {
      metrics_thread.join();
    }
    if (gpu_thread.joinable()) {
      gpu_thread.join();
    }
    std::system(("nsys stop --session=" + session_name + " > /dev/null 2>&1").c_str());
    stop_process(server_pid);
    throw;
  }
}

}  // namespace rlprof::profiler
