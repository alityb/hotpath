#include "hotpath/serve_profiler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <sys/ioctl.h>
#include <unistd.h>

#include "hotpath/batch_analyzer.h"
#include "hotpath/cache_analyzer.h"
#include "hotpath/doctor.h"
#include "hotpath/disagg_model.h"
#include "hotpath/kv_config.h"
#include "hotpath/log_parser.h"
#include "hotpath/phase_analyzer.h"
#include "hotpath/prefix_analyzer.h"
#include "hotpath/profiler/attach.h"
#include "hotpath/profiler/categorizer.h"
#include "hotpath/profiler/parser.h"
#include "hotpath/profiler/server.h"
#include "hotpath/profiler/vllm_metrics.h"
#include "hotpath/recommender.h"
#include "hotpath/request_trace.h"
#include "hotpath/sglang_metrics.h"
#include "hotpath/store.h"
#include "hotpath/traffic_replayer.h"
#include "hotpath/workload_classifier.h"

namespace hotpath {
namespace {

std::string trim(std::string value);

struct ServerTimingMeans {
  double ttft_mean_ms = -1.0;
  double queue_mean_ms = -1.0;
  double prefill_mean_ms = -1.0;
  double decode_mean_ms = -1.0;
};

std::string run_cmd(const std::string& cmd) {
  std::array<char, 4096> buf{};
  std::string out;
  FILE* pipe = popen((cmd + " 2>/dev/null").c_str(), "r");
  if (!pipe) return "";
  while (fgets(buf.data(), static_cast<int>(buf.size()), pipe))
    out.append(buf.data());
  pclose(pipe);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r'))
    out.pop_back();
  return out;
}

std::string run_cmd_checked(const std::string& cmd) {
  std::array<char, 4096> buf{};
  std::string out;
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    throw std::runtime_error("failed to run command: " + cmd);
  }
  while (fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
    out.append(buf.data());
  }
  const int rc = pclose(pipe);
  if (rc != 0) {
    throw std::runtime_error("command failed: " + cmd);
  }
  return trim(std::move(out));
}

double now_seconds() {
  using Clock = std::chrono::system_clock;
  const auto now = Clock::now().time_since_epoch();
  return std::chrono::duration<double>(now).count();
}

double percentile_vec(std::vector<double> v, double p) {
  if (v.empty()) return -1.0;  // sentinel: -1 means "no data", not zero latency
  std::sort(v.begin(), v.end());
  const double idx = p / 100.0 * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(idx);
  const size_t hi = std::min(lo + 1, v.size() - 1);
  const double frac = idx - static_cast<double>(lo);
  return v[lo] * (1.0 - frac) + v[hi] * frac;
}

double histogram_counter_mean_ms(
    const std::vector<MetricSample>& metric_samples,
    const std::string& sum_metric,
    const std::string& count_metric) {
  double min_sum = std::numeric_limits<double>::max();
  double max_sum = -1.0;
  double min_count = std::numeric_limits<double>::max();
  double max_count = -1.0;
  for (const auto& sample : metric_samples) {
    if (sample.source != "cluster") continue;
    if (sample.metric == sum_metric) {
      min_sum = std::min(min_sum, sample.value);
      max_sum = std::max(max_sum, sample.value);
    } else if (sample.metric == count_metric) {
      min_count = std::min(min_count, sample.value);
      max_count = std::max(max_count, sample.value);
    }
  }

  const double delta_count = max_count - min_count;
  const double delta_sum = max_sum - min_sum;
  if (delta_count > 0.0 && delta_sum >= 0.0) {
    return (delta_sum / delta_count) * 1000.0;
  }
  if (max_count > 0.0 && max_sum >= 0.0) {
    return (max_sum / max_count) * 1000.0;
  }
  return -1.0;
}

ServerTimingMeans derive_server_timing_means(
    const std::vector<MetricSample>& metric_samples) {
  ServerTimingMeans means;
  means.ttft_mean_ms = histogram_counter_mean_ms(
      metric_samples,
      "vllm:time_to_first_token_seconds_sum",
      "vllm:time_to_first_token_seconds_count");
  means.queue_mean_ms = histogram_counter_mean_ms(
      metric_samples,
      "vllm:request_queue_time_seconds_sum",
      "vllm:request_queue_time_seconds_count");
  means.prefill_mean_ms = histogram_counter_mean_ms(
      metric_samples,
      "vllm:request_prefill_time_seconds_sum",
      "vllm:request_prefill_time_seconds_count");
  means.decode_mean_ms = histogram_counter_mean_ms(
      metric_samples,
      "vllm:request_decode_time_seconds_sum",
      "vllm:request_decode_time_seconds_count");
  if (means.prefill_mean_ms < 0.0 &&
      means.ttft_mean_ms > 0.0 &&
      means.queue_mean_ms >= 0.0) {
    means.prefill_mean_ms = std::max(0.0, means.ttft_mean_ms - means.queue_mean_ms);
  }
  return means;
}

// ── Live dashboard helpers ────────────────────────────────────────────────────

bool dash_use_control() {
  if (!isatty(STDERR_FILENO)) return false;
  const char* term = std::getenv("TERM");
  if (term != nullptr && std::string(term) == "dumb") return false;
  return true;
}

// Returns true when stderr is an interactive terminal with color support.
bool dash_use_color() {
  if (!isatty(STDERR_FILENO)) return false;
  const char* nc = std::getenv("NO_COLOR");
  if (nc != nullptr) return false;
  const char* term = std::getenv("TERM");
  if (term != nullptr && std::string(term) == "dumb") return false;
  return true;
}

int dash_terminal_columns() {
  struct winsize ws {};
  if (ioctl(STDERR_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
    return ws.ws_col;
  }
  return 80;
}

// Renders a progress bar: `fill` out of `total` using block characters.
std::string prog_bar(int fill, int total, int width = 30) {
  const int filled = (total > 0) ? std::clamp(fill * width / total, 0, width) : 0;
  std::string s;
  s.reserve(static_cast<size_t>(width * 3));
  for (int i = 0; i < width; ++i)
    s += (i < filled) ? "\xe2\x96\x88" : "\xe2\x96\x91";  // █ / ░
  return s;
}

// Snapshot of live server metrics for dashboard display only.
// Fetched independently of the main metrics collection thread.
struct DashSnap {
  double batch = 0, queue = 0, cache = -1;
  bool available = false;
};

// Parse a single metric from Prometheus text exposition format.
// Handles labeled metrics like `name{labels} value [timestamp]`.
double parse_prom_metric(const std::string& body, const std::string& name) {
  size_t pos = 0;
  while ((pos = body.find(name, pos)) != std::string::npos) {
    if (pos > 0 && body[pos - 1] != '\n') { pos += name.size(); continue; }
    const auto eol = body.find('\n', pos);
    const std::string ln = (eol == std::string::npos)
        ? body.substr(pos) : body.substr(pos, eol - pos);
    // Value is the last space-separated token; a trailing timestamp is a large int.
    const auto spc = ln.rfind(' ');
    if (spc == std::string::npos) { pos += name.size(); continue; }
    try {
      const double v = std::stod(ln.substr(spc + 1));
      if (v > 1e10) {  // looks like a Unix-ms timestamp — take the token before it
        const auto spc2 = ln.rfind(' ', spc - 1);
        if (spc2 != std::string::npos)
          return std::stod(ln.substr(spc2 + 1, spc - spc2 - 1));
      }
      return v;
    } catch (...) { pos += name.size(); continue; }
  }
  return -1.0;
}

DashSnap fetch_dash_snap(const std::string& endpoint, const std::string& engine) {
  DashSnap snap;
  // Quick, short-timeout fetch — for display only, never stored.
  std::array<char, 32768> buf{};
  std::string body;
  FILE* p = popen(
      ("curl -fsS --max-time 3 '" + endpoint + "/metrics' 2>/dev/null").c_str(), "r");
  if (!p) return snap;
  while (fgets(buf.data(), static_cast<int>(buf.size()), p))
    body.append(buf.data());
  pclose(p);
  if (body.empty()) return snap;

  snap.available = true;
  if (engine == "sglang") {
    const auto m = parse_sglang_metrics(body);
    snap.batch = m.num_running_req;
    snap.queue = m.num_waiting_req;
    if (m.token_usage > 0) snap.cache = m.token_usage * 100.0;
  } else {
    const double b = parse_prom_metric(body, "vllm:num_requests_running");
    const double q = parse_prom_metric(body, "vllm:num_requests_waiting");
    const double c = parse_prom_metric(body, "vllm:gpu_cache_usage_perc");
    if (b >= 0) snap.batch = b;
    if (q >= 0) snap.queue = q;
    if (c >= 0) snap.cache = c;
  }
  return snap;
}

// ── Convert MetricSamples into MetricSnapshots for analyzers ─────────────────
std::vector<MetricSnapshot> samples_to_snapshots(const std::vector<MetricSample>& samples) {
  // Group by sample_time, extract relevant metrics
  std::map<double, MetricSnapshot> by_time;
  for (const auto& s : samples) {
    if (s.source != "cluster") continue;
    auto& snap = by_time[s.sample_time];
    snap.timestamp_us = static_cast<int64_t>(s.sample_time * 1e6);
    if (s.metric == "vllm:num_requests_running") snap.batch_size = s.value;
    else if (s.metric == "vllm:num_requests_waiting") snap.queue_depth = s.value;
    else if (s.metric == "vllm:num_preemptions_total" ||
             s.metric == "vllm:num_preemption_total") snap.preemption_total = s.value;
    else if (s.metric == "vllm:gpu_cache_usage_perc" ||
             s.metric == "vllm:kv_cache_usage_perc" ||  // vLLM 0.19+
             s.metric == "vllm:cpu_cache_usage_perc") {
      snap.cache_usage = std::max(snap.cache_usage, s.value);
    }
  }
  std::vector<MetricSnapshot> result;
  for (auto& [t, snap] : by_time) result.push_back(std::move(snap));
  return result;
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
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) {
    value.pop_back();
  }
  return value;
}

std::string background_command(const std::string& command,
                               const std::filesystem::path& log_path) {
  return "(" + command + ") >> " + shell_escape(log_path.string()) + " 2>&1 & echo $!";
}

std::vector<std::string> read_log_lines_from_offset(
    const std::filesystem::path& path,
    std::uintmax_t offset_bytes) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("cannot open log file: " + path.string());
  }
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (!ec && offset_bytes > 0 && size >= offset_bytes) {
    in.seekg(static_cast<std::streamoff>(offset_bytes), std::ios::beg);
    if (offset_bytes > 0 && in.good()) {
      std::string partial;
      std::getline(in, partial);
    }
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    lines.push_back(line);
  }
  return lines;
}

pid_t start_background_command(const std::string& command,
                               const std::filesystem::path& log_path) {
  const std::string wrapped =
      "bash -lc " + shell_escape(background_command(command, log_path));
  const std::string output = run_cmd_checked(wrapped);
  return static_cast<pid_t>(std::stol(output));
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

struct EndpointParts {
  std::string host;
  int port = 80;
};

std::optional<EndpointParts> parse_endpoint(const std::string& endpoint) {
  static const std::regex re(R"(https?://([^/:]+)(?::([0-9]+))?.*)",
                             std::regex::icase);
  std::smatch match;
  if (!std::regex_match(endpoint, match, re)) {
    return std::nullopt;
  }
  EndpointParts parts;
  parts.host = match[1].str();
  if (match[2].matched) {
    parts.port = std::stoi(match[2].str());
  } else if (endpoint.rfind("https://", 0) == 0) {
    parts.port = 443;
  }
  return parts;
}

bool hosts_equivalent(const std::string& lhs, const std::string& rhs) {
  auto normalize = [](std::string host) {
    std::transform(host.begin(), host.end(), host.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (host == "localhost") host = "127.0.0.1";
    return host;
  };
  return normalize(lhs) == normalize(rhs);
}

bool endpoint_is_local(const std::string& endpoint) {
  const auto parts = parse_endpoint(endpoint);
  if (!parts.has_value()) {
    return false;
  }
  return hosts_equivalent(parts->host, "127.0.0.1") ||
         hosts_equivalent(parts->host, "::1");
}

std::int64_t listening_pid_for_port(std::int64_t port);

std::int64_t listener_pid_for_endpoint(const std::string& endpoint) {
  if (const char* override_pid = std::getenv("HOTPATH_TEST_LISTENER_PID")) {
    try {
      return std::stoll(override_pid);
    } catch (...) {
      return 0;
    }
  }
  const auto parts = parse_endpoint(endpoint);
  if (!parts.has_value()) {
    return 0;
  }
  return listening_pid_for_port(parts->port);
}

std::optional<std::filesystem::path> listener_fd_regular_file(
    std::int64_t pid,
    int fd_num) {
  namespace fs = std::filesystem;
  const char* override = nullptr;
  if (fd_num == 1) {
    override = std::getenv("HOTPATH_TEST_STDOUT_LOG_PATH");
  } else if (fd_num == 2) {
    override = std::getenv("HOTPATH_TEST_STDERR_LOG_PATH");
  }
  if (override != nullptr && std::string(override).size() > 0) {
    fs::path path(override);
    std::error_code ec;
    return fs::is_regular_file(path, ec) && !ec ? std::optional<fs::path>(path) : std::nullopt;
  }

  if (pid <= 0) {
    return std::nullopt;
  }

  const fs::path fd_path =
      fs::path("/proc") / std::to_string(static_cast<long long>(pid)) / "fd" /
      std::to_string(fd_num);
  std::error_code ec;
  if (!fs::exists(fd_path, ec) || ec) {
    return std::nullopt;
  }

  fs::path target = fs::read_symlink(fd_path, ec);
  if (ec) {
    return std::nullopt;
  }

  std::string target_text = target.string();
  const std::string deleted_suffix = " (deleted)";
  if (target_text.size() > deleted_suffix.size() &&
      target_text.compare(target_text.size() - deleted_suffix.size(),
                          deleted_suffix.size(),
                          deleted_suffix) == 0) {
    target_text.erase(target_text.size() - deleted_suffix.size());
    target = fs::path(target_text);
  }

  if (!target.is_absolute()) {
    return std::nullopt;
  }
  return fs::is_regular_file(target, ec) && !ec ? std::optional<fs::path>(target)
                                                : std::nullopt;
}

std::string listener_fd_target_description(std::int64_t pid, int fd_num) {
  namespace fs = std::filesystem;
  if (pid <= 0) {
    return "";
  }
  const fs::path fd_path =
      fs::path("/proc") / std::to_string(static_cast<long long>(pid)) / "fd" /
      std::to_string(fd_num);
  std::error_code ec;
  if (!fs::exists(fd_path, ec) || ec) {
    return "";
  }
  const fs::path target = fs::read_symlink(fd_path, ec);
  return ec ? std::string() : target.string();
}

std::optional<std::string> process_env_value(
    std::int64_t pid,
    const std::string& key) {
  if (key == "VLLM_LOGGING_LEVEL") {
    if (const char* override = std::getenv("HOTPATH_TEST_VLLM_LOGGING_LEVEL")) {
      return std::string(override);
    }
  }

  if (pid <= 0) {
    return std::nullopt;
  }

  const std::filesystem::path environ_path =
      std::filesystem::path("/proc") /
      std::to_string(static_cast<long long>(pid)) / "environ";
  std::ifstream in(environ_path, std::ios::binary);
  if (!in.is_open()) {
    return std::nullopt;
  }
  std::string blob((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  std::size_t start = 0;
  while (start < blob.size()) {
    const std::size_t end = blob.find('\0', start);
    const std::string entry = blob.substr(start, end - start);
    const std::size_t split = entry.find('=');
    if (split != std::string::npos && entry.substr(0, split) == key) {
      return entry.substr(split + 1);
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return std::nullopt;
}

struct ListenerLogInfo {
  std::int64_t pid = 0;
  std::optional<std::filesystem::path> stdout_path;
  std::optional<std::filesystem::path> stderr_path;
  std::string stdout_target;
  std::string stderr_target;
  std::optional<std::string> vllm_logging_level;
};

ListenerLogInfo inspect_listener_logs_for_endpoint(const std::string& endpoint) {
  ListenerLogInfo info;
  info.pid = listener_pid_for_endpoint(endpoint);
  if (info.pid <= 0) {
    return info;
  }
  info.stdout_path = listener_fd_regular_file(info.pid, 1);
  info.stderr_path = listener_fd_regular_file(info.pid, 2);
  info.stdout_target = listener_fd_target_description(info.pid, 1);
  info.stderr_target = listener_fd_target_description(info.pid, 2);
  info.vllm_logging_level = process_env_value(info.pid, "VLLM_LOGGING_LEVEL");
  return info;
}

std::optional<std::int64_t> extract_apiserver_pid_from_log(
    const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    return std::nullopt;
  }

  static const std::regex apiserver_pid_re(R"(\(APIServer pid=([0-9]+)\))");
  std::optional<std::int64_t> pid;
  std::string line;
  while (std::getline(in, line)) {
    std::smatch match;
    if (std::regex_search(line, match, apiserver_pid_re)) {
      pid = std::stoll(match[1].str());
    }
  }
  return pid;
}

std::optional<profiler::ManagedServerState> managed_server_for_endpoint(
    const std::string& endpoint) {
  const auto target = parse_endpoint(endpoint);
  if (!target.has_value()) return std::nullopt;
  for (const auto& server : profiler::list_managed_servers()) {
    const auto candidate = parse_endpoint(server.server_url);
    if (!candidate.has_value()) continue;
    if (candidate->port == target->port &&
        hosts_equivalent(candidate->host, target->host)) {
      return server;
    }
  }
  return std::nullopt;
}

std::filesystem::path nsys_trace_prefix_for_output(const std::filesystem::path& output_dir) {
  return output_dir / "serve_profile";
}

std::string make_nsys_session_name(const std::filesystem::path& output_prefix) {
  std::string session = "hotpath_";
  const std::string stem = output_prefix.filename().string();
  for (unsigned char ch : stem) {
    session.push_back(std::isalnum(ch) ? static_cast<char>(ch) : '_');
  }
  return session;
}

bool profile_help_mentions_pid_attach(const std::string& nsys_path) {
  try {
    const std::string help =
        run_cmd_checked(shell_escape(nsys_path) + " profile --help");
    return help.find("--pid") != std::string::npos;
  } catch (...) {
    return false;
  }
}

std::string attach_profile_command(
    const std::string& nsys_path,
    const std::filesystem::path& output_prefix,
    const std::string& session_name,
    std::int64_t attach_pid) {
  return shell_escape(nsys_path) +
         " profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none "
         "--export=sqlite --force-overwrite=true --session-new " +
         shell_escape(session_name) + " --start-later=true --wait=all --pid " +
         std::to_string(attach_pid) + " -o " + shell_escape(output_prefix.string());
}

std::int64_t listening_pid_for_port(std::int64_t port) {
  const std::vector<std::string> commands = {
      "bash -lc \"lsof -ti tcp:" + std::to_string(port) +
          " -sTCP:LISTEN 2>/dev/null | head -n 1\"",
      "bash -lc \"ss -ltnp '( sport = :" + std::to_string(port) +
          " )' 2>/dev/null | tail -n +2 | head -n 1\"",
  };
  for (std::size_t i = 0; i < commands.size(); ++i) {
    try {
      const std::string output = trim(run_cmd_checked(commands[i]));
      if (output.empty()) {
        continue;
      }
      if (i == 0) {
        return std::stoll(output);
      }
      std::smatch match;
      if (std::regex_search(output, match, std::regex(R"(pid=([0-9]+))"))) {
        return std::stoll(match[1].str());
      }
    } catch (const std::exception&) {
      continue;
    }
  }
  return 0;
}

bool server_ready(const std::string& endpoint) {
  const std::string command =
      "curl -fsS --connect-timeout 1 --max-time 1 " +
      shell_escape(endpoint + "/health") + " >/dev/null 2>&1";
  return std::system(command.c_str()) == 0;
}

bool wait_for_external_endpoint_ready(
    const std::string& endpoint,
    std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (server_ready(endpoint)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
  return server_ready(endpoint);
}

void wait_for_server_ready_or_throw(
    pid_t server_pid,
    const std::string& endpoint,
    std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (server_ready(endpoint)) {
      return;
    }
    if (server_pid > 0 && !process_alive(server_pid)) {
      throw std::runtime_error("traced vLLM server exited before becoming ready: " + endpoint);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
  throw std::runtime_error("timed out waiting for traced vLLM server: " + endpoint);
}

void wait_for_server_stopped(
    pid_t server_pid,
    const std::string& endpoint,
    std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    const bool pid_gone = server_pid <= 0 || !process_alive(server_pid);
    if (pid_gone && !server_ready(endpoint)) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
  throw std::runtime_error("timed out waiting for source server to stop: " + endpoint);
}

void restore_attach_source_or_throw(
    const profiler::AttachClonePlan& plan,
    const std::filesystem::path& log_path,
    std::chrono::seconds timeout) {
  if (plan.mode != "replace_restore" || plan.restore_command.empty()) {
    return;
  }
  const pid_t restore_pid = start_background_command(plan.restore_command, log_path);
  wait_for_server_ready_or_throw(restore_pid, plan.source_server_url, timeout);
}

bool nsys_sqlite_ready(const std::filesystem::path& path) {
  const std::string cmd =
      "sqlite3 " + shell_escape(path.string()) +
      " \"SELECT 1 FROM sqlite_master WHERE type='table' "
      "AND name='CUPTI_ACTIVITY_KIND_KERNEL' LIMIT 1;\"";
  return trim(run_cmd(cmd)) == "1";
}

bool wait_for_nsys_sqlite(const std::filesystem::path& path, std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (std::filesystem::exists(path) && nsys_sqlite_ready(path)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  return std::filesystem::exists(path) && nsys_sqlite_ready(path);
}

std::vector<MetricSample> fetch_sglang_metrics_once(const std::string& endpoint) {
  const std::string body = run_cmd("curl -fsS --max-time 5 '" + endpoint + "/metrics'");
  if (body.empty()) return {};

  const SglangMetrics metrics = parse_sglang_metrics(body);
  const double sample_time = now_seconds();
  std::vector<MetricSample> samples;
  samples.push_back(MetricSample{
      .sample_time = sample_time,
      .source = "cluster",
      .metric = "vllm:num_requests_running",
      .value = metrics.num_running_req,
  });
  samples.push_back(MetricSample{
      .sample_time = sample_time,
      .source = "cluster",
      .metric = "vllm:num_requests_waiting",
      .value = metrics.num_waiting_req,
  });
  samples.push_back(MetricSample{
      .sample_time = sample_time,
      .source = "cluster",
      .metric = "vllm:gpu_cache_usage_perc",
      .value = metrics.token_usage * 100.0,
  });
  samples.push_back(MetricSample{
      .sample_time = sample_time,
      .source = "cluster",
      .metric = "vllm:prefix_cache_hit_rate",
      .value = metrics.cache_hit_rate,
  });
  samples.push_back(MetricSample{
      .sample_time = sample_time,
      .source = "cluster",
      .metric = "vllm:num_preemptions_total",
      .value = 0.0,
  });
  return samples;
}

// Build request traces from replay results
std::vector<RequestTrace> results_to_traces(const std::vector<ReplayResult>& results,
                                            const std::vector<ReplayRequest>& requests) {
  std::vector<RequestTrace> traces;
  for (size_t i = 0; i < results.size(); ++i) {
    const auto& r = results[i];
    RequestTrace t;
    t.request_id = r.request_id;
    t.arrival_us = r.send_us;
    t.queue_start_us = r.send_us;
    t.prefill_start_us = r.send_us;
    t.prefill_end_us = r.first_token_us;
    t.first_token_us = r.first_token_us;
    t.last_token_us = r.completion_us;
    t.completion_us = r.completion_us;
    if (r.prompt_tokens > 0) {
      t.prompt_tokens = r.prompt_tokens;
      t.prompt_tokens_estimated = r.prompt_tokens_estimated;
    } else {
      t.prompt_tokens =
          (i < requests.size()) ? static_cast<int>(requests[i].prompt.size() / 4) : 0;
      t.prompt_tokens_estimated = true;
    }
    t.output_tokens = r.completion_tokens > 0 ? r.completion_tokens : r.tokens_generated;
    t.cached_tokens = 0;
    t.status = r.success ? "ok" : "error";
    t.prompt_text = (i < requests.size()) ? requests[i].prompt : "";
    if (!r.external_request_id.empty()) {
      t.events.push_back(RequestEvent{
          .event_type = "client_request_id",
          .timestamp_us = r.send_us,
          .detail = "{\"external_request_id\":\"" + r.external_request_id + "\"}",
      });
    }
    traces.push_back(std::move(t));
  }
  return traces;
}

void save_kv(std::map<std::string, std::string>& kv,
             const std::string& prefix, const std::string& key, double v) {
  kv[prefix + "." + key] = std::to_string(v);
}

void save_kv(std::map<std::string, std::string>& kv,
             const std::string& prefix, const std::string& key, int v) {
  kv[prefix + "." + key] = std::to_string(v);
}

void save_kv(std::map<std::string, std::string>& kv,
             const std::string& prefix, const std::string& key, bool v) {
  kv[prefix + "." + key] = v ? "true" : "false";
}

void save_kv(std::map<std::string, std::string>& kv,
             const std::string& prefix, const std::string& key, const std::string& v) {
  kv[prefix + "." + key] = v;
}

std::vector<int> prompt_to_char_ids(const std::string& prompt) {
  std::vector<int> ids;
  ids.reserve(prompt.size());
  for (unsigned char ch : prompt) ids.push_back(static_cast<int>(ch));
  return ids;
}

std::map<std::string, RequestTrace> traces_by_request_id(
    const std::vector<RequestTrace>& traces) {
  std::map<std::string, RequestTrace> by_request_id;
  for (const auto& trace : traces) {
    if (!trace.request_id.empty()) {
      by_request_id[trace.request_id] = trace;
    }
  }
  return by_request_id;
}

bool traces_use_anonymous_v1_ids(const std::vector<RequestTrace>& traces) {
  if (traces.empty()) return false;
  return std::all_of(traces.begin(), traces.end(), [](const RequestTrace& trace) {
    return trace.request_id.rfind("v1-anon-", 0) == 0;
  });
}

bool sane_server_trace(const RequestTrace& trace) {
  if (trace.queue_start_us <= 0 || trace.prefill_start_us <= 0 ||
      trace.prefill_end_us <= 0) {
    return false;
  }
  return trace.queue_start_us <= trace.prefill_start_us &&
         trace.prefill_start_us <= trace.prefill_end_us;
}

void apply_server_trace(RequestTrace& client, const RequestTrace& server) {
  if (server.server_timing_available && sane_server_trace(server)) {
    client.queue_start_us = server.queue_start_us;
    client.prefill_start_us = server.prefill_start_us;
    client.prefill_end_us = server.prefill_end_us;
    if (server.server_last_token_us > 0) {
      client.server_last_token_us = server.server_last_token_us;
    }
    client.server_timing_available = true;
  }
  if (server.prompt_tokens > 0) {
    client.prompt_tokens = server.prompt_tokens;
    client.prompt_tokens_estimated = false;
  }
  if (server.output_tokens > 0 && client.output_tokens == 0) {
    client.output_tokens = server.output_tokens;
  }
  if (server.cached_tokens > 0) {
    client.cached_tokens = server.cached_tokens;
  }
  if (!server.events.empty()) {
    client.events.insert(client.events.end(), server.events.begin(), server.events.end());
  }
}

bool refine_order_matched_v1_timing(
    std::vector<RequestTrace>& traces,
    const ServerTraceCorrelationResult& trace_correlation,
    const ServerTimingMeans& means) {
  if (trace_correlation.method != ServerTraceMatchMethod::ORDER &&
      trace_correlation.method != ServerTraceMatchMethod::ID) {
    return false;
  }

  const double queue_mean_us = means.queue_mean_ms > 0.0 ? means.queue_mean_ms * 1000.0 : 0.0;
  const double prefill_mean_us =
      means.prefill_mean_ms > 0.0 ? means.prefill_mean_ms * 1000.0 : 0.0;
  const double decode_mean_us =
      means.decode_mean_ms > 0.0 ? means.decode_mean_ms * 1000.0 : 0.0;
  const double split_total_us = queue_mean_us + prefill_mean_us;
  if (split_total_us <= 0.0) {
    return false;
  }

  double client_decode_mean_us = -1.0;
  {
    double total_client_decode_us = 0.0;
    int client_decode_samples = 0;
    for (const auto& trace : traces) {
      if (trace.last_token_us > trace.first_token_us) {
        total_client_decode_us += static_cast<double>(trace.last_token_us - trace.first_token_us);
        ++client_decode_samples;
      }
    }
    if (client_decode_samples > 0) {
      client_decode_mean_us =
          total_client_decode_us / static_cast<double>(client_decode_samples);
    }
  }

  bool refined_any = false;
  for (auto& trace : traces) {
    const bool second_quantized =
        trace.queue_start_us > 0 &&
        trace.prefill_start_us > 0 &&
        trace.prefill_end_us > 0 &&
        trace.queue_start_us % 1000000 == 0 &&
        trace.prefill_start_us % 1000000 == 0 &&
        trace.prefill_end_us % 1000000 == 0 &&
        (trace.server_last_token_us <= 0 || trace.server_last_token_us % 1000000 == 0);
    const bool should_refine =
        trace.server_timing_available &&
        trace.arrival_us > 0 &&
        (trace_correlation.method == ServerTraceMatchMethod::ORDER || second_quantized);
    if (!should_refine) {
      continue;
    }

    std::int64_t decode_us_i = 0;
    if (trace.server_last_token_us > 0 &&
        trace.prefill_end_us > 0 &&
        trace.server_last_token_us >= trace.prefill_end_us) {
      decode_us_i = trace.server_last_token_us - trace.prefill_end_us;
    }

    double target_total_us = 0.0;
    if (trace.first_token_us > trace.arrival_us) {
      target_total_us = static_cast<double>(trace.first_token_us - trace.arrival_us);
    } else if (means.ttft_mean_ms > 0.0) {
      target_total_us = means.ttft_mean_ms * 1000.0;
    } else {
      target_total_us = split_total_us;
    }
    if (target_total_us <= 0.0) {
      continue;
    }

    double queue_us = 0.0;
    double prefill_us = 0.0;
    if (queue_mean_us > 0.0 && prefill_mean_us > 0.0) {
      const double scale = target_total_us / split_total_us;
      queue_us = queue_mean_us * scale;
      prefill_us = prefill_mean_us * scale;
    } else if (queue_mean_us > 0.0) {
      queue_us = std::min(queue_mean_us, target_total_us);
      prefill_us = std::max(0.0, target_total_us - queue_us);
    } else {
      prefill_us = std::min(prefill_mean_us, target_total_us);
      queue_us = std::max(0.0, target_total_us - prefill_us);
    }

    std::int64_t queue_us_i = static_cast<std::int64_t>(std::llround(queue_us));
    std::int64_t prefill_us_i = static_cast<std::int64_t>(std::llround(prefill_us));
    const std::int64_t total_us_i = static_cast<std::int64_t>(std::llround(target_total_us));
    if (queue_us_i + prefill_us_i != total_us_i) {
      prefill_us_i = std::max<std::int64_t>(0, total_us_i - queue_us_i);
    }

    trace.queue_start_us = trace.arrival_us;
    trace.prefill_start_us = trace.queue_start_us + queue_us_i;
    trace.prefill_end_us = trace.prefill_start_us + prefill_us_i;
    if (trace.first_token_us > 0) {
      trace.prefill_end_us = std::min(trace.prefill_end_us, trace.first_token_us);
      trace.prefill_start_us = std::min(trace.prefill_start_us, trace.prefill_end_us);
    }
    if (decode_mean_us > 0.0) {
      double target_decode_us = decode_mean_us;
      if (client_decode_mean_us > 0.0 && trace.last_token_us > trace.first_token_us) {
        const double client_decode_us =
            static_cast<double>(trace.last_token_us - trace.first_token_us);
        target_decode_us = client_decode_us * (decode_mean_us / client_decode_mean_us);
      }
      decode_us_i = static_cast<std::int64_t>(std::llround(target_decode_us));
    }

    trace.server_last_token_us = trace.prefill_end_us + decode_us_i;
    if (trace.server_last_token_us < trace.prefill_end_us) {
      trace.server_last_token_us = trace.prefill_end_us;
    }
    trace.server_timing_available = sane_server_trace(trace);
    refined_any = refined_any || trace.server_timing_available;
  }
  return refined_any;
}

std::int64_t server_trace_anchor_us(const RequestTrace& trace) {
  if (trace.queue_start_us > 0) return trace.queue_start_us;
  if (trace.arrival_us > 0) return trace.arrival_us;
  if (!trace.events.empty()) return trace.events.front().timestamp_us;
  return 0;
}

std::vector<KernelEntry> load_phase_kernels(const std::filesystem::path& sqlite_path) {
  std::vector<KernelEntry> kernels;
  for (const auto& event : profiler::parse_nsys_kernel_trace(sqlite_path)) {
    kernels.push_back(KernelEntry{
        .name = event.name,
        .phase = profiler::classify_phase(event.runtime_name, event.grid),
        .start_us = event.start_us,
        .duration_us = event.duration_us,
    });
  }
  return kernels;
}

}  // namespace

ServerTraceCorrelationResult correlate_server_traces(
    std::vector<RequestTrace>& client_traces,
    const std::vector<RequestTrace>& server_traces,
    bool allow_timestamp_fallback,
    std::int64_t max_timestamp_offset_us) {
  ServerTraceCorrelationResult result;
  result.total_requests = static_cast<int>(client_traces.size());

  std::map<std::string, RequestTrace> by_id = traces_by_request_id(server_traces);
  for (auto& client : client_traces) {
    const auto it = by_id.find(client.request_id);
    if (it == by_id.end()) {
      continue;
    }
    apply_server_trace(client, it->second);
    ++result.matched_requests;
  }
  if (result.matched_requests > 0 || !allow_timestamp_fallback) {
    result.method = result.matched_requests > 0 ? ServerTraceMatchMethod::ID
                                                : ServerTraceMatchMethod::NONE;
    return result;
  }

  std::vector<std::pair<std::size_t, std::int64_t>> client_order;
  client_order.reserve(client_traces.size());
  for (std::size_t i = 0; i < client_traces.size(); ++i) {
    if (client_traces[i].arrival_us > 0) {
      client_order.emplace_back(i, client_traces[i].arrival_us);
    }
  }
  std::sort(client_order.begin(), client_order.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  std::vector<std::pair<std::size_t, std::int64_t>> server_order;
  server_order.reserve(server_traces.size());
  for (std::size_t i = 0; i < server_traces.size(); ++i) {
    const auto anchor = server_trace_anchor_us(server_traces[i]);
    if (anchor > 0) {
      server_order.emplace_back(i, anchor);
    }
  }
  std::sort(server_order.begin(), server_order.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  std::vector<bool> server_used(server_traces.size(), false);
  for (const auto& [client_index, client_ts] : client_order) {
    std::optional<std::size_t> best_server_index;
    std::int64_t best_offset = max_timestamp_offset_us + 1;
    for (const auto& [server_index, server_ts] : server_order) {
      if (server_used[server_index]) continue;
      const std::int64_t offset = std::llabs(server_ts - client_ts);
      if (offset <= max_timestamp_offset_us && offset < best_offset) {
        best_offset = offset;
        best_server_index = server_index;
      }
    }
    if (!best_server_index.has_value()) {
      continue;
    }
    server_used[*best_server_index] = true;
    apply_server_trace(client_traces[client_index], server_traces[*best_server_index]);
    ++result.matched_requests;
    result.max_offset_us = std::max(result.max_offset_us, best_offset);
  }

  result.method = result.matched_requests > 0 ? ServerTraceMatchMethod::TIMESTAMP
                                              : ServerTraceMatchMethod::NONE;
  if (result.matched_requests == 0 &&
      traces_use_anonymous_v1_ids(server_traces) &&
      server_traces.size() >= client_traces.size() &&
      !client_traces.empty()) {
    std::vector<std::pair<std::size_t, std::int64_t>> client_order;
    client_order.reserve(client_traces.size());
    for (std::size_t i = 0; i < client_traces.size(); ++i) {
      const std::int64_t anchor = client_traces[i].arrival_us > 0
          ? client_traces[i].arrival_us
          : client_traces[i].completion_us;
      client_order.emplace_back(i, anchor);
    }
    std::stable_sort(client_order.begin(), client_order.end(),
                     [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<std::pair<std::size_t, std::int64_t>> server_order;
    server_order.reserve(server_traces.size());
    for (std::size_t i = 0; i < server_traces.size(); ++i) {
      server_order.emplace_back(i, server_trace_anchor_us(server_traces[i]));
    }
    std::stable_sort(server_order.begin(), server_order.end(),
                     [](const auto& a, const auto& b) { return a.second < b.second; });

    const std::size_t server_start =
        server_order.size() > client_order.size()
            ? server_order.size() - client_order.size()
            : 0;

    for (std::size_t i = 0; i < client_order.size(); ++i) {
      apply_server_trace(client_traces[client_order[i].first],
                         server_traces[server_order[server_start + i].first]);
      ++result.matched_requests;
    }
    result.method = ServerTraceMatchMethod::ORDER;
    result.max_offset_us = 0;
  }
  return result;
}

std::optional<std::filesystem::path> discover_server_log_path(
    const ServeProfileOptions& opts) {
  namespace fs = std::filesystem;
  if (opts.engine != "vllm" || !endpoint_is_local(opts.endpoint)) {
    return std::nullopt;
  }

  const ListenerLogInfo listener_log_info = inspect_listener_logs_for_endpoint(opts.endpoint);
  if (listener_log_info.pid > 0) {
    if (listener_log_info.stdout_path.has_value()) {
      return listener_log_info.stdout_path;
    }
    if (listener_log_info.stderr_path.has_value()) {
      return listener_log_info.stderr_path;
    }
    // We resolved the live listener, and it does not write stdout/stderr to a
    // regular file. Avoid guessing nearby stale files from previous runs.
    return std::nullopt;
  }

  const fs::path output_dir =
      opts.output.empty() ? fs::path(".hotpath/serve_run") : fs::path(opts.output);
  const fs::path output_parent =
      output_dir.has_parent_path() ? output_dir.parent_path() : fs::path(".");

  std::vector<fs::path> candidates = {
      output_parent / "video-server" / "vllm.stderr.log",
      output_parent / "video-server" / "vllm.stdout.log",
      output_parent / "video-server" / "vllm.log",
      output_parent / "video-server" / "server.log",
      output_dir / "vllm.stderr.log",
      output_dir / "vllm.stdout.log",
      output_dir / "vllm.log",
      fs::path(".hotpath/video-server/vllm.stderr.log"),
      fs::path(".hotpath/video-server/vllm.stdout.log"),
      fs::path(".hotpath/video-server/vllm.log"),
      fs::path("vllm.stdout.log"),
      fs::path("vllm.stderr.log"),
      fs::path("vllm.log"),
  };
  candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

  std::optional<fs::path> best_path;
  std::optional<fs::file_time_type> best_mtime;
  for (const fs::path& candidate : candidates) {
    std::error_code ec;
    if (!fs::is_regular_file(candidate, ec)) {
      continue;
    }
    const auto size = fs::file_size(candidate, ec);
    if (ec || size == 0) {
      continue;
    }
    const fs::file_time_type mtime = fs::last_write_time(candidate, ec);
    if (ec) {
      continue;
    }
    if (!best_path.has_value() || mtime > *best_mtime) {
      best_path = candidate;
      best_mtime = mtime;
    }
  }
  return best_path;
}

GpuInfo detect_gpus() {
  GpuInfo info;
  const std::string count_str = run_cmd("nvidia-smi --query-gpu=count --format=csv,noheader -i 0");
  if (!count_str.empty()) {
    try { info.count = std::stoi(count_str); } catch (...) {}
  }
  if (info.count <= 0) {
    // Fallback: count lines
    const std::string list = run_cmd("nvidia-smi --query-gpu=name --format=csv,noheader");
    if (!list.empty()) {
      info.count = 1;
      for (char c : list) if (c == '\n') info.count++;
    }
  }
  info.name = run_cmd("nvidia-smi --query-gpu=name --format=csv,noheader -i 0");
  const std::string mem = run_cmd("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0");
  if (!mem.empty()) {
    try { info.memory_mb = std::stoi(mem); } catch (...) {}
  }
  return info;
}

int run_serve_profile(const ServeProfileOptions& opts) {
  namespace fs = std::filesystem;
  const std::chrono::seconds startup_timeout(300);

  // ── Step 1: Detect GPUs ──
  const auto gpu = detect_gpus();
  if (gpu.count == 0) {
    std::cerr << "error: no NVIDIA GPU detected\n";
    return 1;
  }
  std::cerr << "GPU: " << gpu.count << "x " << gpu.name
            << " (" << gpu.memory_mb << " MB)\n";

  // ── Step 2: Load traffic ──
  std::vector<ReplayRequest> requests;
  if (!opts.traffic_path.empty()) {
    if (!fs::exists(opts.traffic_path)) {
      std::cerr << "error: traffic file not found: " << opts.traffic_path << "\n";
      return 1;
    }
    requests = load_jsonl(opts.traffic_path);
    std::cerr << "Loaded " << requests.size() << " requests from " << opts.traffic_path << "\n";
  }

  // ── Step 3: Create output ──
  const fs::path output_dir(opts.output);
  fs::create_directories(output_dir);
  const fs::path db_path = output_dir / "serve_profile.db";
  if (fs::exists(db_path)) {
    fs::remove(db_path);
  }
  init_db(db_path);

  std::optional<profiler::ManagedServerState> managed_server;
  struct ManagedServerGuard {
    std::optional<profiler::ManagedServerState>* state = nullptr;
    bool owned = false;
    ~ManagedServerGuard() {
      if (!owned || state == nullptr || !state->has_value()) {
        return;
      }
      try {
        profiler::stop_managed_server(**state);
      } catch (...) {
      }
    }
  } managed_server_guard{.state = &managed_server, .owned = false};

  std::string profile_endpoint = opts.endpoint;
  std::string model = opts.model;
  if (opts.launch_managed_server) {
    if (opts.engine != "vllm") {
      std::cerr << "error: --model managed mode currently supports vllm only\n";
      return 1;
    }
    if (model.empty()) {
      std::cerr << "error: --model requires a model name\n";
      return 1;
    }
    const auto unique_name =
        "serve_profile_" + std::to_string(getpid()) + "_" +
        std::to_string(static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()));
    profiler::ManagedServerConfig managed_config;
    managed_config.name = unique_name;
    managed_config.model = model;
    managed_config.port = 8000;
    managed_config.max_model_len = 2048;
    managed_config.wrap_with_nsys = opts.use_nsys;
    managed_config.debug_logging = true;
    managed_config.startup_timeout_s = startup_timeout.count();
    try {
      managed_server = profiler::start_managed_server(managed_config);
      managed_server_guard.owned = true;
      profile_endpoint = managed_server->server_url;
      std::cerr << "Started managed vLLM server at " << profile_endpoint << "\n";
    } catch (const std::exception& e) {
      std::cerr << "error: failed to start managed vLLM server: " << e.what() << "\n";
      return 1;
    }
  } else {
    // ── Step 4: Validate external endpoint ──
    std::cerr << "Checking " << opts.endpoint << " ...\n";
    constexpr auto kExternalEndpointGrace = std::chrono::seconds(20);
    if (!wait_for_external_endpoint_ready(opts.endpoint, kExternalEndpointGrace)) {
      std::cerr << "error: endpoint " << opts.endpoint << " is not reachable\n";
      std::cerr << "Start your " << opts.engine << " server first:\n";
      if (opts.engine == "sglang") {
        std::cerr << "  python -m sglang.launch_server --model-path <model> --port 8000\n";
      } else {
        std::cerr << "  vllm serve <model> --port 8000\n";
        std::cerr << "  or ./examples/start_qwen35_video_server.sh\n";
      }
      return 1;
    }
    if (opts.engine == "vllm") {
      managed_server = managed_server_for_endpoint(opts.endpoint);
    }
    if (model.empty()) {
      model = detect_model(opts.endpoint);
    }
  }
  if (model.empty()) model = "unknown";
  std::cerr << "Model: " << model << "\n";
  const ListenerLogInfo listener_log_info = inspect_listener_logs_for_endpoint(profile_endpoint);

  std::optional<fs::path> server_log_path;
  if (!opts.server_log_path.empty()) {
    server_log_path = fs::path(opts.server_log_path);
  } else if (managed_server.has_value()) {
    server_log_path = managed_server->log_path;
  } else if (auto discovered = discover_server_log_path(opts); discovered.has_value()) {
    server_log_path = *discovered;
    std::cerr << "Auto-discovered vLLM server log: " << server_log_path->string() << "\n";
  }
  if (opts.engine == "vllm" && !server_log_path.has_value()) {
    std::cerr << "Queue wait timing requires access to vLLM server logs.\n";
    std::cerr << "Set VLLM_LOGGING_LEVEL=DEBUG and pass --server-log <path> to enable server-side timing.\n";
    std::cerr << "Demo note: examples/start_qwen35_video_server.sh writes logs to .hotpath/video-server/vllm.stdout.log and .hotpath/video-server/vllm.stderr.log.\n";
    if (listener_log_info.pid > 0) {
      const bool file_backed =
          listener_log_info.stdout_path.has_value() || listener_log_info.stderr_path.has_value();
      if (!file_backed) {
        std::cerr << "Local vLLM listener PID " << listener_log_info.pid
                  << " is not writing stdout/stderr to a regular file.\n";
        if (!listener_log_info.stdout_target.empty()) {
          std::cerr << "  stdout -> " << listener_log_info.stdout_target << "\n";
        }
        if (!listener_log_info.stderr_target.empty() &&
            listener_log_info.stderr_target != listener_log_info.stdout_target) {
          std::cerr << "  stderr -> " << listener_log_info.stderr_target << "\n";
        }
      }
      if (listener_log_info.vllm_logging_level.has_value()) {
        std::string level = *listener_log_info.vllm_logging_level;
        std::transform(level.begin(), level.end(), level.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
        if (level != "DEBUG") {
          std::cerr << "VLLM_LOGGING_LEVEL for listener PID " << listener_log_info.pid
                    << " is " << *listener_log_info.vllm_logging_level
                    << "; per-request DEBUG timing lines will not be emitted.\n";
        }
      } else if (listener_log_info.pid > 0) {
        std::cerr << "VLLM_LOGGING_LEVEL is not set for listener PID "
                  << listener_log_info.pid
                  << "; per-request DEBUG timing lines may be unavailable.\n";
      }
    }
  } else if (opts.engine == "vllm" &&
             listener_log_info.pid > 0 &&
             listener_log_info.vllm_logging_level.has_value()) {
    std::string level = *listener_log_info.vllm_logging_level;
    std::transform(level.begin(), level.end(), level.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    if (level != "DEBUG") {
      std::cerr << "warning: listener PID " << listener_log_info.pid
                << " writes logs to " << server_log_path->string()
                << " but VLLM_LOGGING_LEVEL=" << *listener_log_info.vllm_logging_level
                << ", so per-request DEBUG timing may still be unavailable.\n";
    }
  }

  std::uintmax_t server_log_start_offset = 0;
  if (opts.engine == "vllm" && server_log_path.has_value()) {
    std::error_code ec;
    if (fs::exists(*server_log_path, ec) && !ec) {
      server_log_start_offset = fs::file_size(*server_log_path, ec);
      if (ec) {
        server_log_start_offset = 0;
      }
    }
  }

  std::optional<fs::path> nsys_sqlite_path;
  std::optional<profiler::AttachClonePlan> attach_clone_plan;
  bool nsys_started = false;
  bool nsys_started_later = false;
  pid_t attach_profiler_pid = -1;
  pid_t traced_server_pid = -1;
  std::string nsys_session_name;
  fs::path nsys_log_path = output_dir / "serve_profile_nsys.log";
  if (opts.use_nsys) {
    if (managed_server.has_value()) {
      const fs::path trace_prefix = nsys_trace_prefix_for_output(output_dir);
      nsys_session_name = managed_server->session_name;
      std::string start_command =
          "nsys start --session=" + shell_escape(nsys_session_name) +
          " --sample=none --cpuctxsw=none --output=" + shell_escape(trace_prefix.string()) +
          " --export=sqlite --force-overwrite=true";
      const int start_rc = std::system(start_command.c_str());
      if (start_rc == 0) {
        nsys_started = true;
        nsys_sqlite_path = trace_prefix;
        nsys_sqlite_path->replace_extension(".sqlite");
        std::cerr << "Started nsys capture for session " << nsys_session_name << "\n";
      } else {
        std::cerr << "warning: failed to start nsys capture for session "
                  << nsys_session_name << "\n";
      }
    } else if (opts.engine == "vllm") {
      try {
        const RuntimeEnvironmentInfo environment = inspect_runtime_environment();
        if (!environment.nsys.found || environment.nsys.ambiguous) {
          throw std::runtime_error("nsys unavailable: " + environment.nsys.detail);
        }
        const fs::path trace_prefix = nsys_trace_prefix_for_output(output_dir);
        nsys_sqlite_path = trace_prefix;
        nsys_sqlite_path->replace_extension(".sqlite");
        nsys_session_name = make_nsys_session_name(trace_prefix);

        std::int64_t attach_pid = opts.server_pid;
        if (attach_pid <= 0) {
          if (!endpoint_is_local(opts.endpoint)) {
            throw std::runtime_error(
                "cannot infer a local vLLM PID for a non-local endpoint; pass --server-pid");
          }
          const auto endpoint_parts = parse_endpoint(opts.endpoint);
          if (!endpoint_parts.has_value()) {
            throw std::runtime_error("failed to parse endpoint for PID lookup: " + opts.endpoint);
          }
          attach_pid = listening_pid_for_port(endpoint_parts->port);
        }
        if (attach_pid <= 0) {
          throw std::runtime_error(
              "failed to find a listening vLLM PID for " + opts.endpoint +
              "; pass --server-pid explicitly");
        }

        const bool nsys_supports_pid_attach =
            profile_help_mentions_pid_attach(environment.nsys.resolved_path);
        if (nsys_supports_pid_attach) {
          const std::string attach_command = attach_profile_command(
              environment.nsys.resolved_path, trace_prefix, nsys_session_name, attach_pid);
          attach_profiler_pid = start_background_command(attach_command, nsys_log_path);
        } else {
          attach_clone_plan = profiler::build_attach_clone_plan(
              attach_pid,
              opts.endpoint,
              environment.nsys.resolved_path,
              nsys_session_name,
              trace_prefix);
          if (!attach_clone_plan.has_value()) {
            throw std::runtime_error(
                "installed nsys does not support --pid attach and no local clone plan could be "
                "constructed for " +
                opts.endpoint);
          }
          if (attach_clone_plan->mode == "replace_restore") {
            stop_process(static_cast<pid_t>(attach_clone_plan->source_pid));
            wait_for_server_stopped(
                static_cast<pid_t>(attach_clone_plan->source_pid),
                attach_clone_plan->source_server_url,
                startup_timeout);
          }
          traced_server_pid =
              start_background_command(attach_clone_plan->launch_command, nsys_log_path);
          profile_endpoint = attach_clone_plan->traced_server_url;
          wait_for_server_ready_or_throw(traced_server_pid, profile_endpoint, startup_timeout);
        }

        std::string start_command =
            shell_escape(environment.nsys.resolved_path) +
            " start --session=" + shell_escape(nsys_session_name) +
            " --sample=none --cpuctxsw=none --output=" + shell_escape(trace_prefix.string()) +
            " --export=sqlite --force-overwrite=true";
        const int start_rc = std::system(start_command.c_str());
        if (start_rc == 0) {
          nsys_started = true;
          nsys_started_later = true;
          std::cerr << "Started nsys capture for external vLLM PID " << attach_pid << "\n";
        } else {
          throw std::runtime_error("failed to start nsys session " + nsys_session_name);
        }
      } catch (const std::exception& e) {
        std::cerr << "warning: failed to enable nsys capture: " << e.what() << "\n";
        std::cerr << "GPU phase profiling requires either:\n"
                  << "  (a) nsys with --pid support (nsys >= 2024.1), or\n"
                  << "  (b) letting hotpath manage the server:\n"
                  << "      hotpath serve-profile --model " << model
                  << " --nsys --duration 30 --traffic <file> --output <dir>\n"
                  << "Current nsys does not support --pid attach. Skipping GPU phase.\n";
        if (traced_server_pid > 0) {
          stop_process(traced_server_pid);
          traced_server_pid = -1;
        }
        if (attach_profiler_pid > 0) {
          stop_process(attach_profiler_pid);
          attach_profiler_pid = -1;
        }
        nsys_sqlite_path.reset();
        attach_clone_plan.reset();
        profile_endpoint = opts.endpoint;
      }
    } else {
      std::cerr << "warning: --nsys external attach currently supports vLLM endpoints only\n";
    }
  }

  // ── Step 6: Start metrics polling in background ──
  std::atomic<bool> stop_metrics{false};
  std::vector<MetricSample> metric_samples;
  std::thread metrics_thread([&]() {
    if (opts.engine == "sglang") {
      while (!stop_metrics.load()) {
        const auto batch = fetch_sglang_metrics_once(profile_endpoint);
        metric_samples.insert(metric_samples.end(), batch.begin(), batch.end());
        auto slept = std::chrono::milliseconds::zero();
        const auto interval = std::chrono::milliseconds(1000);
        while (!stop_metrics.load() && slept < interval) {
          constexpr auto kSlice = std::chrono::milliseconds(50);
          std::this_thread::sleep_for(kSlice);
          slept += kSlice;
        }
      }
    } else {
      metric_samples = profiler::poll_metrics(
          profile_endpoint,
          std::chrono::milliseconds(1000),
          stop_metrics);
    }
  });

  // ── Step 7: Replay traffic (or wait) with live dashboard ────────────────────
  // Shared counters updated by the replay callback (main thread) and read by
  // the dashboard thread.  std::atomic is enough — no mutex needed.
  std::atomic<int> dash_done{0}, dash_ok{0}, dash_fail{0};
  const int total_reqs = static_cast<int>(requests.size());
  const double dash_start = now_seconds();
  const bool use_control = dash_use_control();
  const bool use_color = dash_use_color();

  // Server metrics polled by the dashboard thread for display (never stored).
  std::mutex snap_mtx;
  DashSnap dash_snap;

  // Number of dashboard lines rendered in the previous frame.
  int dash_rendered_lines = 0;

  // Render one dashboard frame.  Only called from the dashboard thread.
  auto draw_dash = [&]() {
    const int elapsed_s = static_cast<int>(now_seconds() - dash_start);
    const int done  = dash_done.load(std::memory_order_relaxed);
    const int ok    = dash_ok.load(std::memory_order_relaxed);
    const int fail  = dash_fail.load(std::memory_order_relaxed);
    const double rate = (elapsed_s > 0 && done > 0)
        ? static_cast<double>(done) / elapsed_s : 0.0;
    DashSnap snap;
    { std::lock_guard<std::mutex> lg(snap_mtx); snap = dash_snap; }

    int lines_written = 0;
    if (use_control && dash_rendered_lines > 0) {
      std::cerr << "\033[" << dash_rendered_lines << "F\033[J";
    }

    const int term_cols = dash_terminal_columns();
    const int time_bar_width = std::clamp(term_cols - 26, 8, 30);
    const int request_bar_width = std::clamp(term_cols - 42, 8, 30);

    // Helper: print a line, erasing trailing content if ANSI is available.
    auto ln = [&](const std::string& s) {
      if (use_control) std::cerr << "\r";
      std::cerr << s;
      if (use_control) std::cerr << "\033[K";
      std::cerr << "\n";
      ++lines_written;
    };

    // ── header ──────────────────────────────────────────────────────────────
    {
      std::ostringstream h;
      h << "  ";
      if (use_color) h << "\033[1m\033[36m";
      h << "serve-profile";
      if (use_color) h << "\033[0m";
      if (!model.empty()) {
        if (use_color) h << "  \033[2m"; else h << "  ·  ";
        h << model;
        if (use_color) h << "\033[0m";
      }
      ln(h.str());
    }

    // ── time bar ─────────────────────────────────────────────────────────────
    {
      std::ostringstream s;
      const int shown_elapsed_s =
          opts.duration_seconds > 0 ? std::min(elapsed_s, opts.duration_seconds) : elapsed_s;
      s << "  time      " << prog_bar(shown_elapsed_s, opts.duration_seconds, time_bar_width);
      s << "  ";
      if (use_color) s << "\033[2m";
      s << shown_elapsed_s << "s";
      if (opts.duration_seconds > 0) s << " / " << opts.duration_seconds << "s";
      if (use_color) s << "\033[0m";
      ln(s.str());
    }

    // ── request bar (only shown when replaying traffic) ───────────────────────
    if (total_reqs > 0) {
      std::ostringstream s;
      s << "  requests  " << prog_bar(done, total_reqs, request_bar_width);
      s << "  " << done << " / " << total_reqs;
      if (ok > 0) {
        if (use_color) s << "  \033[32m"; else s << "  ok:";
        s << "\xe2\x9c\x93" << ok;  // ✓
        if (use_color) s << "\033[0m";
      }
      if (fail > 0) {
        if (use_color) s << "  \033[31m"; else s << "  fail:";
        s << "\xe2\x9c\x97" << fail;  // ✗
        if (use_color) s << "\033[0m";
      }
      if (rate > 0.0) {
        if (use_color) s << "  \033[2m"; else s << "  ";
        s << std::fixed << std::setprecision(2) << rate << " req/s";
        if (use_color) s << "\033[0m";
      }
      ln(s.str());
    }

    // ── server metrics ────────────────────────────────────────────────────────
    if (snap.available) {
      std::ostringstream s;
      s << "  server    ";
      s << "batch " << static_cast<int>(snap.batch);
      s << "  \xc2\xb7  queue " << static_cast<int>(snap.queue);  // ·
      if (snap.cache >= 0) {
        s << "  \xc2\xb7  cache ";
        s << std::fixed << std::setprecision(0) << snap.cache << "%";
      }
      ln(s.str());
    } else {
      if (use_color)
        ln("  server    \033[2mconnecting...\033[0m");
      else
        ln("  server    connecting...");
    }

    std::cerr << std::flush;
    dash_rendered_lines = lines_written;
  };

  // Dashboard thread: polls server metrics and redraws every 500 ms.
  std::atomic<bool> stop_dash{false};
  std::thread dash_thread([&]() {
    // Initial metrics fetch before first draw.
    { const DashSnap s = fetch_dash_snap(profile_endpoint, opts.engine);
      std::lock_guard<std::mutex> lg(snap_mtx); dash_snap = s; }
    draw_dash();

    int tick = 0;
    while (!stop_dash.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      if (stop_dash.load()) break;
      if (++tick % 4 == 0) {  // re-fetch metrics every ~2 s
        const DashSnap s = fetch_dash_snap(profile_endpoint, opts.engine);
        std::lock_guard<std::mutex> lg(snap_mtx); dash_snap = s;
      }
      draw_dash();
    }
  });

  // ── traffic replay or timed wait ─────────────────────────────────────────
  std::vector<ReplayResult> replay_results;
  if (!requests.empty()) {
    ReplayConfig rc;
    rc.endpoint = profile_endpoint;
    rc.model = model;
    rc.max_concurrency = opts.max_concurrency;
    rc.max_duration_seconds = opts.duration_seconds;
    rc.rate_limit_rps = 0.0;
    rc.on_request_done = [&](int d, int /*total*/, int o, int f) {
      dash_done.store(d, std::memory_order_relaxed);
      dash_ok.store(o, std::memory_order_relaxed);
      dash_fail.store(f, std::memory_order_relaxed);
    };
    replay_results = replay_traffic(requests, rc);
  } else {
    std::cerr << "No traffic file provided. Collecting metrics only.\n"
              << "Per-request queue, prefill, and decode timing require requests during "
                 "the capture window.\n";
    // No traffic file — sleep until duration elapses.
    const double end = dash_start + static_cast<double>(opts.duration_seconds);
    while (now_seconds() < end)
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // ── Step 8: Stop metrics collection ──────────────────────────────────────
  stop_metrics.store(true);
  stop_dash.store(true);
  metrics_thread.join();
  dash_thread.join();
  const double profile_duration_seconds =
      std::max(0.001, now_seconds() - dash_start);

  // Clear the dashboard and print a compact final summary.
  if (use_control && dash_rendered_lines > 0)
    std::cerr << "\033[" << dash_rendered_lines << "F\033[J";
  {
    const int elapsed_s = static_cast<int>(profile_duration_seconds);
    const int done = dash_done.load(), ok = dash_ok.load(), fail = dash_fail.load();
    std::ostringstream s;
    s << "  ";
    if (use_color) s << "\033[32m\xe2\x9c\x93\033[0m "; else s << "done: ";  // ✓
    s << "serve-profile";
    if (!model.empty()) s << "  \xc2\xb7  " << model;
    if (total_reqs > 0) {
      s << "  \xc2\xb7  " << done << " requests";
      if (fail > 0) s << "  (" << ok << " ok, " << fail << " fail)";
    }
    s << "  \xc2\xb7  " << elapsed_s << "s";
    s << "  \xc2\xb7  " << metric_samples.size() << " metric samples";
    std::cerr << s.str() << "\n";
  }

  // ── Derive server-side metrics from Prometheus counter/histogram deltas ──
  const ServerTimingMeans server_timing_means = derive_server_timing_means(metric_samples);
  const double server_ttft_mean_ms = server_timing_means.ttft_mean_ms;

  // Prefix cache hit rate from vLLM 0.19+ counter deltas (hits / queries).
  // Same sentinel pattern as above: no matching samples → delta < 0 → guard fails → stays -1.
  double prometheus_cache_hit_rate = -1.0;
  {
    double min_hits = std::numeric_limits<double>::max();
    double max_hits = -1.0;
    double min_queries = std::numeric_limits<double>::max();
    double max_queries = -1.0;
    for (const auto& s : metric_samples) {
      if (s.source != "cluster") continue;
      if (s.metric == "vllm:prefix_cache_hits_total") {
        min_hits = std::min(min_hits, s.value);
        max_hits = std::max(max_hits, s.value);
      } else if (s.metric == "vllm:prefix_cache_queries_total") {
        min_queries = std::min(min_queries, s.value);
        max_queries = std::max(max_queries, s.value);
      }
    }
    const double delta_queries = max_queries - min_queries;
    const double delta_hits = max_hits - min_hits;
    if (delta_queries > 0.0 && delta_hits >= 0.0) {
      // Clamp to [0, 1]: counter race conditions can cause delta_hits > delta_queries
      prometheus_cache_hit_rate = std::clamp(delta_hits / delta_queries, 0.0, 1.0);
    } else if (max_queries > 0.0 && max_hits >= 0.0) {
      // All requests completed before first sample window — use cumulative ratio
      prometheus_cache_hit_rate = std::clamp(max_hits / max_queries, 0.0, 1.0);
    }
  }

  if (nsys_started && !nsys_session_name.empty()) {
    const std::string stop_command =
        "nsys stop --session=" + shell_escape(nsys_session_name);
    const int stop_rc = std::system(stop_command.c_str());
    if (stop_rc != 0) {
      std::cerr << "warning: failed to stop nsys session cleanly for "
                << nsys_session_name << "\n";
    }
    if (attach_profiler_pid > 0 && nsys_started_later) {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
      while (std::chrono::steady_clock::now() < deadline && process_alive(attach_profiler_pid)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }
      if (process_alive(attach_profiler_pid)) {
        stop_process(attach_profiler_pid);
        std::cerr << "warning: timed out waiting for external nsys attach session to flush\n";
      }
    }
    if (traced_server_pid > 0) {
      stop_process(traced_server_pid);
      traced_server_pid = -1;
    }
    if (attach_clone_plan.has_value() && attach_clone_plan->mode == "replace_restore") {
      try {
        fs::path restore_log_path = output_dir / "serve_profile_restore.log";
        restore_attach_source_or_throw(*attach_clone_plan, restore_log_path, startup_timeout);
      } catch (const std::exception& e) {
        std::cerr << "warning: failed to restore source vLLM server after traced run: "
                  << e.what() << "\n";
      }
    }
  }

  // ── Step 9: Build request traces ──
  auto traces = results_to_traces(replay_results, requests);

  std::optional<double> aggregate_cache_hit_rate;
  bool per_request_cache_data_available = false;
  ServerTraceCorrelationResult trace_correlation;
  if (opts.engine == "vllm" && server_log_path.has_value() && fs::exists(*server_log_path)) {
    try {
      const auto log_lines =
          read_log_lines_from_offset(*server_log_path, server_log_start_offset);
      const auto log_parse = parse_vllm_log_lines_detailed(log_lines);
      per_request_cache_data_available = std::any_of(
          log_parse.traces.begin(), log_parse.traces.end(), [](const RequestTrace& trace) {
            if (trace.cached_tokens > 0) return true;
            return std::any_of(trace.events.begin(), trace.events.end(),
                               [](const RequestEvent& event) {
                                 return event.event_type == "cache_hit";
                               });
          });
      trace_correlation = correlate_server_traces(traces, log_parse.traces, true, 50000);
      aggregate_cache_hit_rate = log_parse.aggregate_cache_hit_rate;
      const bool no_profiled_requests = traces.empty();
      if (!log_parse.traces.empty()) {
        std::cerr << "Parsed " << log_parse.traces.size()
                  << " server-side request traces from " << server_log_path->string()
                  << " (" << log_lines.size() << " new log lines)\n";
      } else if (log_parse.aggregate_cache_hit_rate.has_value()) {
        if (no_profiled_requests) {
          std::cerr << "Server log: aggregate cache hit rate extracted.\n"
                    << "No requests were observed during this profiling run, so there are "
                       "no per-request traces to correlate.\n";
        } else {
          std::cerr << "Server log: aggregate cache hit rate extracted, but no per-request "
                       "timing traces were parsed from the current log window.\n";
        }
      } else {
        if (no_profiled_requests) {
          std::cerr << "Server log found at " << server_log_path->string()
                    << ", but no requests were observed during this profiling run.\n"
                    << "  Pass --traffic or send live traffic during the capture window for "
                       "per-request timing.\n";
        } else {
          std::cerr << "Server log found at " << server_log_path->string()
                    << " but no per-request DEBUG timing lines were parsed.\n"
                    << "  Restart the server with VLLM_LOGGING_LEVEL=DEBUG and capture stdout/stderr to a file.\n";
        }
        if (listener_log_info.pid > 0) {
          if (!listener_log_info.vllm_logging_level.has_value()) {
            std::cerr << "  Listener PID " << listener_log_info.pid
                      << " has no VLLM_LOGGING_LEVEL in its environment.\n";
          } else {
            std::cerr << "  Listener PID " << listener_log_info.pid
                      << " uses VLLM_LOGGING_LEVEL="
                      << *listener_log_info.vllm_logging_level << ".\n";
          }
        }
      }
      if (trace_correlation.method == ServerTraceMatchMethod::ID) {
        std::cerr << "Matched " << trace_correlation.matched_requests << "/"
                  << trace_correlation.total_requests << " requests by ID\n";
        if (refine_order_matched_v1_timing(traces, trace_correlation, server_timing_means)) {
          trace_correlation.metric_assisted = true;
        }
      } else if (trace_correlation.method == ServerTraceMatchMethod::TIMESTAMP) {
        std::cerr << "ID matching failed. Matched " << trace_correlation.matched_requests << "/"
                  << trace_correlation.total_requests << " requests by timestamp (+/-"
                  << (trace_correlation.max_offset_us / 1000.0) << "ms max offset)\n";
      } else if (trace_correlation.method == ServerTraceMatchMethod::ORDER) {
        std::cerr << "ID/timestamp matching unavailable. Matched "
                  << trace_correlation.matched_requests << "/"
                  << trace_correlation.total_requests
                  << " requests by observed request order\n";
        if (refine_order_matched_v1_timing(traces, trace_correlation, server_timing_means)) {
          trace_correlation.metric_assisted = true;
        }
        std::cerr << "  For exact per-request matching on vLLM v1, start the server with "
                  << "--enable-log-requests so request IDs appear in DEBUG logs.\n";
      } else if (!log_parse.traces.empty() && !traces.empty()) {
        // Only warn with ID details if the IDs looked like real request IDs
        std::cerr
            << "Note: server log contains per-request traces but none matched client trace IDs.\n"
            << "  Server IDs sample: [";
        for (std::size_t i = 0; i < log_parse.traces.size() && i < 3; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << log_parse.traces[i].request_id;
        }
        std::cerr << "] vs client IDs: [";
        for (std::size_t i = 0; i < traces.size() && i < 3; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << traces[i].request_id;
        }
        std::cerr << "]\n"
                  << "  Ensure --server-log points to the log from the same vLLM instance.\n"
                  << "  Per-request server timing will not be available.\n";
      }
    } catch (const std::exception& e) {
      std::cerr << "warning: failed to parse server log " << server_log_path->string()
                << ": " << e.what() << "\n";
    }
  }

  PhaseBreakdown phase;
  bool gpu_phase_available = false;
  if (nsys_sqlite_path.has_value()) {
    try {
      if (wait_for_nsys_sqlite(*nsys_sqlite_path, std::chrono::seconds(180))) {
        const auto kernels = load_phase_kernels(*nsys_sqlite_path);
        phase = analyze_phases(kernels).breakdown;
        gpu_phase_available = phase.prefill_us + phase.decode_us > 0;
        std::cerr << "Parsed " << kernels.size() << " nsys kernels from "
                  << nsys_sqlite_path->string() << "\n";
      } else {
        std::cerr << "warning: timed out waiting for nsys sqlite export readiness: "
                  << nsys_sqlite_path->string() << "\n";
      }
    } catch (const std::exception& e) {
      std::cerr << "warning: failed to analyze nsys trace " << nsys_sqlite_path->string()
                << ": " << e.what() << "\n";
    }
  }

  // Store traces
  for (const auto& t : traces) {
    insert_request_trace(db_path, 1, t);
  }

  // Store raw metrics
  {
    ProfileData pd;
    pd.meta = {{"model_name", model}, {"gpu_name", gpu.name},
               {"gpu_count", std::to_string(gpu.count)},
               {"engine", opts.engine}};
    pd.metrics = metric_samples;
    pd.metrics_summary = profiler::summarize_samples(metric_samples);
    save_profile(db_path, pd);
  }

  // ── Step 10: Run analyzers ──
  std::cerr << "Analyzing ...\n";

  auto snapshots = samples_to_snapshots(metric_samples);
  auto batch = analyze_batches(snapshots);
  // Fall back to Prometheus counter-derived hit rate when no server log provides it
  if (!aggregate_cache_hit_rate.has_value() && prometheus_cache_hit_rate >= 0.0) {
    aggregate_cache_hit_rate = prometheus_cache_hit_rate;
    std::cerr << "Cache hit rate from Prometheus counters: "
              << static_cast<int>(prometheus_cache_hit_rate * 100.0) << "%\n";
  }
  auto cache = analyze_cache(per_request_cache_data_available ? traces
                                                              : std::vector<RequestTrace>{},
                             snapshots, aggregate_cache_hit_rate);
  const bool cache_usage_available = !snapshots.empty();
  const bool cache_hit_rate_available = cache.cache_hit_rate_available;
  const bool cache_histogram_available = cache.hit_rate_histogram_available;
  const bool server_timing_available = std::any_of(
      traces.begin(), traces.end(), [](const RequestTrace& trace) {
        return trace.server_timing_available;
      });
  const bool remote_timestamp_correlation =
      trace_correlation.method == ServerTraceMatchMethod::TIMESTAMP &&
      !endpoint_is_local(profile_endpoint);

  PrefixAnalysis prefix;
  bool prefix_available = false;
  {
    std::vector<std::vector<int>> prompts;
    prompts.reserve(traces.size());
    for (const auto& trace : traces) {
      if (!trace.prompt_text.empty()) {
        prompts.push_back(prompt_to_char_ids(trace.prompt_text));
      }
    }
    if (!prompts.empty()) {
      prefix = analyze_prefixes(prompts);
      prefix_available = true;
    }
  }

  // Compute latency percentiles from traces
  std::vector<double> queue_lat, server_prefill_lat, server_decode_lat;
  std::vector<double> prefill_lat, decode_lat, decode_per_tok, e2e_lat;
  for (const auto& t : traces) {
    if (t.status != "ok") continue;
    if (t.server_timing_available &&
        t.queue_start_us > 0 &&
        t.prefill_start_us > 0 &&
        t.prefill_start_us >= t.queue_start_us) {
      queue_lat.push_back(static_cast<double>(t.prefill_start_us - t.queue_start_us) / 1000.0);
    }
    if (t.server_timing_available &&
        t.prefill_end_us > 0 &&
        t.prefill_start_us > 0 &&
        t.prefill_end_us >= t.prefill_start_us) {
      server_prefill_lat.push_back(
          static_cast<double>(t.prefill_end_us - t.prefill_start_us) / 1000.0);
    }
    if (t.server_timing_available &&
        t.server_last_token_us > 0 &&
        t.prefill_end_us > 0 &&
        t.server_last_token_us >= t.prefill_end_us) {
      server_decode_lat.push_back(
          static_cast<double>(t.server_last_token_us - t.prefill_end_us) / 1000.0);
    }
    if (t.first_token_us > 0 && t.arrival_us > 0 && t.first_token_us >= t.arrival_us)
      prefill_lat.push_back(static_cast<double>(t.first_token_us - t.arrival_us) / 1000.0);
    if (t.last_token_us > 0 && t.first_token_us > 0) {
      double dec = static_cast<double>(t.last_token_us - t.first_token_us) / 1000.0;
      decode_lat.push_back(dec);
      if (t.output_tokens > 1)
        decode_per_tok.push_back(dec / (t.output_tokens - 1));
    }
    if (t.completion_us > 0 && t.arrival_us > 0)
      e2e_lat.push_back(static_cast<double>(t.completion_us - t.arrival_us) / 1000.0);
  }

  double median_prompt = 0, median_output = 0;
  {
    std::vector<double> pt, ot;
    for (const auto& t : traces) {
      pt.push_back(t.prompt_tokens);
      ot.push_back(t.output_tokens);
    }
    median_prompt = std::max(0.0, percentile_vec(pt, 50));
    median_output = std::max(0.0, percentile_vec(ot, 50));
  }

  // Workload classifier
  WorkloadClassifierInput ci;
  ci.phase = phase;
  ci.batch = batch;
  ci.cache = cache;
  ci.prefix = prefix;
  ci.median_prompt_tokens = median_prompt;
  ci.median_output_tokens = median_output;
  ci.request_rate = traces.empty() ? 0 :
      static_cast<double>(traces.size()) / profile_duration_seconds;
  ci.median_decode_latency_us = percentile_vec(decode_lat, 50) * 1000.0;
  ci.p99_decode_latency_us = percentile_vec(decode_lat, 99) * 1000.0;

  auto profile = classify_workload(ci);

  // Disagg model
  DisaggModelInput di;
  di.profile = profile;
  di.total_gpus = gpu.count;
  di.network_bandwidth_gbps = 100.0;  // assume NVLink or fast network
  if (!prefill_lat.empty()) {
    di.measured_prefill_p99_ms = percentile_vec(prefill_lat, 99);
  }
  if (median_prompt > 0.0) {
    const int64_t kv_bytes_per_token = detect_kv_bytes_per_token(model);
    if (kv_bytes_per_token > 0) {
      di.avg_kv_transfer_bytes = static_cast<double>(kv_bytes_per_token) * median_prompt;
    }
  }
  auto disagg = estimate_disaggregation(di);

  const bool prompt_tokens_estimated_any = std::any_of(
      traces.begin(), traces.end(), [](const RequestTrace& trace) {
        return trace.prompt_tokens_estimated;
      });
  std::string advisor_caveat;
  if (prompt_tokens_estimated_any || !gpu_phase_available ||
      !server_timing_available || !cache_hit_rate_available) {
    advisor_caveat = "Note: advisor estimates ";
    bool wrote_clause = false;
    if (prompt_tokens_estimated_any) {
      advisor_caveat += "use approximate prompt token counts (chars/4)";
      wrote_clause = true;
    }
    if (!gpu_phase_available) {
      advisor_caveat += wrote_clause ? ", " : "";
      advisor_caveat += "do not include GPU phase data";
      wrote_clause = true;
    }
    if (!server_timing_available) {
      advisor_caveat += wrote_clause ? ", " : "";
      advisor_caveat += "do not include server-side queue/prefill timing";
      wrote_clause = true;
    }
    if (!cache_hit_rate_available) {
      advisor_caveat += wrote_clause ? ", and " : "";
      advisor_caveat += "do not include cache-hit statistics";
    }
    advisor_caveat += ".";
    if (opts.engine == "sglang") {
      advisor_caveat += " Run against a server with prefix caching enabled for higher-confidence recommendations.";
    } else {
      advisor_caveat += " Run with --nsys, VLLM_LOGGING_LEVEL=DEBUG, and against a server with prefix caching enabled for higher-confidence recommendations.";
    }
  }

  // ── Step 11: Persist analysis to DB ──
  std::map<std::string, std::string> analysis;
  save_kv(analysis, "meta", "model", model);
  save_kv(analysis, "meta", "engine", opts.engine);
  save_kv(analysis, "meta", "gpu_name", gpu.name);
  save_kv(analysis, "meta", "gpu_count", gpu.count);
  save_kv(analysis, "meta", "duration_seconds", profile_duration_seconds);
  save_kv(analysis, "meta", "requested_duration_seconds",
          static_cast<double>(opts.duration_seconds));
  save_kv(analysis, "meta", "total_requests", static_cast<int>(traces.size()));
  save_kv(analysis, "meta", "throughput_rps", ci.request_rate);
  save_kv(analysis, "meta", "prompt_tokens_estimated_any", prompt_tokens_estimated_any);
  // queue_available is true only if we actually collected queue wait samples —
  // server_timing_available can be true without queue_start_us being recorded
  // (e.g. when vLLM logs prefill start but not the "Added request" line).
  save_kv(analysis, "latency", "queue_available", !queue_lat.empty());
  save_kv(analysis, "latency", "server_timing_available", server_timing_available);
  const std::string server_timing_match_method =
      trace_correlation.method == ServerTraceMatchMethod::ID
          ? "id"
          : (trace_correlation.method == ServerTraceMatchMethod::TIMESTAMP
                 ? "timestamp"
                 : (trace_correlation.method == ServerTraceMatchMethod::ORDER
                        ? "order"
                        : "none"));
  save_kv(analysis, "latency", "server_timing_match_method", server_timing_match_method);
  save_kv(analysis, "latency", "server_timing_match_max_offset_ms",
          trace_correlation.max_offset_us / 1000.0);
  save_kv(analysis, "latency", "server_timing_remote_correlation",
          remote_timestamp_correlation);
  save_kv(analysis, "latency", "server_timing_metric_assisted",
          trace_correlation.metric_assisted);

  // Latency percentiles
  save_kv(analysis, "latency", "queue_p50", percentile_vec(queue_lat, 50));
  save_kv(analysis, "latency", "queue_p90", percentile_vec(queue_lat, 90));
  save_kv(analysis, "latency", "queue_p99", percentile_vec(queue_lat, 99));
  save_kv(analysis, "latency", "server_prefill_p50", percentile_vec(server_prefill_lat, 50));
  save_kv(analysis, "latency", "server_prefill_p90", percentile_vec(server_prefill_lat, 90));
  save_kv(analysis, "latency", "server_prefill_p99", percentile_vec(server_prefill_lat, 99));
  save_kv(analysis, "latency", "server_decode_p50", percentile_vec(server_decode_lat, 50));
  save_kv(analysis, "latency", "server_decode_p90", percentile_vec(server_decode_lat, 90));
  save_kv(analysis, "latency", "server_decode_p99", percentile_vec(server_decode_lat, 99));
  save_kv(analysis, "latency", "prefill_p50", percentile_vec(prefill_lat, 50));
  save_kv(analysis, "latency", "prefill_p90", percentile_vec(prefill_lat, 90));
  save_kv(analysis, "latency", "prefill_p99", percentile_vec(prefill_lat, 99));
  save_kv(analysis, "latency", "decode_total_p50", percentile_vec(decode_lat, 50));
  save_kv(analysis, "latency", "decode_total_p90", percentile_vec(decode_lat, 90));
  save_kv(analysis, "latency", "decode_total_p99", percentile_vec(decode_lat, 99));
  save_kv(analysis, "latency", "decode_per_token_p50", percentile_vec(decode_per_tok, 50));
  save_kv(analysis, "latency", "decode_per_token_p90", percentile_vec(decode_per_tok, 90));
  save_kv(analysis, "latency", "decode_per_token_p99", percentile_vec(decode_per_tok, 99));
  save_kv(analysis, "latency", "e2e_p50", percentile_vec(e2e_lat, 50));
  save_kv(analysis, "latency", "e2e_p90", percentile_vec(e2e_lat, 90));
  save_kv(analysis, "latency", "e2e_p99", percentile_vec(e2e_lat, 99));
  // Server-side TTFT mean from Prometheus histogram (vLLM 0.19+); -1 = not available
  save_kv(analysis, "latency", "server_ttft_mean_ms", server_ttft_mean_ms);

  // Batch
  save_kv(analysis, "batch", "avg_batch_size", batch.avg_batch_size);
  save_kv(analysis, "batch", "p50_batch_size", batch.p50_batch_size);
  save_kv(analysis, "batch", "p99_batch_size", batch.p99_batch_size);

  // Phase (GPU breakdown) — only available if nsys data was collected.
  // Do not save pct values when unavailable — saving 0.0 pollutes the DB
  // with uninformative zeros that look like real measurements.
  save_kv(analysis, "phase", "available", gpu_phase_available);
  if (gpu_phase_available) {
    save_kv(analysis, "phase", "prefill_pct", phase.prefill_fraction * 100.0);
    save_kv(analysis, "phase", "decode_pct", phase.decode_fraction * 100.0);
    // Clamp to 0 to absorb floating-point drift that could make this slightly negative.
    save_kv(analysis, "phase", "other_pct",
            std::max(0.0, 1.0 - phase.prefill_fraction - phase.decode_fraction) * 100.0);
  }

  // Prefix
  save_kv(analysis, "prefix", "available", prefix_available);
  save_kv(analysis, "prefix", "unique_prefixes", prefix.unique_prefixes);
  save_kv(analysis, "prefix", "cacheable_tokens_pct", prefix.cacheable_token_fraction * 100.0);

  // Cache
  save_kv(analysis, "cache", "hit_rate_available", cache_hit_rate_available);
  save_kv(analysis, "cache", "usage_available", cache_usage_available);
  save_kv(analysis, "cache", "hit_rate_aggregate_only", cache.cache_hit_rate_aggregate_only);
  save_kv(analysis, "cache", "histogram_available", cache_histogram_available);
  save_kv(analysis, "cache", "hit_rate", cache.cache_hit_rate);
  save_kv(analysis, "cache", "avg_usage", cache.avg_cache_usage);
  save_kv(analysis, "cache", "peak_usage", cache.peak_cache_usage);
  save_kv(analysis, "cache", "evictions", cache.eviction_count);
  for (int i = 0; i < 5; ++i) {
    save_kv(analysis, "cache", "histogram_" + std::to_string(i),
            cache.hit_rate_histogram[static_cast<size_t>(i)]);
  }

  // Disagg
  save_kv(analysis, "disagg", "should", disagg.should_disaggregate);
  save_kv(analysis, "disagg", "optimal_p", disagg.optimal_prefill_gpus);
  save_kv(analysis, "disagg", "optimal_d", disagg.optimal_decode_gpus);
  save_kv(analysis, "disagg", "mono_throughput", disagg.mono_throughput_rps);
  save_kv(analysis, "disagg", "disagg_throughput", disagg.disagg_throughput_rps);
  save_kv(analysis, "disagg", "throughput_improvement", disagg.throughput_improvement);
  save_kv(analysis, "disagg", "mono_p99_ttft", disagg.mono_p99_ttft_ms);
  save_kv(analysis, "disagg", "disagg_p99_ttft", disagg.disagg_p99_ttft_ms);
  save_kv(analysis, "disagg", "kv_transfer_overhead", disagg.kv_transfer_overhead_ms);
  save_kv(analysis, "disagg", "min_bandwidth", disagg.min_bandwidth_gbps);
  save_kv(analysis, "disagg", "reason", disagg.reason);
  save_kv(analysis, "disagg", "caveat", advisor_caveat);

  save_serve_analysis(db_path, analysis);

  // ── Step 12: Print summary ──
  std::cerr << "\n";
  std::cerr << "Results saved to " << db_path.string() << "\n";
  std::cerr << "  Requests: " << traces.size() << "\n";
  std::cerr << "  Throughput: " << ci.request_rate << " req/s\n";
  if (!e2e_lat.empty()) {
    std::cerr << "  p50 e2e: " << percentile_vec(e2e_lat, 50) << " ms\n";
    std::cerr << "  p99 e2e: " << percentile_vec(e2e_lat, 99) << " ms\n";
  }
  std::cerr << "  Disagg: " << (disagg.should_disaggregate ? "recommended" : "not recommended")
            << " — " << disagg.reason << "\n";
  std::cerr << "\nNote: TTFT (client) is measured from the first streamed token chunk.\n";
  std::cerr << "      TTFT (server, mean) comes from the Prometheus histogram when available.\n";
  std::cerr << "      Queue/Prefill/Decode (server) require vLLM DEBUG logs and requests during the capture window.\n";
  std::cerr << "\nRun:  hotpath serve-report " << db_path.string() << "\n";

  return 0;
}

}  // namespace hotpath
