#include "rlprof/profiler/attach.h"

#include <algorithm>
#include <array>
#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>

namespace rlprof::profiler {
namespace {

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

std::string trim(std::string value) {
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) {
    value.pop_back();
  }
  return value;
}

std::vector<std::string> read_proc_cmdline(std::int64_t pid) {
  const std::string path = "/proc/" + std::to_string(pid) + "/cmdline";
  FILE* file = fopen(path.c_str(), "rb");
  if (file == nullptr) {
    return {};
  }
  std::string data;
  std::array<char, 4096> buffer{};
  while (true) {
    const auto read = fread(buffer.data(), 1, buffer.size(), file);
    if (read == 0) {
      break;
    }
    data.append(buffer.data(), read);
  }
  fclose(file);

  std::vector<std::string> argv;
  std::string current;
  for (char ch : data) {
    if (ch == '\0') {
      if (!current.empty()) {
        argv.push_back(current);
        current.clear();
      }
    } else {
      current.push_back(ch);
    }
  }
  if (!current.empty()) {
    argv.push_back(current);
  }
  return argv;
}

std::string host_from_server_url(const std::string& url) {
  const auto scheme_pos = url.find("://");
  const std::size_t host_start = scheme_pos == std::string::npos ? 0 : scheme_pos + 3;
  if (host_start >= url.size()) {
    return "";
  }
  const auto path_start = url.find('/', host_start);
  const auto host_port_end = path_start == std::string::npos ? url.size() : path_start;
  if (url[host_start] == '[') {
    const auto bracket_end = url.find(']', host_start + 1);
    if (bracket_end == std::string::npos || bracket_end >= host_port_end) {
      return "";
    }
    return url.substr(host_start + 1, bracket_end - host_start - 1);
  }
  const auto host_end = url.rfind(':', host_port_end);
  if (host_end == std::string::npos || host_end < host_start) {
    return url.substr(host_start, host_port_end - host_start);
  }
  return url.substr(host_start, host_end - host_start);
}

std::int64_t port_from_server_url(const std::string& url) {
  const auto scheme_pos = url.find("://");
  const std::size_t host_start = scheme_pos == std::string::npos ? 0 : scheme_pos + 3;
  const auto path_start = url.find('/', host_start);
  const auto host_port_end = path_start == std::string::npos ? url.size() : path_start;
  const auto last_colon = url.rfind(':');
  if (last_colon == std::string::npos || last_colon >= host_port_end) {
    return 0;
  }
  if (host_start < url.size() && url[host_start] == '[') {
    const auto bracket_end = url.find(']', host_start + 1);
    if (bracket_end == std::string::npos || last_colon <= bracket_end) {
      return 0;
    }
  }
  if (last_colon < host_start) {
    return 0;
  }
  return std::stoll(url.substr(last_colon + 1, host_port_end - last_colon - 1));
}

std::unordered_set<std::string> local_interface_addresses() {
  std::unordered_set<std::string> values = {"127.0.0.1", "::1"};
  ifaddrs* interfaces = nullptr;
  if (getifaddrs(&interfaces) != 0) {
    return values;
  }
  for (ifaddrs* current = interfaces; current != nullptr; current = current->ifa_next) {
    if (current->ifa_addr == nullptr) {
      continue;
    }
    const int family = current->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6) {
      continue;
    }
    char host[NI_MAXHOST] = {};
    const socklen_t addr_len = family == AF_INET ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
    if (getnameinfo(
            current->ifa_addr,
            addr_len,
            host,
            sizeof(host),
            nullptr,
            0,
            NI_NUMERICHOST) == 0) {
      values.insert(host);
    }
  }
  freeifaddrs(interfaces);
  return values;
}

std::unordered_set<std::string> resolve_host_addresses(const std::string& host) {
  std::unordered_set<std::string> values;
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_ADDRCONFIG;
  addrinfo* result = nullptr;
  if (getaddrinfo(host.c_str(), nullptr, &hints, &result) != 0) {
    return values;
  }
  for (addrinfo* current = result; current != nullptr; current = current->ai_next) {
    char numeric[NI_MAXHOST] = {};
    if (getnameinfo(
            current->ai_addr,
            current->ai_addrlen,
            numeric,
            sizeof(numeric),
            nullptr,
            0,
            NI_NUMERICHOST) == 0) {
      values.insert(numeric);
    }
  }
  freeaddrinfo(result);
  return values;
}

bool port_in_use(std::int64_t port) {
  const std::string ss_command =
      "bash -lc \"ss -ltn '( sport = :" + std::to_string(port) +
      " )' 2>/dev/null | tail -n +2 | grep -q LISTEN\"";
  if (std::system(ss_command.c_str()) == 0) {
    return true;
  }
  const std::string lsof_command =
      "bash -lc \"lsof -iTCP:" + std::to_string(port) +
      " -sTCP:LISTEN >/dev/null 2>&1\"";
  return std::system(lsof_command.c_str()) == 0;
}

std::int64_t choose_clone_port(std::int64_t source_port) {
  std::int64_t candidate = source_port > 0 ? source_port + 1 : 8001;
  for (int attempt = 0; attempt < 64; ++attempt, ++candidate) {
    if (!port_in_use(candidate)) {
      return candidate;
    }
  }
  throw std::runtime_error("failed to find a free local port for attach clone");
}

std::optional<std::int64_t> query_free_gpu_memory_mib() {
  try {
    const std::string output =
        trim(run_command(
            "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1"));
    if (output.empty()) {
      return std::nullopt;
    }
    return std::stoll(output);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::string join_args(const std::vector<std::string>& argv) {
  std::ostringstream out;
  for (std::size_t i = 0; i < argv.size(); ++i) {
    if (i > 0) {
      out << " ";
    }
    out << shell_escape(argv[i]);
  }
  return out.str();
}

std::vector<std::string> cloned_argv(
    const VllmServeProcessInfo& process,
    std::int64_t clone_port) {
  std::vector<std::string> cloned = process.argv;
  bool replaced_port = false;
  for (std::size_t i = 0; i < cloned.size(); ++i) {
    if (cloned[i] == "--port" && i + 1 < cloned.size()) {
      cloned[i + 1] = std::to_string(clone_port);
      replaced_port = true;
    }
  }
  if (!replaced_port) {
    cloned.push_back("--port");
    cloned.push_back(std::to_string(clone_port));
  }
  return cloned;
}

}  // namespace

bool attach_server_is_local(const std::string& attach_server) {
  const std::string host = host_from_server_url(attach_server);
  if (host.empty()) {
    return false;
  }
  if (host == "localhost" || host == "127.0.0.1" || host == "::1") {
    return true;
  }
  char hostname[256] = {};
  if (gethostname(hostname, sizeof(hostname)) == 0) {
    hostname[sizeof(hostname) - 1] = '\0';
    if (host == hostname) {
      return true;
    }
  }
  const auto resolved = resolve_host_addresses(host);
  if (resolved.empty()) {
    return false;
  }
  const auto local = local_interface_addresses();
  for (const auto& value : resolved) {
    if (local.contains(value)) {
      return true;
    }
  }
  return false;
}

std::optional<VllmServeProcessInfo> parse_vllm_serve_argv(
    const std::vector<std::string>& argv) {
  if (argv.empty()) {
    return std::nullopt;
  }
  auto serve_it = std::find(argv.begin() + 1, argv.end(), "serve");
  if (serve_it == argv.end() || serve_it + 1 == argv.end()) {
    return std::nullopt;
  }

  VllmServeProcessInfo info;
  info.executable = argv.front();
  info.argv = argv;
  info.model = *(serve_it + 1);
  for (auto it = serve_it + 2; it != argv.end(); ++it) {
    if (*it == "--port" && it + 1 != argv.end()) {
      info.port = std::stoll(*(it + 1));
      ++it;
    } else if (*it == "--tensor-parallel-size" && it + 1 != argv.end()) {
      info.tp = std::stoll(*(it + 1));
      ++it;
    } else if (*it == "--max-model-len" && it + 1 != argv.end()) {
      info.max_model_len = std::stoll(*(it + 1));
      ++it;
    } else if (*it == "--trust-remote-code") {
      info.trust_remote_code = true;
    }
  }
  return info;
}

std::optional<VllmServeProcessInfo> inspect_vllm_serve_process(
    std::int64_t pid) {
  return parse_vllm_serve_argv(read_proc_cmdline(pid));
}

std::optional<AttachClonePlan> build_attach_clone_plan(
    std::int64_t pid,
    const std::string& attach_server,
    const std::string& nsys_path,
    const std::string& session_name,
    const std::filesystem::path& output_prefix) {
  static_cast<void>(output_prefix);
  const auto process = inspect_vllm_serve_process(pid);
  if (!process.has_value()) {
    return std::nullopt;
  }
  if (!attach_server_is_local(attach_server)) {
    return std::nullopt;
  }
  const std::int64_t attach_port = port_from_server_url(attach_server);
  if (attach_port > 0 && process->port > 0 && attach_port != process->port) {
    return std::nullopt;
  }

  const std::optional<std::int64_t> free_memory_mib = query_free_gpu_memory_mib();
  const bool replace_restore =
      free_memory_mib.has_value() && *free_memory_mib < 4096;

  if (replace_restore) {
    return AttachClonePlan{
        .source_pid = pid,
        .source_server_url = attach_server,
        .mode = "replace_restore",
        .traced_port = process->port,
        .traced_server_url = attach_server,
        .launch_command =
            "VLLM_WORKER_MULTIPROC_METHOD=spawn " + shell_escape(nsys_path) +
            " launch --trace=cuda,nvtx,osrt --session-new " + session_name +
            " --wait=all " + join_args(process->argv),
        .restore_command =
            "setsid nohup env VLLM_WORKER_MULTIPROC_METHOD=spawn " + join_args(process->argv) +
            " < /dev/null",
        .process = *process,
    };
  }

  const std::int64_t clone_port = choose_clone_port(process->port);
  const std::vector<std::string> argv = cloned_argv(*process, clone_port);

  return AttachClonePlan{
      .source_pid = pid,
      .source_server_url = attach_server,
      .mode = "clone",
      .traced_port = clone_port,
      .traced_server_url = "http://127.0.0.1:" + std::to_string(clone_port),
      .launch_command =
          "VLLM_WORKER_MULTIPROC_METHOD=spawn " + shell_escape(nsys_path) +
          " launch --trace=cuda,nvtx,osrt --session-new " + session_name +
          " --wait=all " + join_args(argv),
      .process = *process,
  };
}

}  // namespace rlprof::profiler
