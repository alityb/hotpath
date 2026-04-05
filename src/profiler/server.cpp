#include "hotpath/profiler/server.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <cctype>
#include <regex>
#include <unistd.h>

namespace hotpath::profiler {
namespace {

ManagedServerState parse_state(const std::filesystem::path& path);

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

std::string join_shell_args(const std::vector<std::string>& args) {
  std::ostringstream out;
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (i > 0) {
      out << " ";
    }
    out << shell_escape(args[i]);
  }
  return out.str();
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

std::filesystem::path managed_server_dir() {
  return std::filesystem::path(".hotpath") / "servers";
}

std::filesystem::path managed_server_state_path(const std::string& name) {
  return managed_server_dir() / (name + ".cfg");
}

std::filesystem::path managed_server_lock_path(const std::string& name) {
  return managed_server_dir() / (name + ".lock");
}

std::string sanitize_model_name(std::string value) {
  for (char& ch : value) {
    if (ch == '/') {
      ch = '_';
    }
  }
  return value;
}

bool command_works(const std::string& command) {
  return std::system((command + " >/dev/null 2>&1").c_str()) == 0;
}

std::vector<std::string> vllm_command() {
  // Check both HOTPATH_ and legacy RLPROF_ env vars
  const char* configured_vllm = std::getenv("HOTPATH_VLLM_EXECUTABLE");
  if (!configured_vllm || std::string(configured_vllm).empty()) {
    configured_vllm = std::getenv("RLPROF_VLLM_EXECUTABLE");
  }
  if (configured_vllm != nullptr && std::string(configured_vllm).size() > 0) {
    return {configured_vllm};
  }

  const char* configured_python = std::getenv("HOTPATH_PYTHON_EXECUTABLE");
  if (!configured_python || std::string(configured_python).empty()) {
    configured_python = std::getenv("RLPROF_PYTHON_EXECUTABLE");
  }
  if (configured_python != nullptr && std::string(configured_python).size() > 0) {
    const std::string python = configured_python;
    if (command_works(shell_escape(python) +
                      " -m vllm.entrypoints.openai.api_server --help")) {
      return {python, "-m", "vllm.entrypoints.openai.api_server"};
    }
    // Python exists but vLLM not installed — give a clear error
    if (std::filesystem::exists(python)) {
      throw std::runtime_error(
          "Python at " + python + " does not have vLLM installed.\n"
          "Install it with: " + python + " -m pip install vllm");
    }
    throw std::runtime_error(
        "Python executable not found: " + python + "\n"
        "Check HOTPATH_PYTHON_EXECUTABLE is set to a valid path");
  }

  if (std::filesystem::exists(".venv/bin/vllm") &&
      command_works(shell_escape(".venv/bin/vllm") + " --version")) {
    return {".venv/bin/vllm"};
  }
  if (std::filesystem::exists(".venv/bin/python") &&
      command_works(shell_escape(".venv/bin/python") +
                    " -m vllm.entrypoints.openai.api_server --help")) {
    return {".venv/bin/python", "-m", "vllm.entrypoints.openai.api_server"};
  }
  if (command_works("vllm --version")) {
    return {"vllm"};
  }
  if (command_works("python3 -m vllm.entrypoints.openai.api_server --help")) {
    return {"python3", "-m", "vllm.entrypoints.openai.api_server"};
  }

  throw std::runtime_error(
      "no working vLLM launcher found.\n"
      "Options:\n"
      "  1. Set HOTPATH_PYTHON_EXECUTABLE=/path/to/venv/bin/python (must have vLLM installed)\n"
      "  2. Set HOTPATH_VLLM_EXECUTABLE=/path/to/vllm\n"
      "  3. Install vLLM globally: pip install vllm\n"
      "  4. Profile an existing server: hotpath serve-profile --endpoint http://localhost:8000");
}

std::string make_session_name(const std::string& name) {
  std::string session = "hotpath_managed_";
  for (unsigned char ch : name) {
    session.push_back(std::isalnum(ch) ? static_cast<char>(ch) : '_');
  }
  return session;
}

bool process_alive(std::int64_t pid) {
  if (pid <= 0) {
    return false;
  }
  return std::system(("kill -0 " + std::to_string(pid) + " >/dev/null 2>&1").c_str()) == 0;
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
      const std::string output = trim(run_command(commands[i]));
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

bool server_ready(const std::string& server_url) {
  return std::system(
             ("curl -fsS --connect-timeout 1 --max-time 1 " +
              shell_escape(server_url + "/metrics") + " >/dev/null 2>&1")
                 .c_str()) == 0;
}

bool remove_stale_state_if_needed(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    return false;
  }
  const ManagedServerState state = parse_state(path);
  if (managed_server_ready(state)) {
    return false;
  }
  std::filesystem::remove(path);
  std::error_code ignored;
  std::filesystem::remove_all(managed_server_lock_path(state.name), ignored);
  return true;
}

std::int64_t read_lock_owner_pid(const std::filesystem::path& lock_path) {
  std::ifstream stream(lock_path / "pid");
  if (!stream.good()) {
    return 0;
  }
  std::string line;
  std::getline(stream, line);
  return line.empty() ? 0 : std::stoll(line);
}

void write_lock_owner_pid(const std::filesystem::path& lock_path) {
  std::ofstream stream(lock_path / "pid");
  stream << getpid() << "\n";
}

std::filesystem::path acquire_lock_or_throw(const std::string& name) {
  const auto lock_path = managed_server_lock_path(name);
  std::filesystem::create_directories(lock_path.parent_path());
  if (std::filesystem::create_directory(lock_path)) {
    write_lock_owner_pid(lock_path);
    return lock_path;
  }
  const std::int64_t owner_pid = read_lock_owner_pid(lock_path);
  if (owner_pid <= 0 || !process_alive(owner_pid)) {
    std::error_code ignored;
    std::filesystem::remove_all(lock_path, ignored);
    if (std::filesystem::create_directory(lock_path)) {
      write_lock_owner_pid(lock_path);
      return lock_path;
    }
  }
  throw std::runtime_error("managed server is busy: " + name);
}

void release_lock(const std::filesystem::path& lock_path) {
  if (lock_path.empty()) {
    return;
  }
  std::error_code ignored;
  std::filesystem::remove_all(lock_path, ignored);
}

void wait_for_server_ready_or_throw(
    std::int64_t pid,
    const std::string& server_url,
    std::int64_t startup_timeout_s,
    const std::filesystem::path& log_path) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(startup_timeout_s);
  while (std::chrono::steady_clock::now() < deadline) {
    if (!process_alive(pid)) {
      std::ifstream stream(log_path);
      std::stringstream tail;
      tail << stream.rdbuf();
      throw std::runtime_error("managed server exited before becoming ready\n\n" + trim(tail.str()));
    }
    if (server_ready(server_url)) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  throw std::runtime_error("managed server did not become ready within timeout");
}

ManagedServerState parse_state(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream.good()) {
    throw std::runtime_error("unknown managed server: " + path.stem().string());
  }
  std::map<std::string, std::string> values;
  std::string line;
  while (std::getline(stream, line)) {
    const auto pos = line.find('=');
    if (pos == std::string::npos) {
      continue;
    }
    values[line.substr(0, pos)] = line.substr(pos + 1);
  }
  ManagedServerState state;
  state.name = values["name"];
  state.model = values["model"];
  state.session_name = values["session_name"];
  state.server_url = values["server_url"];
  state.output_prefix = values["output_prefix"];
  state.log_path = values["log_path"];
  state.pid = values["pid"].empty() ? 0 : std::stoll(values["pid"]);
  state.port = values["port"].empty() ? 0 : std::stoll(values["port"]);
  state.tp = values["tp"].empty() ? 1 : std::stoll(values["tp"]);
  state.max_model_len =
      values["max_model_len"].empty() ? 2048 : std::stoll(values["max_model_len"]);
  state.trust_remote_code = values["trust_remote_code"] == "true";
  return state;
}

void save_state(const ManagedServerState& state) {
  const auto path = managed_server_state_path(state.name);
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path);
  stream << "name=" << state.name << "\n";
  stream << "model=" << state.model << "\n";
  stream << "session_name=" << state.session_name << "\n";
  stream << "server_url=" << state.server_url << "\n";
  stream << "output_prefix=" << state.output_prefix.string() << "\n";
  stream << "log_path=" << state.log_path.string() << "\n";
  stream << "pid=" << state.pid << "\n";
  stream << "port=" << state.port << "\n";
  stream << "tp=" << state.tp << "\n";
  stream << "max_model_len=" << state.max_model_len << "\n";
  stream << "trust_remote_code=" << (state.trust_remote_code ? "true" : "false") << "\n";
}

std::int64_t start_background_command(
    const std::string& command,
    const std::filesystem::path& log_path) {
  const std::string wrapped =
      "bash -lc " +
      shell_escape(command + " > " + shell_escape(log_path.string()) + " 2>&1 & echo $!");
  return std::stoll(trim(run_command(wrapped)));
}

}  // namespace

ManagedServerState start_managed_server(const ManagedServerConfig& config) {
  if (config.name.empty()) {
    throw std::runtime_error("managed server requires --name");
  }
  if (config.model.empty()) {
    throw std::runtime_error("managed server requires --model");
  }
  if (find_managed_server(config.name).has_value()) {
    throw std::runtime_error("managed server already exists: " + config.name);
  }
  if (config.max_model_len <= 0) {
    throw std::runtime_error("managed server requires --max-model-len > 0");
  }

  const std::filesystem::path output_prefix =
      std::filesystem::path(".hotpath") /
      ("server_" + config.name + "_" + sanitize_model_name(config.model));
  const std::filesystem::path log_path =
      output_prefix.parent_path() / (output_prefix.stem().string() + "_server.log");
  std::filesystem::create_directories(output_prefix.parent_path());
  const std::string session_name = make_session_name(config.name);
  const std::string server_url = "http://127.0.0.1:" + std::to_string(config.port);
  std::vector<std::string> command = {
      "env",
      "VLLM_WORKER_MULTIPROC_METHOD=spawn",
  };
  if (config.debug_logging) {
    command.push_back("VLLM_LOGGING_LEVEL=DEBUG");
  }
  if (config.wrap_with_nsys) {
    command.push_back("nsys");
    command.push_back("launch");
    command.push_back("--trace=cuda,nvtx,osrt");
    command.push_back("--session-new");
    command.push_back(session_name);
    command.push_back("--wait=all");
  }
  const auto launcher = vllm_command();
  command.insert(command.end(), launcher.begin(), launcher.end());
  command.push_back("serve");
  command.push_back(config.model);
  command.push_back("--port");
  command.push_back(std::to_string(config.port));
  command.push_back("--tensor-parallel-size");
  command.push_back(std::to_string(config.tp));
  command.push_back("--max-model-len");
  command.push_back(std::to_string(config.max_model_len));
  if (config.trust_remote_code) {
    command.push_back("--trust-remote-code");
  }
  const std::string serve_command = join_shell_args(command);

  const std::int64_t launcher_pid = start_background_command(serve_command, log_path);
  wait_for_server_ready_or_throw(launcher_pid, server_url, config.startup_timeout_s, log_path);
  std::int64_t server_pid = 0;
  for (int attempt = 0; attempt < 10 && server_pid <= 0; ++attempt) {
    server_pid = listening_pid_for_port(config.port);
    if (server_pid > 0) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  ManagedServerState state{
      .name = config.name,
      .model = config.model,
      .session_name = session_name,
      .server_url = server_url,
      .output_prefix = output_prefix,
      .log_path = log_path,
      .pid = server_pid > 0 ? server_pid : launcher_pid,
      .port = config.port,
      .tp = config.tp,
      .max_model_len = config.max_model_len,
      .trust_remote_code = config.trust_remote_code,
  };
  save_state(state);
  return state;
}

ManagedServerState load_managed_server(const std::string& name) {
  return parse_state(managed_server_state_path(name));
}

std::vector<ManagedServerState> list_managed_servers() {
  std::vector<ManagedServerState> servers;
  const auto dir = managed_server_dir();
  if (!std::filesystem::exists(dir)) {
    return servers;
  }
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.path().extension() == ".cfg") {
      if (remove_stale_state_if_needed(entry.path())) {
        continue;
      }
      servers.push_back(parse_state(entry.path()));
    }
  }
  std::sort(servers.begin(), servers.end(), [](const auto& a, const auto& b) {
    return a.name < b.name;
  });
  return servers;
}

std::optional<ManagedServerState> find_managed_server(const std::string& name) {
  const auto path = managed_server_state_path(name);
  if (!std::filesystem::exists(path)) {
    return std::nullopt;
  }
  if (remove_stale_state_if_needed(path)) {
    return std::nullopt;
  }
  return parse_state(path);
}

bool managed_server_ready(const ManagedServerState& state) {
  if (!server_ready(state.server_url)) {
    return false;
  }
  return state.pid <= 0 || process_alive(state.pid);
}

std::int64_t stop_target_pid(const ManagedServerState& state) {
  const std::int64_t listener_pid = listening_pid_for_port(state.port);
  if (listener_pid > 0) {
    if (state.pid <= 0 || !process_alive(state.pid) || listener_pid == state.pid) {
      return listener_pid;
    }
  }
  if (state.pid > 0 && process_alive(state.pid)) {
    return state.pid;
  }
  return listener_pid;
}

void stop_managed_server(const ManagedServerState& state) {
  release_lock(managed_server_lock_path(state.name));
  if (state.session_name.size() > 0) {
    const int stop_rc =
        std::system(("nsys shutdown --session=" + state.session_name + " >/dev/null 2>&1").c_str());
    static_cast<void>(stop_rc);
    const int stop_session_rc =
        std::system(("nsys stop --session=" + state.session_name + " >/dev/null 2>&1").c_str());
    static_cast<void>(stop_session_rc);
  }
  const std::int64_t live_pid = stop_target_pid(state);
  if (live_pid > 0) {
    const int term_rc = std::system(("kill " + std::to_string(live_pid) + " >/dev/null 2>&1").c_str());
    static_cast<void>(term_rc);
  }
  std::filesystem::remove(managed_server_state_path(state.name));
}

std::size_t prune_stale_managed_servers() {
  std::size_t removed = 0;
  const auto dir = managed_server_dir();
  if (!std::filesystem::exists(dir)) {
    return 0;
  }
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.path().extension() == ".cfg" &&
        remove_stale_state_if_needed(entry.path())) {
      ++removed;
    }
  }
  return removed;
}

std::string render_managed_servers(const std::vector<ManagedServerState>& servers) {
  std::ostringstream out;
  out << "MANAGED SERVERS\n\n";
  out << "name                 ready   url                          max-len   model\n";
  out << "-----------------------------------------------------------------------------------\n";
  for (const auto& server : servers) {
    out << std::left << std::setw(20) << server.name
        << std::setw(8) << (managed_server_ready(server) ? "yes" : "no")
        << std::setw(29) << server.server_url
        << std::setw(10) << server.max_model_len
        << server.model << "\n";
  }
  return out.str();
}

}  // namespace hotpath::profiler
