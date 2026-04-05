#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace hotpath::profiler {

struct ManagedServerConfig {
  std::string name;
  std::string model;
  std::int64_t port = 8000;
  std::int64_t tp = 1;
  std::int64_t max_model_len = 2048;
  bool trust_remote_code = false;
  bool wrap_with_nsys = true;
  bool debug_logging = false;
  std::int64_t startup_timeout_s = 300;
};

struct ManagedServerState {
  std::string name;
  std::string model;
  std::string session_name;
  std::string server_url;
  std::filesystem::path output_prefix;
  std::filesystem::path log_path;
  std::int64_t pid = 0;
  std::int64_t port = 0;
  std::int64_t tp = 1;
  std::int64_t max_model_len = 2048;
  bool trust_remote_code = false;
};

ManagedServerState start_managed_server(const ManagedServerConfig& config);
ManagedServerState load_managed_server(const std::string& name);
std::vector<ManagedServerState> list_managed_servers();
std::optional<ManagedServerState> find_managed_server(const std::string& name);
bool managed_server_ready(const ManagedServerState& state);
void stop_managed_server(const ManagedServerState& state);
std::size_t prune_stale_managed_servers();
std::string render_managed_servers(const std::vector<ManagedServerState>& servers);

}  // namespace hotpath::profiler
