#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace rlprof::profiler {

struct VllmServeProcessInfo {
  std::string executable;
  std::vector<std::string> argv;
  std::string model;
  std::int64_t port = 8000;
  std::int64_t tp = 1;
  std::int64_t max_model_len = 0;
  bool trust_remote_code = false;
};

struct AttachClonePlan {
  std::int64_t source_pid = 0;
  std::string source_server_url;
  std::string mode;
  std::int64_t traced_port = 0;
  std::string traced_server_url;
  std::string launch_command;
  std::string restore_command;
  VllmServeProcessInfo process;
};

std::optional<VllmServeProcessInfo> parse_vllm_serve_argv(
    const std::vector<std::string>& argv);

std::optional<VllmServeProcessInfo> inspect_vllm_serve_process(
    std::int64_t pid);

bool attach_server_is_local(const std::string& attach_server);

std::optional<AttachClonePlan> build_attach_clone_plan(
    std::int64_t pid,
    const std::string& attach_server,
    const std::string& nsys_path,
    const std::string& session_name,
    const std::filesystem::path& output_prefix);

}  // namespace rlprof::profiler
