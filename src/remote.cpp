#include "rlprof/remote.h"

#include <sstream>
#include <string>

namespace rlprof {
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

std::string join_args(const std::vector<std::string>& args) {
  std::ostringstream out;
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (i > 0) {
      out << " ";
    }
    out << shell_escape(args[i]);
  }
  return out.str();
}

std::string remote_env_prefix(const RemoteTarget& target) {
  std::string prefix;
  if (!target.python_executable.empty()) {
    prefix += "export RLPROF_PYTHON_EXECUTABLE=" + shell_escape(target.python_executable) + "; ";
  }
  if (!target.vllm_executable.empty()) {
    prefix += "export RLPROF_VLLM_EXECUTABLE=" + shell_escape(target.vllm_executable) + "; ";
  }
  return prefix;
}

std::string portable_sha256_command(const std::string& escaped_path) {
  return "(if command -v sha256sum >/dev/null 2>&1; then sha256sum " + escaped_path +
         "; elif command -v shasum >/dev/null 2>&1; then shasum -a 256 " + escaped_path +
         "; else openssl dgst -sha256 " + escaped_path +
         "; fi) | awk 'NF {print $NF}'";
}

}  // namespace

bool has_remote_target(const RemoteTarget& target) {
  return !target.host.empty();
}

std::string remote_join(const RemoteTarget& target, const std::filesystem::path& local_path) {
  const std::string leaf = local_path.filename().string();
  return target.workdir + "/.rlprof/" + leaf;
}

std::string remote_shell_command(
    const RemoteTarget& target,
    const std::string& shell_command) {
  return "ssh " + shell_escape(target.host) + " " +
         shell_escape(remote_env_prefix(target) + shell_command);
}

std::string remote_cli_command(
    const RemoteTarget& target,
    const std::vector<std::string>& args) {
  return remote_shell_command(
      target,
      "cd " + shell_escape(target.workdir) + " && ./build/rlprof " + join_args(args));
}

std::string remote_file_exists_command(
    const RemoteTarget& target,
    const std::string& remote_path) {
  const std::string remote_command = "test -f " + shell_escape(remote_path);
  return "ssh " + shell_escape(target.host) + " " + shell_escape(remote_command);
}

std::string remote_checksum_command(
    const RemoteTarget& target,
    const std::string& remote_path) {
  const std::string remote_command =
      portable_sha256_command(shell_escape(remote_path));
  return "ssh " + shell_escape(target.host) + " " + shell_escape(remote_command);
}

std::string remote_copy_from_command(
    const RemoteTarget& target,
    const std::string& remote_path,
    const std::filesystem::path& local_path) {
  return "scp -q " + shell_escape(target.host + ":" + remote_path) + " " +
         shell_escape(local_path.string());
}

std::string remote_tail_command(
    const RemoteTarget& target,
    const std::string& remote_path,
    int lines) {
  const std::string remote_command =
      "tail -n " + std::to_string(lines) + " " + shell_escape(remote_path) +
      " 2>/dev/null";
  return "ssh " + shell_escape(target.host) + " " + shell_escape(remote_command);
}

std::string remote_epoch_ms_command(const RemoteTarget& target) {
  const std::string remote_command = "date +%s%3N";
  return "ssh " + shell_escape(target.host) + " " + shell_escape(remote_command);
}

}  // namespace rlprof
