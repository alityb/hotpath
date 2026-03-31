#include "rlprof/targets.h"

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace rlprof {
namespace {

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

std::filesystem::path targets_path() {
  const char* xdg_config_home = std::getenv("XDG_CONFIG_HOME");
  if (xdg_config_home != nullptr && std::string(xdg_config_home).size() > 0) {
    return std::filesystem::path(xdg_config_home) / "rlprof" / "targets.cfg";
  }
  const char* home = std::getenv("HOME");
  if (home != nullptr && std::string(home).size() > 0) {
    return std::filesystem::path(home) / ".config" / "rlprof" / "targets.cfg";
  }
  return std::filesystem::path(".rlprof") / "targets.cfg";
}

std::map<std::string, SavedTarget> load_target_map() {
  std::map<std::string, SavedTarget> targets;
  std::ifstream in(targets_path());
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty()) {
      continue;
    }
    std::stringstream stream(line);
    std::string name;
    std::string host;
    std::string workdir;
    std::string python_executable;
    std::string vllm_executable;
    if (!std::getline(stream, name, '|') ||
        !std::getline(stream, host, '|') ||
        !std::getline(stream, workdir, '|')) {
      continue;
    }
    std::getline(stream, python_executable, '|');
    std::getline(stream, vllm_executable);
    targets[name] = SavedTarget{
        .name = trim(name),
        .host = trim(host),
        .workdir = trim(workdir),
        .python_executable = trim(python_executable),
        .vllm_executable = trim(vllm_executable),
    };
  }
  return targets;
}

void save_target_map(const std::map<std::string, SavedTarget>& targets) {
  std::filesystem::create_directories(targets_path().parent_path());
  std::ofstream out(targets_path());
  for (const auto& [name, target] : targets) {
    out << name << "|" << target.host << "|" << target.workdir << "|"
        << target.python_executable << "|" << target.vllm_executable << "\n";
  }
}

bool looks_like_host(const std::string& spec) {
  return spec.find('@') != std::string::npos || spec.find('.') != std::string::npos ||
         spec.find(':') != std::string::npos;
}

}  // namespace

std::vector<SavedTarget> list_targets() {
  std::vector<SavedTarget> targets;
  for (const auto& [_, target] : load_target_map()) {
    targets.push_back(target);
  }
  return targets;
}

void save_target(const SavedTarget& target) {
  auto targets = load_target_map();
  targets[target.name] = target;
  save_target_map(targets);
}

bool remove_target(const std::string& name) {
  auto targets = load_target_map();
  const auto erased = targets.erase(name);
  save_target_map(targets);
  return erased > 0;
}

std::string render_targets(const std::vector<SavedTarget>& targets) {
  std::ostringstream out;
  out << "TARGETS\n\n";
  out << "name                  host                  workdir\n";
  out << "---------------------------------------------------------------\n";
  for (const auto& target : targets) {
    out << target.name;
    if (target.name.size() < 22) {
      out << std::string(22 - target.name.size(), ' ');
    } else {
      out << "  ";
    }
    out << target.host;
    if (target.host.size() < 22) {
      out << std::string(22 - target.host.size(), ' ');
    } else {
      out << "  ";
    }
    out << target.workdir << "\n";
    if (!target.python_executable.empty()) {
      out << "  python: " << target.python_executable << "\n";
    }
    if (!target.vllm_executable.empty()) {
      out << "  vllm: " << target.vllm_executable << "\n";
    }
  }
  return out.str();
}

RemoteTarget resolve_target(const std::string& spec, const std::string& workdir_override) {
  RemoteTarget target;
  if (spec.empty()) {
    return target;
  }
  if (looks_like_host(spec)) {
    target.host = spec;
    target.workdir = workdir_override.empty() ? target.workdir : workdir_override;
    return target;
  }
  const auto targets = load_target_map();
  const auto it = targets.find(spec);
  if (it == targets.end()) {
    if (!workdir_override.empty()) {
      target.host = spec;
      target.workdir = workdir_override;
      return target;
    }
    throw std::runtime_error("unknown target: " + spec);
  }
  target.host = it->second.host;
  target.workdir = workdir_override.empty() ? it->second.workdir : workdir_override;
  target.python_executable = it->second.python_executable;
  target.vllm_executable = it->second.vllm_executable;
  return target;
}

std::string bootstrap_target_command(
    const RemoteTarget& target,
    const std::string& local_repo_root) {
  return "COPYFILE_DISABLE=1 tar -C " + shell_escape(local_repo_root) +
         " --exclude=.git --exclude=build --exclude=.venv --exclude=.rlprof -cf - . | "
         "ssh " + shell_escape(target.host) + " " +
         shell_escape(
             "mkdir -p " + shell_escape(target.workdir) +
             " && tar -xf - -C " + shell_escape(target.workdir) +
             " && cd " + shell_escape(target.workdir) +
             " && cmake -S . -B build && cmake --build build --target rlprof");
}

}  // namespace rlprof
