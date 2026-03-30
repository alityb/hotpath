#include "rlprof/clock_control.h"

#include <array>
#include <cctype>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

std::vector<std::string> split_lines(const std::string& text) {
  std::vector<std::string> lines;
  std::stringstream stream(text);
  std::string line;
  while (std::getline(stream, line)) {
    lines.push_back(line);
  }
  return lines;
}

std::string run_command_capture(const std::string& command) {
  std::array<char, 4096> buffer{};
  std::string output;
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    throw std::runtime_error("failed to run command");
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output.append(buffer.data());
  }
  const int rc = pclose(pipe);
  if (rc != 0) {
    throw std::runtime_error(output.empty() ? "command failed" : trim(output));
  }
  return trim(output);
}

int run_command_status(const std::string& command) {
  return std::system(command.c_str());
}

std::optional<std::string> extract_report_value(
    const std::string& report,
    const std::string& key) {
  for (const auto& line : split_lines(report)) {
    const auto pos = line.find(':');
    if (pos == std::string::npos) {
      continue;
    }
    const std::string left = trim(line.substr(0, pos));
    if (left == key) {
      return trim(line.substr(pos + 1));
    }
  }
  return std::nullopt;
}

std::optional<std::int64_t> parse_mhz_value(const std::string& text) {
  std::string digits;
  for (char ch : text) {
    if (std::isdigit(static_cast<unsigned char>(ch))) {
      digits.push_back(ch);
    } else if (!digits.empty()) {
      break;
    }
  }
  if (digits.empty()) {
    return std::nullopt;
  }
  return std::stoll(digits);
}

std::optional<std::int64_t> extract_locked_sm_clock_mhz(
    const std::string& report) {
  bool in_locked_section = false;
  for (const auto& raw_line : split_lines(report)) {
    const std::string line = trim(raw_line);
    if (line.empty()) {
      continue;
    }
    if (line.find("GPU Locked Clocks") != std::string::npos ||
        line.find("Locked Clocks") != std::string::npos) {
      in_locked_section = true;
      continue;
    }
    if (!in_locked_section) {
      continue;
    }
    if (line.find(':') == std::string::npos) {
      in_locked_section = false;
      continue;
    }
    const auto pos = line.find(':');
    const std::string key = trim(line.substr(0, pos));
    if (key == "SM" || key == "Graphics") {
      return parse_mhz_value(line.substr(pos + 1));
    }
  }
  return std::nullopt;
}

std::string lock_failure_message(const std::string& command) {
  return "failed to lock GPU clocks with nvidia-smi. Run `" + command +
         "` manually (root may be required).";
}

std::string unlock_failure_message() {
  return "failed to unlock GPU clocks with nvidia-smi. Run "
         "`nvidia-smi --reset-gpu-clocks` manually (root may be required).";
}

}  // namespace

ClockPolicyInfo parse_clock_policy_output(
    const std::string& nvidia_smi_report,
    const std::string& max_sm_clock_output) {
  ClockPolicyInfo info;
  info.query_ok = true;
  info.max_sm_clock_mhz = parse_mhz_value(max_sm_clock_output);
  info.locked_sm_clock_mhz = extract_locked_sm_clock_mhz(nvidia_smi_report);

  const auto apps_setting =
      extract_report_value(nvidia_smi_report, "Applications Clocks Setting");
  info.applications_clocks_active =
      apps_setting.has_value() && *apps_setting == "Active";
  info.gpu_clocks_locked = info.locked_sm_clock_mhz.has_value();
  info.lock_status = info.gpu_clocks_locked ? "locked" : "unlocked";
  return info;
}

ClockPolicyInfo query_clock_policy() {
  try {
    return parse_clock_policy_output(
        run_command_capture("nvidia-smi -q"),
        run_command_capture(
            "nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -n 1"));
  } catch (const std::exception&) {
    return ClockPolicyInfo{};
  }
}

std::int64_t query_max_sm_clock_mhz() {
  const std::string output = run_command_capture(
      "nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -n 1");
  const auto value = parse_mhz_value(output);
  if (!value.has_value()) {
    throw std::runtime_error("failed to query max supported SM clock via nvidia-smi");
  }
  return *value;
}

std::string render_clock_policy(const ClockPolicyInfo& info) {
  if (!info.query_ok) {
    return "unknown";
  }
  if (info.gpu_clocks_locked && info.locked_sm_clock_mhz.has_value()) {
    return "locked at " + std::to_string(*info.locked_sm_clock_mhz) + " MHz";
  }
  return "unlocked";
}

std::string gpu_clocks_unlocked_warning() {
  return "GPU clocks are not locked. Run `rlprof lock-clocks` for reproducible "
         "measurements. See: docs.nvidia.com/deploy/nvidia-smi/index.html";
}

void lock_gpu_clocks(std::optional<std::int64_t> freq_mhz) {
  const std::int64_t freq = freq_mhz.value_or(query_max_sm_clock_mhz());
  const std::string command = "nvidia-smi --lock-gpu-clocks=" +
                              std::to_string(freq) + "," +
                              std::to_string(freq);
  if (run_command_status(command + " > /dev/null 2>&1") != 0) {
    throw std::runtime_error(lock_failure_message(command));
  }
}

void unlock_gpu_clocks() {
  if (run_command_status("nvidia-smi --reset-gpu-clocks > /dev/null 2>&1") != 0) {
    throw std::runtime_error(unlock_failure_message());
  }
}

}  // namespace rlprof
