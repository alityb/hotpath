#include "hotpath/doctor.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "hotpath/clock_control.h"
#include "hotpath/serve_profiler.h"

namespace hotpath {
namespace {

struct CommandResult {
  bool ok = false;
  std::string output;
};

CommandResult run_command(const std::string& command) {
  std::array<char, 4096> buffer{};
  std::string output;
  FILE* pipe = popen((command + " 2>&1").c_str(), "r");
  if (pipe == nullptr) {
    return {};
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output.append(buffer.data());
  }
  const int rc = pclose(pipe);
  while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
    output.pop_back();
  }
  return CommandResult{.ok = rc == 0, .output = output};
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

std::string first_line(const std::string& text) {
  const auto pos = text.find('\n');
  return pos == std::string::npos ? text : text.substr(0, pos);
}

std::string last_line(const std::string& text) {
  if (text.empty()) {
    return "";
  }
  const auto pos = text.rfind('\n');
  return pos == std::string::npos ? text : text.substr(pos + 1);
}

std::string configured_value(const char* key) {
  const char* value = std::getenv(key);
  if (value == nullptr) {
    return "";
  }
  return value;
}

std::optional<std::string> resolve_candidate_path(
    const std::string& candidate,
    bool path_lookup) {
  if (path_lookup) {
    const auto resolved = run_command("command -v " + shell_escape(candidate));
    if (!resolved.ok || resolved.output.empty()) {
      return std::nullopt;
    }
    return first_line(resolved.output);
  }
  if (!std::filesystem::exists(candidate)) {
    return std::nullopt;
  }
  return std::filesystem::absolute(candidate).lexically_normal().string();
}

RuntimeToolInfo resolve_tool(
    const std::string& explicit_path,
    const std::vector<std::vector<std::pair<std::string, bool>>>& candidate_tiers,
    const std::string& version_arg) {
  RuntimeToolInfo tool;
  tool.requested_path = explicit_path;
  tool.explicit_config = !explicit_path.empty();

  if (tool.explicit_config) {
    tool.resolved_path = explicit_path;
    const auto version =
        run_command(shell_escape(explicit_path) + " " + version_arg);
    tool.found = version.ok;
    tool.version = version.ok ? first_line(version.output) : "";
    tool.detail = version.ok ? first_line(version.output)
                             : last_line(version.output.empty() ? "missing" : version.output);
    return tool;
  }

  for (const auto& tier : candidate_tiers) {
    for (const auto& [candidate, path_lookup] : tier) {
      const auto resolved = resolve_candidate_path(candidate, path_lookup);
      if (!resolved.has_value()) {
        continue;
      }
      tool.resolved_path = *resolved;
      const auto version =
          run_command(shell_escape(tool.resolved_path) + " " + version_arg);
      tool.found = version.ok;
      tool.version = version.ok ? first_line(version.output) : "";
      tool.detail = version.ok ? first_line(version.output)
                               : last_line(version.output.empty() ? "missing" : version.output);
      if (tool.found) {
        return tool;
      }
    }
    if (!tool.resolved_path.empty()) {
      return tool;
    }
  }

  tool.detail = "not found";
  return tool;
}

DoctorStatus tool_status(const RuntimeToolInfo& tool) {
  if (tool.ambiguous || !tool.found) {
    return DoctorStatus::kFail;
  }
  return DoctorStatus::kPass;
}

std::string tool_detail(const RuntimeToolInfo& tool) {
  if (tool.ambiguous || !tool.found) {
    return tool.detail;
  }
  if (tool.resolved_path.empty()) {
    return tool.version;
  }
  return tool.resolved_path + " (" + tool.version + ")";
}

std::string status_label(DoctorStatus status) {
  switch (status) {
    case DoctorStatus::kPass:
      return "PASS";
    case DoctorStatus::kWarn:
      return "WARN";
    case DoctorStatus::kFail:
      return "FAIL";
  }
  return "FAIL";
}

}  // namespace

RuntimeEnvironmentInfo inspect_runtime_environment() {
  RuntimeEnvironmentInfo environment;
  environment.python = resolve_tool(
      configured_value("RLPROF_PYTHON_EXECUTABLE"),
      {
          {{".venv/bin/python", false}},
          {{"venv/bin/python", false}},
          {{"python3", true}, {"python", true}},
      },
      "--version");
  environment.vllm = resolve_tool(
      configured_value("RLPROF_VLLM_EXECUTABLE"),
      {
          {{".venv/bin/vllm", false}},
          {{"venv/bin/vllm", false}},
          {{"vllm", true}},
      },
      "--version");
  environment.nsys = resolve_tool(
      "",
      {
          {{"nsys", true}},
      },
      "--version");

  const auto gpu = run_command(
      "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits | head -n 1");
  if (gpu.ok && !gpu.output.empty()) {
    const auto comma = gpu.output.find(',');
    if (comma == std::string::npos) {
      environment.gpu_name = first_line(gpu.output);
    } else {
      environment.gpu_name = first_line(gpu.output.substr(0, comma));
      environment.driver_version = first_line(gpu.output.substr(comma + 1));
      while (!environment.driver_version.empty() &&
             environment.driver_version.front() == ' ') {
        environment.driver_version.erase(environment.driver_version.begin());
      }
    }
  }

  const auto visible_devices = run_command(
      "nvidia-smi --query-gpu=index --format=csv,noheader,nounits");
  if (visible_devices.ok) {
    std::stringstream stream(visible_devices.output);
    std::string line;
    std::vector<std::string> ids;
    while (std::getline(stream, line)) {
      line = first_line(line);
      if (!line.empty()) {
        ids.push_back(line);
      }
    }
    environment.cuda_visible_device_count = static_cast<std::int64_t>(ids.size());
    std::ostringstream joined;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (i > 0) {
        joined << ",";
      }
      joined << ids[i];
    }
    environment.cuda_visible_devices = joined.str();
  }

  if (environment.python.found && !environment.python.ambiguous &&
      environment.vllm.found && !environment.vllm.ambiguous) {
    const auto bench = run_command(
        shell_escape(environment.python.resolved_path) +
        " -c \"import torch, vllm, hotpath_py.bench_cuda; "
        "raise SystemExit(0 if torch.cuda.is_available() else 1)\"");
    environment.bench_helper_ok = bench.ok;
    environment.bench_helper_detail =
        bench.ok ? "torch, vllm, and CUDA available" : last_line(bench.output);
  } else {
    environment.bench_helper_detail = "python/vllm unresolved";
  }

  const auto nsys_env = run_command("nsys status -e");
  environment.nsys_environment_ok = nsys_env.ok;
  environment.nsys_environment_detail = first_line(nsys_env.output);

  return environment;
}

std::vector<DoctorCheck> doctor_checks_from_environment(
    const RuntimeEnvironmentInfo& environment) {
  std::vector<DoctorCheck> checks;
  checks.push_back(DoctorCheck{
      .name = "python",
      .status = tool_status(environment.python),
      .detail = tool_detail(environment.python),
  });
  checks.push_back(DoctorCheck{
      .name = "vllm",
      .status = tool_status(environment.vllm),
      .detail = tool_detail(environment.vllm),
  });
  checks.push_back(DoctorCheck{
      .name = "nsys",
      .status = tool_status(environment.nsys),
      .detail = tool_detail(environment.nsys),
  });
  checks.push_back(DoctorCheck{
      .name = "cuda visibility",
      .status = environment.cuda_visible_device_count > 0 ? DoctorStatus::kPass
                                                          : DoctorStatus::kFail,
      .detail = environment.cuda_visible_device_count > 0
                    ? environment.gpu_name + " visible GPUs=" +
                          std::to_string(environment.cuda_visible_device_count)
                    : "no visible NVIDIA GPUs",
  });
  checks.push_back(DoctorCheck{
      .name = "driver",
      .status = !environment.driver_version.empty() ? DoctorStatus::kPass
                                                    : DoctorStatus::kFail,
      .detail = environment.driver_version.empty() ? "driver unknown"
                                                   : environment.driver_version,
  });
  checks.push_back(DoctorCheck{
      .name = "bench helper",
      .status = environment.bench_helper_ok ? DoctorStatus::kPass : DoctorStatus::kWarn,
      .detail = environment.bench_helper_detail,
  });
  return checks;
}

std::map<std::string, std::string> runtime_environment_metadata(
    const RuntimeEnvironmentInfo& environment) {
  return {
      {"measurement_python_path", environment.python.resolved_path},
      {"measurement_python_version", environment.python.version},
      {"measurement_vllm_path", environment.vllm.resolved_path},
      {"measurement_vllm_version", environment.vllm.version},
      {"measurement_nsys_path", environment.nsys.resolved_path},
      {"measurement_nsys_version", environment.nsys.version},
      {"measurement_nvidia_driver_version", environment.driver_version},
      {"measurement_cuda_visible_devices", environment.cuda_visible_devices},
      {"measurement_cuda_visible_device_count",
       std::to_string(environment.cuda_visible_device_count)},
  };
}

std::vector<DoctorCheck> run_doctor() {
  const RuntimeEnvironmentInfo environment = inspect_runtime_environment();
  auto checks = doctor_checks_from_environment(environment);

  const ClockPolicyInfo clock_policy = query_clock_policy();
  checks.push_back(DoctorCheck{
      .name = "clock policy",
      .status = !clock_policy.query_ok
                    ? DoctorStatus::kWarn
                    : (clock_policy.gpu_clocks_locked ? DoctorStatus::kPass
                                                      : DoctorStatus::kWarn),
      .detail = render_clock_policy(clock_policy),
  });
  checks.push_back(DoctorCheck{
      .name = "nsys environment",
      .status = environment.nsys_environment_ok ? DoctorStatus::kPass
                                                : DoctorStatus::kWarn,
      .detail = environment.nsys_environment_detail,
  });

  // Serving readiness checks
  const auto gpu = detect_gpus();
  checks.push_back(DoctorCheck{
      .name = "serve-profile GPUs",
      .status = (gpu.count >= 2) ? DoctorStatus::kPass
                : (gpu.count == 1) ? DoctorStatus::kWarn
                : DoctorStatus::kFail,
      .detail = (gpu.count > 0)
                    ? (std::to_string(gpu.count) + "x " + gpu.name + " (" +
                       std::to_string(gpu.memory_mb) + " MB)" +
                       (gpu.count < 2 ? " — disaggregation needs >= 2 GPUs" : ""))
                    : "no NVIDIA GPU detected",
  });

  // Check serving engines using the detected python
  const std::string py = environment.python.found
      ? shell_escape(environment.python.resolved_path) : "python3";

  const auto sglang_check = run_command(
      py + " -c \"import sglang; print(sglang.__version__)\"");
  checks.push_back(DoctorCheck{
      .name = "sglang (serve-profile)",
      .status = sglang_check.ok ? DoctorStatus::kPass : DoctorStatus::kWarn,
      .detail = sglang_check.ok
                    ? ("sglang " + first_line(sglang_check.output))
                    : "not found (pip install sglang[all])",
  });

  return checks;
}

std::string render_doctor_report(const std::vector<DoctorCheck>& checks, bool color) {
  const char* reset = color ? "\033[0m" : "";
  const char* bold = color ? "\033[1m" : "";
  const char* green = color ? "\033[32m" : "";
  const char* yellow = color ? "\033[33m" : "";
  const char* red = color ? "\033[31m" : "";

  std::ostringstream out;
  out << "DOCTOR\n\n";
  out << std::left << std::setw(18) << "check" << "  "
      << std::setw(6) << "status" << "  detail\n";
  out << std::string(72, '-') << "\n";
  for (const auto& check : checks) {
    const char* status_color = reset;
    if (check.status == DoctorStatus::kPass) {
      status_color = green;
    } else if (check.status == DoctorStatus::kWarn) {
      status_color = yellow;
    } else {
      status_color = red;
    }
    out << std::left << std::setw(18) << check.name << "  "
        << status_color << bold << std::setw(6) << status_label(check.status)
        << reset << "  " << check.detail << "\n";
  }
  return out.str();
}

}  // namespace hotpath
