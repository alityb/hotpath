#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace rlprof {

struct ClockPolicyInfo {
  bool query_ok = false;
  bool gpu_clocks_locked = false;
  bool applications_clocks_active = false;
  std::optional<std::int64_t> locked_sm_clock_mhz;
  std::optional<std::int64_t> max_sm_clock_mhz;
  std::string lock_status = "unknown";
};

ClockPolicyInfo parse_clock_policy_output(
    const std::string& nvidia_smi_report,
    const std::string& max_sm_clock_output);

ClockPolicyInfo query_clock_policy();
std::int64_t query_max_sm_clock_mhz();
std::string render_clock_policy(const ClockPolicyInfo& info);
std::string gpu_clocks_unlocked_warning();
void lock_gpu_clocks(std::optional<std::int64_t> freq_mhz = std::nullopt);
void unlock_gpu_clocks();

}  // namespace rlprof
