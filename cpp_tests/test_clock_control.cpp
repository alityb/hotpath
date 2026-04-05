#include <cstdlib>
#include <iostream>
#include <string>

#include "hotpath/clock_control.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  const std::string unlocked_report =
      "Clocks Event Reasons\n"
      "    Applications Clocks Setting                    : Not Active\n"
      "Clocks\n"
      "    SM                                             : 210 MHz\n";
  // current clock (210) != max (1710) → not locked
  const auto unlocked =
      hotpath::parse_clock_policy_output(unlocked_report, "1710", "210");
  expect_true(unlocked.query_ok, "unlocked parse should mark query ok");
  expect_true(!unlocked.gpu_clocks_locked, "unlocked parse should stay unlocked");
  expect_true(unlocked.max_sm_clock_mhz.has_value() && *unlocked.max_sm_clock_mhz == 1710,
              "max sm clock should be parsed");
  expect_true(hotpath::render_clock_policy(unlocked) == "unlocked",
              "unlocked render should be stable");

  const std::string locked_report =
      "Clocks Event Reasons\n"
      "    Applications Clocks Setting                    : Active\n"
      "GPU Locked Clocks\n"
      "    SM                                             : 1500 MHz\n";
  const auto locked = hotpath::parse_clock_policy_output(locked_report, "1710", "1500");
  expect_true(locked.gpu_clocks_locked, "locked parse should detect a locked section");
  expect_true(locked.locked_sm_clock_mhz.has_value() && *locked.locked_sm_clock_mhz == 1500,
              "locked sm clock should be parsed");
  expect_true(hotpath::render_clock_policy(locked) == "locked at 1500 MHz",
              "locked render should include the explicit frequency");

  // Fallback: no "GPU Locked Clocks" section but current == max (e.g. A10G on AWS)
  const std::string at_max_report =
      "Clocks Event Reasons\n"
      "    Applications Clocks Setting                    : Not Active\n"
      "Clocks\n"
      "    SM                                             : 1710 MHz\n";
  const auto at_max = hotpath::parse_clock_policy_output(at_max_report, "1710", "1710");
  expect_true(at_max.gpu_clocks_locked, "at-max fallback should mark as locked");
  expect_true(at_max.locked_sm_clock_mhz.has_value() && *at_max.locked_sm_clock_mhz == 1710,
              "at-max fallback should set locked_sm_clock_mhz");
  expect_true(hotpath::render_clock_policy(at_max) == "locked at 1710 MHz",
              "at-max fallback render should show frequency");
  expect_true(
      hotpath::gpu_clocks_unlocked_warning().find("hotpath lock-clocks") !=
          std::string::npos,
      "warning text should point at the lock-clocks command");

  return 0;
}
