#include <cstdlib>
#include <iostream>
#include <string>

#include "rlprof/clock_control.h"

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
  const auto unlocked =
      rlprof::parse_clock_policy_output(unlocked_report, "1710");
  expect_true(unlocked.query_ok, "unlocked parse should mark query ok");
  expect_true(!unlocked.gpu_clocks_locked, "unlocked parse should stay unlocked");
  expect_true(unlocked.max_sm_clock_mhz.has_value() && *unlocked.max_sm_clock_mhz == 1710,
              "max sm clock should be parsed");
  expect_true(rlprof::render_clock_policy(unlocked) == "unlocked",
              "unlocked render should be stable");

  const std::string locked_report =
      "Clocks Event Reasons\n"
      "    Applications Clocks Setting                    : Active\n"
      "GPU Locked Clocks\n"
      "    SM                                             : 1500 MHz\n";
  const auto locked = rlprof::parse_clock_policy_output(locked_report, "1710");
  expect_true(locked.gpu_clocks_locked, "locked parse should detect a locked section");
  expect_true(locked.locked_sm_clock_mhz.has_value() && *locked.locked_sm_clock_mhz == 1500,
              "locked sm clock should be parsed");
  expect_true(rlprof::render_clock_policy(locked) == "locked at 1500 MHz",
              "locked render should include the explicit frequency");
  expect_true(
      rlprof::gpu_clocks_unlocked_warning().find("rlprof lock-clocks") !=
          std::string::npos,
      "warning text should point at the lock-clocks command");

  return 0;
}
