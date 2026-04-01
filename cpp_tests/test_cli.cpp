#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

struct CommandResult {
  int rc = -1;
  std::string output;
};

CommandResult run_command(const std::string& command) {
  std::array<char, 4096> buffer{};
  std::string output;
  FILE* pipe = popen((command + " 2>&1").c_str(), "r");
  if (pipe == nullptr) {
    throw std::runtime_error("failed to run command");
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output.append(buffer.data());
  }
  const int rc = pclose(pipe);
  return CommandResult{.rc = rc, .output = output};
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

}  // namespace

int main(int argc, char** argv) {
  namespace fs = std::filesystem;
  const fs::path exe_dir = fs::absolute(argv[0]).parent_path();
  const fs::path rlprof = exe_dir / "rlprof";
  const fs::path temp_root = fs::temp_directory_path() / "rlprof_test_cli";
  fs::remove_all(temp_root);
  fs::create_directories(temp_root);

  const std::string export_command =
      "cd " + shell_escape(temp_root.string()) + " && " +
      shell_escape(rlprof.string()) + " export --format json";
  const auto export_result = run_command(export_command);
  expect_true(export_result.rc == 0, "expected export with no profiles to exit cleanly");
  expect_true(
      export_result.output.find("No profiles found in .rlprof/") != std::string::npos,
      "expected clean no-profile export message");

  const std::string validate_command =
      "cd " + shell_escape(temp_root.string()) + " && " +
      shell_escape(rlprof.string()) + " validate";
  const auto validate_result = run_command(validate_command);
  expect_true(validate_result.rc != 0, "expected validate with no profiles to fail cleanly");
  expect_true(
      validate_result.output.find("No profile database found in .rlprof/") != std::string::npos,
      "expected clean no-profile validate message");

  fs::remove_all(temp_root);
  return 0;
}
