#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "rlprof/targets.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  namespace fs = std::filesystem;
  const fs::path temp_root = fs::temp_directory_path() / "rlprof_test_targets";
  fs::remove_all(temp_root);
  fs::create_directories(temp_root / "config");
  setenv("XDG_CONFIG_HOME", (temp_root / "config").c_str(), 1);

  rlprof::save_target({
      .name = "a10g",
      .host = "ubuntu@a10g-box",
      .workdir = "/srv/rlprof",
      .python_executable = "/opt/venvs/rlprof/bin/python",
      .vllm_executable = "/opt/venvs/rlprof/bin/vllm",
  });

  const auto listed = rlprof::list_targets();
  expect_true(listed.size() == 1, "expected one saved target");
  expect_true(listed[0].name == "a10g", "expected saved target name");

  const auto resolved = rlprof::resolve_target("a10g");
  expect_true(resolved.host == "ubuntu@a10g-box", "expected resolved host");
  expect_true(resolved.workdir == "/srv/rlprof", "expected resolved workdir");
  expect_true(
      resolved.python_executable == "/opt/venvs/rlprof/bin/python",
      "expected resolved python executable");
  expect_true(
      resolved.vllm_executable == "/opt/venvs/rlprof/bin/vllm",
      "expected resolved vllm executable");

  const auto direct = rlprof::resolve_target("ubuntu@direct-box", "/tmp/rlprof");
  expect_true(direct.host == "ubuntu@direct-box", "expected direct host");
  expect_true(direct.workdir == "/tmp/rlprof", "expected direct workdir");

  const auto alias = rlprof::resolve_target("gpu-a10g", "/opt/rlprof");
  expect_true(alias.host == "gpu-a10g", "expected alias host fallback");
  expect_true(alias.workdir == "/opt/rlprof", "expected alias workdir fallback");

  const auto rendered = rlprof::render_targets(listed);
  expect_true(rendered.find("TARGETS") != std::string::npos, "expected target table header");
  expect_true(rendered.find("ubuntu@a10g-box") != std::string::npos, "expected target host");
  expect_true(
      rendered.find("/opt/venvs/rlprof/bin/python") != std::string::npos,
      "expected rendered python executable");

  const auto bootstrap = rlprof::bootstrap_target_command(resolved, "/home/ubuntu/rlprof");
  expect_true(bootstrap.find("cmake -S . -B build") != std::string::npos,
              "expected bootstrap build command");
  expect_true(bootstrap.find("cmake --build build --target rlprof") != std::string::npos,
              "expected bootstrap rlprof target build");

  expect_true(rlprof::remove_target("a10g"), "expected target removal");
  expect_true(rlprof::list_targets().empty(), "expected empty target registry");

  unsetenv("XDG_CONFIG_HOME");
  fs::remove_all(temp_root);
  return 0;
}
