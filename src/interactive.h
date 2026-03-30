#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace rlprof::interactive {

struct ProfileConfig {
  std::string model;
  int prompts = 64;
  int rollouts = 4;
  int min_tokens = 256;
  int max_tokens = 1024;
  int input_len = 512;
  int port = 8000;
  int tp = 1;
  std::string peer_servers;
  bool trust_remote_code = false;
  bool discard_first_run = false;
  int repeat = 1;
  std::string output = "auto";
};

struct BenchConfig {
  std::string kernel = "silu_and_mul";
  std::string shapes = "64x4096,256x4096";
  std::string dtype = "bf16";
  int warmup = 20;
  int n_iter = 200;
  int repeats = 3;
};

using ProgressCallback = std::function<void(const std::string&)>;

std::optional<std::string> prompt_string(
    const std::string& label,
    const std::string& default_val = "");
std::optional<int> prompt_int(
    const std::string& label,
    int default_val);
std::optional<bool> prompt_bool(
    const std::string& label,
    bool default_yes = false);
std::optional<int> prompt_choice(
    const std::string& label,
    const std::vector<std::string>& options,
    int default_index = 0);

void print_header(const std::string& text);
void print_warning(const std::string& text);
void print_info(const std::string& label, const std::string& value);

std::string detect_gpu_name();
bool are_clocks_locked();
std::string clock_status_label();
std::vector<std::string> list_recent_profiles(int max_count = 10);
ProfileConfig load_profile_defaults();
BenchConfig load_bench_defaults();
void save_profile_defaults(const ProfileConfig& config);
void save_bench_defaults(const BenchConfig& config);
void clear_saved_defaults();

std::vector<std::string> build_profile_args(const ProfileConfig& config);
std::vector<std::string> build_bench_args(const BenchConfig& config);

void run_with_progress(
    const std::string& initial_status,
    const std::function<void(const ProgressCallback&)>& action);

}  // namespace rlprof::interactive
