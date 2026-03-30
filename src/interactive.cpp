#include "interactive.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "rlprof/clock_control.h"

namespace rlprof::interactive {
namespace {

constexpr const char* RESET = "\033[0m";
constexpr const char* BOLD = "\033[1m";
constexpr const char* DIM = "\033[2m";
constexpr const char* CYAN = "\033[36m";
constexpr const char* GREEN = "\033[32m";
constexpr const char* YELLOW = "\033[33m";
constexpr const char* WHITE = "\033[37m";

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

std::string run_command_capture(const std::string& command) {
  std::array<char, 4096> buffer{};
  std::string output;
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    return "";
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output.append(buffer.data());
  }
  const int rc = pclose(pipe);
  if (rc != 0) {
    return "";
  }
  return trim(output);
}

std::optional<std::string> read_line() {
  std::string line;
  if (!std::getline(std::cin, line)) {
    return std::nullopt;
  }
  if (line == "q" || line == "Q") {
    return std::nullopt;
  }
  return trim(line);
}

void print_prompt_prefix(const std::string& label, const std::string& default_value) {
  std::cout << "  " << CYAN << "? " << RESET << BOLD << label << RESET;
  if (!default_value.empty()) {
    std::cout << " " << DIM << "(" << default_value << ")" << RESET;
  }
  std::cout << ": " << GREEN << "> " << RESET;
  std::flush(std::cout);
}

std::string format_timestamp(const std::filesystem::path& path) {
  namespace fs = std::filesystem;
  try {
    const auto file_time = fs::last_write_time(path);
    const auto system_time =
        std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            file_time - fs::file_time_type::clock::now() +
            std::chrono::system_clock::now());
    const std::time_t time = std::chrono::system_clock::to_time_t(system_time);
    std::tm tm{};
    gmtime_r(&time, &tm);
    std::ostringstream stream;
    stream << std::put_time(&tm, "%Y-%m-%d %H:%M");
    return stream.str();
  } catch (const std::exception&) {
    return "";
  }
}

}  // namespace

std::optional<std::string> prompt_string(
    const std::string& label,
    const std::string& default_val) {
  print_prompt_prefix(label, default_val);
  const auto line = read_line();
  if (!line.has_value()) {
    return std::nullopt;
  }
  if (line->empty()) {
    return default_val;
  }
  return *line;
}

std::optional<int> prompt_int(const std::string& label, int default_val) {
  while (true) {
    const auto line = prompt_string(label, std::to_string(default_val));
    if (!line.has_value()) {
      return std::nullopt;
    }
    try {
      const int value = std::stoi(*line);
      if (value > 0) {
        return value;
      }
    } catch (const std::exception&) {
    }
    print_warning(label + " must be a positive integer");
  }
}

std::optional<bool> prompt_bool(const std::string& label, bool default_yes) {
  const std::string suffix = default_yes ? "Y/n" : "y/N";
  while (true) {
    print_prompt_prefix(label + "?", suffix);
    const auto line = read_line();
    if (!line.has_value()) {
      return std::nullopt;
    }
    if (line->empty()) {
      return default_yes;
    }
    std::string value = *line;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
      return static_cast<char>(std::tolower(ch));
    });
    if (value == "y" || value == "yes") {
      return true;
    }
    if (value == "n" || value == "no") {
      return false;
    }
    print_warning("enter y or n");
  }
}

std::optional<int> prompt_choice(
    const std::string& label,
    const std::vector<std::string>& options,
    int default_index) {
  if (options.empty()) {
    return std::nullopt;
  }
  std::cout << "  " << CYAN << "? " << RESET << BOLD << label << RESET << "\n";
  for (std::size_t i = 0; i < options.size(); ++i) {
    std::cout << "    [" << (i + 1) << "] " << options[i] << "\n";
  }
  while (true) {
    print_prompt_prefix("Select", std::to_string(default_index + 1));
    const auto line = read_line();
    if (!line.has_value()) {
      return std::nullopt;
    }
    if (line->empty()) {
      return default_index;
    }
    try {
      const int selection = std::stoi(*line);
      if (selection >= 1 && selection <= static_cast<int>(options.size())) {
        return selection - 1;
      }
    } catch (const std::exception&) {
    }
    print_warning("enter a number between 1 and " + std::to_string(options.size()));
  }
}

void print_header(const std::string& text) {
  std::cout << "\n  " << BOLD << CYAN << text << RESET << "\n\n";
}

void print_warning(const std::string& text) {
  std::cout << "  " << YELLOW << "!" << RESET << " " << text << "\n";
}

void print_info(const std::string& label, const std::string& value) {
  std::cout << "  " << WHITE << label << RESET;
  if (!value.empty()) {
    std::cout << " " << DIM << value << RESET;
  }
  std::cout << "\n";
}

std::string detect_gpu_name() {
  const std::string gpu_name = run_command_capture(
      "nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1");
  return gpu_name.empty() ? "unknown" : gpu_name;
}

bool are_clocks_locked() {
  const auto info = rlprof::query_clock_policy();
  return info.query_ok && info.gpu_clocks_locked;
}

std::string clock_status_label() {
  return rlprof::render_clock_policy(rlprof::query_clock_policy());
}

std::vector<std::string> list_recent_profiles(int max_count) {
  namespace fs = std::filesystem;
  std::vector<std::pair<fs::file_time_type, std::string>> entries;
  const fs::path root = ".rlprof";
  if (!fs::exists(root)) {
    return {};
  }

  for (const auto& entry : fs::directory_iterator(root)) {
    if (entry.path().extension() == ".db") {
      entries.push_back({entry.last_write_time(), entry.path().string()});
    }
  }

  std::sort(entries.begin(), entries.end(), [](const auto& left, const auto& right) {
    return left.first > right.first;
  });

  std::vector<std::string> paths;
  for (std::size_t i = 0; i < entries.size() && i < static_cast<std::size_t>(max_count); ++i) {
    paths.push_back(entries[i].second);
  }
  return paths;
}

std::vector<std::string> build_profile_args(const ProfileConfig& config) {
  std::vector<std::string> args = {
      "profile",
      "--model",
      config.model,
      "--prompts",
      std::to_string(config.prompts),
      "--rollouts",
      std::to_string(config.rollouts),
      "--min-tokens",
      std::to_string(config.min_tokens),
      "--max-tokens",
      std::to_string(config.max_tokens),
      "--input-len",
      std::to_string(config.input_len),
      "--port",
      std::to_string(config.port),
      "--tp",
      std::to_string(config.tp),
      "--repeat",
      std::to_string(config.repeat),
  };
  if (config.trust_remote_code) {
    args.push_back("--trust-remote-code");
  }
  const std::string output = trim(config.output);
  if (!output.empty() && output != "auto") {
    args.push_back("--output");
    args.push_back(output);
  }
  return args;
}

std::vector<std::string> build_bench_args(const BenchConfig& config) {
  return {
      "bench",
      "--kernel",
      config.kernel,
      "--shapes",
      config.shapes,
      "--dtype",
      config.dtype,
      "--warmup",
      std::to_string(config.warmup),
      "--n-iter",
      std::to_string(config.n_iter),
      "--repeats",
      std::to_string(config.repeats),
  };
}

void run_with_progress(
    const std::string& initial_status,
    const std::function<void(const ProgressCallback&)>& action) {
  std::atomic<bool> done = false;
  std::mutex status_mutex;
  std::string status = initial_status;
  std::exception_ptr error;
  int frame = 0;
  const std::vector<std::string> spinner = {"|", "/", "-", "\\"};

  const ProgressCallback progress = [&](const std::string& next_status) {
    std::lock_guard<std::mutex> lock(status_mutex);
    status = next_status;
  };

  std::thread worker([&] {
    try {
      action(progress);
    } catch (...) {
      error = std::current_exception();
    }
    done.store(true);
  });

  while (!done.load()) {
    std::string current_status;
    {
      std::lock_guard<std::mutex> lock(status_mutex);
      current_status = status;
    }
    std::cout << "\r  " << CYAN << spinner[static_cast<std::size_t>(frame)] << RESET
              << " " << current_status << "    " << std::flush;
    frame = (frame + 1) % static_cast<int>(spinner.size());
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
  }

  if (worker.joinable()) {
    worker.join();
  }

  std::cout << "\r\033[2K" << std::flush;
  if (error) {
    std::rethrow_exception(error);
  }
}

}  // namespace rlprof::interactive
