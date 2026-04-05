#include "interactive.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include "hotpath/bench/runner.h"
#include "hotpath/clock_control.h"

namespace hotpath::interactive {
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

bool use_color() {
  const char* no_color = std::getenv("NO_COLOR");
  if (no_color != nullptr) {
    return false;
  }
  const char* term = std::getenv("TERM");
  if (term != nullptr && std::string(term) == "dumb") {
    return false;
  }
  return isatty(STDOUT_FILENO);
}

bool use_ansi_control() {
  return isatty(STDOUT_FILENO);
}

const char* color_code(const char* code) {
  return use_color() ? code : "";
}

int terminal_columns() {
  struct winsize ws{};
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
    return ws.ws_col;
  }
  return 80;
}

std::string truncate_display_text(const std::string& text, int max_columns) {
  if (max_columns <= 0) {
    return "";
  }
  if (static_cast<int>(text.size()) <= max_columns) {
    return text;
  }
  if (max_columns <= 3) {
    return text.substr(0, static_cast<std::size_t>(max_columns));
  }
  return text.substr(0, static_cast<std::size_t>(max_columns - 3)) + "...";
}

std::filesystem::path defaults_path() {
  const char* explicit_path = std::getenv("RLPROF_DEFAULTS_PATH");
  if (explicit_path != nullptr && std::string(explicit_path).size() > 0) {
    return std::filesystem::path(explicit_path);
  }
  const char* xdg_state_home = std::getenv("XDG_STATE_HOME");
  if (xdg_state_home != nullptr && std::string(xdg_state_home).size() > 0) {
    return std::filesystem::path(xdg_state_home) / "hotpath" / "interactive_defaults.cfg";
  }
  const char* home = std::getenv("HOME");
  if (home != nullptr && std::string(home).size() > 0) {
    return std::filesystem::path(home) / ".local" / "state" / "hotpath" /
           "interactive_defaults.cfg";
  }
  return std::filesystem::path(".hotpath") / "interactive_defaults.cfg";
}

std::map<std::string, std::string> load_defaults_map() {
  std::map<std::string, std::string> values;
  std::ifstream in(defaults_path());
  std::string line;
  while (std::getline(in, line)) {
    const std::size_t split = line.find('=');
    if (split == std::string::npos) {
      continue;
    }
    values[trim(line.substr(0, split))] = trim(line.substr(split + 1));
  }
  return values;
}

void save_defaults_map(const std::map<std::string, std::string>& values) {
  std::filesystem::create_directories(defaults_path().parent_path());
  std::ofstream out(defaults_path());
  for (const auto& [key, value] : values) {
    out << key << "=" << value << "\n";
  }
}

bool parse_bool_value(const std::map<std::string, std::string>& values, const std::string& key, bool fallback) {
  const auto it = values.find(key);
  if (it == values.end()) {
    return fallback;
  }
  return it->second == "true" || it->second == "1" || it->second == "yes";
}

int parse_int_value(const std::map<std::string, std::string>& values, const std::string& key, int fallback) {
  const auto it = values.find(key);
  if (it == values.end() || it->second.empty()) {
    return fallback;
  }
  try {
    return std::stoi(it->second);
  } catch (const std::exception&) {
    return fallback;
  }
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
  std::cout << "  " << color_code(CYAN) << "? " << color_code(RESET)
            << color_code(BOLD) << label << color_code(RESET);
  if (!default_value.empty()) {
    std::cout << " " << color_code(DIM) << "(" << default_value << ")"
              << color_code(RESET);
  }
  std::cout << ": " << color_code(GREEN) << "> " << color_code(RESET);
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

  // Fall back to numbered input when stdin is not a terminal.
  if (!isatty(STDIN_FILENO)) {
    std::cout << "  " << color_code(CYAN) << "? " << color_code(RESET)
              << color_code(BOLD) << label << color_code(RESET) << "\n";
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

  // Interactive arrow-key selection with viewport scrolling.
  // We only render a bounded window of options so the total height never
  // exceeds the terminal — which would cause scrolling that breaks the
  // cursor-up redraw logic.
  struct termios saved_termios{};
  tcgetattr(STDIN_FILENO, &saved_termios);
  struct termios raw = saved_termios;
  raw.c_lflag &= ~static_cast<tcflag_t>(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &raw);

  // Determine how many items to show at once.
  struct winsize ws{};
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws);
  const int term_rows = (ws.ws_row > 6) ? ws.ws_row - 6 : 8;
  const int term_cols = terminal_columns();
  const int n = static_cast<int>(options.size());
  const int max_vis = std::min(n, std::max(4, term_rows));

  int selected = std::clamp(default_index, 0, n - 1);
  int vp_start = 0;          // index of first visible item
  int rendered_lines = 0;

  const auto render = [&]() {
    // Keep selected item inside the viewport.
    if (selected < vp_start) vp_start = selected;
    if (selected >= vp_start + max_vis) vp_start = selected - max_vis + 1;

    rendered_lines = 0;
    const std::string header =
        truncate_display_text(label + "  (arrows, enter, q)", std::max(1, term_cols - 4));
    std::cout << "\r  " << color_code(CYAN) << "? " << color_code(RESET)
              << color_code(BOLD) << header << color_code(RESET) << "\033[K\n";
    ++rendered_lines;

    for (int i = vp_start; i < vp_start + max_vis; ++i) {
      const std::string option =
          truncate_display_text(options[static_cast<std::size_t>(i)], std::max(1, term_cols - 4));
      if (i == selected) {
        std::cout << "\r  " << color_code(CYAN) << "> " << color_code(RESET)
                  << option << "\033[K\n";
      } else {
        std::cout << "\r    " << color_code(DIM)
                  << option
                  << color_code(RESET) << "\033[K\n";
      }
      ++rendered_lines;
    }
    std::flush(std::cout);
  };

  if (use_ansi_control()) {
    std::cout << "\033[?25l";  // hide cursor while drawing

    // Reserve vertical space so the initial render never causes the terminal
    // to scroll (which would break the cursor-save/restore logic below).
    const int reserve = max_vis + 1;
    for (int i = 0; i < reserve; ++i) std::cout << '\n';
    // CUU (up N) + CR to reach column 0 — both universally supported VT100.
    std::cout << "\033[" << reserve << "A\r";
    // Save the top-of-menu cursor position so redraws can return here cleanly.
    std::cout << "\033[s";
  }
  render();

  std::optional<int> result;
  bool running = true;
  while (running) {
    char c = 0;
    if (read(STDIN_FILENO, &c, 1) != 1) break;

    bool needs_redraw = false;
    if (c == '\n' || c == '\r') {
      result = selected;
      running = false;
    } else if (c == 'q' || c == 'Q') {
      running = false;
    } else if (c == '\033') {
      char seq[2] = {};
      if (read(STDIN_FILENO, &seq[0], 1) == 1 &&
          read(STDIN_FILENO, &seq[1], 1) == 1 &&
          seq[0] == '[') {
        if (seq[1] == 'A') {
          selected = (selected - 1 + n) % n;  // up
          needs_redraw = true;
        } else if (seq[1] == 'B') {
          selected = (selected + 1) % n;       // down
          needs_redraw = true;
        }
      }
    }

    if (needs_redraw && use_ansi_control()) {
      std::cout << "\033[u";
      render();
    }
  }

  if (use_ansi_control()) {
    std::cout << "\033[?25h";  // restore cursor
  }
  tcsetattr(STDIN_FILENO, TCSANOW, &saved_termios);

  // Replace the menu with a single confirmation line.
  if (use_ansi_control()) {
    std::cout << "\033[u\033[J";
  }
  if (result.has_value()) {
    std::cout << "  " << color_code(CYAN) << "\xe2\x9c\x93 " << color_code(RESET)
              << color_code(BOLD) << label << color_code(RESET)
              << "  " << color_code(DIM)
              << options[static_cast<std::size_t>(*result)] << color_code(RESET) << "\n";
  }
  return result;
}

void print_header(const std::string& text) {
  std::cout << "\n  " << color_code(BOLD) << color_code(CYAN)
            << text << color_code(RESET) << "\n\n";
}

void print_warning(const std::string& text) {
  std::cout << "  " << color_code(YELLOW) << "!" << color_code(RESET)
            << " " << text << "\n";
}

void print_info(const std::string& label, const std::string& value) {
  std::cout << "  " << color_code(WHITE) << label << color_code(RESET);
  if (!value.empty()) {
    std::cout << " " << color_code(DIM) << value << color_code(RESET);
  }
  std::cout << "\n";
}

std::string detect_gpu_name() {
  const std::string gpu_name = run_command_capture(
      "nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1");
  return gpu_name.empty() ? "unknown" : gpu_name;
}

bool are_clocks_locked() {
  const auto info = hotpath::query_clock_policy();
  return info.query_ok && info.gpu_clocks_locked;
}

std::string clock_status_label() {
  return hotpath::render_clock_policy(hotpath::query_clock_policy());
}

std::vector<std::string> list_recent_profiles(int max_count) {
  namespace fs = std::filesystem;
  std::vector<std::pair<fs::file_time_type, std::string>> entries;
  const fs::path root = ".hotpath";
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

std::vector<std::string> list_recent_bench_results(int max_count) {
  namespace fs = std::filesystem;
  std::vector<std::pair<fs::file_time_type, std::string>> entries;
  const fs::path root = ".hotpath";
  if (!fs::exists(root)) {
    return {};
  }

  for (const auto& entry : fs::directory_iterator(root)) {
    if (entry.path().extension() == ".json") {
      try {
        std::ifstream stream(entry.path());
        std::stringstream buffer;
        buffer << stream.rdbuf();
        const auto parsed =
            hotpath::bench::parse_bench_json(buffer.str());
        if (!parsed.results.empty()) {
          entries.push_back({entry.last_write_time(), entry.path().string()});
        }
      } catch (const std::exception&) {
      }
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

ProfileConfig load_profile_defaults() {
  const auto values = load_defaults_map();
  ProfileConfig config;
  const auto get = [&](const std::string& key, const std::string& fallback = "") {
    const auto it = values.find(key);
    return it == values.end() ? fallback : it->second;
  };
  config.model = get("profile.model");
  config.target = get("profile.target");
  config.target_workdir = get("profile.target_workdir");
  config.prompts = parse_int_value(values, "profile.prompts", config.prompts);
  config.rollouts = parse_int_value(values, "profile.rollouts", config.rollouts);
  config.min_tokens = parse_int_value(values, "profile.min_tokens", config.min_tokens);
  config.max_tokens = parse_int_value(values, "profile.max_tokens", config.max_tokens);
  config.input_len = parse_int_value(values, "profile.input_len", config.input_len);
  config.port = parse_int_value(values, "profile.port", config.port);
  config.tp = parse_int_value(values, "profile.tp", config.tp);
  config.peer_servers = get("profile.peer_servers");
  config.trust_remote_code =
      parse_bool_value(values, "profile.trust_remote_code", config.trust_remote_code);
  config.discard_first_run =
      parse_bool_value(values, "profile.discard_first_run", config.discard_first_run);
  config.repeat = parse_int_value(values, "profile.repeat", config.repeat);
  const std::string output = get("profile.output");
  if (!output.empty()) {
    config.output = output;
  }
  return config;
}

BenchConfig load_bench_defaults() {
  const auto values = load_defaults_map();
  BenchConfig config;
  const auto get = [&](const std::string& key, const std::string& fallback = "") {
    const auto it = values.find(key);
    return it == values.end() ? fallback : it->second;
  };
  const std::string kernel = get("bench.kernel");
  if (!kernel.empty()) {
    config.kernel = kernel;
  }
  config.target = get("bench.target");
  config.target_workdir = get("bench.target_workdir");
  const std::string shapes = get("bench.shapes");
  if (!shapes.empty()) {
    config.shapes = shapes;
  }
  const std::string dtype = get("bench.dtype");
  if (!dtype.empty()) {
    config.dtype = dtype;
  }
  config.warmup = parse_int_value(values, "bench.warmup", config.warmup);
  config.n_iter = parse_int_value(values, "bench.n_iter", config.n_iter);
  config.repeats = parse_int_value(values, "bench.repeats", config.repeats);
  return config;
}

void save_profile_defaults(const ProfileConfig& config) {
  auto values = load_defaults_map();
  values["profile.model"] = config.model;
  values["profile.target"] = config.target;
  values["profile.target_workdir"] = config.target_workdir;
  values["profile.prompts"] = std::to_string(config.prompts);
  values["profile.rollouts"] = std::to_string(config.rollouts);
  values["profile.min_tokens"] = std::to_string(config.min_tokens);
  values["profile.max_tokens"] = std::to_string(config.max_tokens);
  values["profile.input_len"] = std::to_string(config.input_len);
  values["profile.port"] = std::to_string(config.port);
  values["profile.tp"] = std::to_string(config.tp);
  values["profile.peer_servers"] = config.peer_servers;
  values["profile.trust_remote_code"] = config.trust_remote_code ? "true" : "false";
  values["profile.discard_first_run"] = config.discard_first_run ? "true" : "false";
  values["profile.repeat"] = std::to_string(config.repeat);
  values["profile.output"] = config.output;
  save_defaults_map(values);
}

void save_bench_defaults(const BenchConfig& config) {
  auto values = load_defaults_map();
  values["bench.kernel"] = config.kernel;
  values["bench.target"] = config.target;
  values["bench.target_workdir"] = config.target_workdir;
  values["bench.shapes"] = config.shapes;
  values["bench.dtype"] = config.dtype;
  values["bench.warmup"] = std::to_string(config.warmup);
  values["bench.n_iter"] = std::to_string(config.n_iter);
  values["bench.repeats"] = std::to_string(config.repeats);
  save_defaults_map(values);
}

void clear_saved_defaults() {
  std::error_code ec;
  std::filesystem::remove(defaults_path(), ec);
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
  const std::string target = trim(config.target);
  if (!target.empty()) {
    args.push_back("--target");
    args.push_back(target);
  }
  const std::string target_workdir = trim(config.target_workdir);
  if (!target_workdir.empty()) {
    args.push_back("--target-workdir");
    args.push_back(target_workdir);
  }
  if (config.discard_first_run) {
    args.push_back("--discard-first-run");
  }
  if (config.trust_remote_code) {
    args.push_back("--trust-remote-code");
  }
  const std::string peer_servers = trim(config.peer_servers);
  if (!peer_servers.empty()) {
    args.push_back("--peer-servers");
    args.push_back(peer_servers);
  }
  const std::string output = trim(config.output);
  if (!output.empty() && output != "auto") {
    args.push_back("--output");
    args.push_back(output);
  }
  return args;
}

std::vector<std::string> build_bench_args(const BenchConfig& config) {
  std::vector<std::string> args = {
      "bench",
      "--kernel",
      config.kernel,
  };
  const std::string target = trim(config.target);
  if (!target.empty()) {
    args.push_back("--target");
    args.push_back(target);
  }
  const std::string target_workdir = trim(config.target_workdir);
  if (!target_workdir.empty()) {
    args.push_back("--target-workdir");
    args.push_back(target_workdir);
  }
  args.insert(
      args.end(),
      {"--shapes",
       config.shapes,
       "--dtype",
       config.dtype,
       "--warmup",
       std::to_string(config.warmup),
       "--n-iter",
       std::to_string(config.n_iter),
       "--repeats",
       std::to_string(config.repeats)});
  return args;
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
  const bool tty_stdout = isatty(STDOUT_FILENO);

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
    if (tty_stdout) {
      std::cout << "\r\033[2K";
    } else {
      std::cout << "\r";
    }
    std::cout << "  " << color_code(CYAN)
              << spinner[static_cast<std::size_t>(frame)] << color_code(RESET)
              << " " << current_status << std::flush;
    frame = (frame + 1) % static_cast<int>(spinner.size());
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
  }

  if (worker.joinable()) {
    worker.join();
  }

  if (tty_stdout) {
    std::cout << "\r\033[2K" << std::flush;
  } else {
    std::cout << "\n";
  }
  if (error) {
    std::rethrow_exception(error);
  }
}

}  // namespace hotpath::interactive
