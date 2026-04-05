#include "hotpath/bench/runner.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <variant>

namespace hotpath::bench {
namespace {

class ChronoBackend final : public BenchmarkBackend {
 public:
  double measure_ms(const std::function<void()>& fn) override {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  }
};

// Write to a buffer larger than the CPU LLC to evict all cache lines.
// Typical LLC sizes: 30-60 MB on server CPUs, 6-16 MB on consumer.
// We use 64 MB to be safe.
void flush_cpu_cache() {
  static std::vector<char> flush_buf(64 * 1024 * 1024, 0);
  // volatile write so the compiler can't optimize it away
  for (size_t i = 0; i < flush_buf.size(); i += 64) {
    static_cast<volatile char&>(flush_buf[i]) = static_cast<char>(i);
  }
}

double percentile(std::vector<double> values, double quantile) {
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(
      std::ceil((values.size() - 1) * quantile));
  return values[index];
}

std::size_t dtype_size(const std::string& dtype) {
  if (dtype == "bf16" || dtype == "fp16") {
    return 2;
  }
  if (dtype == "fp32") {
    return 4;
  }
  throw std::runtime_error("unsupported dtype: " + dtype);
}

using JsonValue = std::variant<std::nullptr_t, bool, double, std::string,
                               std::vector<std::variant<std::nullptr_t, bool, double, std::string,
                                                        std::vector<std::variant<std::nullptr_t, bool, double, std::string,
                                                                                 std::vector<int>, std::map<std::string, int>>>,
                                                        std::map<std::string, int>>>,
                               std::map<std::string, std::variant<std::nullptr_t, bool, double, std::string,
                                                                  std::vector<std::variant<std::nullptr_t, bool, double, std::string,
                                                                                           std::vector<int>, std::map<std::string, int>>>,
                                                                  std::map<std::string, int>>>>;

class JsonParser {
 public:
  explicit JsonParser(std::string text) : text_(std::move(text)) {}

  struct Value;
  using Array = std::vector<Value>;
  using Object = std::map<std::string, Value>;
  struct Value {
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> data;
  };

  Value parse() {
    skip_ws();
    Value value = parse_value();
    skip_ws();
    if (index_ != text_.size()) {
      throw std::runtime_error("unexpected trailing JSON");
    }
    return value;
  }

 private:
  Value parse_value() {
    skip_ws();
    if (index_ >= text_.size()) {
      throw std::runtime_error("unexpected end of JSON");
    }
    const char ch = text_[index_];
    if (ch == '{') {
      return Value{parse_object()};
    }
    if (ch == '[') {
      return Value{parse_array()};
    }
    if (ch == '"') {
      return Value{parse_string()};
    }
    if (ch == 't') {
      consume("true");
      return Value{true};
    }
    if (ch == 'f') {
      consume("false");
      return Value{false};
    }
    if (ch == 'n') {
      consume("null");
      return Value{nullptr};
    }
    return Value{parse_number()};
  }

  Object parse_object() {
    expect('{');
    skip_ws();
    Object object;
    if (peek('}')) {
      expect('}');
      return object;
    }
    while (true) {
      const std::string key = parse_string();
      skip_ws();
      expect(':');
      object[key] = parse_value();
      skip_ws();
      if (peek('}')) {
        expect('}');
        break;
      }
      expect(',');
    }
    return object;
  }

  Array parse_array() {
    expect('[');
    skip_ws();
    Array array;
    if (peek(']')) {
      expect(']');
      return array;
    }
    while (true) {
      array.push_back(parse_value());
      skip_ws();
      if (peek(']')) {
        expect(']');
        break;
      }
      expect(',');
    }
    return array;
  }

  std::string parse_string() {
    expect('"');
    std::string value;
    while (index_ < text_.size()) {
      const char ch = text_[index_++];
      if (ch == '"') {
        return value;
      }
      if (ch == '\\') {
        if (index_ >= text_.size()) {
          throw std::runtime_error("unterminated JSON escape");
        }
        const char escaped = text_[index_++];
        switch (escaped) {
          case '"':
          case '\\':
          case '/':
            value.push_back(escaped);
            break;
          case 'b':
            value.push_back('\b');
            break;
          case 'f':
            value.push_back('\f');
            break;
          case 'n':
            value.push_back('\n');
            break;
          case 'r':
            value.push_back('\r');
            break;
          case 't':
            value.push_back('\t');
            break;
          default:
            throw std::runtime_error("unsupported JSON escape");
        }
      } else {
        value.push_back(ch);
      }
    }
    throw std::runtime_error("unterminated JSON string");
  }

  double parse_number() {
    const std::size_t start = index_;
    if (text_[index_] == '-') {
      ++index_;
    }
    while (index_ < text_.size() &&
           std::isdigit(static_cast<unsigned char>(text_[index_]))) {
      ++index_;
    }
    if (index_ < text_.size() && text_[index_] == '.') {
      ++index_;
      while (index_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[index_]))) {
        ++index_;
      }
    }
    if (index_ < text_.size() &&
        (text_[index_] == 'e' || text_[index_] == 'E')) {
      ++index_;
      if (index_ < text_.size() &&
          (text_[index_] == '+' || text_[index_] == '-')) {
        ++index_;
      }
      while (index_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[index_]))) {
        ++index_;
      }
    }
    return std::stod(text_.substr(start, index_ - start));
  }

  void skip_ws() {
    while (index_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[index_]))) {
      ++index_;
    }
  }

  void expect(char ch) {
    skip_ws();
    if (index_ >= text_.size() || text_[index_] != ch) {
      throw std::runtime_error("unexpected JSON token");
    }
    ++index_;
  }

  bool peek(char ch) {
    skip_ws();
    return index_ < text_.size() && text_[index_] == ch;
  }

  void consume(const std::string& token) {
    if (text_.compare(index_, token.size(), token) != 0) {
      throw std::runtime_error("unexpected JSON literal");
    }
    index_ += token.size();
  }

  std::string text_;
  std::size_t index_ = 0;
};

const JsonParser::Object& as_object(const JsonParser::Value& value) {
  return std::get<JsonParser::Object>(value.data);
}

const JsonParser::Array& as_array(const JsonParser::Value& value) {
  return std::get<JsonParser::Array>(value.data);
}

const std::string& as_string(const JsonParser::Value& value) {
  return std::get<std::string>(value.data);
}

double as_number(const JsonParser::Value& value) {
  return std::get<double>(value.data);
}

bool as_bool(const JsonParser::Value& value) {
  return std::get<bool>(value.data);
}

bool is_null(const JsonParser::Value& value) {
  return std::holds_alternative<std::nullptr_t>(value.data);
}

std::vector<std::string> parse_string_array(
    const JsonParser::Object& object,
    const std::string& key) {
  std::vector<std::string> values;
  const auto it = object.find(key);
  if (it == object.end()) {
    return values;
  }
  for (const auto& item : as_array(it->second)) {
    values.push_back(as_string(item));
  }
  return values;
}

Shape parse_shape_spec(const std::string& value) {
  return parse_shapes(value).front();
}

std::string json_escape(const std::string& input) {
  std::string output;
  for (char ch : input) {
    switch (ch) {
      case '\\':
        output += "\\\\";
        break;
      case '"':
        output += "\\\"";
        break;
      case '\n':
        output += "\\n";
        break;
      default:
        output.push_back(ch);
        break;
    }
  }
  return output;
}

}  // namespace

std::vector<Shape> parse_shapes(const std::string& spec) {
  std::vector<Shape> shapes;
  std::stringstream ss(spec);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }
    Shape shape;
    std::stringstream shape_stream(token);
    std::string dim;
    while (std::getline(shape_stream, dim, 'x')) {
      const std::int64_t value = std::stoll(dim);
      if (value <= 0) {
        throw std::runtime_error("invalid shape: " + token);
      }
      shape.push_back(value);
    }
    if (shape.empty()) {
      throw std::runtime_error("invalid shape: " + token);
    }
    shapes.push_back(shape);
  }
  if (shapes.empty()) {
    throw std::runtime_error("at least one shape is required");
  }
  return shapes;
}

std::vector<BenchResult> benchmark_impl(
    const std::string& category,
    const KernelImpl& implementation,
    const std::vector<Shape>& shapes,
    const std::string& dtype,
    std::int64_t warmup,
    std::int64_t n_iter,
    BenchmarkBackend* backend,
    bool flush_cache) {
  if (warmup < 0 || n_iter <= 0) {
    throw std::runtime_error("warmup must be >= 0 and n_iter must be > 0");
  }
  if (std::find(implementation.dtypes.begin(), implementation.dtypes.end(), dtype) ==
      implementation.dtypes.end()) {
    throw std::runtime_error("unsupported dtype for implementation: " + dtype);
  }

  ChronoBackend default_backend;
  BenchmarkBackend* active_backend = backend == nullptr ? &default_backend : backend;

  std::vector<BenchResult> results;
  for (const Shape& shape : shapes) {
    std::any state = implementation.setup(shape, dtype);
    for (std::int64_t i = 0; i < warmup; ++i) {
      implementation.fn(state);
    }
    active_backend->synchronize();

    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(n_iter));
    for (std::int64_t i = 0; i < n_iter; ++i) {
      if (flush_cache) flush_cpu_cache();
      times.push_back(active_backend->measure_ms([&]() { implementation.fn(state); }));
      active_backend->synchronize();
    }

    const double avg_ms =
        std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
    const double min_ms = *std::min_element(times.begin(), times.end());
    const double p50_ms = percentile(times, 0.50);
    const double p99_ms = percentile(times, 0.99);
    const double bandwidth_gb_s =
        static_cast<double>(implementation.bytes_processed(shape, dtype)) / (avg_ms / 1000.0) / 1e9;

    results.push_back(BenchResult{
        .kernel = category,
        .implementation = implementation.name,
        .shape = shape,
        .dtype = dtype,
        .avg_ms = avg_ms,
        .stddev_ms = 0.0,
        .repeat_cv_pct = 0.0,
        .min_ms = min_ms,
        .p50_ms = p50_ms,
        .p99_ms = p99_ms,
        .bandwidth_gb_s = bandwidth_gb_s,
        .validation_passed = true,
        .validation_max_abs_error = 0.0,
        .deterministic_passed = true,
        .determinism_max_abs_error = 0.0,
        .has_timing_warning = false,
        .has_environment_warning = false,
        .unstable = false,
    });
  }
  return results;
}

std::vector<BenchResult> benchmark_category(
    const std::string& category,
    const std::vector<Shape>& shapes,
    const std::string& dtype,
    std::int64_t warmup,
    std::int64_t n_iter,
    BenchmarkBackend* backend,
    bool flush_cache) {
  std::vector<BenchResult> results;
  for (const KernelImpl& implementation : get_kernel_impls(category)) {
    const auto impl_results =
        benchmark_impl(category, implementation, shapes, dtype, warmup, n_iter, backend, flush_cache);
    results.insert(results.end(), impl_results.begin(), impl_results.end());
  }
  return results;
}

BenchRunOutput parse_bench_json(const std::string& json_text) {
  const JsonParser::Value root = JsonParser(json_text).parse();
  const JsonParser::Object& object = as_object(root);

  BenchRunOutput output;
  const auto gpu_it = object.find("gpu");
  if (gpu_it != object.end() && !is_null(gpu_it->second)) {
    const auto& gpu = as_object(gpu_it->second);
    output.gpu = BenchGpuInfo{
        .name = as_string(gpu.at("name")),
        .driver_version = as_string(gpu.at("driver_version")),
        .sm_clock_mhz = as_number(gpu.at("sm_clock_mhz")),
        .mem_clock_mhz = as_number(gpu.at("mem_clock_mhz")),
        .temp_c = as_number(gpu.at("temp_c")),
        .power_draw_w = as_number(gpu.at("power_draw_w")),
        .power_limit_w = as_number(gpu.at("power_limit_w")),
    };
  }

  output.correctness_failures = parse_string_array(object, "correctness_failures");
  output.timing_warnings = parse_string_array(object, "timing_warnings");
  output.environment_warnings = parse_string_array(object, "environment_warnings");

  for (const auto& item : as_array(object.at("results"))) {
    const auto& result = as_object(item);
    output.results.push_back(BenchResult{
        .kernel = as_string(result.at("kernel")),
        .implementation = as_string(result.at("implementation")),
        .shape = parse_shape_spec(as_string(result.at("shape"))),
        .dtype = as_string(result.at("dtype")),
        .avg_ms = as_number(result.at("avg_us")) / 1000.0,
        .stddev_ms = as_number(result.at("stddev_us")) / 1000.0,
        .repeat_cv_pct = as_number(result.at("cv_pct")),
        .min_ms = as_number(result.at("min_us")) / 1000.0,
        .p50_ms = as_number(result.at("p50_us")) / 1000.0,
        .p99_ms = as_number(result.at("p99_us")) / 1000.0,
        .bandwidth_gb_s = as_number(result.at("bandwidth_gb_s")),
        .validation_passed = as_bool(result.at("valid")),
        .validation_max_abs_error = as_number(result.at("validation_max_abs_error")),
        .deterministic_passed = as_bool(result.at("deterministic")),
        .determinism_max_abs_error = as_number(result.at("determinism_max_abs_error")),
        .has_timing_warning = as_bool(result.at("timing_warning")),
        .has_environment_warning = as_bool(result.at("environment_warning")),
        .unstable = as_bool(result.at("unstable")),
    });
  }

  return output;
}

std::string render_bench_results(const std::vector<BenchResult>& results) {
  return render_bench_output(BenchRunOutput{
      .gpu = std::nullopt,
      .results = results,
      .correctness_failures = {},
      .timing_warnings = {},
      .environment_warnings = {},
  });
}

std::string render_bench_comparison(
    const BenchRunOutput& left,
    const BenchRunOutput& right) {
  struct Row {
    std::string key;
    std::optional<double> left_us;
    std::optional<double> right_us;
  };

  std::map<std::string, Row> rows;
  const auto add = [&](const BenchRunOutput& output, bool is_left) {
    for (const auto& result : output.results) {
      std::ostringstream key;
      key << result.kernel << " | " << result.implementation << " | ";
      for (std::size_t i = 0; i < result.shape.size(); ++i) {
        if (i > 0) {
          key << "x";
        }
        key << result.shape[i];
      }
      auto& row = rows[key.str()];
      row.key = key.str();
      if (is_left) {
        row.left_us = result.avg_ms * 1000.0;
      } else {
        row.right_us = result.avg_ms * 1000.0;
      }
    }
  };
  add(left, true);
  add(right, false);

  const auto format_optional = [](const std::optional<double>& value) {
    if (!value.has_value()) {
      return std::string("missing");
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << *value;
    return out.str();
  };

  std::ostringstream out;
  out << std::left << std::setw(48) << "benchmark" << "  "
      << std::right << std::setw(10) << "A avg us" << "  "
      << std::setw(10) << "B avg us" << "  "
      << std::setw(10) << "delta us" << "  "
      << std::setw(10) << "delta %" << "\n";
  out << std::string(98, '-') << "\n";
  for (const auto& [_, row] : rows) {
    std::optional<double> delta;
    std::optional<double> delta_pct;
    if (row.left_us.has_value() && row.right_us.has_value()) {
      delta = *row.right_us - *row.left_us;
      if (*row.left_us != 0.0) {
        delta_pct = (*delta / *row.left_us) * 100.0;
      }
    }
    out << std::left << std::setw(48) << row.key << "  "
        << std::right << std::setw(10) << format_optional(row.left_us) << "  "
        << std::setw(10) << format_optional(row.right_us) << "  "
        << std::setw(10) << format_optional(delta) << "  "
        << std::setw(10) << format_optional(delta_pct) << "\n";
  }
  return out.str();
}

std::string serialize_bench_output_json(const BenchRunOutput& output) {
  std::ostringstream out;
  out << "{";
  out << "\"gpu\":";
  if (output.gpu.has_value()) {
    out << "{"
        << "\"name\":\"" << json_escape(output.gpu->name) << "\","
        << "\"driver_version\":\"" << json_escape(output.gpu->driver_version) << "\","
        << "\"sm_clock_mhz\":" << output.gpu->sm_clock_mhz << ","
        << "\"mem_clock_mhz\":" << output.gpu->mem_clock_mhz << ","
        << "\"temp_c\":" << output.gpu->temp_c << ","
        << "\"power_draw_w\":" << output.gpu->power_draw_w << ","
        << "\"power_limit_w\":" << output.gpu->power_limit_w
        << "}";
  } else {
    out << "null";
  }
  auto write_string_array = [&](const std::string& key, const std::vector<std::string>& values) {
    out << ",\"" << key << "\":[";
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (i > 0) {
        out << ",";
      }
      out << "\"" << json_escape(values[i]) << "\"";
    }
    out << "]";
  };
  write_string_array("correctness_failures", output.correctness_failures);
  write_string_array("timing_warnings", output.timing_warnings);
  write_string_array("environment_warnings", output.environment_warnings);
  out << ",\"results\":[";
  for (std::size_t i = 0; i < output.results.size(); ++i) {
    const auto& result = output.results[i];
    if (i > 0) {
      out << ",";
    }
    std::ostringstream shape_stream;
    for (std::size_t j = 0; j < result.shape.size(); ++j) {
      if (j > 0) {
        shape_stream << "x";
      }
      shape_stream << result.shape[j];
    }
    out << "{"
        << "\"kernel\":\"" << json_escape(result.kernel) << "\","
        << "\"implementation\":\"" << json_escape(result.implementation) << "\","
        << "\"shape\":\"" << shape_stream.str() << "\","
        << "\"dtype\":\"" << json_escape(result.dtype) << "\","
        << "\"avg_us\":" << (result.avg_ms * 1000.0) << ","
        << "\"stddev_us\":" << (result.stddev_ms * 1000.0) << ","
        << "\"cv_pct\":" << result.repeat_cv_pct << ","
        << "\"min_us\":" << (result.min_ms * 1000.0) << ","
        << "\"p50_us\":" << (result.p50_ms * 1000.0) << ","
        << "\"p99_us\":" << (result.p99_ms * 1000.0) << ","
        << "\"bandwidth_gb_s\":" << result.bandwidth_gb_s << ","
        << "\"valid\":" << (result.validation_passed ? "true" : "false") << ","
        << "\"validation_max_abs_error\":" << result.validation_max_abs_error << ","
        << "\"deterministic\":" << (result.deterministic_passed ? "true" : "false") << ","
        << "\"determinism_max_abs_error\":" << result.determinism_max_abs_error << ","
        << "\"timing_warning\":" << (result.has_timing_warning ? "true" : "false") << ","
        << "\"environment_warning\":" << (result.has_environment_warning ? "true" : "false") << ","
        << "\"unstable\":" << (result.unstable ? "true" : "false")
        << "}";
  }
  out << "]}";
  return out.str();
}

std::filesystem::path resolve_bench_output_path(
    const std::string& kernel,
    const std::string& output_spec) {
  if (output_spec.empty() || output_spec == "none") {
    return {};
  }
  if (output_spec != "auto") {
    return output_spec;
  }
  const auto now = std::chrono::system_clock::now().time_since_epoch().count();
  return std::filesystem::path(".hotpath") /
         ("bench_" + kernel + "_" + std::to_string(now) + ".json");
}

std::string render_bench_output(const BenchRunOutput& output) {
  std::ostringstream out;
  if (output.gpu.has_value()) {
    out << "gpu: " << output.gpu->name << " | driver: " << output.gpu->driver_version
        << " | sm clock: " << std::fixed << std::setprecision(0) << output.gpu->sm_clock_mhz
        << " mhz | mem clock: " << output.gpu->mem_clock_mhz
        << " mhz | temp: " << output.gpu->temp_c << " c | power: "
        << std::setprecision(1) << output.gpu->power_draw_w << "/"
        << output.gpu->power_limit_w << " w\n\n";
  }

  out << std::left << std::setw(18) << "kernel" << "  "
      << std::setw(18) << "implementation" << "  "
      << std::setw(12) << "shape" << "  "
      << std::right << std::setw(8) << "avg us" << "  "
      << std::setw(8) << "stddev" << "  "
      << std::setw(7) << "cv %" << "  "
      << std::setw(8) << "min us" << "  "
      << std::setw(8) << "p50 us" << "  "
      << std::setw(8) << "p99 us" << "  "
      << std::setw(11) << "GB/s" << "  "
      << std::setw(5) << "valid" << "  "
      << std::setw(5) << "det" << "  "
      << std::setw(6) << "timing" << "  "
      << std::setw(6) << "env" << "  "
      << std::setw(8) << "unstable" << "\n";
  out << std::string(165, '-') << "\n";
  for (const BenchResult& result : output.results) {
    std::ostringstream shape_stream;
    for (std::size_t i = 0; i < result.shape.size(); ++i) {
      if (i > 0) {
        shape_stream << "x";
      }
      shape_stream << result.shape[i];
    }
    out << std::left << std::setw(18) << result.kernel << "  "
        << std::setw(18) << result.implementation << "  "
        << std::setw(12) << shape_stream.str() << "  "
        << std::right << std::fixed << std::setprecision(3)
        << std::setw(8) << (result.avg_ms * 1000.0) << "  "
        << std::setw(8) << (result.stddev_ms * 1000.0) << "  "
        << std::setw(7) << result.repeat_cv_pct << "  "
        << std::setw(8) << (result.min_ms * 1000.0) << "  "
        << std::setw(8) << (result.p50_ms * 1000.0) << "  "
        << std::setw(8) << (result.p99_ms * 1000.0) << "  "
        << std::setw(11) << result.bandwidth_gb_s << "  "
        << std::setw(5) << (result.validation_passed ? "yes" : "no") << "  "
        << std::setw(5) << (result.deterministic_passed ? "yes" : "no") << "  "
        << std::setw(6) << (result.has_timing_warning ? "yes" : "no") << "  "
        << std::setw(6) << (result.has_environment_warning ? "yes" : "no") << "  "
        << std::setw(8) << (result.unstable ? "yes" : "no") << "\n";
  }
  if (!output.correctness_failures.empty()) {
    out << "\nCORRECTNESS FAILURES\n";
    for (const auto& warning : output.correctness_failures) {
      out << "- " << warning << "\n";
    }
  }
  if (!output.timing_warnings.empty()) {
    out << "\nTIMING WARNINGS\n";
    for (const auto& warning : output.timing_warnings) {
      out << "- " << warning << "\n";
    }
  }
  if (!output.environment_warnings.empty()) {
    out << "\nENVIRONMENT WARNINGS\n";
    for (const auto& warning : output.environment_warnings) {
      out << "- " << warning << "\n";
    }
  }
  return out.str();
}

}  // namespace hotpath::bench
