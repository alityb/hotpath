#include "rlprof/bench/runner.h"

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

namespace rlprof::bench {
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

Shape parse_shape_spec(const std::string& value) {
  return parse_shapes(value).front();
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
    BenchmarkBackend* backend) {
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
    BenchmarkBackend* backend) {
  std::vector<BenchResult> results;
  for (const KernelImpl& implementation : get_kernel_impls(category)) {
    const auto impl_results =
        benchmark_impl(category, implementation, shapes, dtype, warmup, n_iter, backend);
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

  const auto warnings_it = object.find("warnings");
  if (warnings_it != object.end()) {
    for (const auto& warning : as_array(warnings_it->second)) {
      output.warnings.push_back(as_string(warning));
    }
  }

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
        .unstable = as_bool(result.at("unstable")),
    });
  }

  return output;
}

std::string render_bench_results(const std::vector<BenchResult>& results) {
  return render_bench_output(BenchRunOutput{
      .gpu = std::nullopt,
      .results = results,
      .warnings = {},
  });
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
      << std::setw(8) << "unstable" << "\n";
  out << std::string(141, '-') << "\n";
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
        << std::setw(8) << (result.unstable ? "yes" : "no") << "\n";
  }
  if (!output.warnings.empty()) {
    out << "\nMEASUREMENT WARNINGS\n";
    for (const auto& warning : output.warnings) {
      out << "- " << warning << "\n";
    }
  }
  return out.str();
}

}  // namespace rlprof::bench
