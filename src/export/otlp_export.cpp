#include "hotpath/otlp_export.h"

#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>

namespace hotpath {
namespace {

std::string generate_hex_id(std::mt19937_64& rng, int bytes) {
  std::ostringstream ss;
  for (int i = 0; i < bytes; ++i) {
    ss << std::hex << std::setw(2) << std::setfill('0')
       << (static_cast<int>(rng()) & 0xFF);
  }
  return ss.str();
}

// Convert microseconds to nanoseconds string
std::string us_to_ns(int64_t us) {
  return std::to_string(us * 1000);
}

std::string json_escape(const std::string& s) {
  std::string result;
  for (char c : s) {
    if (c == '"') result += "\\\"";
    else if (c == '\\') result += "\\\\";
    else if (c == '\n') result += "\\n";
    else result += c;
  }
  return result;
}

struct Span {
  std::string trace_id;
  std::string span_id;
  std::string parent_span_id;
  std::string name;
  int64_t start_us;
  int64_t end_us;
  std::vector<std::pair<std::string, std::string>> attributes;
};

std::string span_to_json(const Span& span) {
  std::ostringstream ss;
  ss << "{"
     << "\"traceId\":\"" << span.trace_id << "\","
     << "\"spanId\":\"" << span.span_id << "\",";
  if (!span.parent_span_id.empty()) {
    ss << "\"parentSpanId\":\"" << span.parent_span_id << "\",";
  }
  ss << "\"name\":\"" << json_escape(span.name) << "\","
     << "\"kind\":1,"  // SPAN_KIND_INTERNAL
     << "\"startTimeUnixNano\":\"" << us_to_ns(span.start_us) << "\","
     << "\"endTimeUnixNano\":\"" << us_to_ns(span.end_us) << "\"";

  if (!span.attributes.empty()) {
    ss << ",\"attributes\":[";
    for (size_t i = 0; i < span.attributes.size(); ++i) {
      if (i > 0) ss << ",";
      ss << "{\"key\":\"" << span.attributes[i].first
         << "\",\"value\":{\"intValue\":\"" << span.attributes[i].second << "\"}}";
    }
    ss << "]";
  }

  ss << "}";
  return ss.str();
}

}  // namespace

std::string export_otlp_json(const std::vector<RequestTrace>& traces,
                             const std::string& service_name) {
  std::ostringstream ss;

  ss << "{\"resourceSpans\":[{";
  ss << "\"resource\":{\"attributes\":[{\"key\":\"service.name\","
     << "\"value\":{\"stringValue\":\"" << json_escape(service_name) << "\"}}]},";
  ss << "\"scopeSpans\":[{\"scope\":{\"name\":\"hotpath\"},\"spans\":[";

  // Local RNG seeded deterministically so identical input always produces
  // identical output, regardless of how many times the function is called.
  std::mt19937_64 rng(42);

  bool first = true;
  for (const auto& trace : traces) {
    const std::string trace_id = generate_hex_id(rng, 16);

    // Root span: llm.request
    Span root;
    root.trace_id = trace_id;
    root.span_id = generate_hex_id(rng, 8);
    root.name = "llm.request";
    root.start_us = trace.arrival_us;
    root.end_us = trace.completion_us > 0 ? trace.completion_us : trace.arrival_us;

    if (!first) ss << ",";
    first = false;
    ss << span_to_json(root);

    // Child: llm.queue
    if (trace.prefill_start_us > 0) {
      Span queue;
      queue.trace_id = trace_id;
      queue.span_id = generate_hex_id(rng, 8);
      queue.parent_span_id = root.span_id;
      queue.name = "llm.queue";
      queue.start_us = trace.arrival_us;
      queue.end_us = trace.prefill_start_us;
      ss << "," << span_to_json(queue);
    }

    // Child: llm.prefill
    if (trace.prefill_start_us > 0 && trace.prefill_end_us > 0) {
      Span prefill;
      prefill.trace_id = trace_id;
      prefill.span_id = generate_hex_id(rng, 8);
      prefill.parent_span_id = root.span_id;
      prefill.name = "llm.prefill";
      prefill.start_us = trace.prefill_start_us;
      prefill.end_us = trace.prefill_end_us;
      prefill.attributes.emplace_back("prompt_tokens", std::to_string(trace.prompt_tokens));
      prefill.attributes.emplace_back("cached_tokens", std::to_string(trace.cached_tokens));
      ss << "," << span_to_json(prefill);
    }

    // Child: llm.decode
    if (trace.first_token_us > 0 && trace.last_token_us > 0) {
      Span decode;
      decode.trace_id = trace_id;
      decode.span_id = generate_hex_id(rng, 8);
      decode.parent_span_id = root.span_id;
      decode.name = "llm.decode";
      decode.start_us = trace.first_token_us;
      decode.end_us = trace.last_token_us;
      decode.attributes.emplace_back("output_tokens", std::to_string(trace.output_tokens));
      ss << "," << span_to_json(decode);
    }
  }

  ss << "]}]}]}";
  return ss.str();
}

void export_otlp_file(const std::vector<RequestTrace>& traces,
                      const std::filesystem::path& output_path,
                      const std::string& service_name) {
  const auto json = export_otlp_json(traces, service_name);
  std::ofstream file(output_path);
  if (!file.is_open()) {
    throw std::runtime_error("cannot open output file: " + output_path.string());
  }
  file << json;
}

}  // namespace hotpath
