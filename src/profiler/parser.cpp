#include "rlprof/profiler/parser.h"

#include <sqlite3.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "rlprof/profiler/categorizer.h"

namespace rlprof::profiler {
namespace {

using SqliteDbPtr = std::unique_ptr<sqlite3, decltype(&sqlite3_close)>;
using SqliteStmtPtr = std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)>;

struct NvtxRange {
  std::int64_t start = 0;
  std::int64_t end = 0;
  std::string text;
  std::string category;
};

struct KernelEvent {
  std::int64_t start = 0;
  std::int64_t end = 0;
  std::string name;
  std::string runtime_name;
  std::int64_t registers = 0;
  std::int64_t shared_mem = 0;
};

struct Aggregate {
  std::int64_t total_ns = 0;
  std::int64_t calls = 0;
  std::int64_t min_ns = std::numeric_limits<std::int64_t>::max();
  std::int64_t max_ns = 0;
  std::int64_t registers = 0;
  std::int64_t shared_mem = 0;
  std::map<std::string, std::int64_t> category_ns;
};

std::int64_t column_int64(sqlite3_stmt* statement, int index) {
  return sqlite3_column_int64(statement, index);
}

std::string column_text(sqlite3_stmt* statement, int index) {
  const unsigned char* text = sqlite3_column_text(statement, index);
  return text == nullptr ? std::string() : reinterpret_cast<const char*>(text);
}

bool table_exists(sqlite3* db, const char* table_name) {
  sqlite3_stmt* raw_statement = nullptr;
  constexpr const char* kSql =
      "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1";
  if (sqlite3_prepare_v2(db, kSql, -1, &raw_statement, nullptr) != SQLITE_OK) {
    throw std::runtime_error(sqlite3_errmsg(db));
  }
  SqliteStmtPtr statement(raw_statement, sqlite3_finalize);
  if (sqlite3_bind_text(statement.get(), 1, table_name, -1, SQLITE_STATIC) != SQLITE_OK) {
    throw std::runtime_error(sqlite3_errmsg(db));
  }
  const int step_rc = sqlite3_step(statement.get());
  if (step_rc != SQLITE_ROW && step_rc != SQLITE_DONE) {
    throw std::runtime_error(sqlite3_errmsg(db));
  }
  return step_rc == SQLITE_ROW;
}

std::string build_kernel_events_query(bool has_string_ids, bool has_runtime) {
  const std::string kernel_name =
      has_string_ids
          ? "COALESCE(short_names.value, CAST(kernels.shortName AS TEXT))"
          : "CAST(kernels.shortName AS TEXT)";
  const std::string runtime_name =
      has_runtime
          ? (has_string_ids
                 ? "COALESCE(runtime_names.value, CAST(runtime.nameId AS TEXT))"
                 : "CAST(runtime.nameId AS TEXT)")
          : "''";

  std::string query =
      "SELECT kernels.start, kernels.end, " + kernel_name + " AS shortName, " +
      runtime_name +
      " AS runtimeName, kernels.registersPerThread, "
      "(kernels.staticSharedMemory + kernels.dynamicSharedMemory) AS sharedMem "
      "FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels ";

  if (has_string_ids) {
    query += "LEFT JOIN StringIds AS short_names ON short_names.id = kernels.shortName ";
  }
  if (has_runtime) {
    query +=
        "LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS runtime "
        "ON runtime.correlationId = kernels.correlationId ";
    if (has_string_ids) {
      query +=
          "LEFT JOIN StringIds AS runtime_names ON runtime_names.id = runtime.nameId ";
    }
  }
  query += "ORDER BY kernels.start ASC";
  return query;
}

std::optional<std::string> categorize_non_other(std::string_view text) {
  const std::string category = std::string(categorize(text));
  if (category == "other") {
    return std::nullopt;
  }
  return category;
}

std::string to_lower(std::string_view text) {
  std::string lowered;
  lowered.reserve(text.size());
  for (unsigned char ch : text) {
    lowered.push_back(static_cast<char>(std::tolower(ch)));
  }
  return lowered;
}

std::optional<std::string> categorize_nvtx_text(std::string_view text) {
  const std::string lowered = to_lower(text);
  const auto contains = [&](std::string_view needle) {
    return lowered.find(needle) != std::string::npos;
  };

  if (contains("attention")) {
    return "attention";
  }
  if (contains("mamba") || contains("selective_scan") || contains("ssm")) {
    return "mamba";
  }
  if (contains("moe") || contains("expert")) {
    return "moe";
  }
  if (contains("rms_norm") || contains("layer_norm") || contains("norm")) {
    return "norm";
  }
  if (contains("rotary") || contains("rope") || contains("position")) {
    return "position";
  }
  if (contains("cache") || contains("kv")) {
    return "cache";
  }
  if (contains("sample") || contains("topk") || contains("topp") || contains("argmax")) {
    return "sampling";
  }
  if (contains("allreduce") || contains("allgather") || contains("nccl") ||
      contains("reduce_scatter")) {
    return "comm";
  }
  if (contains("memcpy") || contains("memset")) {
    return "memory";
  }
  if (contains("silu") || contains("gelu") || contains("relu") ||
      contains("swiglu") || contains("act_and_mul")) {
    return "activation";
  }
  if (contains("gemm") || contains("matmul") || contains("linear")) {
    return "gemm";
  }
  return categorize_non_other(text);
}

std::vector<NvtxRange> load_nvtx_ranges(sqlite3* db, bool has_string_ids) {
  if (!table_exists(db, "NVTX_EVENTS")) {
    return {};
  }

  const std::string text_expr = has_string_ids
                                    ? "COALESCE(nvtx.text, text_ids.value, '')"
                                    : "COALESCE(nvtx.text, '')";
  std::string query =
      "SELECT nvtx.start, nvtx.end, " + text_expr +
      " AS text "
      "FROM NVTX_EVENTS AS nvtx ";
  if (has_string_ids) {
    query += "LEFT JOIN StringIds AS text_ids ON text_ids.id = nvtx.textId ";
  }
  query += "WHERE nvtx.end IS NOT NULL ORDER BY nvtx.start ASC";

  sqlite3_stmt* raw_statement = nullptr;
  if (sqlite3_prepare_v2(db, query.c_str(), -1, &raw_statement, nullptr) != SQLITE_OK) {
    throw std::runtime_error(sqlite3_errmsg(db));
  }
  SqliteStmtPtr statement(raw_statement, sqlite3_finalize);

  std::vector<NvtxRange> ranges;
  while (true) {
    const int step_rc = sqlite3_step(statement.get());
    if (step_rc == SQLITE_DONE) {
      break;
    }
    if (step_rc != SQLITE_ROW) {
      throw std::runtime_error(sqlite3_errmsg(db));
    }

    const std::string text = column_text(statement.get(), 2);
    const auto category = categorize_nvtx_text(text);
    if (!category.has_value()) {
      continue;
    }
    ranges.push_back(NvtxRange{
        .start = column_int64(statement.get(), 0),
        .end = column_int64(statement.get(), 1),
        .text = text,
        .category = *category,
    });
  }

  return ranges;
}

std::optional<std::string> category_from_nvtx(
    const KernelEvent& event,
    const std::vector<NvtxRange>& ranges,
    std::size_t* first_candidate_index) {
  while (*first_candidate_index < ranges.size() &&
         ranges[*first_candidate_index].end <= event.start) {
    ++(*first_candidate_index);
  }

  std::optional<std::string> best_category;
  std::int64_t best_overlap = 0;
  for (std::size_t i = *first_candidate_index; i < ranges.size(); ++i) {
    if (ranges[i].start >= event.end) {
      break;
    }
    const std::int64_t overlap =
        std::min(event.end, ranges[i].end) - std::max(event.start, ranges[i].start);
    if (overlap > best_overlap) {
      best_overlap = overlap;
      best_category = ranges[i].category;
    }
  }
  return best_category;
}

std::vector<KernelEvent> load_kernel_events(sqlite3* db, bool has_string_ids, bool has_runtime) {
  const std::string query = build_kernel_events_query(has_string_ids, has_runtime);
  sqlite3_stmt* raw_statement = nullptr;
  if (sqlite3_prepare_v2(db, query.c_str(), -1, &raw_statement, nullptr) != SQLITE_OK) {
    throw std::runtime_error(sqlite3_errmsg(db));
  }
  SqliteStmtPtr statement(raw_statement, sqlite3_finalize);

  std::vector<KernelEvent> events;
  while (true) {
    const int step_rc = sqlite3_step(statement.get());
    if (step_rc == SQLITE_DONE) {
      break;
    }
    if (step_rc != SQLITE_ROW) {
      throw std::runtime_error(sqlite3_errmsg(db));
    }

    events.push_back(KernelEvent{
        .start = column_int64(statement.get(), 0),
        .end = column_int64(statement.get(), 1),
        .name = column_text(statement.get(), 2),
        .runtime_name = column_text(statement.get(), 3),
        .registers = column_int64(statement.get(), 4),
        .shared_mem = column_int64(statement.get(), 5),
    });
  }
  return events;
}

std::string choose_category(const Aggregate& aggregate) {
  std::string category = "other";
  std::int64_t best_ns = -1;
  for (const auto& [candidate, total_ns] : aggregate.category_ns) {
    if (total_ns > best_ns) {
      best_ns = total_ns;
      category = candidate;
    }
  }
  return category;
}

}  // namespace

std::vector<KernelRecord> parse_nsys_sqlite(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("SQLite file does not exist: " + path.string());
  }

  sqlite3* raw_db = nullptr;
  const int open_rc = sqlite3_open_v2(path.c_str(), &raw_db, SQLITE_OPEN_READONLY, nullptr);
  if (open_rc != SQLITE_OK) {
    const std::string message =
        raw_db == nullptr ? "failed to open sqlite database" : sqlite3_errmsg(raw_db);
    if (raw_db != nullptr) {
      sqlite3_close(raw_db);
    }
    throw std::runtime_error(message);
  }
  SqliteDbPtr db(raw_db, sqlite3_close);

  const bool has_string_ids = table_exists(db.get(), "StringIds");
  const bool has_runtime = table_exists(db.get(), "CUPTI_ACTIVITY_KIND_RUNTIME");
  const std::vector<NvtxRange> nvtx_ranges = load_nvtx_ranges(db.get(), has_string_ids);
  const std::vector<KernelEvent> kernel_events =
      load_kernel_events(db.get(), has_string_ids, has_runtime);

  std::size_t nvtx_index = 0;
  std::map<std::string, Aggregate> aggregates;
  for (const KernelEvent& event : kernel_events) {
    const std::int64_t duration_ns = event.end - event.start;
    const std::optional<std::string> nvtx_category =
        category_from_nvtx(event, nvtx_ranges, &nvtx_index);
    const std::optional<std::string> runtime_category =
        categorize_non_other(event.runtime_name);
    const std::string category = nvtx_category.has_value()
                                     ? *nvtx_category
                                     : runtime_category.has_value()
                                           ? *runtime_category
                                           : std::string(categorize(event.name));

    Aggregate& aggregate = aggregates[event.name];
    aggregate.total_ns += duration_ns;
    aggregate.calls += 1;
    aggregate.min_ns = std::min(aggregate.min_ns, duration_ns);
    aggregate.max_ns = std::max(aggregate.max_ns, duration_ns);
    aggregate.registers = std::max(aggregate.registers, event.registers);
    aggregate.shared_mem = std::max(aggregate.shared_mem, event.shared_mem);
    aggregate.category_ns[category] += duration_ns;
  }

  std::vector<KernelRecord> records;
  records.reserve(aggregates.size());
  for (const auto& [name, aggregate] : aggregates) {
    records.push_back(KernelRecord{
        .name = name,
        .category = choose_category(aggregate),
        .total_ns = aggregate.total_ns,
        .calls = aggregate.calls,
        .avg_ns = aggregate.calls == 0 ? 0 : aggregate.total_ns / aggregate.calls,
        .min_ns = aggregate.calls == 0 ? 0 : aggregate.min_ns,
        .max_ns = aggregate.max_ns,
        .registers = aggregate.registers,
        .shared_mem = aggregate.shared_mem,
    });
  }

  std::sort(
      records.begin(),
      records.end(),
      [](const KernelRecord& lhs, const KernelRecord& rhs) {
        if (lhs.total_ns != rhs.total_ns) {
          return lhs.total_ns > rhs.total_ns;
        }
        return lhs.name < rhs.name;
      });
  return records;
}

}  // namespace rlprof::profiler
