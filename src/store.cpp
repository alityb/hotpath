#include "hotpath/store.h"

#include <sqlite3.h>

#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace hotpath {
namespace {

using SqliteDbPtr = std::unique_ptr<sqlite3, decltype(&sqlite3_close)>;
using SqliteStmtPtr = std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)>;

constexpr const char* kSchema = R"SQL(
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kernels (
    name TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    total_ns INTEGER NOT NULL,
    calls INTEGER NOT NULL,
    avg_ns INTEGER NOT NULL,
    min_ns INTEGER NOT NULL,
    max_ns INTEGER NOT NULL,
    registers INTEGER NOT NULL,
    shared_mem INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS vllm_metrics (
    sample_time REAL NOT NULL,
    source TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS vllm_metrics_summary (
    metric TEXT PRIMARY KEY,
    avg REAL,
    peak REAL,
    min REAL
);

CREATE TABLE IF NOT EXISTS traffic_stats (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS request_traces (
    id INTEGER PRIMARY KEY,
    profile_id INTEGER NOT NULL,
    request_id TEXT NOT NULL,
    arrival_us INTEGER,
    queue_start_us INTEGER,
    prefill_start_us INTEGER,
    prefill_end_us INTEGER,
    server_last_token_us INTEGER,
    first_token_us INTEGER,
    last_token_us INTEGER,
    completion_us INTEGER,
    prompt_tokens INTEGER,
    output_tokens INTEGER,
    cached_tokens INTEGER,
    status TEXT,
    server_timing_available INTEGER NOT NULL DEFAULT 0,
    prompt_tokens_estimated INTEGER NOT NULL DEFAULT 0,
    prompt_text TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS request_events (
    id INTEGER PRIMARY KEY,
    trace_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    timestamp_us INTEGER NOT NULL,
    detail TEXT,
    FOREIGN KEY (trace_id) REFERENCES request_traces(id)
);

CREATE TABLE IF NOT EXISTS serve_analysis (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
)SQL";

SqliteDbPtr open_db(const std::filesystem::path& path) {
  sqlite3* raw_db = nullptr;
  if (sqlite3_open(path.c_str(), &raw_db) != SQLITE_OK) {
    const std::string message =
        raw_db == nullptr ? "failed to open sqlite db" : sqlite3_errmsg(raw_db);
    if (raw_db != nullptr) {
      sqlite3_close(raw_db);
    }
    throw std::runtime_error(message);
  }
  return SqliteDbPtr(raw_db, sqlite3_close);
}

void exec_sql(sqlite3* db, const char* sql) {
  char* error_message = nullptr;
  const int rc = sqlite3_exec(db, sql, nullptr, nullptr, &error_message);
  if (rc != SQLITE_OK) {
    const std::string message = error_message == nullptr ? "sqlite error" : error_message;
    sqlite3_free(error_message);
    throw std::runtime_error(message);
  }
}

SqliteStmtPtr prepare(sqlite3* db, const char* sql) {
  sqlite3_stmt* raw_stmt = nullptr;
  if (sqlite3_prepare_v2(db, sql, -1, &raw_stmt, nullptr) != SQLITE_OK) {
    throw std::runtime_error(sqlite3_errmsg(db));
  }
  return SqliteStmtPtr(raw_stmt, sqlite3_finalize);
}

void bind_text(sqlite3_stmt* stmt, int index, const std::string& value) {
  sqlite3_bind_text(stmt, index, value.c_str(), -1, SQLITE_TRANSIENT);
}

void bind_optional_double(sqlite3_stmt* stmt, int index, const std::optional<double>& value) {
  if (value.has_value()) {
    sqlite3_bind_double(stmt, index, *value);
  } else {
    sqlite3_bind_null(stmt, index);
  }
}

std::string optional_to_string(const std::optional<double>& value) {
  return value.has_value() ? std::to_string(*value) : "";
}

std::optional<double> string_to_optional_double(const std::string& value) {
  if (value.empty()) {
    return std::nullopt;
  }
  return std::stod(value);
}

std::int64_t string_to_i64_or(const std::string& value, std::int64_t fallback) {
  if (value.empty()) {
    return fallback;
  }
  return std::stoll(value);
}

std::string column_text(sqlite3_stmt* stmt, int index) {
  const unsigned char* text = sqlite3_column_text(stmt, index);
  return text == nullptr ? std::string() : reinterpret_cast<const char*>(text);
}

bool table_has_column(sqlite3* db, const std::string& table, const std::string& column) {
  const auto pragma = "PRAGMA table_info(" + table + ")";
  auto stmt = prepare(db, pragma.c_str());
  while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
    if (column_text(stmt.get(), 1) == column) {
      return true;
    }
  }
  return false;
}

}  // namespace

std::filesystem::path init_db(const std::filesystem::path& path) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  SqliteDbPtr db = open_db(path);
  exec_sql(db.get(), kSchema);
  if (!table_has_column(db.get(), "vllm_metrics", "source")) {
    exec_sql(
        db.get(),
        "ALTER TABLE vllm_metrics ADD COLUMN source TEXT NOT NULL DEFAULT ''");
  }
  if (!table_has_column(db.get(), "request_traces", "prompt_tokens_estimated")) {
    exec_sql(
        db.get(),
        "ALTER TABLE request_traces "
        "ADD COLUMN prompt_tokens_estimated INTEGER NOT NULL DEFAULT 0");
  }
  if (!table_has_column(db.get(), "request_traces", "prompt_text")) {
    exec_sql(
        db.get(),
        "ALTER TABLE request_traces "
        "ADD COLUMN prompt_text TEXT NOT NULL DEFAULT ''");
  }
  if (!table_has_column(db.get(), "request_traces", "server_last_token_us")) {
    exec_sql(
        db.get(),
        "ALTER TABLE request_traces "
        "ADD COLUMN server_last_token_us INTEGER");
  }
  if (!table_has_column(db.get(), "request_traces", "server_timing_available")) {
    exec_sql(
        db.get(),
        "ALTER TABLE request_traces "
        "ADD COLUMN server_timing_available INTEGER NOT NULL DEFAULT 0");
  }
  return path;
}

std::filesystem::path save_profile(
    const std::filesystem::path& path,
    const ProfileData& profile) {
  init_db(path);
  SqliteDbPtr db = open_db(path);

  exec_sql(db.get(), "BEGIN");
  exec_sql(db.get(), "DELETE FROM meta");
  exec_sql(db.get(), "DELETE FROM kernels");
  exec_sql(db.get(), "DELETE FROM vllm_metrics");
  exec_sql(db.get(), "DELETE FROM vllm_metrics_summary");
  exec_sql(db.get(), "DELETE FROM traffic_stats");

  {
    auto stmt = prepare(db.get(), "INSERT INTO meta (key, value) VALUES (?, ?)");
    for (const auto& [key, value] : profile.meta) {
      sqlite3_reset(stmt.get());
      sqlite3_clear_bindings(stmt.get());
      bind_text(stmt.get(), 1, key);
      bind_text(stmt.get(), 2, value);
      if (sqlite3_step(stmt.get()) != SQLITE_DONE) {
        throw std::runtime_error(sqlite3_errmsg(db.get()));
      }
    }
  }

  {
    auto stmt = prepare(
        db.get(),
        "INSERT INTO kernels "
        "(name, category, total_ns, calls, avg_ns, min_ns, max_ns, registers, shared_mem) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");
    for (const profiler::KernelRecord& kernel : profile.kernels) {
      sqlite3_reset(stmt.get());
      sqlite3_clear_bindings(stmt.get());
      bind_text(stmt.get(), 1, kernel.name);
      bind_text(stmt.get(), 2, kernel.category);
      sqlite3_bind_int64(stmt.get(), 3, kernel.total_ns);
      sqlite3_bind_int64(stmt.get(), 4, kernel.calls);
      sqlite3_bind_int64(stmt.get(), 5, kernel.avg_ns);
      sqlite3_bind_int64(stmt.get(), 6, kernel.min_ns);
      sqlite3_bind_int64(stmt.get(), 7, kernel.max_ns);
      sqlite3_bind_int64(stmt.get(), 8, kernel.registers);
      sqlite3_bind_int64(stmt.get(), 9, kernel.shared_mem);
      if (sqlite3_step(stmt.get()) != SQLITE_DONE) {
        throw std::runtime_error(sqlite3_errmsg(db.get()));
      }
    }
  }

  {
    auto stmt = prepare(
        db.get(),
        "INSERT INTO vllm_metrics (sample_time, source, metric, value) VALUES (?, ?, ?, ?)");
    for (const MetricSample& metric : profile.metrics) {
      sqlite3_reset(stmt.get());
      sqlite3_clear_bindings(stmt.get());
      sqlite3_bind_double(stmt.get(), 1, metric.sample_time);
      bind_text(stmt.get(), 2, metric.source);
      bind_text(stmt.get(), 3, metric.metric);
      sqlite3_bind_double(stmt.get(), 4, metric.value);
      if (sqlite3_step(stmt.get()) != SQLITE_DONE) {
        throw std::runtime_error(sqlite3_errmsg(db.get()));
      }
    }
  }

  {
    auto stmt = prepare(
        db.get(),
        "INSERT INTO vllm_metrics_summary (metric, avg, peak, min) VALUES (?, ?, ?, ?)");
    for (const MetricSummary& summary : profile.metrics_summary) {
      sqlite3_reset(stmt.get());
      sqlite3_clear_bindings(stmt.get());
      bind_text(stmt.get(), 1, summary.metric);
      bind_optional_double(stmt.get(), 2, summary.avg);
      bind_optional_double(stmt.get(), 3, summary.peak);
      bind_optional_double(stmt.get(), 4, summary.min);
      if (sqlite3_step(stmt.get()) != SQLITE_DONE) {
        throw std::runtime_error(sqlite3_errmsg(db.get()));
      }
    }
  }

  {
    auto stmt = prepare(db.get(), "INSERT INTO traffic_stats (key, value) VALUES (?, ?)");
    const std::vector<std::pair<std::string, std::string>> traffic_rows = {
        {"total_requests", std::to_string(profile.traffic_stats.total_requests)},
        {"completion_length_mean", optional_to_string(profile.traffic_stats.completion_length_mean)},
        {"completion_length_p50", optional_to_string(profile.traffic_stats.completion_length_p50)},
        {"completion_length_p99", optional_to_string(profile.traffic_stats.completion_length_p99)},
        {"max_median_ratio", optional_to_string(profile.traffic_stats.max_median_ratio)},
        {"errors", std::to_string(profile.traffic_stats.errors)},
        {"completion_length_samples", std::to_string(profile.traffic_stats.completion_length_samples)},
    };
    for (const auto& [key, value] : traffic_rows) {
      sqlite3_reset(stmt.get());
      sqlite3_clear_bindings(stmt.get());
      bind_text(stmt.get(), 1, key);
      bind_text(stmt.get(), 2, value);
      if (sqlite3_step(stmt.get()) != SQLITE_DONE) {
        throw std::runtime_error(sqlite3_errmsg(db.get()));
      }
    }
  }

  exec_sql(db.get(), "COMMIT");
  return path;
}

ProfileData load_profile(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("profile db does not exist: " + path.string());
  }

  SqliteDbPtr db = open_db(path);
  ProfileData profile;

  {
    auto stmt = prepare(db.get(), "SELECT key, value FROM meta ORDER BY key");
    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
      profile.meta[column_text(stmt.get(), 0)] = column_text(stmt.get(), 1);
    }
  }

  {
    auto stmt = prepare(
        db.get(),
        "SELECT name, category, total_ns, calls, avg_ns, min_ns, max_ns, registers, shared_mem "
        "FROM kernels ORDER BY total_ns DESC, name ASC");
    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
      profile.kernels.push_back(profiler::KernelRecord{
          .name = column_text(stmt.get(), 0),
          .category = column_text(stmt.get(), 1),
          .total_ns = sqlite3_column_int64(stmt.get(), 2),
          .calls = sqlite3_column_int64(stmt.get(), 3),
          .avg_ns = sqlite3_column_int64(stmt.get(), 4),
          .min_ns = sqlite3_column_int64(stmt.get(), 5),
          .max_ns = sqlite3_column_int64(stmt.get(), 6),
          .registers = sqlite3_column_int64(stmt.get(), 7),
          .shared_mem = sqlite3_column_int64(stmt.get(), 8),
      });
    }
  }

  {
    const bool has_source = table_has_column(db.get(), "vllm_metrics", "source");
    auto stmt = prepare(
        db.get(),
        has_source
            ? "SELECT sample_time, source, metric, value FROM vllm_metrics "
              "ORDER BY sample_time ASC, source ASC, metric ASC"
            : "SELECT sample_time, metric, value FROM vllm_metrics "
              "ORDER BY sample_time ASC, metric ASC");
    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
      if (has_source) {
        profile.metrics.push_back(MetricSample{
            .sample_time = sqlite3_column_double(stmt.get(), 0),
            .source = column_text(stmt.get(), 1),
            .metric = column_text(stmt.get(), 2),
            .value = sqlite3_column_double(stmt.get(), 3),
        });
      } else {
        profile.metrics.push_back(MetricSample{
            .sample_time = sqlite3_column_double(stmt.get(), 0),
            .source = "",
            .metric = column_text(stmt.get(), 1),
            .value = sqlite3_column_double(stmt.get(), 2),
        });
      }
    }
  }

  {
    auto stmt = prepare(
        db.get(),
        "SELECT metric, avg, peak, min FROM vllm_metrics_summary ORDER BY metric ASC");
    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
      profile.metrics_summary.push_back(MetricSummary{
          .metric = column_text(stmt.get(), 0),
          .avg = sqlite3_column_type(stmt.get(), 1) == SQLITE_NULL
                     ? std::nullopt
                     : std::optional<double>(sqlite3_column_double(stmt.get(), 1)),
          .peak = sqlite3_column_type(stmt.get(), 2) == SQLITE_NULL
                      ? std::nullopt
                      : std::optional<double>(sqlite3_column_double(stmt.get(), 2)),
          .min = sqlite3_column_type(stmt.get(), 3) == SQLITE_NULL
                     ? std::nullopt
                     : std::optional<double>(sqlite3_column_double(stmt.get(), 3)),
      });
    }
  }

  {
    auto stmt = prepare(db.get(), "SELECT key, value FROM traffic_stats ORDER BY key");
    std::map<std::string, std::string> traffic_map;
    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
      traffic_map[column_text(stmt.get(), 0)] = column_text(stmt.get(), 1);
    }
    const std::int64_t total_requests = string_to_i64_or(traffic_map["total_requests"], 0);
    const std::int64_t errors = string_to_i64_or(traffic_map["errors"], 0);
    const auto completion_length_mean =
        string_to_optional_double(traffic_map["completion_length_mean"]);
    std::int64_t completion_length_samples =
        string_to_i64_or(traffic_map["completion_length_samples"], 0);
    if (completion_length_samples == 0 && completion_length_mean.has_value()) {
      completion_length_samples = std::max<std::int64_t>(0, total_requests - errors);
    }
    profile.traffic_stats = TrafficStats{
        .total_requests = total_requests,
        .completion_length_mean = completion_length_mean,
        .completion_length_p50 = string_to_optional_double(traffic_map["completion_length_p50"]),
        .completion_length_p99 = string_to_optional_double(traffic_map["completion_length_p99"]),
        .max_median_ratio = string_to_optional_double(traffic_map["max_median_ratio"]),
        .errors = errors,
        .completion_length_samples = completion_length_samples,
    };
  }

  return profile;
}

int64_t insert_request_trace(const std::filesystem::path& db_path,
                             int64_t profile_id,
                             const RequestTrace& trace) {
  init_db(db_path);
  SqliteDbPtr db = open_db(db_path);

  auto stmt = prepare(
      db.get(),
      "INSERT INTO request_traces "
      "(profile_id, request_id, arrival_us, queue_start_us, prefill_start_us, "
      "prefill_end_us, server_last_token_us, first_token_us, last_token_us, completion_us, "
      "prompt_tokens, output_tokens, cached_tokens, status, server_timing_available, "
      "prompt_tokens_estimated, prompt_text) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
  sqlite3_bind_int64(stmt.get(), 1, profile_id);
  bind_text(stmt.get(), 2, trace.request_id);
  sqlite3_bind_int64(stmt.get(), 3, trace.arrival_us);
  sqlite3_bind_int64(stmt.get(), 4, trace.queue_start_us);
  sqlite3_bind_int64(stmt.get(), 5, trace.prefill_start_us);
  sqlite3_bind_int64(stmt.get(), 6, trace.prefill_end_us);
  sqlite3_bind_int64(stmt.get(), 7, trace.server_last_token_us);
  sqlite3_bind_int64(stmt.get(), 8, trace.first_token_us);
  sqlite3_bind_int64(stmt.get(), 9, trace.last_token_us);
  sqlite3_bind_int64(stmt.get(), 10, trace.completion_us);
  sqlite3_bind_int(stmt.get(), 11, trace.prompt_tokens);
  sqlite3_bind_int(stmt.get(), 12, trace.output_tokens);
  sqlite3_bind_int(stmt.get(), 13, trace.cached_tokens);
  bind_text(stmt.get(), 14, trace.status);
  sqlite3_bind_int(stmt.get(), 15, trace.server_timing_available ? 1 : 0);
  sqlite3_bind_int(stmt.get(), 16, trace.prompt_tokens_estimated ? 1 : 0);
  bind_text(stmt.get(), 17, trace.prompt_text);
  if (sqlite3_step(stmt.get()) != SQLITE_DONE) {
    throw std::runtime_error(sqlite3_errmsg(db.get()));
  }
  const int64_t trace_id = sqlite3_last_insert_rowid(db.get());

  if (!trace.events.empty()) {
    auto event_stmt = prepare(
        db.get(),
        "INSERT INTO request_events (trace_id, event_type, timestamp_us, detail) "
        "VALUES (?, ?, ?, ?)");
    for (const RequestEvent& event : trace.events) {
      sqlite3_reset(event_stmt.get());
      sqlite3_clear_bindings(event_stmt.get());
      sqlite3_bind_int64(event_stmt.get(), 1, trace_id);
      bind_text(event_stmt.get(), 2, event.event_type);
      sqlite3_bind_int64(event_stmt.get(), 3, event.timestamp_us);
      bind_text(event_stmt.get(), 4, event.detail);
      if (sqlite3_step(event_stmt.get()) != SQLITE_DONE) {
        throw std::runtime_error(sqlite3_errmsg(db.get()));
      }
    }
  }

  return trace_id;
}

namespace {

std::vector<RequestTrace> load_traces_from_stmt(sqlite3* db, sqlite3_stmt* stmt) {
  std::vector<RequestTrace> traces;
  std::map<int64_t, size_t> trace_index;

  while (sqlite3_step(stmt) == SQLITE_ROW) {
    const int64_t id = sqlite3_column_int64(stmt, 0);
    RequestTrace t;
    t.request_id = column_text(stmt, 1);
    t.arrival_us = sqlite3_column_int64(stmt, 2);
    t.queue_start_us = sqlite3_column_int64(stmt, 3);
    t.prefill_start_us = sqlite3_column_int64(stmt, 4);
    t.prefill_end_us = sqlite3_column_int64(stmt, 5);
    t.server_last_token_us = sqlite3_column_int64(stmt, 6);
    t.first_token_us = sqlite3_column_int64(stmt, 7);
    t.last_token_us = sqlite3_column_int64(stmt, 8);
    t.completion_us = sqlite3_column_int64(stmt, 9);
    t.prompt_tokens = sqlite3_column_int(stmt, 10);
    t.output_tokens = sqlite3_column_int(stmt, 11);
    t.cached_tokens = sqlite3_column_int(stmt, 12);
    t.status = column_text(stmt, 13);
    t.server_timing_available = sqlite3_column_int(stmt, 14) != 0;
    t.prompt_tokens_estimated = sqlite3_column_int(stmt, 15) != 0;
    t.prompt_text = column_text(stmt, 16);
    trace_index[id] = traces.size();
    traces.push_back(std::move(t));
  }

  if (!traces.empty()) {
    // Load events for all traces
    auto event_stmt = prepare(
        db,
        "SELECT trace_id, event_type, timestamp_us, detail FROM request_events "
        "ORDER BY trace_id, timestamp_us");
    while (sqlite3_step(event_stmt.get()) == SQLITE_ROW) {
      const int64_t tid = sqlite3_column_int64(event_stmt.get(), 0);
      auto it = trace_index.find(tid);
      if (it != trace_index.end()) {
        RequestEvent ev;
        ev.event_type = column_text(event_stmt.get(), 1);
        ev.timestamp_us = sqlite3_column_int64(event_stmt.get(), 2);
        ev.detail = column_text(event_stmt.get(), 3);
        traces[it->second].events.push_back(std::move(ev));
      }
    }
  }

  return traces;
}

}  // namespace

std::vector<RequestTrace> load_request_traces(const std::filesystem::path& db_path,
                                              int64_t profile_id) {
  init_db(db_path);
  SqliteDbPtr db = open_db(db_path);
  auto stmt = prepare(
      db.get(),
      "SELECT id, request_id, arrival_us, queue_start_us, prefill_start_us, "
      "prefill_end_us, server_last_token_us, first_token_us, last_token_us, completion_us, "
      "prompt_tokens, output_tokens, cached_tokens, status, server_timing_available, "
      "prompt_tokens_estimated, prompt_text "
      "FROM request_traces WHERE profile_id = ? ORDER BY arrival_us");
  sqlite3_bind_int64(stmt.get(), 1, profile_id);
  return load_traces_from_stmt(db.get(), stmt.get());
}

std::vector<RequestTrace> query_traces_prefill_gt(const std::filesystem::path& db_path,
                                                  int64_t profile_id,
                                                  int64_t min_prefill_us) {
  init_db(db_path);
  SqliteDbPtr db = open_db(db_path);
  auto stmt = prepare(
      db.get(),
      "SELECT id, request_id, arrival_us, queue_start_us, prefill_start_us, "
      "prefill_end_us, server_last_token_us, first_token_us, last_token_us, completion_us, "
      "prompt_tokens, output_tokens, cached_tokens, status, server_timing_available, "
      "prompt_tokens_estimated, prompt_text "
      "FROM request_traces "
      "WHERE profile_id = ? AND (prefill_end_us - prefill_start_us) > ? "
      "ORDER BY arrival_us");
  sqlite3_bind_int64(stmt.get(), 1, profile_id);
  sqlite3_bind_int64(stmt.get(), 2, min_prefill_us);
  return load_traces_from_stmt(db.get(), stmt.get());
}

std::vector<RequestTrace> query_traces_cached_gt(const std::filesystem::path& db_path,
                                                 int64_t profile_id,
                                                 int min_cached_tokens) {
  init_db(db_path);
  SqliteDbPtr db = open_db(db_path);
  auto stmt = prepare(
      db.get(),
      "SELECT id, request_id, arrival_us, queue_start_us, prefill_start_us, "
      "prefill_end_us, server_last_token_us, first_token_us, last_token_us, completion_us, "
      "prompt_tokens, output_tokens, cached_tokens, status, server_timing_available, "
      "prompt_tokens_estimated, prompt_text "
      "FROM request_traces "
      "WHERE profile_id = ? AND cached_tokens > ? "
      "ORDER BY arrival_us");
  sqlite3_bind_int64(stmt.get(), 1, profile_id);
  sqlite3_bind_int(stmt.get(), 2, min_cached_tokens);
  return load_traces_from_stmt(db.get(), stmt.get());
}

void save_serve_analysis(const std::filesystem::path& db_path,
                         const std::map<std::string, std::string>& kv) {
  init_db(db_path);
  SqliteDbPtr db = open_db(db_path);
  exec_sql(db.get(), "BEGIN");
  exec_sql(db.get(), "DELETE FROM serve_analysis");
  auto stmt = prepare(db.get(),
      "INSERT OR REPLACE INTO serve_analysis (key, value) VALUES (?, ?)");
  for (const auto& [key, value] : kv) {
    sqlite3_reset(stmt.get());
    sqlite3_clear_bindings(stmt.get());
    bind_text(stmt.get(), 1, key);
    bind_text(stmt.get(), 2, value);
    if (sqlite3_step(stmt.get()) != SQLITE_DONE) {
      throw std::runtime_error(sqlite3_errmsg(db.get()));
    }
  }
  exec_sql(db.get(), "COMMIT");
}

std::map<std::string, std::string> load_serve_analysis(const std::filesystem::path& db_path) {
  SqliteDbPtr db = open_db(db_path);
  // Check table exists
  auto check = prepare(db.get(),
      "SELECT name FROM sqlite_master WHERE type='table' AND name='serve_analysis'");
  if (sqlite3_step(check.get()) != SQLITE_ROW) {
    return {};
  }
  auto stmt = prepare(db.get(), "SELECT key, value FROM serve_analysis ORDER BY key");
  std::map<std::string, std::string> result;
  while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
    result[column_text(stmt.get(), 0)] = column_text(stmt.get(), 1);
  }
  return result;
}

}  // namespace hotpath
