#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <sqlite3.h>

#include "rlprof/profiler/parser.h"

namespace {

void exec_sql(sqlite3* db, const char* sql) {
  char* error_message = nullptr;
  const int rc = sqlite3_exec(db, sql, nullptr, nullptr, &error_message);
  if (rc != SQLITE_OK) {
    const std::string message = error_message == nullptr ? "sqlite error" : error_message;
    sqlite3_free(error_message);
    throw std::runtime_error(message);
  }
}

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  namespace fs = std::filesystem;
  const fs::path temp_dir = fs::temp_directory_path() / "rlprof_cpp_tests";
  fs::create_directories(temp_dir);
  const auto verify_records = [](const std::vector<rlprof::profiler::KernelRecord>& records) {
    expect_true(records.size() == 2, "expected two aggregated kernel records");
    expect_true(records[0].name == "sm80_xmma_gemm_bf16", "unexpected first kernel");
    expect_true(records[0].category == "gemm", "unexpected first category");
    expect_true(records[0].total_ns == 260, "unexpected first total_ns");
    expect_true(records[0].calls == 2, "unexpected first calls");
    expect_true(records[0].avg_ns == 130, "unexpected first avg_ns");
    expect_true(records[0].min_ns == 100, "unexpected first min_ns");
    expect_true(records[0].max_ns == 160, "unexpected first max_ns");
    expect_true(records[0].registers == 72, "unexpected first registers");
    expect_true(records[0].shared_mem == 160, "unexpected first shared_mem");

    expect_true(records[1].name == "flash_fwd_splitkv_bf16_sm80", "unexpected second kernel");
    expect_true(records[1].category == "attention", "unexpected second category");
    expect_true(records[1].total_ns == 50, "unexpected second total_ns");
    expect_true(records[1].calls == 1, "unexpected second calls");
    expect_true(records[1].avg_ns == 50, "unexpected second avg_ns");
    expect_true(records[1].min_ns == 50, "unexpected second min_ns");
    expect_true(records[1].max_ns == 50, "unexpected second max_ns");
    expect_true(records[1].registers == 48, "unexpected second registers");
    expect_true(records[1].shared_mem == 64, "unexpected second shared_mem");
  };

  const auto verify_nvtx_records = [](const std::vector<rlprof::profiler::KernelRecord>& records) {
    expect_true(records.size() == 1, "expected one aggregated kernel record");
    expect_true(records[0].name == "Kernel2", "unexpected nvtx kernel");
    expect_true(records[0].category == "attention", "nvtx overlap should override name fallback");
    expect_true(records[0].total_ns == 100, "unexpected nvtx total_ns");
    expect_true(records[0].calls == 2, "unexpected nvtx calls");
    expect_true(records[0].avg_ns == 50, "unexpected nvtx avg_ns");
    expect_true(records[0].min_ns == 40, "unexpected nvtx min_ns");
    expect_true(records[0].max_ns == 60, "unexpected nvtx max_ns");
    expect_true(records[0].registers == 32, "unexpected nvtx registers");
    expect_true(records[0].shared_mem == 64, "unexpected nvtx shared_mem");
  };

  const auto build_text_schema_db = [&](const fs::path& db_path) {
    fs::remove(db_path);
    sqlite3* db = nullptr;
    if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK) {
      throw std::runtime_error("failed to create text-schema sqlite db");
    }
    try {
      exec_sql(
          db,
          R"SQL(
          CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
              shortName TEXT NOT NULL,
              start INTEGER NOT NULL,
              "end" INTEGER NOT NULL,
              registersPerThread INTEGER NOT NULL,
              staticSharedMemory INTEGER NOT NULL,
              dynamicSharedMemory INTEGER NOT NULL
          )
          )SQL");
      exec_sql(
          db,
          R"SQL(
          INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL
              (shortName, start, "end", registersPerThread, staticSharedMemory, dynamicSharedMemory)
          VALUES
              ('sm80_xmma_gemm_bf16', 0, 100, 64, 128, 0),
              ('sm80_xmma_gemm_bf16', 100, 260, 72, 128, 32),
              ('flash_fwd_splitkv_bf16_sm80', 0, 50, 48, 64, 0)
          )SQL");
      sqlite3_close(db);
    } catch (...) {
      sqlite3_close(db);
      throw;
    }
  };

  const auto build_string_ids_db = [&](const fs::path& db_path) {
    fs::remove(db_path);
    sqlite3* db = nullptr;
    if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK) {
      throw std::runtime_error("failed to create StringIds sqlite db");
    }
    try {
      exec_sql(
          db,
          R"SQL(
          CREATE TABLE StringIds (
              id INTEGER NOT NULL PRIMARY KEY,
              value TEXT NOT NULL
          )
          )SQL");
      exec_sql(
          db,
          R"SQL(
          CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
              shortName INTEGER NOT NULL,
              start INTEGER NOT NULL,
              "end" INTEGER NOT NULL,
              registersPerThread INTEGER NOT NULL,
              staticSharedMemory INTEGER NOT NULL,
              dynamicSharedMemory INTEGER NOT NULL
          )
          )SQL");
      exec_sql(
          db,
          R"SQL(
          INSERT INTO StringIds (id, value)
          VALUES
              (1719, 'sm80_xmma_gemm_bf16'),
              (1738, 'flash_fwd_splitkv_bf16_sm80')
          )SQL");
      exec_sql(
          db,
          R"SQL(
          INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL
              (shortName, start, "end", registersPerThread, staticSharedMemory, dynamicSharedMemory)
          VALUES
              (1719, 0, 100, 64, 128, 0),
              (1719, 100, 260, 72, 128, 32),
              (1738, 0, 50, 48, 64, 0)
          )SQL");
      sqlite3_close(db);
    } catch (...) {
      sqlite3_close(db);
      throw;
    }
  };

  const auto build_nvtx_db = [&](const fs::path& db_path) {
    fs::remove(db_path);
    sqlite3* db = nullptr;
    if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK) {
      throw std::runtime_error("failed to create NVTX sqlite db");
    }
    try {
      exec_sql(
          db,
          R"SQL(
          CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
              start INTEGER NOT NULL,
              "end" INTEGER NOT NULL,
              eventClass INTEGER NOT NULL,
              globalTid INTEGER,
              correlationId INTEGER,
              nameId INTEGER NOT NULL,
              returnValue INTEGER NOT NULL,
              callchainId INTEGER
          )
          )SQL");
      exec_sql(
          db,
          R"SQL(
          CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
              start INTEGER NOT NULL,
              "end" INTEGER NOT NULL,
              correlationId INTEGER,
              shortName TEXT NOT NULL,
              registersPerThread INTEGER NOT NULL,
              staticSharedMemory INTEGER NOT NULL,
              dynamicSharedMemory INTEGER NOT NULL
          )
          )SQL");
      exec_sql(
          db,
          R"SQL(
          CREATE TABLE NVTX_EVENTS (
              start INTEGER NOT NULL,
              "end" INTEGER,
              text TEXT,
              textId INTEGER
          )
          )SQL");
      exec_sql(
          db,
          R"SQL(
          INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME
              (start, "end", eventClass, globalTid, correlationId, nameId, returnValue, callchainId)
          VALUES
              (0, 1, 0, 0, 111, 0, 0, NULL),
              (1, 2, 0, 0, 112, 0, 0, NULL)
          )SQL");
      exec_sql(
          db,
          R"SQL(
          INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL
              (start, "end", correlationId, shortName, registersPerThread, staticSharedMemory, dynamicSharedMemory)
          VALUES
              (10, 50, 111, 'Kernel2', 24, 32, 0),
              (60, 120, 112, 'Kernel2', 32, 32, 32)
          )SQL");
      exec_sql(
          db,
          R"SQL(
          INSERT INTO NVTX_EVENTS (start, "end", text, textId)
          VALUES
              (0, 200, 'attention', NULL)
          )SQL");
      sqlite3_close(db);
    } catch (...) {
      sqlite3_close(db);
      throw;
    }
  };

  try {
    const fs::path text_db_path = temp_dir / "mock_nsys.sqlite";
    build_text_schema_db(text_db_path);
    verify_records(rlprof::profiler::parse_nsys_sqlite(text_db_path));
    fs::remove(text_db_path);

    const fs::path string_ids_db_path = temp_dir / "mock_nsys_string_ids.sqlite";
    build_string_ids_db(string_ids_db_path);
    verify_records(rlprof::profiler::parse_nsys_sqlite(string_ids_db_path));
    fs::remove(string_ids_db_path);

    const fs::path nvtx_db_path = temp_dir / "mock_nsys_nvtx.sqlite";
    build_nvtx_db(nvtx_db_path);
    verify_nvtx_records(rlprof::profiler::parse_nsys_sqlite(nvtx_db_path));
    fs::remove(nvtx_db_path);
  } catch (const std::exception& exc) {
    std::cerr << exc.what() << "\n";
    return 1;
  }

  return 0;
}
