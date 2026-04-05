#pragma once

#include <filesystem>
#include <vector>

#include "hotpath/profiler/kernel_record.h"
#include "hotpath/profiler/categorizer.h"

namespace hotpath::profiler {

struct KernelTraceEvent {
  std::string name;
  std::string runtime_name;
  std::int64_t start_us = 0;
  std::int64_t duration_us = 0;
  GridDim grid;
  std::int64_t registers = 0;
  std::int64_t shared_mem = 0;
};

std::vector<KernelRecord> parse_nsys_sqlite(const std::filesystem::path& path);
std::vector<KernelTraceEvent> parse_nsys_kernel_trace(const std::filesystem::path& path);

}  // namespace hotpath::profiler
