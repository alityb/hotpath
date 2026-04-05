#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "hotpath/batch_analyzer.h"
#include "hotpath/request_trace.h"

namespace hotpath {

struct CacheAnalysis {
    double cache_hit_rate = 0.0;
    double avg_cache_usage = 0.0;
    double peak_cache_usage = 0.0;
    double cache_pressure_seconds = 0.0;
    int eviction_count = 0;
    bool cache_hit_rate_available = false;
    bool cache_hit_rate_aggregate_only = false;
    bool hit_rate_histogram_available = true;
    // histogram: 0%, 1-25%, 25-50%, 50-75%, 75-100% cache hit per request
    std::array<int, 5> hit_rate_histogram = {};
};

CacheAnalysis analyze_cache(const std::vector<RequestTrace>& traces,
                            const std::vector<MetricSnapshot>& snapshots,
                            std::optional<double> aggregate_cache_hit_rate = std::nullopt);

}  // namespace hotpath
