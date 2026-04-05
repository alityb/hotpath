#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hotpath {

struct RequestEvent {
    std::string event_type;
    int64_t timestamp_us = 0;
    std::string detail;  // JSON
};

struct RequestTrace {
    std::string request_id;
    int64_t arrival_us = 0;
    int64_t queue_start_us = 0;
    int64_t prefill_start_us = 0;
    int64_t prefill_end_us = 0;
    int64_t server_last_token_us = 0;
    int64_t first_token_us = 0;
    int64_t last_token_us = 0;
    int64_t completion_us = 0;
    int prompt_tokens = 0;
    int output_tokens = 0;
    int cached_tokens = 0;
    std::string status;
    bool server_timing_available = false;
    bool prompt_tokens_estimated = false;
    std::string prompt_text;
    std::vector<RequestEvent> events;
};

}  // namespace hotpath
