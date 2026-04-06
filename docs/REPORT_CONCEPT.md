# serve-report target output

This is the target terminal output for `hotpath serve-report`.
Reference screenshot: docs/report_concept.png

## Layout (top to bottom)

1. Header: tool name, model, engine version, GPU, data source flags
2. Summary cards: requests, duration, req/s, tok/s
3. Latency table: queue wait, prefill (server), decode (server), decode (per-tok), end-to-end — each with p50/p90/p99
4. GPU phase: horizontal colored bars — prefill (purple), decode (green), schedule (amber), idle (gray)
5. Throughput graph: tok/s over time, ASCII line chart, x-axis = seconds
6. Batch size graph: requests in batch over time, same style
7. KV cache + prefix sharing: side by side key-value pairs
8. Cache hit distribution: horizontal histogram with 5 buckets (0%, 1-25%, 25-50%, 50-75%, 75%+)
9. Disaggregation advisor: workload class, contention level, recommendation badge, current vs projected table, config hint
10. Footer: kernel count, match method, DB path

## Colors (ANSI)

- Purple (prefill bars): \033[35m
- Green (decode bars, positive improvements): \033[32m
- Amber/yellow (schedule bars, warning): \033[33m
- Gray (idle, borders, secondary text): \033[90m
- White/bold (headers, emphasis): \033[1m
- Reset: \033[0m

## ASCII graph style

Use box-drawing characters for axes and unicode line-drawing for curves.
Y-axis labels left-aligned with `┤`. X-axis labels below.
