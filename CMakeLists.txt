cmake_minimum_required(VERSION 3.28)

project(hotpath VERSION 0.1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SQLite3 REQUIRED)

add_library(
  hotpath_core
  src/bench/kernels/rms_norm.cpp
  src/bench/kernels/rotary_emb.cpp
  src/bench/kernels/silu_mul.cpp
  src/bench/registry.cpp
  src/bench/runner.cpp
  src/aggregate.cpp
  src/artifacts.cpp
  src/clock_control.cpp
  src/diff.cpp
  src/doctor.cpp
  src/export.cpp
  src/ops.cpp
  src/remote.cpp
  src/stability.cpp
  src/targets.cpp
  src/traffic.cpp
  src/validate.cpp
  src/profiler/categorizer.cpp
  src/profiler/attach.cpp
  src/profiler/parser.cpp
  src/profiler/runner.cpp
  src/profiler/server.cpp
  src/profiler/vllm_metrics.cpp
  src/report.cpp
  src/store.cpp
  src/serving/log_parser.cpp
  src/serving/phase_analyzer.cpp
  src/serving/batch_analyzer.cpp
  src/serving/cache_analyzer.cpp
  src/serving/prefix_analyzer.cpp
  src/advisor/workload_classifier.cpp
  src/advisor/disagg_model.cpp
  src/advisor/recommender.cpp
  src/advisor/kv_config.cpp
  src/serving/serve_profiler.cpp
  src/serving/traffic_replayer.cpp
  src/profiler/sglang_metrics.cpp
  src/export/otlp_export.cpp
)

target_include_directories(
  hotpath_core
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(
  hotpath_core
  PUBLIC
    SQLite::SQLite3
)

add_library(
  hotpath_interactive
  src/interactive.cpp
)

target_include_directories(
  hotpath_interactive
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(
  hotpath_interactive
  PUBLIC
    hotpath_core
)

include(CTest)

if(BUILD_TESTING)
  add_executable(
    test_categorizer
    cpp_tests/test_categorizer.cpp
  )
  target_link_libraries(test_categorizer PRIVATE hotpath_core)
  add_test(NAME test_categorizer COMMAND test_categorizer)

  add_executable(
    test_attach
    cpp_tests/test_attach.cpp
  )
  target_link_libraries(test_attach PRIVATE hotpath_core)
  add_test(NAME test_attach COMMAND test_attach)

  add_executable(
    test_parser
    cpp_tests/test_parser.cpp
  )
  target_link_libraries(test_parser PRIVATE hotpath_core)
  add_test(NAME test_parser COMMAND test_parser)

  add_executable(
    test_report
    cpp_tests/test_report.cpp
  )
  target_link_libraries(test_report PRIVATE hotpath_core)
  add_test(NAME test_report COMMAND test_report)

  add_executable(
    test_store
    cpp_tests/test_store.cpp
  )
  target_link_libraries(test_store PRIVATE hotpath_core)
  add_test(NAME test_store COMMAND test_store)

  add_executable(
    test_vllm_metrics
    cpp_tests/test_vllm_metrics.cpp
  )
  target_link_libraries(test_vllm_metrics PRIVATE hotpath_core)
  add_test(NAME test_vllm_metrics COMMAND test_vllm_metrics)

  add_executable(
    test_diff
    cpp_tests/test_diff.cpp
  )
  target_link_libraries(test_diff PRIVATE hotpath_core)
  add_test(NAME test_diff COMMAND test_diff)

  add_executable(
    test_bench
    cpp_tests/test_bench.cpp
  )
  target_link_libraries(test_bench PRIVATE hotpath_core)
  add_test(NAME test_bench COMMAND test_bench)

  add_executable(
    test_bench_json
    cpp_tests/test_bench_json.cpp
  )
  target_link_libraries(test_bench_json PRIVATE hotpath_core)
  add_test(NAME test_bench_json COMMAND test_bench_json)

  add_executable(
    test_stability
    cpp_tests/test_stability.cpp
  )
  target_link_libraries(test_stability PRIVATE hotpath_core)
  add_test(NAME test_stability COMMAND test_stability)

  add_executable(
    test_clock_control
    cpp_tests/test_clock_control.cpp
  )
  target_link_libraries(test_clock_control PRIVATE hotpath_core)
  add_test(NAME test_clock_control COMMAND test_clock_control)

  add_executable(
    test_export
    cpp_tests/test_export.cpp
  )
  target_link_libraries(test_export PRIVATE hotpath_core)
  add_test(NAME test_export COMMAND test_export)

  add_executable(
    test_aggregate
    cpp_tests/test_aggregate.cpp
  )
  target_link_libraries(test_aggregate PRIVATE hotpath_core)
  add_test(NAME test_aggregate COMMAND test_aggregate)

  add_executable(
    test_artifacts
    cpp_tests/test_artifacts.cpp
  )
  target_link_libraries(test_artifacts PRIVATE hotpath_core)
  add_test(NAME test_artifacts COMMAND test_artifacts)

  add_executable(
    test_doctor
    cpp_tests/test_doctor.cpp
  )
  target_link_libraries(test_doctor PRIVATE hotpath_core)
  add_test(NAME test_doctor COMMAND test_doctor)

  add_executable(
    test_validate
    cpp_tests/test_validate.cpp
  )
  target_link_libraries(test_validate PRIVATE hotpath_core)
  add_test(NAME test_validate COMMAND test_validate)

  add_executable(
    test_traffic
    cpp_tests/test_traffic.cpp
  )
  target_link_libraries(test_traffic PRIVATE hotpath_core)
  add_test(NAME test_traffic COMMAND test_traffic)

  add_executable(
    test_remote
    cpp_tests/test_remote.cpp
  )
  target_link_libraries(test_remote PRIVATE hotpath_core)
  add_test(NAME test_remote COMMAND test_remote)

  add_executable(
    test_targets
    cpp_tests/test_targets.cpp
  )
  target_link_libraries(test_targets PRIVATE hotpath_core)
  add_test(NAME test_targets COMMAND test_targets)

  add_executable(
    test_server
    cpp_tests/test_server.cpp
  )
  target_link_libraries(test_server PRIVATE hotpath_core)
  add_test(NAME test_server COMMAND test_server)

  add_executable(
    test_request_trace
    cpp_tests/test_request_trace.cpp
  )
  target_link_libraries(test_request_trace PRIVATE hotpath_core)
  add_test(NAME test_request_trace COMMAND test_request_trace)

  add_executable(
    test_log_parser
    cpp_tests/test_log_parser.cpp
  )
  target_link_libraries(test_log_parser PRIVATE hotpath_core)
  add_test(NAME test_log_parser COMMAND test_log_parser)

  add_executable(
    test_phase_analyzer
    cpp_tests/test_phase_analyzer.cpp
  )
  target_link_libraries(test_phase_analyzer PRIVATE hotpath_core)
  add_test(NAME test_phase_analyzer COMMAND test_phase_analyzer)

  add_executable(
    test_batch_analyzer
    cpp_tests/test_batch_analyzer.cpp
  )
  target_link_libraries(test_batch_analyzer PRIVATE hotpath_core)
  add_test(NAME test_batch_analyzer COMMAND test_batch_analyzer)

  add_executable(
    test_cache_analyzer
    cpp_tests/test_cache_analyzer.cpp
  )
  target_link_libraries(test_cache_analyzer PRIVATE hotpath_core)
  add_test(NAME test_cache_analyzer COMMAND test_cache_analyzer)

  add_executable(
    test_prefix_analyzer
    cpp_tests/test_prefix_analyzer.cpp
  )
  target_link_libraries(test_prefix_analyzer PRIVATE hotpath_core)
  add_test(NAME test_prefix_analyzer COMMAND test_prefix_analyzer)

  add_executable(
    test_workload_classifier
    cpp_tests/test_workload_classifier.cpp
  )
  target_link_libraries(test_workload_classifier PRIVATE hotpath_core)
  add_test(NAME test_workload_classifier COMMAND test_workload_classifier)

  add_executable(
    test_disagg_model
    cpp_tests/test_disagg_model.cpp
  )
  target_link_libraries(test_disagg_model PRIVATE hotpath_core)
  add_test(NAME test_disagg_model COMMAND test_disagg_model)

  add_executable(
    test_recommender
    cpp_tests/test_recommender.cpp
  )
  target_link_libraries(test_recommender PRIVATE hotpath_core)
  add_test(NAME test_recommender COMMAND test_recommender)

  add_executable(
    test_serve_report
    cpp_tests/test_serve_report.cpp
  )
  target_link_libraries(test_serve_report PRIVATE hotpath_core)
  add_test(NAME test_serve_report COMMAND test_serve_report)

  add_executable(
    test_traffic_replayer
    cpp_tests/test_traffic_replayer.cpp
  )
  target_link_libraries(test_traffic_replayer PRIVATE hotpath_core)
  add_test(NAME test_traffic_replayer COMMAND test_traffic_replayer)

  add_executable(
    test_sglang_metrics
    cpp_tests/test_sglang_metrics.cpp
  )
  target_link_libraries(test_sglang_metrics PRIVATE hotpath_core)
  add_test(NAME test_sglang_metrics COMMAND test_sglang_metrics)

  add_executable(
    test_otlp_export
    cpp_tests/test_otlp_export.cpp
  )
  target_link_libraries(test_otlp_export PRIVATE hotpath_core)
  add_test(NAME test_otlp_export COMMAND test_otlp_export)

  add_executable(
    test_audit
    cpp_tests/test_audit.cpp
  )
  target_link_libraries(test_audit PRIVATE hotpath_core)
  add_test(NAME test_audit COMMAND test_audit)

  add_executable(
    test_interactive
    cpp_tests/test_interactive.cpp
  )
  target_link_libraries(test_interactive PRIVATE hotpath_interactive)
  add_test(NAME test_interactive COMMAND test_interactive)

  add_executable(
    test_cli
    cpp_tests/test_cli.cpp
  )
  add_dependencies(test_cli hotpath)
  add_test(NAME test_cli COMMAND test_cli)
endif()

add_executable(
  hotpath
  src/main.cpp
)
target_include_directories(
  hotpath
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(hotpath PRIVATE hotpath_interactive)

include(GNUInstallDirs)

install(
  TARGETS hotpath
  RUNTIME DESTINATION hotpath_py/bin
)
