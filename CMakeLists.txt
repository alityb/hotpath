cmake_minimum_required(VERSION 3.28)

project(rlprof VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SQLite3 REQUIRED)

add_library(
  rlprof_core
  src/bench/kernels/rms_norm.cpp
  src/bench/kernels/rotary_emb.cpp
  src/bench/kernels/silu_mul.cpp
  src/bench/registry.cpp
  src/bench/runner.cpp
  src/clock_control.cpp
  src/diff.cpp
  src/export.cpp
  src/stability.cpp
  src/traffic.cpp
  src/profiler/categorizer.cpp
  src/profiler/parser.cpp
  src/profiler/runner.cpp
  src/profiler/vllm_metrics.cpp
  src/report.cpp
  src/store.cpp
)

target_include_directories(
  rlprof_core
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(
  rlprof_core
  PUBLIC
    SQLite::SQLite3
)

include(CTest)

if(BUILD_TESTING)
  add_executable(
    test_categorizer
    cpp_tests/test_categorizer.cpp
  )
  target_link_libraries(test_categorizer PRIVATE rlprof_core)
  add_test(NAME test_categorizer COMMAND test_categorizer)

  add_executable(
    test_parser
    cpp_tests/test_parser.cpp
  )
  target_link_libraries(test_parser PRIVATE rlprof_core)
  add_test(NAME test_parser COMMAND test_parser)

  add_executable(
    test_report
    cpp_tests/test_report.cpp
  )
  target_link_libraries(test_report PRIVATE rlprof_core)
  add_test(NAME test_report COMMAND test_report)

  add_executable(
    test_store
    cpp_tests/test_store.cpp
  )
  target_link_libraries(test_store PRIVATE rlprof_core)
  add_test(NAME test_store COMMAND test_store)

  add_executable(
    test_vllm_metrics
    cpp_tests/test_vllm_metrics.cpp
  )
  target_link_libraries(test_vllm_metrics PRIVATE rlprof_core)
  add_test(NAME test_vllm_metrics COMMAND test_vllm_metrics)

  add_executable(
    test_diff
    cpp_tests/test_diff.cpp
  )
  target_link_libraries(test_diff PRIVATE rlprof_core)
  add_test(NAME test_diff COMMAND test_diff)

  add_executable(
    test_bench
    cpp_tests/test_bench.cpp
  )
  target_link_libraries(test_bench PRIVATE rlprof_core)
  add_test(NAME test_bench COMMAND test_bench)

  add_executable(
    test_bench_json
    cpp_tests/test_bench_json.cpp
  )
  target_link_libraries(test_bench_json PRIVATE rlprof_core)
  add_test(NAME test_bench_json COMMAND test_bench_json)

  add_executable(
    test_stability
    cpp_tests/test_stability.cpp
  )
  target_link_libraries(test_stability PRIVATE rlprof_core)
  add_test(NAME test_stability COMMAND test_stability)

  add_executable(
    test_clock_control
    cpp_tests/test_clock_control.cpp
  )
  target_link_libraries(test_clock_control PRIVATE rlprof_core)
  add_test(NAME test_clock_control COMMAND test_clock_control)

  add_executable(
    test_export
    cpp_tests/test_export.cpp
  )
  target_link_libraries(test_export PRIVATE rlprof_core)
  add_test(NAME test_export COMMAND test_export)
endif()

add_executable(
  rlprof
  src/main.cpp
)
target_link_libraries(rlprof PRIVATE rlprof_core)
