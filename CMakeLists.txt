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

add_library(
  rlprof_interactive
  src/interactive.cpp
)

target_include_directories(
  rlprof_interactive
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(
  rlprof_interactive
  PUBLIC
    rlprof_core
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
    test_attach
    cpp_tests/test_attach.cpp
  )
  target_link_libraries(test_attach PRIVATE rlprof_core)
  add_test(NAME test_attach COMMAND test_attach)

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

  add_executable(
    test_aggregate
    cpp_tests/test_aggregate.cpp
  )
  target_link_libraries(test_aggregate PRIVATE rlprof_core)
  add_test(NAME test_aggregate COMMAND test_aggregate)

  add_executable(
    test_artifacts
    cpp_tests/test_artifacts.cpp
  )
  target_link_libraries(test_artifacts PRIVATE rlprof_core)
  add_test(NAME test_artifacts COMMAND test_artifacts)

  add_executable(
    test_doctor
    cpp_tests/test_doctor.cpp
  )
  target_link_libraries(test_doctor PRIVATE rlprof_core)
  add_test(NAME test_doctor COMMAND test_doctor)

  add_executable(
    test_validate
    cpp_tests/test_validate.cpp
  )
  target_link_libraries(test_validate PRIVATE rlprof_core)
  add_test(NAME test_validate COMMAND test_validate)

  add_executable(
    test_traffic
    cpp_tests/test_traffic.cpp
  )
  target_link_libraries(test_traffic PRIVATE rlprof_core)
  add_test(NAME test_traffic COMMAND test_traffic)

  add_executable(
    test_remote
    cpp_tests/test_remote.cpp
  )
  target_link_libraries(test_remote PRIVATE rlprof_core)
  add_test(NAME test_remote COMMAND test_remote)

  add_executable(
    test_targets
    cpp_tests/test_targets.cpp
  )
  target_link_libraries(test_targets PRIVATE rlprof_core)
  add_test(NAME test_targets COMMAND test_targets)

  add_executable(
    test_server
    cpp_tests/test_server.cpp
  )
  target_link_libraries(test_server PRIVATE rlprof_core)
  add_test(NAME test_server COMMAND test_server)

  add_executable(
    test_interactive
    cpp_tests/test_interactive.cpp
  )
  target_link_libraries(test_interactive PRIVATE rlprof_interactive)
  add_test(NAME test_interactive COMMAND test_interactive)

  add_executable(
    test_cli
    cpp_tests/test_cli.cpp
  )
  add_dependencies(test_cli rlprof)
  add_test(NAME test_cli COMMAND test_cli)
endif()

add_executable(
  rlprof
  src/main.cpp
)
target_include_directories(
  rlprof
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(rlprof PRIVATE rlprof_interactive)

include(GNUInstallDirs)

install(
  TARGETS rlprof
  RUNTIME DESTINATION rlprof_py/bin
)
