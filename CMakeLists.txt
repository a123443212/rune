cmake_minimum_required(VERSION 3.20)
project(rune VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RUNE_BUILD_TESTS "Build C++ tests" ON)
option(RUNE_BUILD_BENCH "Build C++ benchmarks" ON)
option(RUNE_BUILD_BINDINGS "Build pybind11 module" ON)
option(RUNE_ENABLE_AVX2 "Enable AVX2 paths where available" OFF)

add_compile_options(-O2 -Wall -Wextra)

include_directories(${CMAKE_SOURCE_DIR})

add_library(rune_core STATIC
  core/board/board_state.cpp
  core/board/board_attacks.cpp
  core/board/move_gen.cpp
  core/board/move_make.cpp
  core/position/position.cpp
  core/features/feature_set.cpp
  core/accumulators/grouped_accumulator.cpp
  core/accumulators/token_layout.cpp
  core/accumulators/flex_accumulator.cpp
  core/architectures/base/architecture.cpp
  core/architectures/mlp/mlp.cpp
  core/architectures/attention/attention.cpp
  core/architectures/relational/context.cpp
  core/architectures/relational/relational.cpp
  core/inference/evaluator.cpp
  core/model_io/model_io.cpp
  core/model_io/model_save.cpp
  core/model_io/model_factory.cpp
)

target_include_directories(rune_core PUBLIC ${CMAKE_SOURCE_DIR})
set_target_properties(rune_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(RUNE_BUILD_TESTS)
  enable_testing()
  add_executable(rune_tests
    tests/cpp/main.cpp
    tests/cpp/test_board.cpp
    tests/cpp/test_features.cpp
    tests/cpp/test_arch.cpp
    tests/cpp/test_model.cpp
    tests/cpp/test_relational.cpp
  )
  target_link_libraries(rune_tests PRIVATE rune_core)
  add_test(NAME rune_cpp_tests COMMAND rune_tests)
endif()

if(RUNE_BUILD_BENCH)
  add_executable(rune_bench benchmarks/cpp_bench.cpp)
  target_link_libraries(rune_bench PRIVATE rune_core)
  add_executable(rune_stage_bench benchmarks/stage_bench.cpp)
  target_link_libraries(rune_stage_bench PRIVATE rune_core)
endif()

if(RUNE_BUILD_BINDINGS)
  execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(pybind11_DIR)
    find_package(pybind11 CONFIG QUIET)
  endif()
  if(pybind11_FOUND)
    pybind11_add_module(rune_bindings bindings/rune_bindings.cpp bindings/relational_bindings.cpp)
    target_link_libraries(rune_bindings PRIVATE rune_core)
  else()
    message(STATUS "pybind11 not found, skipping bindings target")
  endif()
endif()
