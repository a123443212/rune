cmake_minimum_required(VERSION 3.20)
project(rune VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RUNE_BUILD_TESTS "Build C++ tests" ON)
option(RUNE_BUILD_BENCH "Build C++ benchmarks" ON)
option(RUNE_BUILD_BINDINGS "Build pybind11 module" ON)

add_compile_options(-O2 -Wall -Wextra)

include_directories(${CMAKE_SOURCE_DIR})

add_library(rune_core STATIC
  core/board/board_state.cpp
  core/board/board_attacks.cpp
  core/board/move_gen.cpp
  core/board/move_make.cpp
  core/shogi/shogi_board.cpp
  core/shogi/shogi_features.cpp
  core/shogi/shogi_evaluator.cpp
  core/xiangqi/xiangqi_board.cpp
  core/xiangqi/xiangqi_features.cpp
  core/xiangqi/xiangqi_evaluator.cpp
  core/position/position.cpp
  core/features/feature_set.cpp
  core/kernels/ref_kernels.cpp
  core/kernels/simd_kernels.cpp
  core/kernels/smallmat.cpp
  core/kernels/fused.cpp
  core/kernels/quant_specialized.cpp
  core/kernels/accum_specialized.cpp
  core/compiler/ir.cpp
  core/compiler/plan.cpp
  core/compiler/artifact.cpp
  core/runtime/arena.cpp
  core/runtime/compiled_loader.cpp
  core/runtime/compiled_evaluator.cpp
  core/engine/eval_contract.cpp
  core/engine/eval_cache.cpp
  core/engine/search.cpp
  core/accumulators/grouped_accumulator.cpp
  core/incremental/token_delta.cpp
  core/incremental/interaction_graph.cpp
  core/incremental/relational_cache.cpp
  core/incremental/incremental_mixer.cpp
  core/accumulators/token_layout.cpp
  core/accumulators/flex_accumulator.cpp
  core/architectures/base/architecture.cpp
  core/architectures/mlp/mlp.cpp
  core/architectures/attention/attention.cpp
  core/architectures/attention/multi_head.cpp
  core/architectures/relational/context.cpp
  core/architectures/relational/relational.cpp
  core/architectures/dense/var_accum.cpp
  core/architectures/dense/pooling.cpp
  core/architectures/dense/dense.cpp
  core/architectures/adaptive/adaptive.cpp
  core/inference/evaluator.cpp
  core/inference/dump.cpp
  core/model_io/model_io.cpp
  core/model_io/model_save.cpp
  core/model_io/model_factory.cpp
  core/model_io/model_error.cpp
)

target_include_directories(rune_core PUBLIC ${CMAKE_SOURCE_DIR})
set_target_properties(rune_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(RUNE_BUILD_TESTS)
  enable_testing()
  add_executable(rune_tests
    tests/cpp/main.cpp
    tests/cpp/test_board.cpp
    tests/cpp/test_features.cpp
    tests/cpp/test_shogi.cpp
    tests/cpp/test_xiangqi.cpp
    tests/cpp/test_arch.cpp
    tests/cpp/test_model.cpp
    tests/cpp/test_relational.cpp
    tests/cpp/test_dense.cpp
    tests/cpp/test_adaptive.cpp
    tests/cpp/test_v10.cpp
    tests/cpp/test_multihead.cpp
    tests/cpp/test_v11.cpp
    tests/cpp/test_v12.cpp
    tests/cpp/test_v13.cpp
  )
  target_link_libraries(rune_tests PRIVATE rune_core)
  add_test(NAME rune_cpp_tests COMMAND rune_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

if(RUNE_BUILD_BENCH)
  add_executable(rune_bench benchmarks/cpp_bench.cpp)
  target_link_libraries(rune_bench PRIVATE rune_core)
  add_executable(rune_stage_bench benchmarks/stage_bench.cpp)
  target_link_libraries(rune_stage_bench PRIVATE rune_core)
  add_executable(rune_diff tools/diff/rune_diff.cpp)
  target_link_libraries(rune_diff PRIVATE rune_core)
  add_executable(rune_bench_v10 tools/bench/rune_bench.cpp)
  target_link_libraries(rune_bench_v10 PRIVATE rune_core)
  add_executable(rune_eval tools/eval/rune_eval.cpp)
  target_link_libraries(rune_eval PRIVATE rune_core)
  add_executable(rune_compile_support tools/compile/rune_compile_support.cpp)
  target_link_libraries(rune_compile_support PRIVATE rune_core)
  add_executable(rune_bench_compile tools/bench/rune_bench_compile.cpp)
  target_link_libraries(rune_bench_compile PRIVATE rune_core)
  add_executable(rune_model_info tools/inspect/rune_model_info.cpp)
  target_link_libraries(rune_model_info PRIVATE rune_core)
  add_executable(rune_search tools/search/rune_search.cpp)
  target_link_libraries(rune_search PRIVATE rune_core)
  add_executable(rune_search_diff tools/search/rune_search_diff.cpp)
  target_link_libraries(rune_search_diff PRIVATE rune_core)
  add_executable(rune_bench_v13 tools/bench/rune_bench_v13.cpp)
  target_link_libraries(rune_bench_v13 PRIVATE rune_core)
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
    pybind11_add_module(rune_bindings bindings/rune_bindings.cpp bindings/relational_bindings.cpp bindings/dense_bindings.cpp bindings/adaptive_bindings.cpp)
    target_link_libraries(rune_bindings PRIVATE rune_core)
  else()
    message(STATUS "pybind11 not found, skipping bindings target")
  endif()
endif()
