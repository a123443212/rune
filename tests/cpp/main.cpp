/*
RUNE — Relational Unified Neural Evaluator
Copyright (C) 2026 a123443212

SPDX-License-Identifier: MIT OR Apache-2.0

This project is dual-licensed under the MIT License and the
Apache License, Version 2.0. You may choose either license
when using, copying, modifying, or distributing this software.

MIT License: https://opensource.org/license/mit
Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, this
software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
*/

#include "tests/cpp/test_framework.h"
#include "tests/cpp/test_multihead.h"
#include "tests/cpp/test_v10.h"
#include "tests/cpp/test_v11.h"
#include "tests/cpp/test_v12.h"
#include "tests/cpp/test_v13.h"

static int g_failures = 0;

int runeTestFailures() { return g_failures; }

void runeTestFail(const char* file, int line, const char* cond) {
  std::printf("FAIL %s:%d: %s\n", file, line, cond);
  ++g_failures;
}

void runeTestFailClose(const char* file, int line, double a, double b, double tol) {
  std::printf("FAIL %s:%d: |%f - %f| = %f > %f\n", file, line, a, b, std::fabs(a - b), tol);
  ++g_failures;
}

int main() {
  testBoard();
  testMakeUnmakeConsistency();
  testKingTwoSquareOffRank();
  testEnPassantUnmake();
  testHashKeyEp();
  testFeatures();
  testAccumulatorIncremental();
  testAttentionMath();
  testSerialization();
  testPairHead();
  testShogi();
  testShogiLegal();
  testXiangqi();
  testXiangqiLegal();
  testGo();
  testGoV02();
  testQuantization();
  testEvaluator();
  testModelIO();
  testRelationalMath();
  testDynamicBiasBounded();
  testContextValues();
  testTokenLayouts();
  testFlexModelIO();
  testInt16FixedPath();
  testVarWidths();
  testVarAccumulatorIncremental();
  testPoolGateMath();
  testDenseModelIO();
  testDenseParamAccounting();
  testDenseStudentWidths();
  testAdaptiveConfigure();
  testAdaptiveRoutingModes();
  testAdaptiveHysteresis();
  testRouteSearchHigh();
  testAdaptiveIncremental();
  testAdaptiveModelIO();
  testAdaptiveParamAccounting();
  testAdaptiveStudentWidths();
  testUncertaintyBounds();
  testSearchRoutingModes();
  testUncertaintyModelIO();
  testUncertaintyParamAccounting();
  runV10Tests();
  runMultiHeadTests();
  runV11Tests();
  runV12Tests();
  runV13Tests();
  if (g_failures == 0) {
    std::printf("ALL CPP TESTS PASSED\n");
    return 0;
  }
  std::printf("%d FAILURES\n", g_failures);
  return 1;
}
