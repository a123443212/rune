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

#pragma once

#include <cmath>
#include <cstdio>

int runeTestFailures();
void runeTestFail(const char* file, int line, const char* cond);
void runeTestFailClose(const char* file, int line, double a, double b, double tol);

#define CHECK(cond)                                      \
  do {                                                   \
    if (!(cond)) runeTestFail(__FILE__, __LINE__, #cond); \
  } while (0)

#define CHECK_CLOSE(a, b, tol)                           \
  do {                                                   \
    double _a = static_cast<double>(a);                  \
    double _b = static_cast<double>(b);                  \
    double _d = std::fabs(_a - _b);                      \
    if (_d > (tol)) runeTestFailClose(__FILE__, __LINE__, _a, _b, tol); \
  } while (0)

void testBoard();
void testMakeUnmakeConsistency();
void testKingTwoSquareOffRank();
void testEnPassantUnmake();
void testHashKeyEp();
void testFeatures();
void testAccumulatorIncremental();
void testAttentionMath();
void testSerialization();
void testPairHead();
void testShogi();
void testShogiLegal();
void testXiangqi();
void testXiangqiLegal();
void testGo();
void testGoV02();
void testQuantization();
void testEvaluator();
void testModelIO();
void testRelationalMath();
void testDynamicBiasBounded();
void testContextValues();
void testTokenLayouts();
void testFlexModelIO();
void testInt16FixedPath();
void testVarWidths();
void testVarAccumulatorIncremental();
void testPoolGateMath();
void testDenseModelIO();
void testDenseParamAccounting();
void testAdaptiveConfigure();
void testAdaptiveRoutingModes();
void testAdaptiveHysteresis();
void testRouteSearchHigh();
void testAdaptiveIncremental();
void testAdaptiveModelIO();
void testAdaptiveParamAccounting();
void testUncertaintyBounds();
void testSearchRoutingModes();
void testUncertaintyModelIO();
void testUncertaintyParamAccounting();
void testDenseStudentWidths();
void testAdaptiveStudentWidths();
