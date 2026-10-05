#include "tests/cpp/test_framework.h"
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
  testFeatures();
  testAccumulatorIncremental();
  testAttentionMath();
  testSerialization();
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
  testAdaptiveIncremental();
  testAdaptiveModelIO();
  testAdaptiveParamAccounting();
  testAdaptiveStudentWidths();
  testUncertaintyBounds();
  testSearchRoutingModes();
  testUncertaintyModelIO();
  testUncertaintyParamAccounting();
  runV10Tests();
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
