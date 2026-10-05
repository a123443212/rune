#pragma once
#include <cstdint>
#include <string>
namespace rune {
namespace eng {
struct EvalOutput {
  float value = 0.0f;
  float wdl[3] = {0.0f, 1.0f, 0.0f};
  float uncertainty = 0.0f;
  bool refine = false;
};
float canonicalValue(float raw);
int engineScore(float value);
void normalizeWdl(float w[3]);
float wdlValue(const float w[3]);
bool valueWdlConsistent(float value, const float w[3], float tol);
bool isExtreme(float value, float bound);
std::string evalCacheKey(const std::string& fen, const std::string& modelHash, const std::string& mode);
}
}
