#include "core/engine/eval_contract.h"
#include <cmath>
namespace rune {
namespace eng {
float canonicalValue(float raw) {
  if (!(raw == raw)) return 0.0f;
  if (raw < -1.0f) return -1.0f;
  if (raw > 1.0f) return 1.0f;
  return raw;
}
int engineScore(float value) {
  float v = canonicalValue(value);
  int s = (int)(v * 1000.0f + (v >= 0 ? 0.5f : -0.5f));
  if (s >= 9000) s = 8999;
  if (s <= -9000) s = -8999;
  return s;
}
void normalizeWdl(float w[3]) {
  float s = w[0] + w[1] + w[2];
  if (s <= 0.0f) {
    w[0] = 0.0f;
    w[1] = 1.0f;
    w[2] = 0.0f;
    return;
  }
  w[0] /= s;
  w[1] /= s;
  w[2] /= s;
}
float wdlValue(const float w[3]) {
  float s = w[0] + w[1] + w[2];
  if (s <= 0.0f) return 0.0f;
  return (w[0] - w[2]) / s;
}
bool valueWdlConsistent(float value, const float w[3], float tol) {
  float d = canonicalValue(value) - wdlValue(w);
  return d < 0 ? -d <= tol : d <= tol;
}
bool isExtreme(float value, float bound) {
  float v = canonicalValue(value);
  return v <= -bound || v >= bound;
}
std::string evalCacheKey(const std::string& fen, const std::string& modelHash, const std::string& mode) {
  size_t sp = fen.find(' ');
  size_t sp2 = std::string::npos;
  size_t sp3 = std::string::npos;
  size_t sp4 = std::string::npos;
  if (sp != std::string::npos) sp2 = fen.find(' ', sp + 1);
  if (sp2 != std::string::npos) sp3 = fen.find(' ', sp2 + 1);
  if (sp3 != std::string::npos) sp4 = fen.find(' ', sp3 + 1);
  std::string core = fen.substr(0, sp4);
  std::string raw = core + "|" + modelHash + "|" + mode;
  uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : raw) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  char b[17];
  snprintf(b, sizeof(b), "%016llx", (unsigned long long)h);
  return b;
}
}
}
