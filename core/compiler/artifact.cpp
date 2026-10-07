#include "core/compiler/artifact.h"
#include <fstream>
#include "core/kernels/simd_kernels.h"
namespace rune {
namespace vart {
namespace {
std::string pullStr(const std::string& h, const std::string& key) {
  std::string pat = "\"" + key + "\":\"";
  size_t p = h.find(pat);
  if (p == std::string::npos) return "";
  p += pat.size();
  size_t q = h.find('"', p);
  if (q == std::string::npos) return "";
  return h.substr(p, q - p);
}
long pullInt(const std::string& h, const std::string& key, long fb) {
  std::string pat = "\"" + key + "\":";
  size_t p = h.find(pat);
  if (p == std::string::npos) return fb;
  p += pat.size();
  size_t q = h.find_first_of(",}", p);
  if (q == std::string::npos) return fb;
  try {
    return std::stol(h.substr(p, q - p));
  } catch (...) {
    return fb;
  }
}
}
uint64_t fnv1a64Bytes(const uint8_t* data, size_t n) {
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n; ++i) {
    h ^= data[i];
    h *= 1099511628211ULL;
  }
  return h;
}
bool checkIsaSupported(const std::string& isa, std::string& err) {
  if (isa != "portable" && isa != "avx2" && isa != "avx512") {
    err = "unsupported isa";
    return false;
  }
  if (isa == "avx2" && !kern::hasAvx2()) {
    err = "avx2 artifact on non-avx2 cpu";
    return false;
  }
  if (isa == "avx512" && !kern::hasAvx512()) {
    err = "avx512 artifact on non-avx512 cpu";
    return false;
  }
  return true;
}
bool readCompiledHeader(const std::string& path, std::string& headerOut, CompiledInfo& info, std::string& err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    err = "cannot open file";
    return false;
  }
  char magic[4];
  f.read(magic, 4);
  if (f.gcount() != 4 || magic[0] != 'R' || magic[1] != 'U' || magic[2] != 'N' || magic[3] != 'E') {
    err = "bad magic";
    return false;
  }
  uint32_t hlen = 0;
  f.read(reinterpret_cast<char*>(&hlen), 4);
  if (hlen > 1000000) {
    err = "header too large";
    return false;
  }
  headerOut.assign(hlen, '\0');
  f.read(headerOut.data(), hlen);
  if (!f) {
    err = "truncated header";
    return false;
  }
  std::string comp = headerOut.find("\"compiled\":true") != std::string::npos ? "1" : "";
  if (comp.empty()) {
    err = "not a compiled artifact";
    return false;
  }
  info.compiled = true;
  info.irVersion = pullStr(headerOut, "rune_ir_version");
  info.compilerVersion = pullStr(headerOut, "compiler_version");
  info.targetIsa = pullStr(headerOut, "target_isa");
  info.targetCpu = pullStr(headerOut, "target_cpu");
  info.sourceHash = pullStr(headerOut, "source_hash");
  info.planHash = pullStr(headerOut, "kernel_plan_hash");
  info.modelHash = pullStr(headerOut, "model_hash");
  info.arenaBytes = (size_t)pullInt(headerOut, "arena_bytes", 0);
  size_t p = 0;
  int n = 0;
  while (true) {
    size_t q = headerOut.find("\"kernel_id\"", p);
    if (q == std::string::npos) break;
    ++n;
    p = q + 1;
  }
  info.kernelCount = n;
  if (info.irVersion != "1.0") {
    err = "ir version mismatch";
    return false;
  }
  if (!checkIsaSupported(info.targetIsa, err)) return false;
  if (n == 0) {
    err = "missing kernel plan";
    return false;
  }
  return true;
}
std::string cacheKeyFor(const std::string& src, const std::string& arch, const std::string& quant, const std::string& isa) {
  std::string raw = src + "|" + arch + "|" + quant + "|" + isa + "|0.11.0";
  uint64_t h = fnv1a64Bytes(reinterpret_cast<const uint8_t*>(raw.data()), raw.size());
  char b[32];
  snprintf(b, sizeof(b), "%016llx", (unsigned long long)h);
  return b;
}
}
}
