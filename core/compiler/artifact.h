#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace rune {
namespace vart {
struct CompiledInfo {
  bool compiled = false;
  std::string irVersion;
  std::string compilerVersion;
  std::string targetIsa;
  std::string targetCpu;
  std::string sourceHash;
  std::string planHash;
  std::string modelHash;
  size_t arenaBytes = 0;
  int kernelCount = 0;
};
bool readCompiledHeader(const std::string& path, std::string& headerOut, CompiledInfo& info, std::string& err);
bool checkIsaSupported(const std::string& isa, std::string& err);
uint64_t fnv1a64Bytes(const uint8_t* data, size_t n);
std::string cacheKeyFor(const std::string& src, const std::string& arch, const std::string& quant, const std::string& isa);
}
}
