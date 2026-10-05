#include <cstdio>
#include <string>
#include "core/compiler/artifact.h"
#include "core/model_io/model_io.h"
int main(int argc, char** argv) {
  std::string model;
  for (int i = 1; i + 1 < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model") model = argv[i + 1];
  }
  if (model.empty()) {
    std::printf("usage: rune_model_info --model <path>\n");
    return 1;
  }
  rune::RuneFile rf;
  std::string err;
  if (!rune::loadRuneFile(model, rf, err)) {
    std::printf("load FAILED %s\n", err.c_str());
    std::string header;
    rune::vart::CompiledInfo info;
    std::string e2;
    if (rune::vart::readCompiledHeader(model, header, info, e2)) {
      std::printf("compiled isa %s plan %s kernels %d\n", info.targetIsa.c_str(), info.planHash.c_str(), info.kernelCount);
    } else {
      std::printf("artifact check %s\n", e2.c_str());
    }
    return 2;
  }
  std::printf("arch %s\n", rf.spec.arch.c_str());
  std::printf("tokens %d dim %d\n", rf.spec.tokens, rf.spec.tokenDim);
  std::printf("quant %s\n", rf.spec.quantization.c_str());
  std::string header;
  rune::vart::CompiledInfo info;
  std::string e2;
  if (rune::vart::readCompiledHeader(model, header, info, e2)) {
    std::printf("compiled true isa %s compiler %s plan %s kernels %d\n", info.targetIsa.c_str(), info.compilerVersion.c_str(), info.planHash.c_str(), info.kernelCount);
  } else {
    std::printf("compiled false generic artifact\n");
  }
  return 0;
}
