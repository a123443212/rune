#pragma once
#include <string>
#include "core/compiler/artifact.h"
namespace rune {
namespace rt {
struct CompiledModel {
  vart::CompiledInfo info;
  std::string header;
};
bool loadCompiledInfo(const std::string& path, CompiledModel& out, std::string& err);
}
}
