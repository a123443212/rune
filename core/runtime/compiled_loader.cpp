#include "core/runtime/compiled_loader.h"
namespace rune {
namespace rt {
bool loadCompiledInfo(const std::string& path, CompiledModel& out, std::string& err) {
  return vart::readCompiledHeader(path, out.header, out.info, err);
}
}
}
