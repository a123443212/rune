#pragma once

#include <string>

#include "core/go/go_resnet.h"

namespace rune {
namespace go {

struct GoResnetFile {
  GoResnetWeights weights;
  GoResnetSizes sizes;
  std::string featureVersion;
};

bool loadGoResnetFile(const std::string& path, GoResnetFile& out, std::string& err);

}
}
