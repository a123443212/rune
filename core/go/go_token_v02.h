#pragma once

#include <vector>

#include "core/features/feature_set.h"
#include "core/go/go_state.h"

namespace rune {
namespace go {

class GoTokenV02 {
 public:
  static const char* version();
  static void extract(const GoState& st, std::vector<ActiveFeature>& out);
  static int phaseFromGroup7(const std::vector<ActiveFeature>& feats);
};

}
}
