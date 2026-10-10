#pragma once

#include <vector>

#include "core/go/go_state.h"

namespace rune {
namespace go {

constexpr int kPlanesV02 = 8;

void extractPlanesV02(const GoState& st, std::vector<float>& out);
void extractPlanesV02Into(const GoState& st, float* out);

}
}
