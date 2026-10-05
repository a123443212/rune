#pragma once
#include <string>
#include <vector>
#include "core/compiler/ir.h"
namespace rune {
namespace vplan {
void selectKernels(vir::RuneIr& ir);
size_t planArenaBytes(int tokens, int dim, int h1, int h2);
double kernelCost(const std::string& shape, const std::string& kernelId, const std::string& isa);
std::string fusionOf(const std::string& kind);
}
}
