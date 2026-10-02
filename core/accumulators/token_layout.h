#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rune {

struct TokenSource {
  uint8_t group = 0;
  uint16_t lo = 0;
  uint16_t hi = 0;
};

struct TokenLayout {
  int tokens = 8;
  int dim = 32;
  std::vector<std::vector<TokenSource>> sources;

  static bool make(int tokens, int dim, TokenLayout& out, std::string& err);
  int findToken(uint8_t group, uint16_t index) const;
};

}
