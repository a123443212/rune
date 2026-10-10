#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/go/go_board.h"

namespace rune {
namespace go {

struct GoState {
  GoBoard board;
  bool hasKo = false;
  int ko = -1;
  float komi = 7.5f;
  uint32_t moveNo = 1;
  uint32_t passNo = 0;

  bool setState(const std::string& state);
  std::string encode() const;
  std::string normalize() const;
  void contextV02(float* out) const;
  int phaseV02() const;
};

int goKomiBucket(float komi);
int goMoveBucket(uint32_t moveNo);
int goGamePhase(uint32_t moveNo, int n);
float goClamp01(float v);

}
}
