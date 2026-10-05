#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "core/features/feature_set.h"
namespace rune {
struct IntermediateDump {
  std::vector<ActiveFeature> features;
  std::vector<float> accumulator;
  std::vector<float> tokens;
  std::vector<float> q;
  std::vector<float> k;
  std::vector<float> v;
  std::vector<float> scores;
  std::vector<float> gates;
  std::vector<float> mixed;
  std::vector<float> h1;
  std::vector<float> h2;
  float value = 0.0f;
  float wdl[3] = {0.0f, 0.0f, 0.0f};
  bool refine = false;
  float difficulty = 0.0f;
  bool writeText(const std::string& path) const;
};
}
