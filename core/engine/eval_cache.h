#pragma once
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>
namespace rune {
namespace eng {
class EvalCache {
 public:
  EvalCache(const std::string& modelHash, const std::string& mode, size_t capacity);
  bool get(const std::string& fen, float& value) const;
  void put(const std::string& fen, float value);
  double hitRate() const;
  size_t size() const;
 private:
  std::string model_;
  std::string mode_;
  size_t cap_;
  mutable size_t hits_ = 0;
  mutable size_t misses_ = 0;
  std::unordered_map<std::string, float> map_;
  std::vector<std::string> order_;
};
}
}
