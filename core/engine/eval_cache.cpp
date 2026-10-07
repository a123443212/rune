#include "core/engine/eval_cache.h"
#include "core/engine/eval_contract.h"
namespace rune {
namespace eng {
EvalCache::EvalCache(const std::string& modelHash, const std::string& mode, size_t capacity) : model_(modelHash), mode_(mode), cap_(capacity) {}
bool EvalCache::get(const std::string& fen, float& value) const {
  std::string k = evalCacheKey(fen, model_, mode_);
  auto it = map_.find(k);
  if (it == map_.end()) {
    ++misses_;
    return false;
  }
  ++hits_;
  value = it->second;
  return true;
}
void EvalCache::put(const std::string& fen, float value) {
  std::string k = evalCacheKey(fen, model_, mode_);
  auto it = map_.find(k);
  if (it != map_.end()) {
    it->second = value;
    return;
  }
  if (cap_ == 0) {
    return;
  }
  if (map_.size() >= cap_ && !order_.empty()) {
    map_.erase(order_.front());
    order_.erase(order_.begin());
  }
  map_[k] = value;
  order_.push_back(k);
}
double EvalCache::hitRate() const {
  size_t n = hits_ + misses_;
  return n ? (double)hits_ / (double)n : 0.0;
}
size_t EvalCache::size() const {
  return map_.size();
}
}
}
