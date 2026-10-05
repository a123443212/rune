#pragma once

#include <string>
#include <utility>
#include <vector>

namespace rune {
namespace v13 {

extern const char* kGraphVersion;
extern const char* kInvalidationVersion;

struct EdgeMeta {
  int source = 0;
  int target = 0;
};

class InteractionGraph {
 public:
  explicit InteractionGraph(int tokens);
  InteractionGraph(int tokens, const std::vector<std::pair<int, int>>& edges);

  int tokens() const { return tokens_; }
  int numEdges() const { return static_cast<int>(edges_.size()); }
  bool isDense() const { return static_cast<int>(edges_.size()) == tokens_ * tokens_; }
  const char* version() const { return kGraphVersion; }

  std::vector<std::pair<int, int>> affectedEdges(const std::vector<int>& changed) const;
  std::vector<int> affectedRows(const std::vector<int>& changed) const;
  EdgeMeta edgeMeta(int a, int b) const;
  InteractionGraph pruned(const std::vector<std::pair<int, int>>& keep) const;

 private:
  int tokens_ = 8;
  std::vector<std::pair<int, int>> edges_;
};

InteractionGraph buildDenseGraph(int tokens);

}
}
