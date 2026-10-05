#include "core/runtime/compiled_evaluator.h"
#include "core/kernels/accum_specialized.h"
#include "core/kernels/fused.h"
#include "core/kernels/smallmat.h"
namespace rune {
namespace rt {
CompiledEvaluator::CompiledEvaluator(const EmbeddingTables* tables, const IArchitecture* arch, const std::string& isa, size_t arenaBytes) : arch_(arch), isa_(isa), arena_(arenaBytes == 0 ? 8192 : arenaBytes) {
  acc_.bind(tables);
  tok_.assign(256, 0.0f);
}
void CompiledEvaluator::refresh(const Board& board) {
  std::vector<ActiveFeature> feats;
  GroupedFeatureSet::extract(board, feats);
  aspec::GroupOffsets g = aspec::offsetsFor(32);
  float tmp[256];
  for (int i = 0; i < 256; ++i) tmp[i] = 0.0f;
  acc_.refresh(feats);
}
void CompiledEvaluator::updateIncremental(const std::vector<ActiveFeature>& added, const std::vector<ActiveFeature>& removed) {
  acc_.applyDiff(added, removed);
}
EvalResult CompiledEvaluator::evaluate() {
  float* tok = arena_.at(0, 256);
  acc_.tokens(tok);
  EvalResult r;
  arch_->forward(tok, r.value, r.wdl);
  return r;
}
EvalResult CompiledEvaluator::evaluateBoard(const Board& board) {
  refresh(board);
  return evaluate();
}
const std::string& CompiledEvaluator::targetIsa() const {
  return isa_;
}
size_t CompiledEvaluator::arenaBytes() const {
  return arena_.bytes();
}
}
}
