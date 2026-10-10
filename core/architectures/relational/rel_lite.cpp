#include "core/architectures/relational/rel_lite.h"
#include "core/model_io/model_factory.h"
namespace rune {
RelLiteModel::RelLiteModel() {
  std::string err;
  FlexBuildSpec spec;
  spec.tokens = 6;
  spec.dim = 24;
  spec.gate = "clip";
  spec.alpha = 1.0f;
  spec.dynamicBias = false;
  RelationalModel tmp;
  std::string e2;
  RelationalConfig cfg;
  cfg.tokens = 6;
  cfg.dim = 24;
  cfg.gate = GateFn::Clip;
  cfg.alpha = 1.0f;
  cfg.dynamicBias = false;
  tmp.configure(cfg, e2);
  inner = std::move(tmp);
}
void RelLiteModel::forward(const float* tokens, float& value, float* wdl, int phase) const {
  inner.forward(tokens, value, wdl, phase);
}
size_t RelLiteModel::parameterCount() const {
  return inner.parameterCount();
}
size_t RelLiteModel::modelSizeBytes() const {
  return inner.modelSizeBytes();
}
void RelLiteModel::getTensors(std::vector<std::string>& names, std::vector<std::vector<int>>& shapes,
                              std::vector<const float*>& data) const {
  inner.getTensors(names, shapes, data);
}
bool RelLiteModel::setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) {
  return inner.setTensors(names, flat);
}
ModelSpec RelLiteModel::spec() const {
  ModelSpec s = inner.spec();
  s.arch = archId();
  return s;
}
std::unique_ptr<RelLiteModel> createRelLite(std::string& err) {
  FlexBuildSpec spec;
  spec.tokens = 6;
  spec.dim = 24;
  spec.gate = "clip";
  spec.alpha = 1.0f;
  spec.dynamicBias = false;
  auto m = std::unique_ptr<RelLiteModel>(new RelLiteModel());
  (void)err;
  (void)spec;
  return m;
}
}
