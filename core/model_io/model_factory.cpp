#include "core/model_io/model_factory.h"

#include "core/architectures/attention/attention.h"
#include "core/architectures/mlp/mlp.h"

namespace rune {

std::unique_ptr<IArchitecture> createArchitecture(const std::string& archId) {
  if (archId == "RUNE-MLP") return std::unique_ptr<IArchitecture>(new GroupedMlp());
  if (archId == "RUNE-SFNN") return std::unique_ptr<IArchitecture>(new SfnnBaseline());
  if (archId == "RUNE-ATTN") return std::unique_ptr<IArchitecture>(new RuneAttnModel(false));
  if (archId == "RUNE-ATTN-GAB") return std::unique_ptr<IArchitecture>(new RuneAttnModel(true));
  return nullptr;
}

std::unique_ptr<RelationalModel> createRelational(const FlexBuildSpec& spec, std::string& err) {
  RelationalConfig cfg;
  cfg.tokens = spec.tokens;
  cfg.dim = spec.dim;
  cfg.alpha = spec.alpha;
  cfg.dynamicBias = spec.dynamicBias;
  if (!gateFromString(spec.gate, cfg.gate)) {
    err = "unknown gate";
    return nullptr;
  }
  std::unique_ptr<RelationalModel> m(new RelationalModel());
  if (!m->configure(cfg, err)) return nullptr;
  return m;
}

}
