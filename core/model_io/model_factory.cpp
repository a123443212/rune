#include "core/model_io/model_factory.h"

#include "core/architectures/attention/attention.h"
#include "core/architectures/attention/dual_attention.h"
#include "core/architectures/attention/multi_head.h"
#include "core/architectures/mlp/mlp.h"
#include "core/architectures/mlp/mlp_small.h"
#include "core/architectures/sfnn/sfnn.h"
#include "core/architectures/sfnn/sfnn_compact.h"

namespace rune {

std::unique_ptr<IArchitecture> createArchitecture(const std::string& archId, GateFn gate) {
  if (archId == "RUNE-MLP") return std::unique_ptr<IArchitecture>(new GroupedMlp());
  if (archId == "RUNE-MLP-S") return std::unique_ptr<IArchitecture>(new GroupedMlpSmall());
  if (archId == "RUNE-SFNN") return std::unique_ptr<IArchitecture>(new SfnnBaseline());
  if (archId == "RUNE-SFNN-C") return std::unique_ptr<IArchitecture>(new SfnnCompact());
  if (archId == "RUNE-ATTN") return std::unique_ptr<IArchitecture>(new RuneAttnModel(false, gate));
  if (archId == "RUNE-ATTN-GAB") return std::unique_ptr<IArchitecture>(new RuneAttnModel(true, gate));
  if (archId == "RUNE-ATTN-DUAL") return std::unique_ptr<IArchitecture>(new RuneDualAttentionModel(gate));
  if (archId == "RUNE-ATTN-MH4") return std::unique_ptr<IArchitecture>(new RuneAttnMhModel(gate));
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

bool isSupportedVersion(const std::string& version) {
  return version == "0.1.0" || version == "0.2.0" || version == "0.3.0" || version == "0.4.0" ||
         version == "0.5.0";
}

std::unique_ptr<DenseModel> createDense(const DenseBuildSpec& spec, std::string& err) {
  std::unique_ptr<DenseModel> m(new DenseModel());
  if (!m->configure(spec, err)) return nullptr;
  return m;
}

std::unique_ptr<AdaptiveModel> createAdaptive(const AdaptiveBuildSpec& spec, std::string& err) {
  std::unique_ptr<AdaptiveModel> m(new AdaptiveModel());
  if (!m->configure(spec, err)) return nullptr;
  return m;
}

}
