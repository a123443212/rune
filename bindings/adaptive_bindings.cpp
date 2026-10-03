#include "bindings/adaptive_bindings.h"

#include <pybind11/stl.h>

#include "core/architectures/adaptive/adaptive.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"
#include "core/model_io/model_io.h"

using namespace rune;

class AdaptiveBindingModel {
 public:
  AdaptiveBindingModel(int dim, const std::string& cheapPooling, float threshold) {
    AdaptiveBuildSpec spec;
    spec.dim = dim;
    spec.cheapPooling = cheapPooling;
    spec.threshold = threshold;
    spec.tHigh = threshold;
    std::string err;
    if (!model_.configure(spec, err)) throw std::runtime_error(err);
    VarWidths gw;
    for (int g = 0; g < 8; ++g) gw.w[g] = (cheapPooling == "shared") ? 32 : dim;
    embeddings_.configure(gw);
    if (!eval_.configure(&embeddings_, &model_, err)) throw std::runtime_error(err);
  }

  void setEmbedding(int group, const std::vector<float>& flat) {
    int w = embeddings_.groupWidth(group);
    size_t expect = static_cast<size_t>(GroupedFeatureSet::vocabSize(group)) * w;
    if (flat.size() != expect) throw std::runtime_error("embedding size mismatch");
    for (size_t i = 0; i < flat.size(); ++i) {
      embeddings_.set(group, static_cast<int>(i) / w, static_cast<int>(i) % w, flat[i]);
    }
  }

  void setArchTensors(const std::vector<std::string>& names, const std::vector<float>& flat) {
    if (!model_.setTensors(names, flat)) throw std::runtime_error("arch tensor mismatch");
  }

  py::tuple evalFen(const std::string& fen, const std::string& mode, float threshold) {
    AdaptiveMode m;
    if (!adaptiveModeFromString(mode, m)) throw std::runtime_error("unknown mode");
    Board b(fen);
    float t = (threshold < -1e29f) ? model_.buildSpec().threshold : threshold;
    AdaptiveEvalResult r = eval_.evaluateBoard(b, m, t, true, false);
    return py::make_tuple(r.value,
                          std::vector<float>{r.wdl[0], r.wdl[1], r.wdl[2]}, r.difficulty,
                          r.refined);
  }

  py::tuple forwardTokens(const std::vector<float>& tok, const std::string& mode,
                          float threshold) {
    AdaptiveMode m;
    if (!adaptiveModeFromString(mode, m)) throw std::runtime_error("unknown mode");
    int total = model_.spec().tokenDim * 8;
    if (static_cast<int>(tok.size()) != total) throw std::runtime_error("tokens size mismatch");
    float value;
    float wdl[3];
    float diff = 0.0f;
    bool refined = false;
    if (m == AdaptiveMode::Cheap) {
      std::vector<float> cheap(total);
      model_.cheapForward(tok.data(), cheap.data(), value, wdl, diff);
    } else if (m == AdaptiveMode::Always) {
      model_.forward(tok.data(), value, wdl);
      refined = true;
    } else {
      std::vector<float> cheap(total);
      model_.cheapForward(tok.data(), cheap.data(), value, wdl, diff);
      float t = (threshold < -1e29f) ? model_.buildSpec().threshold : threshold;
      refined = model_.route(diff, false, t);
      if (refined) model_.refineForward(cheap.data(), value, wdl);
    }
    return py::make_tuple(value, std::vector<float>{wdl[0], wdl[1], wdl[2]}, diff, refined);
  }

  float difficultyForFen(const std::string& fen) {
    Board b(fen);
    eval_.refresh(b);
    int total = model_.spec().tokenDim * 8;
    std::vector<float> raw(total);
    std::vector<float> cheap(total);
    eval_.currentTokens(raw.data());
    float value;
    float wdl[3];
    float diff = 0.0f;
    model_.cheapForward(raw.data(), cheap.data(), value, wdl, diff);
    return diff;
  }

  size_t parameterCount() const { return model_.parameterCount() + embeddings_.numFloats(); }

  size_t cheapParameterCount() const {
    return model_.cheapParameterCount() + embeddings_.numFloats();
  }

 private:
  VarEmbeddings embeddings_;
  AdaptiveModel model_;
  AdaptiveEvaluator eval_;
};

void registerAdaptive(py::module_& m) {
  py::class_<AdaptiveBindingModel>(m, "AdaptiveModel")
      .def(py::init<int, const std::string&, float>())
      .def("set_embedding", &AdaptiveBindingModel::setEmbedding)
      .def("set_arch_tensors", &AdaptiveBindingModel::setArchTensors)
      .def("eval_fen", &AdaptiveBindingModel::evalFen)
      .def("forward_tokens", &AdaptiveBindingModel::forwardTokens)
      .def("difficulty_for_fen", &AdaptiveBindingModel::difficultyForFen)
      .def("parameter_count", &AdaptiveBindingModel::parameterCount)
      .def("cheap_parameter_count", &AdaptiveBindingModel::cheapParameterCount);
}
