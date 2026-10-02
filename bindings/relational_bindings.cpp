#include "bindings/relational_bindings.h"

#include <pybind11/stl.h>

#include "core/architectures/relational/relational.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"

using namespace rune;

class FlexModel {
 public:
  FlexModel(int tokens, int dim, const std::string& gate, float alpha, bool dynamicBias) {
    RelationalConfig cfg;
    cfg.tokens = tokens;
    cfg.dim = dim;
    cfg.alpha = alpha;
    cfg.dynamicBias = dynamicBias;
    std::string err;
    if (!gateFromString(gate, cfg.gate)) throw std::runtime_error("unknown gate");
    if (!TokenLayout::make(tokens, dim, layout_, err)) throw std::runtime_error(err);
    embeddings_.configure(dim);
    if (!model_.configure(cfg, err)) throw std::runtime_error(err);
    if (!eval_.configure(&embeddings_, &layout_, &model_, err)) throw std::runtime_error(err);
  }

  void setEmbedding(int group, const std::vector<float>& flat) {
    size_t expect = static_cast<size_t>(GroupedFeatureSet::vocabSize(group)) * layout_.dim;
    if (flat.size() != expect) throw std::runtime_error("embedding size mismatch");
    for (size_t i = 0; i < flat.size(); ++i) {
      embeddings_.set(group, static_cast<int>(i) / layout_.dim, static_cast<int>(i) % layout_.dim,
                      flat[i]);
    }
  }

  void setArchTensors(const std::vector<std::string>& names, const std::vector<float>& flat) {
    if (!model_.setTensors(names, flat)) throw std::runtime_error("arch tensor mismatch");
  }

  std::vector<float> tokensForFen(const std::string& fen) {
    Board b(fen);
    eval_.refresh(b);
    std::vector<float> out(layout_.tokens * layout_.dim);
    eval_.currentTokens(out.data());
    return out;
  }

  std::vector<float> contextForFen(const std::string& fen) {
    Board b(fen);
    float ctx[ContextSpec::kDim];
    computeContext(b, ctx);
    return std::vector<float>(ctx, ctx + ContextSpec::kDim);
  }

  py::tuple evalFen(const std::string& fen) {
    Board b(fen);
    EvalResultFlex r = eval_.evaluateBoard(b);
    return py::make_tuple(r.value, std::vector<float>{r.wdl[0], r.wdl[1], r.wdl[2]});
  }

  py::tuple forwardTokens(const std::vector<float>& tok, const std::vector<float>& ctx) {
    if (tok.size() != static_cast<size_t>(layout_.tokens * layout_.dim)) {
      throw std::runtime_error("token size mismatch");
    }
    if (ctx.size() != ContextSpec::kDim) throw std::runtime_error("context size mismatch");
    float value;
    float wdl[3];
    model_.forwardWithContext(tok.data(), ctx.data(), value, wdl);
    return py::make_tuple(value, std::vector<float>{wdl[0], wdl[1], wdl[2]});
  }

  size_t parameterCount() const { return model_.parameterCount() + embeddings_.numFloats(); }

 private:
  TokenLayout layout_;
  FlexEmbeddings embeddings_;
  RelationalModel model_;
  RelationalEvaluator eval_;
};

void registerRelational(py::module_& m) {
  py::class_<FlexModel>(m, "FlexModel")
      .def(py::init<int, int, const std::string&, float, bool>())
      .def("set_embedding", &FlexModel::setEmbedding)
      .def("set_arch_tensors", &FlexModel::setArchTensors)
      .def("tokens_for_fen", &FlexModel::tokensForFen)
      .def("context_for_fen", &FlexModel::contextForFen)
      .def("eval_fen", &FlexModel::evalFen)
      .def("forward_tokens", &FlexModel::forwardTokens)
      .def("parameter_count", &FlexModel::parameterCount);
}
