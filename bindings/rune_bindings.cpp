#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/accumulators/grouped_accumulator.h"
#include "core/architectures/attention/attention.h"
#include "core/architectures/mlp/mlp.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"
#include "core/inference/evaluator.h"
#include "bindings/dense_bindings.h"
#include "bindings/relational_bindings.h"
#include "core/model_io/model_factory.h"
#include "core/model_io/model_io.h"

namespace py = pybind11;
using namespace rune;

class PyModel {
 public:
  explicit PyModel(const std::string& archId) {
    arch_ = createArchitecture(archId);
    if (!arch_) throw std::runtime_error("unknown arch");
    eval_ = std::make_unique<Evaluator>(&tables_, arch_.get());
  }

  void setEmbedding(int group, const std::vector<float>& flat) {
    size_t expect =
        static_cast<size_t>(GroupedFeatureSet::vocabSize(group)) * GroupedFeatureSet::kTokenDim;
    if (flat.size() != expect) throw std::runtime_error("embedding size mismatch");
    for (size_t i = 0; i < flat.size(); ++i) {
      tables_.set(group, static_cast<int>(i) / 32, static_cast<int>(i) % 32, flat[i]);
    }
  }

  void setArchTensors(const std::vector<std::string>& names, const std::vector<float>& flat) {
    if (!arch_->setTensors(names, flat)) throw std::runtime_error("arch tensor mismatch");
  }

  std::vector<float> tokensForFen(const std::string& fen) {
    Board b(fen);
    eval_->refresh(b);
    std::vector<float> out(256);
    eval_->currentTokens(out.data());
    return out;
  }

  py::tuple evalFen(const std::string& fen) {
    Board b(fen);
    EvalResult r = eval_->evaluateBoard(b);
    return py::make_tuple(r.value, std::vector<float>{r.wdl[0], r.wdl[1], r.wdl[2]});
  }

  py::tuple forwardTokens(const std::vector<float>& tok) {
    if (tok.size() != 256) throw std::runtime_error("tokens must be 256 floats");
    float value;
    float wdl[3];
    arch_->forward(tok.data(), value, wdl);
    return py::make_tuple(value, std::vector<float>{wdl[0], wdl[1], wdl[2]});
  }

  size_t parameterCount() const { return arch_->parameterCount() + tables_.numFloats(); }

 private:
  EmbeddingTables tables_;
  std::unique_ptr<IArchitecture> arch_;
  std::unique_ptr<Evaluator> eval_;
};

PYBIND11_MODULE(rune_bindings, m) {
  m.doc() = "RUNE v0.1 core bindings";

  py::class_<Move>(m, "Move")
      .def(py::init<>())
      .def_readwrite("from_sq", &Move::from)
      .def_readwrite("to_sq", &Move::to)
      .def_property(
          "promo", [](const Move& mv) { return static_cast<int>(mv.promotion); },
          [](Move& mv, int v) { mv.promotion = static_cast<PieceType>(v); });

  py::class_<Board>(m, "Board")
      .def(py::init<>())
      .def(py::init<const std::string&>())
      .def("set_fen", &Board::setFen)
      .def("to_fen", &Board::toFen)
      .def("make_move", &Board::makeMove)
      .def("unmake_move", &Board::unmakeMove)
      .def("is_legal_position", &Board::isLegalPosition)
      .def("hash_key", &Board::hashKey)
      .def("game_phase", &Board::gamePhase)
      .def("piece_count", &Board::pieceCount)
      .def("side_to_move", [](const Board& b) { return b.sideToMove() == Color::White ? 0 : 1; })
      .def("legal_moves", [](Board& b) {
        std::vector<Move> pseudo;
        b.generatePseudoLegalMoves(pseudo);
        std::vector<std::vector<int>> out;
        for (const Move& mv : pseudo) {
          Board copy = b;
          if (copy.makeMove(mv)) {
            out.push_back({mv.from, mv.to, static_cast<int>(mv.promotion)});
          }
        }
        return out;
      })
      .def("pseudo_count", [](const Board& b) {
        std::vector<Move> pseudo;
        b.generatePseudoLegalMoves(pseudo);
        return pseudo.size();
      });

  m.def("extract_features", [](const std::string& fen) {
    Board b(fen);
    std::vector<ActiveFeature> feats;
    GroupedFeatureSet::extract(b, feats);
    std::vector<std::vector<int>> out;
    for (const ActiveFeature& f : feats) out.push_back({f.group, f.index});
    return out;
  });
  m.def("feature_version", []() { return std::string(GroupedFeatureSet::version()); });

  py::class_<PyModel>(m, "RuneModel")
      .def(py::init<const std::string&>())
      .def("set_embedding", &PyModel::setEmbedding)
      .def("set_arch_tensors", &PyModel::setArchTensors)
      .def("tokens_for_fen", &PyModel::tokensForFen)
      .def("eval_fen", &PyModel::evalFen)
      .def("forward_tokens", &PyModel::forwardTokens)
      .def("parameter_count", &PyModel::parameterCount);

  registerRelational(m);
  registerDense(m);
}
