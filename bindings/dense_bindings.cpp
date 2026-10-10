/*
RUNE — Relational Unified Neural Evaluator
Copyright (C) 2026 a123443212

SPDX-License-Identifier: MIT OR Apache-2.0

This project is dual-licensed under the MIT License and the
Apache License, Version 2.0. You may choose either license
when using, copying, modifying, or distributing this software.

MIT License: https://opensource.org/license/mit
Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, this
software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
*/

#include "bindings/dense_bindings.h"

#include <pybind11/stl.h>

#include "core/architectures/dense/dense.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"
#include "core/model_io/model_io.h"

using namespace rune;

class DenseBindingModel {
 public:
  DenseBindingModel(const std::string& variant, const std::vector<int>& dims,
                    const std::string& pooling, bool poolClip, bool gateOn,
                    int headH1 = 128, int headH2 = 32) {
    DenseBuildSpec spec;
    spec.variant = variant;
    spec.dims = dims;
    spec.pooling = pooling;
    spec.poolClip = poolClip;
    spec.gateOn = gateOn;
    spec.sharedWidth = 32;
    spec.headH1 = headH1;
    spec.headH2 = headH2;
    std::string err;
    if (!model_.configure(spec, err)) throw std::runtime_error(err);
    VarWidths gw;
    for (int g = 0; g < 8; ++g) gw.w[g] = (pooling == "shared") ? 32 : dims[g];
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

  std::vector<float> tokensForFen(const std::string& fen) {
    Board b(fen);
    eval_.refresh(b);
    int n = 0;
    for (int t = 0; t < 8; ++t) n += embeddings_.groupWidth(t);
    std::vector<float> raw(n);
    eval_.currentTokens(raw.data());
    return raw;
  }

  py::tuple evalFen(const std::string& fen) {
    Board b(fen);
    float value;
    float wdl[3];
    eval_.evaluateBoard(b, value, wdl);
    return py::make_tuple(value, std::vector<float>{wdl[0], wdl[1], wdl[2]});
  }

  py::tuple forwardTokens(const std::vector<float>& tok) {
    int expect = 0;
    for (int t = 0; t < 8; ++t) expect += model_.spec().tokenDims[t];
    if (static_cast<int>(tok.size()) != expect) throw std::runtime_error("tokens size mismatch");
    float value;
    float wdl[3];
    model_.forward(tok.data(), value, wdl);
    return py::make_tuple(value, std::vector<float>{wdl[0], wdl[1], wdl[2]});
  }

  size_t parameterCount() const { return model_.parameterCount() + embeddings_.numFloats(); }

 private:
  VarEmbeddings embeddings_;
  DenseModel model_;
  DenseEvaluator eval_;
};

void registerDense(py::module_& m) {
  m.def("verify_rune_file", [](const std::string& path) -> py::tuple {
    RuneFile loaded;
    std::string err;
    if (!loadRuneFile(path, loaded, err)) return py::make_tuple(false, err);
    return py::make_tuple(true, std::string("ok"));
  });
  py::class_<DenseBindingModel>(m, "DenseModel")
      .def(py::init<const std::string&, const std::vector<int>&, const std::string&, bool, bool>())
      .def(py::init<const std::string&, const std::vector<int>&, const std::string&, bool, bool, int, int>())
      .def("set_embedding", &DenseBindingModel::setEmbedding)
      .def("set_arch_tensors", &DenseBindingModel::setArchTensors)
      .def("tokens_for_fen", &DenseBindingModel::tokensForFen)
      .def("eval_fen", &DenseBindingModel::evalFen)
      .def("forward_tokens", &DenseBindingModel::forwardTokens)
      .def("parameter_count", &DenseBindingModel::parameterCount);
}
