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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/accumulators/flex_accumulator.h"
#include "core/accumulators/grouped_accumulator.h"
#include "core/architectures/base/architecture.h"
#include "core/architectures/dense/var_accum.h"
#include "core/model_io/model_factory.h"

namespace rune {

struct RuneFile {
  ModelSpec spec;
  EmbeddingTables embeddings;
  std::unique_ptr<IArchitecture> arch;
  QuantEmbeddingTables qembeddings;
  QuantScales scales;
  bool isInt8 = false;
  Quant16Tables q16embeddings;
  bool isInt16 = false;
  bool isFlex = false;
  TokenLayout layout;
  FlexEmbeddings flexEmbeddings;
  FlexQuantTables flexQ;
  FlexScales flexScales;
  bool isDense = false;
  VarWidths varWidths;
  VarEmbeddings varEmbeddings;
  VarQuantTables varQ;
  VarScales varScales;
  bool isAdaptive = false;
  bool isUncertainty = false;
};

bool loadRuneFile(const std::string& path, RuneFile& out, std::string& err);
bool saveRuneFile(const std::string& path, const ModelSpec& spec, const EmbeddingTables& embeddings,
                  const IArchitecture& arch, const std::string& quantization, std::string& err);
bool saveFlexRuneFile(const std::string& path, const ModelSpec& spec,
                      const FlexEmbeddings& embeddings, const IArchitecture& arch,
                      const std::string& quantization, std::string& err);
bool saveDenseRuneFile(const std::string& path, const ModelSpec& spec,
                       const VarEmbeddings& embeddings, const IArchitecture& arch,
                       const std::string& quantization, std::string& err);
bool saveAdaptiveRuneFile(const std::string& path, const ModelSpec& spec,
                          const VarEmbeddings& embeddings, const IArchitecture& arch,
                          const std::string& quantization, std::string& err);

}
