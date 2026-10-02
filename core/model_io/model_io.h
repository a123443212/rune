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

}
