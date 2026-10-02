#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/accumulators/grouped_accumulator.h"
#include "core/architectures/base/architecture.h"

namespace rune {

struct RuneFile {
  ModelSpec spec;
  EmbeddingTables embeddings;
  std::unique_ptr<IArchitecture> arch;
  QuantEmbeddingTables qembeddings;
  QuantScales scales;
  bool isInt8 = false;
};

std::unique_ptr<IArchitecture> createArchitecture(const std::string& archId);

bool loadRuneFile(const std::string& path, RuneFile& out, std::string& err);
bool saveRuneFile(const std::string& path, const ModelSpec& spec, const EmbeddingTables& embeddings,
                  const IArchitecture& arch, const std::string& quantization, std::string& err);

}
