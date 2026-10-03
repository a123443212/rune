#include "core/model_io/model_io.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>

namespace rune {
namespace {

void writeHeaderFields(std::ostringstream& h, const ModelSpec& spec, const std::string& quantization,
                       const float* scales) {
  h << "{\"format\":1";
  h << ",\"arch\":\"" << spec.arch << "\"";
  h << ",\"arch_version\":\"" << spec.archVersion << "\"";
  h << ",\"feature_set\":\"" << spec.featureSet << "\"";
  h << ",\"tokens\":" << spec.tokens;
  h << ",\"token_dim\":" << spec.tokenDim;
  h << ",\"attention\":\"" << spec.attention << "\"";
  h << ",\"geometric_bias\":\"" << spec.geometricBias << "\"";
  h << ",\"head\":\"" << spec.head << "\"";
  h << ",\"quantization\":\"" << quantization << "\"";
  h << ",\"gate\":\"" << spec.gate << "\"";
  h << ",\"alpha\":" << spec.alpha;
  h << ",\"context_dim\":" << ContextSpec::kDim;
  h << ",\"scales\":{";
  for (int g = 0; g < 8; ++g) {
    if (g > 0) h << ",";
    h << "\"emb" << g << "\":" << scales[g];
  }
  h << "}";
}

void writeTensorList(std::ostringstream& h, const std::vector<std::string>& names,
                     const std::vector<std::vector<int>>& shapes, const std::string& embDtype,
                     int dim) {
  h << ",\"tensors\":[";
  bool first = true;
  auto emit = [&](const std::string& n, const std::string& shape, const std::string& dtype) {
    if (!first) h << ",";
    first = false;
    h << "{\"name\":\"" << n << "\",\"shape\":" << shape << ",\"dtype\":\"" << dtype << "\"}";
  };
  for (int g = 0; g < 8; ++g) {
    int v = GroupedFeatureSet::vocabSize(g);
    emit("emb" + std::to_string(g), "[" + std::to_string(v) + "," + std::to_string(dim) + "]",
         embDtype);
  }
  for (size_t i = 0; i < names.size(); ++i) {
    std::string shape = "[";
    for (size_t j = 0; j < shapes[i].size(); ++j) {
      if (j > 0) shape += ",";
      shape += std::to_string(shapes[i][j]);
    }
    shape += "]";
    emit(names[i], shape, "float32");
  }
  h << "]}";
}

}  // namespace

bool saveRuneFile(const std::string& path, const ModelSpec& spec, const EmbeddingTables& embeddings,
                  const IArchitecture& arch, const std::string& quantization, std::string& err) {
  std::vector<std::string> names;
  std::vector<std::vector<int>> shapes;
  std::vector<const float*> data;
  arch.getTensors(names, shapes, data);
  QuantEmbeddingTables qe;
  Quant16Tables q16;
  QuantScales sc;
  for (int g = 0; g < 8; ++g) sc.embedding[g] = 1.0f;
  std::string embDtype = "float32";
  if (quantization == "int8") {
    qe.quantizeFrom(embeddings, sc);
    embDtype = "int8";
  }
  if (quantization == "int16") {
    q16.quantizeFrom(embeddings, sc);
    embDtype = "int16";
  }
  float scales[8];
  for (int g = 0; g < 8; ++g) scales[g] = sc.embedding[g];
  std::ostringstream h;
  writeHeaderFields(h, spec, quantization, scales);
  writeTensorList(h, names, shapes, embDtype, 32);
  std::string header = h.str();
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    err = "cannot open output";
    return false;
  }
  f.write("RUNE", 4);
  uint32_t hlen = static_cast<uint32_t>(header.size());
  f.write(reinterpret_cast<const char*>(&hlen), 4);
  f.write(header.data(), header.size());
  for (int g = 0; g < 8; ++g) {
    const std::vector<float>& d = embeddings.groupData(g);
    if (quantization == "int8") {
      const std::vector<int8_t>& q = qe.groupData(g);
      f.write(reinterpret_cast<const char*>(q.data()), q.size());
    } else if (quantization == "int16") {
      const std::vector<int16_t>& q = q16.groupData(g);
      f.write(reinterpret_cast<const char*>(q.data()), q.size() * 2);
    } else {
      f.write(reinterpret_cast<const char*>(d.data()), d.size() * 4);
    }
  }
  for (size_t i = 0; i < data.size(); ++i) {
    size_t n = 1;
    for (int s : shapes[i]) n *= static_cast<size_t>(s);
    f.write(reinterpret_cast<const char*>(data[i]), n * 4);
  }
  return true;
}

bool saveFlexRuneFile(const std::string& path, const ModelSpec& spec,                      const FlexEmbeddings& embeddings, const IArchitecture& arch,
                      const std::string& quantization, std::string& err) {
  std::vector<std::string> names;
  std::vector<std::vector<int>> shapes;
  std::vector<const float*> data;
  arch.getTensors(names, shapes, data);
  int dim = spec.tokenDim;
  TokenLayout layout;
  if (!TokenLayout::make(spec.tokens, dim, layout, err)) return false;
  FlexQuantTables qt;
  qt.configure(dim, quantization == "int16");
  FlexScales sc;
  for (int g = 0; g < 8; ++g) sc.embedding[g] = 1.0f;
  std::string embDtype = "float32";
  if (quantization == "int8" || quantization == "int16") {
    qt.quantizeFrom(embeddings, layout, sc);
    embDtype = quantization;
  }
  float scales[8];
  for (int g = 0; g < 8; ++g) scales[g] = sc.embedding[g];
  std::ostringstream h;
  writeHeaderFields(h, spec, quantization, scales);
  writeTensorList(h, names, shapes, embDtype, dim);
  std::string header = h.str();
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    err = "cannot open output";
    return false;
  }
  f.write("RUNE", 4);
  uint32_t hlen = static_cast<uint32_t>(header.size());
  f.write(reinterpret_cast<const char*>(&hlen), 4);
  f.write(header.data(), header.size());
  for (int g = 0; g < 8; ++g) {
    const std::vector<float>& d = embeddings.groupData(g);
    if (quantization == "int8") {
      for (float v : d) {
        int q = static_cast<int>(std::lround(v / sc.embedding[g]));
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        int8_t b = static_cast<int8_t>(q);
        f.write(reinterpret_cast<const char*>(&b), 1);
      }
    } else if (quantization == "int16") {
      for (float v : d) {
        int q = static_cast<int>(std::lround(v / sc.embedding[g]));
        if (q > 32767) q = 32767;
        if (q < -32767) q = -32767;
        int16_t w = static_cast<int16_t>(q);
        f.write(reinterpret_cast<const char*>(&w), 2);
      }
    } else {
      f.write(reinterpret_cast<const char*>(d.data()), d.size() * 4);
    }
  }
  for (size_t i = 0; i < data.size(); ++i) {
    size_t n = 1;
    for (int s : shapes[i]) n *= static_cast<size_t>(s);
    f.write(reinterpret_cast<const char*>(data[i]), n * 4);
  }
  return true;
}

namespace {

std::string toHex(uint64_t v) {
  static const char* digits = "0123456789abcdef";
  std::string s(16, '0');
  for (int i = 15; i >= 0; --i) {
    s[i] = digits[v & 15];
    v >>= 4;
  }
  return s;
}

}  // namespace

bool saveDenseRuneFile(const std::string& path, const ModelSpec& spec,
                       const VarEmbeddings& embeddings, const IArchitecture& arch,
                       const std::string& quantization, std::string& err) {
  std::vector<std::string> names;
  std::vector<std::vector<int>> shapes;
  std::vector<const float*> data;
  arch.getTensors(names, shapes, data);
  int sw = 32;
  VarWidths gw;
  for (int g = 0; g < 8; ++g) gw.w[g] = (spec.pooling == "shared") ? 32 : spec.tokenDims[g];
  VarQuantTables qt;
  qt.configure(gw, quantization == "int16");
  VarScales sc;
  std::string embDtype = "float32";
  if (quantization == "int8" || quantization == "int16") {
    qt.quantizeFrom(embeddings, sc);
    embDtype = quantization;
  } else {
    for (int g = 0; g < 8; ++g) sc.token[g] = 1.0f;
  }
  std::string payload;
  for (int g = 0; g < 8; ++g) {
    int w = gw.w[g];
    size_t count = static_cast<size_t>(GroupedFeatureSet::vocabSize(g)) * w;
    const std::vector<float>& d = embeddings.groupData(g);
    if (embDtype == "int8") {
      for (size_t i = 0; i < count; ++i) {
        int q = static_cast<int>(std::lround(d[i] / sc.token[g]));
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        payload.push_back(static_cast<char>(q));
      }
    } else if (embDtype == "int16") {
      for (size_t i = 0; i < count; ++i) {
        int q = static_cast<int>(std::lround(d[i] / sc.token[g]));
        if (q > 32767) q = 32767;
        if (q < -32767) q = -32767;
        int16_t v = static_cast<int16_t>(q);
        payload.append(reinterpret_cast<const char*>(&v), 2);
      }
    } else {
      payload.append(reinterpret_cast<const char*>(d.data()), count * 4);
    }
  }
  for (size_t i = 0; i < data.size(); ++i) {
    size_t n = 1;
    for (int s : shapes[i]) n *= static_cast<size_t>(s);
    payload.append(reinterpret_cast<const char*>(data[i]), n * 4);
  }
  std::string checksum = toHex(fnv1aHash(reinterpret_cast<const uint8_t*>(payload.data()),
                                         payload.size()));
  std::ostringstream h;
  h << "{\"format\":1";
  h << ",\"arch\":\"" << spec.arch << "\"";
  h << ",\"arch_version\":\"" << spec.archVersion << "\"";
  h << ",\"feature_set\":\"" << spec.featureSet << "\"";
  h << ",\"tokens\":" << spec.tokens;
  h << ",\"token_dim\":" << spec.tokenDim;
  h << ",\"attention\":\"" << spec.attention << "\"";
  h << ",\"geometric_bias\":\"" << spec.geometricBias << "\"";
  h << ",\"head\":\"" << spec.head << "\"";
  h << ",\"quantization\":\"" << quantization << "\"";
  h << ",\"gate\":\"" << spec.gate << "\"";
  h << ",\"alpha\":" << spec.alpha;
  h << ",\"context_dim\":" << ContextSpec::kDim;
  h << ",\"variant\":\"" << spec.variant << "\"";
  h << ",\"token_dims\":[";
  for (size_t i = 0; i < spec.tokenDims.size(); ++i) {
    if (i > 0) h << ",";
    h << spec.tokenDims[i];
  }
  h << "]";
  h << ",\"pooling\":\"" << spec.pooling << "\"";
  h << ",\"pool_clip\":" << (spec.poolClip ? 1 : 0);
  h << ",\"gate_on\":" << (spec.gateOn ? 1 : 0);
  h << ",\"shared_width\":" << sw;
  h << ",\"scales\":{";
  for (int g = 0; g < 8; ++g) {
    if (g > 0) h << ",";
    h << "\"emb" << g << "\":" << sc.token[g];
  }
  h << "}";
  h << ",\"tensors\":[";
  bool first = true;
  auto emit = [&](const std::string& n, const std::string& shape, const std::string& dtype) {
    if (!first) h << ",";
    first = false;
    h << "{\"name\":\"" << n << "\",\"shape\":" << shape << ",\"dtype\":\"" << dtype << "\"}";
  };
  for (int g = 0; g < 8; ++g) {
    int v = GroupedFeatureSet::vocabSize(g);
    emit("emb" + std::to_string(g), "[" + std::to_string(v) + "," + std::to_string(gw.w[g]) + "]",
         embDtype);
  }
  for (size_t i = 0; i < names.size(); ++i) {
    std::string shape = "[";
    for (size_t j = 0; j < shapes[i].size(); ++j) {
      if (j > 0) shape += ",";
      shape += std::to_string(shapes[i][j]);
    }
    shape += "]";
    emit(names[i], shape, "float32");
  }
  h << "]";
  h << ",\"checksum\":\"" << checksum << "\"}";
  std::string header = h.str();
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    err = "cannot open output";
    return false;
  }
  f.write("RUNE", 4);
  uint32_t hlen = static_cast<uint32_t>(header.size());
  f.write(reinterpret_cast<const char*>(&hlen), 4);
  f.write(header.data(), header.size());
  f.write(payload.data(), payload.size());
  return true;
}

bool saveAdaptiveRuneFile(const std::string& path, const ModelSpec& spec,
                          const VarEmbeddings& embeddings, const IArchitecture& arch,
                          const std::string& quantization, std::string& err) {
  std::vector<std::string> names;
  std::vector<std::vector<int>> shapes;
  std::vector<const float*> data;
  arch.getTensors(names, shapes, data);
  int dim = spec.tokenDim;
  VarWidths gw;
  for (int g = 0; g < 8; ++g) gw.w[g] = (spec.cheapPooling == "shared") ? 32 : dim;
  VarQuantTables qt;
  qt.configure(gw, quantization == "int16");
  VarScales sc;
  std::string embDtype = "float32";
  if (quantization == "int8" || quantization == "int16") {
    qt.quantizeFrom(embeddings, sc);
    embDtype = quantization;
  } else {
    for (int g = 0; g < 8; ++g) sc.token[g] = 1.0f;
  }
  std::string payload;
  for (int g = 0; g < 8; ++g) {
    int w = gw.w[g];
    size_t count = static_cast<size_t>(GroupedFeatureSet::vocabSize(g)) * w;
    const std::vector<float>& d = embeddings.groupData(g);
    if (embDtype == "int8") {
      for (size_t i = 0; i < count; ++i) {
        int q = static_cast<int>(std::lround(d[i] / sc.token[g]));
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        payload.push_back(static_cast<char>(q));
      }
    } else if (embDtype == "int16") {
      for (size_t i = 0; i < count; ++i) {
        int q = static_cast<int>(std::lround(d[i] / sc.token[g]));
        if (q > 32767) q = 32767;
        if (q < -32767) q = -32767;
        int16_t v = static_cast<int16_t>(q);
        payload.append(reinterpret_cast<const char*>(&v), 2);
      }
    } else {
      payload.append(reinterpret_cast<const char*>(d.data()), count * 4);
    }
  }
  for (size_t i = 0; i < data.size(); ++i) {
    size_t n = 1;
    for (int s : shapes[i]) n *= static_cast<size_t>(s);
    payload.append(reinterpret_cast<const char*>(data[i]), n * 4);
  }
  std::string checksum = toHex(fnv1aHash(reinterpret_cast<const uint8_t*>(payload.data()),
                                         payload.size()));
  std::ostringstream h;
  h << "{\"format\":1";
  h << ",\"arch\":\"" << spec.arch << "\"";
  h << ",\"arch_version\":\"" << spec.archVersion << "\"";
  h << ",\"feature_set\":\"" << spec.featureSet << "\"";
  h << ",\"tokens\":" << spec.tokens;
  h << ",\"token_dim\":" << spec.tokenDim;
  h << ",\"attention\":\"" << spec.attention << "\"";
  h << ",\"geometric_bias\":\"" << spec.geometricBias << "\"";
  h << ",\"head\":\"" << spec.head << "\"";
  h << ",\"quantization\":\"" << quantization << "\"";
  h << ",\"gate\":\"" << spec.gate << "\"";
  h << ",\"alpha\":" << spec.alpha;
  h << ",\"context_dim\":" << ContextSpec::kDim;
  h << ",\"cheap_pooling\":\"" << spec.cheapPooling << "\"";
  h << ",\"threshold\":" << spec.threshold;
  h << ",\"t_high\":" << spec.tHigh;
  if (spec.hasTLow) h << ",\"t_low\":" << spec.tLow;
  else h << ",\"t_low\":null";
  h << ",\"refine_precision\":\"" << spec.refinePrecision << "\"";
  h << ",\"uncertainty\":" << (spec.hasUncertainty ? "true" : "false");
  h << ",\"stability_head\":" << (spec.hasStabilityHead ? "true" : "false");
  h << ",\"pruned_pairs\":[";
  for (size_t i = 0; i < spec.prunedPairs.size(); ++i) {
    if (i > 0) h << ",";
    h << "[" << spec.prunedPairs[i].first << "," << spec.prunedPairs[i].second << "]";
  }
  h << "]";
  h << ",\"token_dims\":[";
  for (int g = 0; g < 8; ++g) {
    if (g > 0) h << ",";
    h << dim;
  }
  h << "]";
  h << ",\"scales\":{";
  for (int g = 0; g < 8; ++g) {
    if (g > 0) h << ",";
    h << "\"emb" << g << "\":" << sc.token[g];
  }
  h << "}";
  h << ",\"tensors\":[";
  bool first = true;
  auto emit = [&](const std::string& n, const std::string& shape, const std::string& dtype) {
    if (!first) h << ",";
    first = false;
    h << "{\"name\":\"" << n << "\",\"shape\":" << shape << ",\"dtype\":\"" << dtype << "\"}";
  };
  for (int g = 0; g < 8; ++g) {
    int v = GroupedFeatureSet::vocabSize(g);
    emit("emb" + std::to_string(g), "[" + std::to_string(v) + "," + std::to_string(gw.w[g]) + "]",
         embDtype);
  }
  for (size_t i = 0; i < names.size(); ++i) {
    std::string shape = "[";
    for (size_t j = 0; j < shapes[i].size(); ++j) {
      if (j > 0) shape += ",";
      shape += std::to_string(shapes[i][j]);
    }
    shape += "]";
    emit(names[i], shape, "float32");
  }
  h << "]";
  h << ",\"checksum\":\"" << checksum << "\"}";
  std::string header = h.str();
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    err = "cannot open output";
    return false;
  }
  f.write("RUNE", 4);
  uint32_t hlen = static_cast<uint32_t>(header.size());
  f.write(reinterpret_cast<const char*>(&hlen), 4);
  f.write(header.data(), header.size());
  f.write(payload.data(), payload.size());
  return true;
}

}
