#include "core/model_io/model_io.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>

namespace rune {

namespace {

std::string extractString(const std::string& h, const std::string& key) {
  std::string pat = "\"" + key + "\":\"";
  size_t p = h.find(pat);
  if (p == std::string::npos) return "";
  p += pat.size();
  size_t q = h.find('"', p);
  return h.substr(p, q - p);
}

long extractInt(const std::string& h, const std::string& key, long fallback) {
  std::string pat = "\"" + key + "\":";
  size_t p = h.find(pat);
  if (p == std::string::npos) return fallback;
  p += pat.size();
  size_t q = h.find_first_of(",}", p);
  try {
    return std::stol(h.substr(p, q - p));
  } catch (...) {
    return fallback;
  }
}

double extractNumber(const std::string& h, const std::string& key, double fallback) {
  std::string pat = "\"" + key + "\":";
  size_t p = h.find(pat);
  if (p == std::string::npos) return fallback;
  size_t v = p + pat.size();
  if (h[v] == '"') return fallback;
  size_t q = h.find_first_of(",}", v);
  try {
    return std::stod(h.substr(v, q - v));
  } catch (...) {
    return fallback;
  }
}

struct TensorMeta {
  std::string name;
  std::vector<int> shape;
  std::string dtype;
};

bool parseTensors(const std::string& h, std::vector<TensorMeta>& out) {
  size_t p = h.find("\"tensors\":[");
  if (p == std::string::npos) return false;
  p += 11;
  while (true) {
    size_t nb = h.find('{', p);
    size_t end = h.find(']', p);
    if (nb == std::string::npos || (end != std::string::npos && end < nb)) break;
    size_t ne = h.find('}', nb);
    if (ne == std::string::npos) return false;
    std::string obj = h.substr(nb, ne - nb + 1);
    TensorMeta t;
    size_t pn = obj.find("\"name\":\"");
    if (pn == std::string::npos) return false;
    pn += 8;
    t.name = obj.substr(pn, obj.find('"', pn) - pn);
    size_t ps = obj.find("\"shape\":[");
    if (ps == std::string::npos) return false;
    ps += 9;
    size_t pe = obj.find(']', ps);
    std::string dims = obj.substr(ps, pe - ps);
    std::stringstream ss(dims);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      if (!tok.empty()) t.shape.push_back(std::stoi(tok));
    }
    size_t pd = obj.find("\"dtype\":\"");
    if (pd == std::string::npos) return false;
    pd += 9;
    t.dtype = obj.substr(pd, obj.find('"', pd) - pd);
    out.push_back(t);
    p = ne + 1;
  }
  return !out.empty();
}

bool readHeader(std::ifstream& f, std::string& header, std::string& err) {
  char magic[4];
  f.read(magic, 4);
  if (f.gcount() != 4 || magic[0] != 'R' || magic[1] != 'U' || magic[2] != 'N' || magic[3] != 'E') {
    err = "bad magic";
    return false;
  }
  uint32_t hlen = 0;
  f.read(reinterpret_cast<char*>(&hlen), 4);
  if (hlen > 1000000) {
    err = "header too large";
    return false;
  }
  header.assign(hlen, '\0');
  f.read(header.data(), hlen);
  return true;
}

void fillSpec(const std::string& header, ModelSpec& spec) {
  spec.arch = extractString(header, "arch");
  spec.archVersion = extractString(header, "arch_version");
  spec.featureSet = extractString(header, "feature_set");
  spec.tokens = static_cast<int>(extractInt(header, "tokens", 8));
  spec.tokenDim = static_cast<int>(extractInt(header, "token_dim", 32));
  spec.attention = extractString(header, "attention");
  spec.geometricBias = extractString(header, "geometric_bias");
  spec.head = extractString(header, "head");
  spec.quantization = extractString(header, "quantization");
  std::string gate = extractString(header, "gate");
  spec.gate = gate.empty() ? "clip" : gate;
  spec.alpha = static_cast<float>(extractNumber(header, "alpha", 1.0));
}

}  // namespace

bool loadRuneFile(const std::string& path, RuneFile& out, std::string& err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    err = "cannot open file";
    return false;
  }
  std::string header;
  if (!readHeader(f, header, err)) return false;
  fillSpec(header, out.spec);
  std::string arch = out.spec.arch;
  out.isInt8 = (out.spec.quantization == "int8");
  out.isInt16 = (out.spec.quantization == "int16");
  out.isFlex = (arch == "RUNE-REL-02");
  std::vector<TensorMeta> metas;
  if (!parseTensors(header, metas)) {
    err = "tensor list parse failed";
    return false;
  }
  TokenLayout layout;
  FlexEmbeddings* flexEmb = nullptr;
  if (out.isFlex) {
    FlexBuildSpec bs;
    bs.tokens = out.spec.tokens;
    bs.dim = out.spec.tokenDim;
    bs.gate = out.spec.gate;
    bs.alpha = out.spec.alpha;
    bs.dynamicBias = (out.spec.geometricBias == "dynamic");
    int ctxDim = static_cast<int>(extractInt(header, "context_dim", 8));
    if (ctxDim != ContextSpec::kDim) {
      err = "context dim mismatch";
      return false;
    }
    out.arch = createRelational(bs, err);
    if (!out.arch) {
      if (err.empty()) err = "relational build failed";
      return false;
    }
    if (!TokenLayout::make(bs.tokens, bs.dim, layout, err)) return false;
    out.layout = layout;
    out.flexEmbeddings.configure(bs.dim);
    flexEmb = &out.flexEmbeddings;
  } else {
    out.arch = createArchitecture(arch);
    if (!out.arch) {
      err = "unknown arch " + arch;
      return false;
    }
  }
  int dim = out.isFlex ? out.spec.tokenDim : 32;
  std::vector<std::string> archNames;
  std::vector<float> archFlat;
  for (const TensorMeta& t : metas) {
    size_t count = 1;
    for (int s : t.shape) count *= static_cast<size_t>(s);
    if (t.name.rfind("emb", 0) == 0) {
      int g = std::stoi(t.name.substr(3));
      double scale = extractNumber(header, t.name, 1.0);
      if (t.dtype == "int8" || t.dtype == "int16") {
        if (t.dtype == "int8") {
          std::vector<char> buf(count);
          f.read(buf.data(), count);
          if (!f) {
            err = "truncated payload";
            return false;
          }
          for (size_t i = 0; i < count; ++i) {
            float v = static_cast<float>(buf[i]) * static_cast<float>(scale);
            if (out.isFlex) flexEmb->set(g, static_cast<int>(i) / dim, static_cast<int>(i) % dim, v);
            else out.embeddings.set(g, static_cast<int>(i) / 32, static_cast<int>(i) % 32, v);
          }
        } else {
          std::vector<int16_t> buf(count);
          f.read(reinterpret_cast<char*>(buf.data()), count * 2);
          if (!f) {
            err = "truncated payload";
            return false;
          }
          for (size_t i = 0; i < count; ++i) {
            float v = static_cast<float>(buf[i]) * static_cast<float>(scale);
            if (out.isFlex) flexEmb->set(g, static_cast<int>(i) / dim, static_cast<int>(i) % dim, v);
            else out.embeddings.set(g, static_cast<int>(i) / 32, static_cast<int>(i) % 32, v);
          }
        }
        if (out.isFlex) out.flexScales.embedding[g] = static_cast<float>(scale);
        else out.scales.embedding[g] = static_cast<float>(scale);
      } else {
        std::vector<float> buf(count);
        f.read(reinterpret_cast<char*>(buf.data()), count * 4);
        if (!f) {
          err = "truncated payload";
          return false;
        }
        for (size_t i = 0; i < count; ++i) {
          if (out.isFlex) flexEmb->set(g, static_cast<int>(i) / dim, static_cast<int>(i) % dim, buf[i]);
          else out.embeddings.set(g, static_cast<int>(i) / 32, static_cast<int>(i) % 32, buf[i]);
        }
      }
    } else {
      std::vector<float> buf(count);
      f.read(reinterpret_cast<char*>(buf.data()), count * 4);
      if (!f) {
        err = "truncated payload";
        return false;
      }
      archNames.push_back(t.name);
      for (float v : buf) archFlat.push_back(v);
    }
  }
  if (!out.isFlex) {
    if (out.isInt8) out.qembeddings.quantizeFrom(out.embeddings, out.scales);
    if (out.isInt16) out.q16embeddings.quantizeFrom(out.embeddings, out.scales);
  } else {
    out.flexQ.configure(out.spec.tokenDim, out.isInt16);
    if (out.isInt8 || out.isInt16) out.flexQ.quantizeFrom(out.flexEmbeddings, out.layout, out.flexScales);
  }
  if (!out.arch->setTensors(archNames, archFlat)) {
    err = "arch tensor mismatch";
    return false;
  }
  return true;
}

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

bool saveFlexRuneFile(const std::string& path, const ModelSpec& spec,
                      const FlexEmbeddings& embeddings, const IArchitecture& arch,
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

}
