#include "core/model_io/model_io.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>

#include "core/architectures/attention/attention.h"
#include "core/architectures/mlp/mlp.h"

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

double extractScale(const std::string& h, const std::string& key) {
  std::string pat = "\"" + key + "\":";
  size_t p = h.find(pat);
  if (p == std::string::npos) return 1.0;
  p += pat.size();
  size_t q = h.find_first_of(",}", p);
  return std::stod(h.substr(p, q - p));
}

}  // namespace

std::unique_ptr<IArchitecture> createArchitecture(const std::string& archId) {
  if (archId == "RUNE-MLP") return std::unique_ptr<IArchitecture>(new GroupedMlp());
  if (archId == "RUNE-SFNN") return std::unique_ptr<IArchitecture>(new SfnnBaseline());
  if (archId == "RUNE-ATTN") return std::unique_ptr<IArchitecture>(new RuneAttnModel(false));
  if (archId == "RUNE-ATTN-GAB") return std::unique_ptr<IArchitecture>(new RuneAttnModel(true));
  return nullptr;
}

bool loadRuneFile(const std::string& path, RuneFile& out, std::string& err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    err = "cannot open file";
    return false;
  }
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
  std::string header(hlen, '\0');
  f.read(header.data(), hlen);
  std::string arch = extractString(header, "arch");
  out.arch = createArchitecture(arch);
  if (!out.arch) {
    err = "unknown arch " + arch;
    return false;
  }
  out.spec.arch = arch;
  out.spec.archVersion = extractString(header, "arch_version");
  out.spec.featureSet = extractString(header, "feature_set");
  out.spec.attention = extractString(header, "attention");
  out.spec.geometricBias = extractString(header, "geometric_bias");
  out.spec.head = extractString(header, "head");
  out.spec.quantization = extractString(header, "quantization");
  out.isInt8 = (out.spec.quantization == "int8");
  std::vector<TensorMeta> metas;
  if (!parseTensors(header, metas)) {
    err = "tensor list parse failed";
    return false;
  }
  std::vector<std::string> archNames;
  std::vector<float> archFlat;
  for (const TensorMeta& t : metas) {
    size_t count = 1;
    for (int s : t.shape) count *= static_cast<size_t>(s);
    if (t.name.rfind("emb", 0) == 0) {
      int g = std::stoi(t.name.substr(3));
      if (t.dtype == "int8") {
        std::vector<char> buf(count);
        f.read(buf.data(), count);
        double scale = extractScale(header, t.name);
        out.scales.embedding[g] = static_cast<float>(scale);
        for (size_t i = 0; i < count; ++i) {
          int r = static_cast<int>(i) / 32;
          int d = static_cast<int>(i) % 32;
          float v = static_cast<float>(buf[i]) * static_cast<float>(scale);
          out.embeddings.set(g, r, d, v);
        }
      } else {
        std::vector<float> buf(count);
        f.read(reinterpret_cast<char*>(buf.data()), count * 4);
        for (size_t i = 0; i < count; ++i) {
          out.embeddings.set(g, static_cast<int>(i) / 32, static_cast<int>(i) % 32, buf[i]);
        }
        out.scales.embedding[g] = 1.0f;
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
  if (out.isInt8) {
    out.qembeddings.quantizeFrom(out.embeddings, out.scales);
  }
  if (!out.arch->setTensors(archNames, archFlat)) {
    err = "arch tensor mismatch";
    return false;
  }
  return true;
}

bool saveRuneFile(const std::string& path, const ModelSpec& spec, const EmbeddingTables& embeddings,
                  const IArchitecture& arch, const std::string& quantization, std::string& err) {
  std::vector<std::string> names;
  std::vector<std::vector<int>> shapes;
  std::vector<const float*> data;
  arch.getTensors(names, shapes, data);
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
  h << ",\"scales\":{";
  QuantEmbeddingTables qe;
  QuantScales sc;
  if (quantization == "int8") qe.quantizeFrom(embeddings, sc);
  for (int g = 0; g < 8; ++g) {
    if (g > 0) h << ",";
    h << "\"emb" << g << "\":";
    if (quantization == "int8") h << sc.embedding[g];
    else h << "1.0";
  }
  h << "},\"tensors\":[";
  bool first = true;
  auto emit = [&](const std::string& n, const std::string& shape, const std::string& dtype) {
    if (!first) h << ",";
    first = false;
    h << "{\"name\":\"" << n << "\",\"shape\":" << shape << ",\"dtype\":\"" << dtype << "\"}";
  };
  for (int g = 0; g < 8; ++g) {
    int v = GroupedFeatureSet::vocabSize(g);
    emit("emb" + std::to_string(g), "[" + std::to_string(v) + ",32]",
         quantization == "int8" ? "int8" : "float32");
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
