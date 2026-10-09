#include "core/model_io/model_io.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <new>
#include <set>
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

bool extractBool(const std::string& h, const std::string& key, bool fallback) {
  std::string pat = "\"" + key + "\":";
  size_t p = h.find(pat);
  if (p == std::string::npos) return fallback;
  size_t v = p + pat.size();
  size_t q = h.find_first_of(",}", v);
  std::string val = h.substr(v, q - v);
  if (val == "true" || val == "1") return true;
  if (val == "false" || val == "0") return false;
  try {
    return std::stod(val) != 0.0;
  } catch (...) {
    return fallback;
  }
}

struct TensorMeta {
  std::string name;
  std::vector<int> shape;
  std::string dtype;
};

bool parseDecimalInt(const std::string& s, int& out) {
  size_t b = 0;
  size_t e = s.size();
  while (b < e && (s[b] == ' ' || s[b] == '\t')) ++b;
  while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t')) --e;
  if (b >= e || e - b > 10) return false;
  bool neg = false;
  if (s[b] == '-' || s[b] == '+') {
    neg = (s[b] == '-');
    ++b;
    if (b >= e) return false;
  }
  long v = 0;
  for (size_t i = b; i < e; ++i) {
    if (s[i] < '0' || s[i] > '9') return false;
    v = v * 10 + (s[i] - '0');
    if (v > 2000000000L) return false;
  }
  out = neg ? -static_cast<int>(v) : static_cast<int>(v);
  return true;
}

bool parseGroupIndex(const std::string& name, int& g) {
  if (name.size() < 4 || name.compare(0, 3, "emb") != 0) return false;
  if (!parseDecimalInt(name.substr(3), g)) return false;
  return g >= 0 && g < 9;
}

bool parseTensors(const std::string& h, std::vector<TensorMeta>& out) {
  size_t p = h.find("\"tensor_metadata\":[");
  if (p == std::string::npos) p = h.find("\"tensors\":[");
  if (p == std::string::npos) return false;
  p = h.find('[', p);
  p += 1;
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
    size_t qn = obj.find('"', pn);
    if (qn == std::string::npos) return false;
    t.name = obj.substr(pn, qn - pn);
    size_t ps = obj.find("\"shape\":[");
    if (ps == std::string::npos) return false;
    ps += 9;
    size_t pe = obj.find(']', ps);
    if (pe == std::string::npos) return false;
    std::string dims = obj.substr(ps, pe - ps);
    std::stringstream ss(dims);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      if (!tok.empty()) {
        int dv = 0;
        if (!parseDecimalInt(tok, dv)) return false;
        t.shape.push_back(dv);
      }
    }
    size_t pd = obj.find("\"dtype\":\"");
    if (pd == std::string::npos) return false;
    pd += 9;
    size_t qd = obj.find('"', pd);
    if (qd == std::string::npos) return false;
    t.dtype = obj.substr(pd, qd - pd);
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
  if (!f) {
    err = "truncated header";
    return false;
  }
  return true;
}

bool parseIntList(const std::string& header, const std::string& key, std::vector<int>& out) {
  std::string pat = "\"" + key + "\":[";
  size_t p = header.find(pat);
  if (p == std::string::npos) return false;
  p += pat.size();
  size_t q = header.find(']', p);
  if (q == std::string::npos) return false;
  std::stringstream ss(header.substr(p, q - p));
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.empty()) continue;
    try {
      out.push_back(std::stoi(tok));
    } catch (...) {
      return false;
    }
  }
  return true;
}

bool isDenseArch(const std::string& arch) { return arch.rfind("RUNE-03-", 0) == 0; }

bool isAdaptiveArch(const std::string& arch) { return arch == "RUNE-04"; }

bool isUncertaintyArch(const std::string& arch) { return arch == "RUNE-05"; }

bool parseIntPairs(const std::string& header, const std::string& key,
                   std::vector<std::pair<int, int>>& out) {
  std::string pat = "\"" + key + "\":";
  size_t p = header.find(pat);
  if (p == std::string::npos) return true;
  p += pat.size();
  if (header.compare(p, 4, "null") == 0) return true;
  if (header.compare(p, 2, "[]") == 0) return true;
  size_t q = header.find(']', p);
  if (q == std::string::npos) return false;
  size_t end = header.find(']', q + 1);
  if (end == std::string::npos) return false;
  std::string inner = header.substr(p, end - p + 1);
  size_t pos = 0;
  while (true) {
    size_t a = inner.find('[', pos);
    if (a == std::string::npos) break;
    if (a == 0) {
      pos = 1;
      continue;
    }
    size_t b = inner.find(']', a);
    if (b == std::string::npos) return false;
    std::string pair = inner.substr(a + 1, b - a - 1);
    size_t c = pair.find(',');
    if (c == std::string::npos) return false;
    try {
      out.push_back({std::stoi(pair.substr(0, c)), std::stoi(pair.substr(c + 1))});
    } catch (...) {
      return false;
    }
    pos = b + 1;
  }
  return true;
}

bool hasNonNull(const std::string& header, const std::string& key) {
  std::string pat = "\"" + key + "\":";
  size_t p = header.find(pat);
  if (p == std::string::npos) return false;
  p += pat.size();
  return header.compare(p, 4, "null") != 0;
}

std::string pickString(const std::string& h, const std::string& a, const std::string& b) {
  std::string v = extractString(h, a);
  if (!v.empty()) return v;
  return extractString(h, b);
}

void fillSpec(const std::string& header, ModelSpec& spec) {
  spec.arch = pickString(header, "architecture_id", "arch");
  spec.archVersion = pickString(header, "architecture_version", "arch_version");
  spec.featureSet = pickString(header, "feature_version", "feature_set");
  spec.tokens = static_cast<int>(extractInt(header, "tokens", 8));
  spec.tokenDim = static_cast<int>(extractInt(header, "token_dim", 32));
  spec.attention = extractString(header, "attention");
  spec.geometricBias = extractString(header, "geometric_bias");
  spec.head = extractString(header, "head");
  spec.quantization = extractString(header, "quantization");
  std::string gate = extractString(header, "gate");
  spec.gate = gate.empty() ? "clip" : gate;
  spec.alpha = static_cast<float>(extractNumber(header, "alpha", 1.0));
  spec.variant = extractString(header, "variant");
  spec.pooling = extractString(header, "pooling");
  if (spec.pooling.empty()) spec.pooling = "none";
  spec.gateOn = extractBool(header, "gate_on", false);
  spec.headH1 = static_cast<int>(extractInt(header, "head_h1", 128));
  spec.headH2 = static_cast<int>(extractInt(header, "head_h2", 32));
  spec.headBuckets = static_cast<int>(extractInt(header, "head_buckets", 1));
  spec.tokenDims.clear();
  std::vector<int> td;
  if (parseIntList(header, "token_dims", td)) spec.tokenDims = td;
  spec.cheapPooling = extractString(header, "cheap_pooling");
  if (spec.cheapPooling.empty()) spec.cheapPooling = "none";
  spec.threshold = static_cast<float>(extractNumber(header, "threshold", 0.5));
  spec.tHigh = static_cast<float>(extractNumber(header, "t_high", spec.threshold));
  spec.hasTLow = hasNonNull(header, "t_low");
  spec.tLow = static_cast<float>(extractNumber(header, "t_low", spec.threshold));
  spec.refinePrecision = extractString(header, "refine_precision");
  if (spec.refinePrecision.empty()) spec.refinePrecision = "fp32";
  spec.prunedPairs.clear();
  std::vector<std::pair<int, int>> pp;
  if (parseIntPairs(header, "pruned_pairs", pp)) spec.prunedPairs = pp;
  spec.cheapHidden = static_cast<int>(extractInt(header, "cheap_hidden", 32));
  spec.refH1 = static_cast<int>(extractInt(header, "ref_h1", 128));
  spec.refH2 = static_cast<int>(extractInt(header, "ref_h2", 32));
  spec.hasUncertainty = extractBool(header, "uncertainty", false);
  spec.hasStabilityHead = extractBool(header, "stability_head", false);
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
  long fmt = extractInt(header, "format", 1);
  if (fmt != 1 && fmt != 2) {
    err = "unsupported format version";
    return false;
  }
  fillSpec(header, out.spec);
  if (out.spec.headBuckets != 1 && out.spec.headBuckets != 3) {
    err = "unsupported head_buckets (want 1 or 3)";
    return false;
  }
  if (out.spec.featureSet != "grouped_hkav2_fullthreats_v02") {
    err = "feature version mismatch: " + out.spec.featureSet;
    return false;
  }
  if (!isSupportedVersion(out.spec.archVersion)) {
    err = "unsupported arch version " + out.spec.archVersion;
    return false;
  }
  if (out.spec.quantization != "fp32" && out.spec.quantization != "int8" &&
      out.spec.quantization != "int16") {
    err = "unsupported quantization " + out.spec.quantization;
    return false;
  }
  const int* hyperDims[] = {&out.spec.headH1, &out.spec.headH2, &out.spec.cheapHidden,
                            &out.spec.refH1, &out.spec.refH2};
  for (const int* hp : hyperDims) {
    if (*hp < 1 || *hp > 4096) {
      err = "hyperparameter dim out of range";
      return false;
    }
  }
  std::string arch = out.spec.arch;
  out.isInt8 = (out.spec.quantization == "int8");
  out.isInt16 = (out.spec.quantization == "int16");
  out.isFlex = (arch == "RUNE-REL-02");
  out.isDense = isDenseArch(arch);
  out.isAdaptive = isAdaptiveArch(arch);
  out.isUncertainty = isUncertaintyArch(arch);
  const bool useVar =
      out.isDense || out.isAdaptive || out.isUncertainty;
  if (out.isFlex) {
    if (out.spec.tokens != 6 && out.spec.tokens != 8 && out.spec.tokens != 10) {
      err = "token count mismatch";
      return false;
    }
  } else if (out.spec.tokens != 8) {
    err = "token count mismatch";
    return false;
  }
  if (!out.isFlex && !useVar && out.spec.tokenDim != 32) {
    err = "token dim mismatch";
    return false;
  }
  std::vector<TensorMeta> metas;
  if (!parseTensors(header, metas)) {
    err = "tensor list parse failed";
    return false;
  }
  TokenLayout layout;
  FlexEmbeddings* flexEmb = nullptr;
  VarEmbeddings* varEmb = nullptr;
  VarWidths varWidths;
  VarWidths embWidths;
  if (out.isDense) {
    DenseBuildSpec ds;
    ds.variant = arch.substr(8);
    std::vector<int> dims;
    if (!parseIntList(header, "token_dims", dims)) {
      err = "missing token_dims";
      return false;
    }
    ds.dims = dims;
    ds.pooling = out.spec.pooling;
    ds.poolClip = extractBool(header, "pool_clip", true);
    ds.gateOn = out.spec.gateOn;
    ds.sharedWidth = static_cast<int>(extractInt(header, "shared_width", 32));
    ds.headH1 = out.spec.headH1;
    ds.headH2 = out.spec.headH2;
    if (ds.sharedWidth < 1 || ds.sharedWidth > 64) {
      err = "hyperparameter dim out of range";
      return false;
    }
    out.arch = createDense(ds, err);
    if (!out.arch) {
      if (err.empty()) err = "dense build failed";
      return false;
    }
    if (!VarWidths::make(dims, varWidths, err)) return false;
    out.varWidths = varWidths;
    VarWidths gw = varWidths;
    if (ds.pooling == "shared") {
      for (int g = 0; g < 9; ++g) gw.w[g] = ds.sharedWidth;
    }
    embWidths = gw;
    out.varEmbeddings.configure(gw);
    varEmb = &out.varEmbeddings;
  } else if (out.isAdaptive || out.isUncertainty) {
    int dim = out.spec.tokenDim;
    if (dim < 8 || dim > 64) {
      err = "bad adaptive dim";
      return false;
    }
    AdaptiveBuildSpec as;
    as.dim = dim;
    as.cheapPooling = out.spec.cheapPooling;
    as.alpha = out.spec.alpha;
    as.threshold = out.spec.threshold;
    as.tHigh = out.spec.tHigh;
    as.hasTLow = out.spec.hasTLow;
    as.tLow = out.spec.tLow;
    as.prunedPairs = out.spec.prunedPairs;
    as.refinePrecision = out.spec.refinePrecision;
    as.hasUncertainty = out.isUncertainty;
    as.hasStabilityHead = out.spec.hasStabilityHead;
    as.cheapHidden = out.spec.cheapHidden;
    as.refH1 = out.spec.refH1;
    as.refH2 = out.spec.refH2;
    out.arch = createAdaptive(as, err);
    if (!out.arch) {
      if (err.empty()) err = "adaptive build failed";
      return false;
    }
    for (int g = 0; g < 9; ++g) varWidths.w[g] = dim;
    out.varWidths = varWidths;
    VarWidths gw = varWidths;
    if (as.cheapPooling == "shared") {
      for (int g = 0; g < 9; ++g) gw.w[g] = 32;
    }
    embWidths = gw;
    out.varEmbeddings.configure(gw);
    varEmb = &out.varEmbeddings;
  } else if (out.isFlex) {
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
  std::set<std::string> seenNames;
  size_t totalBytes = 0;
  try {
  for (const TensorMeta& t : metas) {
    if (!seenNames.insert(t.name).second) {
      err = "duplicate tensor " + t.name;
      return false;
    }
    size_t count = 1;
    for (int s : t.shape) {
      if (s < 0 || s > 1000000) {
        err = "shape value out of range";
        return false;
      }
      size_t prev = count;
      count *= static_cast<size_t>(s);
      if (s != 0 && count / static_cast<size_t>(s) != prev) {
        err = "tensor size overflow";
        return false;
      }
    }
    if (count == 0 || count > 200000000) {
      err = "tensor size out of range";
      return false;
    }
    if (t.dtype != "float32" && t.dtype != "int8" && t.dtype != "int16") {
      err = "unsupported dtype " + t.dtype;
      return false;
    }
    size_t elemBytes = t.dtype == "float32" ? 4 : (t.dtype == "int16" ? 2 : 1);
    totalBytes += count * elemBytes;
    if (totalBytes > 800000000) {
      err = "payload too large";
      return false;
    }
    if (t.name.rfind("emb", 0) == 0) {
      int g = -1;
      if (!parseGroupIndex(t.name, g)) {
        err = "bad embedding tensor name " + t.name;
        return false;
      }
      int wantCols = useVar ? embWidths.w[g] : (out.isFlex ? dim : 32);
      if (t.shape.size() != 2 || t.shape[0] != GroupedFeatureSet::vocabSize(g) ||
          t.shape[1] != wantCols) {
        err = "embedding shape mismatch " + t.name;
        return false;
      }
      double scale = extractNumber(header, t.name, 1.0);
      if (t.dtype == "int8" || t.dtype == "int16") {
        std::string spat = "\"" + t.name + "\":";
        if (header.find(spat) == std::string::npos) {
          err = "missing embedding scale " + t.name;
          return false;
        }
        if (!(scale > 0.0) || !(scale == scale)) {
          err = "bad embedding scale " + t.name;
          return false;
        }
        if (t.dtype == "int8") {
          std::vector<int8_t> buf(count);
          f.read(reinterpret_cast<char*>(buf.data()), count);
          if (!f) {
            err = "truncated payload";
            return false;
          }
          for (size_t i = 0; i < count; ++i) {
            float v = static_cast<float>(buf[i]) * static_cast<float>(scale);
            if (useVar) {
              int gw = embWidths.w[g];
              varEmb->set(g, static_cast<int>(i) / gw, static_cast<int>(i) % gw, v);
            } else if (out.isFlex) flexEmb->set(g, static_cast<int>(i) / dim, static_cast<int>(i) % dim, v);
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
            if (useVar) {
              int gw = embWidths.w[g];
              varEmb->set(g, static_cast<int>(i) / gw, static_cast<int>(i) % gw, v);
            } else if (out.isFlex) flexEmb->set(g, static_cast<int>(i) / dim, static_cast<int>(i) % dim, v);
            else out.embeddings.set(g, static_cast<int>(i) / 32, static_cast<int>(i) % 32, v);
          }
        }
        if (useVar) out.varScales.token[g] = static_cast<float>(scale);
        else if (out.isFlex) out.flexScales.embedding[g] = static_cast<float>(scale);
        else out.scales.embedding[g] = static_cast<float>(scale);
      } else {
        std::vector<float> buf(count);
        f.read(reinterpret_cast<char*>(buf.data()), count * 4);
        if (!f) {
          err = "truncated payload";
          return false;
        }
        for (size_t i = 0; i < count; ++i) {
          if (useVar) {
            int gw = embWidths.w[g];
            varEmb->set(g, static_cast<int>(i) / gw, static_cast<int>(i) % gw, buf[i]);
          } else if (out.isFlex) flexEmb->set(g, static_cast<int>(i) / dim, static_cast<int>(i) % dim, buf[i]);
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
  if (f.peek() != std::ifstream::traits_type::eof()) {
    err = "trailing payload bytes";
    return false;
  }
  } catch (const std::bad_alloc&) {
    err = "oversized allocation";
    return false;
  }
  if (useVar) {
    out.varQ.configure(embWidths, out.isInt16);
    if (out.isInt8 || out.isInt16) out.varQ.quantizeFrom(out.varEmbeddings, out.varScales);
  } else if (!out.isFlex) {
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
  {
    std::string cs = pickString(header, "model_hash", "checksum");
    bool needHash = (fmt == 2) || useVar;
    if (needHash && cs.empty()) {
      err = "model missing checksum";
      return false;
    }
    if (!cs.empty()) {
      uint64_t want = 0;
      try {
        want = std::stoull(cs, nullptr, 16);
      } catch (...) {
        err = "bad checksum format";
        return false;
      }
      std::ifstream g(path, std::ios::binary);
      if (!g) {
        err = "cannot reopen file";
        return false;
      }
      g.seekg(0, std::ios::end);
      std::streampos end = g.tellg();
      std::streampos start = 8 + static_cast<std::streampos>(header.size());
      if (end < start) {
        err = "bad payload range";
        return false;
      }
      size_t n = static_cast<size_t>(end - start);
      if (n > 800000000) {
        err = "payload too large";
        return false;
      }
      std::string payload(n, '\0');
      g.seekg(start);
      g.read(payload.data(), n);
      if (!g) {
        err = "cannot read payload";
        return false;
      }
      uint64_t got = fnv1aHash(reinterpret_cast<const uint8_t*>(payload.data()), n);
      if (got != want) {
        err = "checksum mismatch";
        return false;
      }
    }
  }
  return true;
}

}
