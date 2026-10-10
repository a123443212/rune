#include "core/go/go_model.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

namespace rune {
namespace go {

namespace {

std::string findString(const std::string& h, const std::string& key) {
  std::string pat = "\"" + key + "\":\"";
  size_t p = h.find(pat);
  if (p == std::string::npos) return "";
  p += pat.size();
  size_t q = h.find('"', p);
  if (q == std::string::npos) return "";
  return h.substr(p, q - p);
}

long findInt(const std::string& h, const std::string& key, long fallback) {
  std::string pat = "\"" + key + "\":";
  size_t p = h.find(pat);
  if (p == std::string::npos) return fallback;
  p += pat.size();
  size_t q = h.find_first_of(",}", p);
  if (q == std::string::npos) return fallback;
  try {
    return std::stol(h.substr(p, q - p));
  } catch (...) {
    return fallback;
  }
}

bool findFirstDouble(const std::string& h, const std::string& key, double& out) {
  std::string pat = "\"" + key + "\":";
  size_t p = h.find(pat);
  if (p == std::string::npos) return false;
  size_t v = p + pat.size();
  if (v >= h.size() || h[v] == '"') return false;
  size_t q = h.find_first_of(",}", v);
  if (q == std::string::npos) return false;
  try {
    out = std::stod(h.substr(v, q - v));
    return true;
  } catch (...) {
    return false;
  }
}

std::string pickStr(const std::string& h, const std::string& a, const std::string& b) {
  std::string v = findString(h, a);
  if (!v.empty()) return v;
  return findString(h, b);
}

}

bool loadGoResnetFile(const std::string& path, GoResnetFile& out, std::string& err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    err = "cannot open file";
    return false;
  }
  char magic[4];
  f.read(magic, 4);
  if (f.gcount() != 4 || std::memcmp(magic, "RUNE", 4) != 0) {
    err = "bad magic";
    return false;
  }
  uint32_t hlen = 0;
  f.read(reinterpret_cast<char*>(&hlen), 4);
  if (f.gcount() != 4 || hlen == 0 || hlen > (1u << 24)) {
    err = "bad header length";
    return false;
  }
  std::string header(static_cast<size_t>(hlen), '\0');
  f.read(&header[0], hlen);
  if (static_cast<uint32_t>(f.gcount()) != hlen) {
    err = "truncated header";
    return false;
  }
  long fmt = findInt(header, "format", 1);
  if (fmt != 1 && fmt != 2) {
    err = "unsupported format version";
    return false;
  }
  std::string arch = pickStr(header, "architecture_id", "arch");
  if (arch != "RUNE-RESNET-01") {
    err = "unsupported arch " + arch;
    return false;
  }
  std::string game = findString(header, "game");
  if (game.empty()) game = "chess";
  if (game != "go") {
    err = "game mismatch " + game;
    return false;
  }
  std::string feat = pickStr(header, "feature_version", "feature_set");
  if (feat != "go_planes_v01" && feat != "go_planes_v02") {
    err = "feature version mismatch " + feat;
    return false;
  }
  std::string quant = findString(header, "quantization");
  if (!quant.empty() && quant != "fp32") {
    err = "unsupported quantization " + quant;
    return false;
  }
  int board = static_cast<int>(findInt(header, "board_size", 0));
  if (board == 0) board = static_cast<int>(findInt(header, "tokens", 0));
  int channels = static_cast<int>(findInt(header, "channels", 0));
  if (channels == 0) channels = static_cast<int>(findInt(header, "token_dim", 0));
  int blocks = static_cast<int>(findInt(header, "num_blocks", 0));
  int inPlanes = static_cast<int>(findInt(header, "in_planes", 1));
  int h2 = static_cast<int>(findInt(header, "head_h2", 256));
  int policy = static_cast<int>(findInt(header, "policy_size", 0));
  if (board != 9 && board != 13 && board != 19) {
    err = "bad board size";
    return false;
  }
  if (channels < 1 || channels > 256) {
    err = "bad channels";
    return false;
  }
  if (blocks < 0 || blocks > 64) {
    err = "bad blocks";
    return false;
  }
  if (inPlanes < 1 || inPlanes > 16) {
    err = "bad in_planes";
    return false;
  }
  if (h2 < 1 || h2 > 4096) {
    err = "bad head_h2";
    return false;
  }
  if (policy <= 0) policy = board * board + 1;
  if (policy != board * board + 1) {
    err = "bad policy size";
    return false;
  }
  if (feat == "go_planes_v02" && inPlanes != 8) {
    err = "v02 wants 8 planes";
    return false;
  }
  if (feat == "go_planes_v01" && inPlanes != 1) {
    err = "v01 wants 1 plane";
    return false;
  }
  std::vector<std::string> order;
  order.push_back("stem_w");
  order.push_back("stem_b");
  for (int b = 0; b < blocks; ++b) {
    order.push_back("b" + std::to_string(b) + "_w1");
    order.push_back("b" + std::to_string(b) + "_b1");
    order.push_back("b" + std::to_string(b) + "_w2");
    order.push_back("b" + std::to_string(b) + "_b2");
  }
  order.push_back("vh1");
  order.push_back("bh1");
  order.push_back("wv");
  order.push_back("bv");
  order.push_back("wwdl");
  order.push_back("bwdl");
  order.push_back("wpol");
  order.push_back("bpol");
  std::vector<size_t> counts;
  counts.push_back(static_cast<size_t>(channels) * inPlanes * 9);
  counts.push_back(static_cast<size_t>(channels));
  for (int b = 0; b < blocks; ++b) {
    (void)b;
    counts.push_back(static_cast<size_t>(channels) * channels * 9);
    counts.push_back(static_cast<size_t>(channels));
    counts.push_back(static_cast<size_t>(channels) * channels * 9);
    counts.push_back(static_cast<size_t>(channels));
  }
  counts.push_back(static_cast<size_t>(h2) * channels);
  counts.push_back(static_cast<size_t>(h2));
  counts.push_back(static_cast<size_t>(h2));
  counts.push_back(1);
  counts.push_back(static_cast<size_t>(3) * h2);
  counts.push_back(3);
  counts.push_back(static_cast<size_t>(policy) * channels * board * board);
  counts.push_back(static_cast<size_t>(policy));
  std::vector<float> flat;
  flat.reserve(1024);
  for (size_t t = 0; t < order.size(); ++t) {
    size_t need = counts[t] * 4;
    std::vector<char> buf(need);
    f.read(buf.data(), static_cast<std::streamsize>(need));
    if (static_cast<size_t>(f.gcount()) != need) {
      err = "truncated payload " + order[t];
      return false;
    }
    size_t base = flat.size();
    flat.resize(base + counts[t]);
    std::memcpy(&flat[base], buf.data(), need);
  }
  GoResnetWeights wt;
  GoResnetSizes sz;
  sz.board = board;
  sz.channels = channels;
  sz.blocks = blocks;
  sz.policySize = policy;
  sz.valueH2 = h2;
  sz.inPlanes = inPlanes;
  size_t pos = 0;
  auto take = [&](size_t n, std::vector<float>& v) {
    v.assign(flat.begin() + pos, flat.begin() + pos + n);
    pos += n;
  };
  take(counts[0], wt.stemW);
  take(counts[1], wt.stemB);
  wt.blockW1.resize(static_cast<size_t>(blocks));
  wt.blockB1.resize(static_cast<size_t>(blocks));
  wt.blockW2.resize(static_cast<size_t>(blocks));
  wt.blockB2.resize(static_cast<size_t>(blocks));
  size_t k = 2;
  for (int b = 0; b < blocks; ++b) {
    take(counts[k++], wt.blockW1[static_cast<size_t>(b)]);
    take(counts[k++], wt.blockB1[static_cast<size_t>(b)]);
    take(counts[k++], wt.blockW2[static_cast<size_t>(b)]);
    take(counts[k++], wt.blockB2[static_cast<size_t>(b)]);
  }
  take(counts[k++], wt.vh1);
  take(counts[k++], wt.bh1);
  take(counts[k++], wt.wv);
  std::vector<float> bv;
  take(counts[k++], bv);
  wt.bv = bv[0];
  take(counts[k++], wt.wwdl);
  take(counts[k++], wt.bwdl);
  take(counts[k++], wt.wpol);
  take(counts[k++], wt.bpol);
  out.weights = wt;
  out.sizes = sz;
  out.featureVersion = feat;
  return true;
}

}
}
