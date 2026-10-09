#include "core/shogi/shogi_board.h"

#include <cctype>
#include <vector>

namespace rune {
namespace shogi {

namespace {

int handTypeIndex(char c) {
  switch (c) {
    case 'p': return 0;
    case 'l': return 1;
    case 'n': return 2;
    case 's': return 3;
    case 'g': return 4;
    case 'b': return 5;
    case 'r': return 6;
    default: return -1;
  }
}

int boardKind(char c, bool promo) {
  int base = handTypeIndex(c);
  if (base < 0) {
    if (c == 'k') return kKing;
    return -1;
  }
  if (!promo) return base;
  switch (base) {
    case 0: return kProPawn;
    case 1: return kProLance;
    case 2: return kProKnight;
    case 3: return kProSilver;
    case 5: return kHorse;
    case 6: return kDragon;
    default: return -1;
  }
}

void stepMoves(uint8_t kind, uint8_t color, int* dfs, int* drs, int* n) {
  int f = (color == kBlack) ? -1 : 1;
  *n = 0;
  auto push = [&](int df, int dr) {
    dfs[*n] = df;
    drs[*n] = dr;
    (*n)++;
  };
  switch (kind) {
    case kPawn: push(0, f); break;
    case kKnight: push(-1, 2 * f); push(1, 2 * f); break;
    case kSilver: push(0, f); push(-1, f); push(1, f); push(-1, -f); push(1, -f); break;
    case kGold:
    case kProPawn:
    case kProLance:
    case kProKnight:
    case kProSilver:
      push(0, f); push(-1, f); push(1, f); push(-1, 0); push(1, 0); push(0, -f);
      break;
    case kKing:
      push(-1, -1); push(0, -1); push(1, -1); push(-1, 0);
      push(1, 0); push(-1, 1); push(0, 1); push(1, 1);
      break;
    case kHorse: push(-1, 0); push(1, 0); push(0, -1); push(0, 1); break;
    case kDragon: push(-1, -1); push(1, -1); push(-1, 1); push(1, 1); break;
    default: break;
  }
}

int slideDirs(uint8_t kind, int* dfs, int* drs) {
  switch (kind) {
    case kBishop:
    case kHorse:
      dfs[0] = -1; drs[0] = -1; dfs[1] = 1; drs[1] = -1;
      dfs[2] = -1; drs[2] = 1; dfs[3] = 1; drs[3] = 1;
      return 4;
    case kRook:
    case kDragon:
      dfs[0] = -1; drs[0] = 0; dfs[1] = 1; drs[1] = 0;
      dfs[2] = 0; drs[2] = -1; dfs[3] = 0; drs[3] = 1;
      return 4;
    default: return 0;
  }
}

bool parseRank(const std::string& rank, int row, std::array<ShogiPiece, 81>& sq) {
  int f = 0;
  for (size_t i = 0; i < rank.size(); ++i) {
    char c = rank[i];
    if (c >= '0' && c <= '9') {
      f += c - '0';
    } else {
      bool promo = false;
      if (c == '+') {
        promo = true;
        if (++i >= rank.size()) return false;
        c = rank[i];
      }
      if (c < 'A' || (c > 'Z' && c < 'a') || c > 'z') return false;
      uint8_t color = (c <= 'Z') ? kBlack : kWhite;
      char lo = static_cast<char>(c | 32);
      int kind = boardKind(lo, promo);
      if (kind < 0 || f >= 9) return false;
      ShogiPiece p;
      p.kind = static_cast<uint8_t>(kind);
      p.color = color;
      p.present = true;
      sq[static_cast<size_t>(row * 9 + f)] = p;
      f++;
    }
  }
  return f == 9;
}

std::vector<std::string> splitWs(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == ' ' || c == '\t') {
      if (!cur.empty()) {
        out.push_back(cur);
        cur.clear();
      }
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

}  // namespace

ShogiBoard::ShogiBoard() {
  for (auto& p : squares_) p = ShogiPiece();
}

ShogiBoard::ShogiBoard(const std::string& sfen) : ShogiBoard() { setSfen(sfen); }

bool ShogiBoard::setSfen(const std::string& sfen) {
  for (auto& p : squares_) p = ShogiPiece();
  side_ = kBlack;
  for (int c = 0; c < 2; ++c)
    for (int t = 0; t < kHandTypes; ++t) hand_[c][t] = 0;
  moveNo_ = 1;
  std::vector<std::string> parts = splitWs(sfen);
  if (parts.empty()) return false;
  std::vector<std::string> ranks;
  std::string cur;
  for (char c : parts[0]) {
    if (c == '/') {
      ranks.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  ranks.push_back(cur);
  if (ranks.size() != 9) return false;
  for (int r = 0; r < 9; ++r) {
    if (!parseRank(ranks[static_cast<size_t>(r)], r, squares_)) return false;
  }
  if (parts.size() > 1) {
    if (parts[1] == "b") side_ = kBlack;
    else if (parts[1] == "w") side_ = kWhite;
    else return false;
  }
  if (parts.size() > 2 && parts[2] != "-") {
    int num = 0;
    bool haveNum = false;
    for (char c : parts[2]) {
      if (c >= '0' && c <= '9') {
        num = num * 10 + (c - '0');
        haveNum = true;
      } else {
        int n = haveNum ? num : 1;
        num = 0;
        haveNum = false;
        if (c < 'A' || (c > 'Z' && c < 'a') || c > 'z') return false;
        uint8_t color = (c <= 'Z') ? 0 : 1;
        int ti = handTypeIndex(static_cast<char>(c | 32));
        if (ti < 0) return false;
        int v = static_cast<int>(hand_[color][ti]) + n;
        if (v > 255) v = 255;
        hand_[color][ti] = static_cast<uint8_t>(v);
      }
    }
  }
  if (parts.size() > 3) {
    int m = 0;
    for (char c : parts[3]) {
      if (c < '0' || c > '9') break;
      m = m * 10 + (c - '0');
    }
    if (m >= 1) moveNo_ = static_cast<uint32_t>(m);
  }
  return true;
}

uint32_t ShogiBoard::handTotal() const {
  uint32_t n = 0;
  for (int c = 0; c < 2; ++c)
    for (int t = 0; t < kHandTypes; ++t) n += hand_[c][t];
  return n;
}

uint32_t ShogiBoard::promoCount() const {
  uint32_t n = 0;
  for (const auto& p : squares_) {
    if (p.present && p.kind >= kProPawn) n++;
  }
  return n;
}

bool shogiAttacks(const ShogiBoard& board, int from, int to) {
  ShogiPiece pc = board.at(from);
  if (!pc.present) return false;
  int ff = sqFile(from), rf = sqRank(from);
  int tf = sqFile(to), rt = sqRank(to);
  if (pc.kind == kLance) {
    int f = (pc.color == kBlack) ? -1 : 1;
    if (tf != ff) return false;
    int step = (rt > rf) ? 1 : -1;
    if (step != f) return false;
    for (int r = rf + step; r != rt; r += step) {
      if (board.at(makeSq(tf, r)).present) return false;
    }
    return true;
  }
  int dfs[8], drs[8], n = 0;
  stepMoves(pc.kind, pc.color, dfs, drs, &n);
  for (int i = 0; i < n; ++i) {
    if (ff + dfs[i] == tf && rf + drs[i] == rt) return true;
  }
  int sdfs[4], sdrs[4];
  int nd = slideDirs(pc.kind, sdfs, sdrs);
  for (int i = 0; i < nd; ++i) {
    int f = ff + sdfs[i], r = rf + sdrs[i];
    while (onBoard(f, r)) {
      if (f == tf && r == rt) return true;
      if (board.at(makeSq(f, r)).present) break;
      f += sdfs[i];
      r += sdrs[i];
    }
  }
  return false;
}

}
}
