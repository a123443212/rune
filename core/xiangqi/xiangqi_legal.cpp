#include "core/xiangqi/xiangqi_legal.h"

#include <algorithm>

#include "core/xiangqi/xiangqi_moves.h"

namespace rune {
namespace xiangqi {

namespace {

bool parseTwo(const std::string& s, size_t pos, int& v) {
  if (pos + 2 > s.size()) return false;
  char a = s[pos];
  char b = s[pos + 1];
  if (a < '0' || a > '9' || b < '0' || b > '9') return false;
  v = (a - '0') * 10 + (b - '0');
  return v >= 0 && v < 90;
}

std::string twoDigits(int v) {
  std::string s;
  s += static_cast<char>('0' + v / 10);
  s += static_cast<char>('0' + v % 10);
  return s;
}

bool destHas(const std::vector<int>& dests, int to) {
  for (int d : dests) {
    if (d == to) return true;
  }
  return false;
}

std::string encodeGrid(const XiangqiBoard& board) {
  std::string grid;
  for (int r = 9; r >= 0; --r) {
    if (r != 9) grid += "/";
    std::string rank;
    int gap = 0;
    for (int f = 0; f < 9; ++f) {
      XiangqiPiece p = board.at(makeSq(f, r));
      if (!p.present) {
        ++gap;
        continue;
      }
      if (gap > 0) {
        rank += static_cast<char>('0' + gap);
        gap = 0;
      }
      char ch = 'k';
      if (p.kind == kPawn) ch = 'p';
      else if (p.kind == kHorse) ch = 'h';
      else if (p.kind == kRook) ch = 'r';
      else if (p.kind == kCannon) ch = 'c';
      else if (p.kind == kAdvisor) ch = 'a';
      else if (p.kind == kElephant) ch = 'e';
      rank += (p.color == kRed) ? static_cast<char>(ch - 32) : ch;
    }
    if (gap > 0) rank += static_cast<char>('0' + gap);
    grid += rank;
  }
  return grid;
}

}

std::string xiangqiEncodeMove(int fr, int to) { return twoDigits(fr) + twoDigits(to); }

bool xiangqiDecodeMove(const std::string& mv, int& fr, int& to) {
  if (mv.size() != 4) return false;
  return parseTwo(mv, 0, fr) && parseTwo(mv, 2, to);
}

bool xiangqiKingInCheck(const XiangqiBoard& board, uint8_t color) {
  int kus = -1;
  for (int sq = 0; sq < 90; ++sq) {
    XiangqiPiece p = board.at(sq);
    if (p.present && p.kind == kKing && p.color == color) {
      kus = sq;
      break;
    }
  }
  if (kus < 0) return true;
  for (int asq = 0; asq < 90; ++asq) {
    XiangqiPiece p = board.at(asq);
    if (!p.present || p.color == color) continue;
    if (xiangqiAttacks(board, asq, kus)) return true;
  }
  return false;
}

bool xiangqiLegalMoves(const std::string& state, std::vector<std::string>& out) {
  out.clear();
  XiangqiBoard board;
  if (!board.setFen(state)) return false;
  uint8_t stm = board.sideToMove();
  std::vector<int> dests;
  for (int fr = 0; fr < 90; ++fr) {
    XiangqiPiece p = board.at(fr);
    if (!p.present || p.color != stm) continue;
    xiangqiPseudoDests(board, fr, dests);
    for (int to : dests) {
      XiangqiPiece t = board.at(to);
      if (t.present && t.color == stm) continue;
      XiangqiBoard nb = board;
      nb.setSquare(to, p);
      nb.setSquare(fr, XiangqiPiece());
      if (xiangqiKingInCheck(nb, stm)) continue;
      out.push_back(xiangqiEncodeMove(fr, to));
    }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return true;
}

bool xiangqiApplyMove(const std::string& state, const std::string& mv, std::string& out) {
  int fr = -1;
  int to = -1;
  if (!xiangqiDecodeMove(mv, fr, to)) return false;
  XiangqiBoard board;
  if (!board.setFen(state)) return false;
  uint8_t stm = board.sideToMove();
  XiangqiPiece p = board.at(fr);
  if (!p.present || p.color != stm) return false;
  std::vector<int> dests;
  xiangqiPseudoDests(board, fr, dests);
  if (!destHas(dests, to)) return false;
  XiangqiPiece t = board.at(to);
  if (t.present && t.color == stm) return false;
  XiangqiBoard nb = board;
  nb.setSquare(to, p);
  nb.setSquare(fr, XiangqiPiece());
  if (xiangqiKingInCheck(nb, stm)) return false;
  uint8_t nstm = (stm == kRed) ? kBlack : kRed;
  std::string side = (nstm == kRed) ? "w" : "b";
  uint32_t nmove = board.moveNo() + ((stm == kBlack) ? 1 : 0);
  out = encodeGrid(nb) + " " + side + " - - 0 " + std::to_string(nmove);
  return true;
}

std::string xiangqiGameResult(const std::string& state) {
  std::vector<std::string> moves;
  if (!xiangqiLegalMoves(state, moves) || moves.empty()) {
    XiangqiBoard board;
    if (!board.setFen(state)) return "*";
    if (board.sideToMove() == kRed) return "0-1";
    return "1-0";
  }
  return "*";
}

}
}
