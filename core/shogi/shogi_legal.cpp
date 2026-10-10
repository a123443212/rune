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

#include "core/shogi/shogi_legal.h"

#include <algorithm>

namespace rune {
namespace shogi {

namespace {

std::string twoDigits(int v) {
  std::string s;
  s += static_cast<char>('0' + v / 10);
  s += static_cast<char>('0' + v % 10);
  return s;
}

bool parseTwo(const std::string& s, size_t pos, int& v) {
  if (pos + 2 > s.size()) return false;
  char a = s[pos];
  char b = s[pos + 1];
  if (a < '0' || a > '9' || b < '0' || b > '9') return false;
  v = (a - '0') * 10 + (b - '0');
  return v >= 0 && v < 81;
}

char kindLetter(uint8_t kind) {
  if (kind == kPawn) return 'p';
  if (kind == kLance) return 'l';
  if (kind == kKnight) return 'n';
  if (kind == kSilver) return 's';
  if (kind == kGold) return 'g';
  if (kind == kBishop) return 'b';
  return 'r';
}

bool letterKind(char c, uint8_t& kind) {
  if (c == 'p') kind = kPawn;
  else if (c == 'l') kind = kLance;
  else if (c == 'n') kind = kKnight;
  else if (c == 's') kind = kSilver;
  else if (c == 'g') kind = kGold;
  else if (c == 'b') kind = kBishop;
  else if (c == 'r') kind = kRook;
  else return false;
  return true;
}

uint8_t promoteOf(uint8_t kind) {
  if (kind == kPawn) return kProPawn;
  if (kind == kLance) return kProLance;
  if (kind == kKnight) return kProKnight;
  if (kind == kSilver) return kProSilver;
  if (kind == kBishop) return kHorse;
  return kDragon;
}

bool sameMove(ShogiMove a, ShogiMove b) {
  if (a.drop != b.drop) return false;
  if (a.drop) return a.to == b.to && a.piece == b.piece;
  return a.fr == b.fr && a.to == b.to && a.promo == b.promo;
}

bool inPseudo(const std::vector<ShogiMove>& moves, ShogiMove mv) {
  for (const auto& m : moves) {
    if (sameMove(m, mv)) return true;
  }
  return false;
}

bool doMove(ShogiBoard& nb, ShogiMove mv, uint8_t stm) {
  if (mv.drop) {
    int ti = -1;
    const uint8_t order[7] = {kPawn, kLance, kKnight, kSilver, kGold, kBishop, kRook};
    for (int i = 0; i < 7; ++i) {
      if (order[i] == mv.piece) ti = i;
    }
    if (ti < 0 || nb.hand(stm, ti) == 0) return false;
    if (nb.at(mv.to).present) return false;
    nb.setHand(stm, ti, nb.hand(stm, ti) - 1);
    ShogiPiece p;
    p.kind = mv.piece;
    p.color = stm;
    p.present = true;
    nb.setSquare(mv.to, p);
    return true;
  }
  ShogiPiece p = nb.at(mv.fr);
  if (!p.present || p.color != stm) return false;
  ShogiPiece t = nb.at(mv.to);
  if (t.present && t.color == stm) return false;
  uint8_t nk = p.kind;
  if (mv.promo) {
    if (p.kind != kPawn && p.kind != kLance && p.kind != kKnight && p.kind != kSilver &&
        p.kind != kBishop && p.kind != kRook)
      return false;
    nk = promoteOf(p.kind);
  }
  if (t.present) {
    uint8_t base = shogiUnpromote(t.kind);
    const uint8_t order[7] = {kPawn, kLance, kKnight, kSilver, kGold, kBishop, kRook};
    for (int i = 0; i < 7; ++i) {
      if (order[i] == base) nb.setHand(stm, i, nb.hand(stm, i) + 1);
    }
  }
  ShogiPiece np;
  np.kind = nk;
  np.color = stm;
  np.present = true;
  nb.setSquare(mv.to, np);
  nb.setSquare(mv.fr, ShogiPiece());
  return true;
}

bool pawnDropMate(const ShogiBoard& board, int to) {
  uint8_t stm = board.sideToMove();
  uint8_t foe = (stm == kBlack) ? kWhite : kBlack;
  ShogiBoard nb = board;
  ShogiPiece p;
  p.kind = kPawn;
  p.color = stm;
  p.present = true;
  nb.setSquare(to, p);
  if (!shogiKingInCheck(nb, foe)) return false;
  ShogiBoard fb = nb;
  fb.setSide(foe);
  std::vector<ShogiMove> pseudo;
  shogiPseudoMoves(fb, pseudo);
  for (const auto& mv : pseudo) {
    if (mv.drop) continue;
    ShogiBoard trial = nb;
    if (!doMove(trial, mv, foe)) continue;
    if (!shogiKingInCheck(trial, foe)) return false;
  }
  return true;
}

std::string encodeGrid(const ShogiBoard& board) {
  std::string grid;
  for (int ri = 0; ri < 9; ++ri) {
    if (ri) grid += "/";
    std::string rank;
    int gap = 0;
    for (int f = 0; f < 9; ++f) {
      ShogiPiece p = board.at(ri * 9 + f);
      if (!p.present) {
        ++gap;
        continue;
      }
      if (gap > 0) {
        rank += static_cast<char>('0' + gap);
        gap = 0;
      }
      uint8_t base = shogiUnpromote(p.kind);
      if (base != p.kind) rank += "+";
      char ch = kindLetter(base);
      if (base == kKing) ch = 'k';
      rank += (p.color == kBlack) ? static_cast<char>(ch - 32) : ch;
    }
    if (gap > 0) rank += static_cast<char>('0' + gap);
    grid += rank;
  }
  return grid;
}

std::string encodeHands(const ShogiBoard& board) {
  std::string s;
  for (int c = 0; c < 2; ++c) {
    const char* letters[7] = {"r", "b", "g", "s", "n", "l", "p"};
    for (int i = 0; i < 7; ++i) {
      uint8_t n = board.hand(c, i);
      if (n == 0) continue;
      if (n > 1) s += std::to_string(n);
      char ch = letters[i][0];
      s += (c == kBlack) ? static_cast<char>(ch - 32) : ch;
    }
  }
  if (s.empty()) s = "-";
  return s;
}

}

std::string shogiEncodeMove(ShogiMove mv) {
  if (mv.drop) return "D" + std::to_string(mv.to) + kindLetter(mv.piece);
  if (mv.promo) return twoDigits(mv.fr) + twoDigits(mv.to) + "+";
  return twoDigits(mv.fr) + twoDigits(mv.to);
}

bool shogiDecodeMove(const std::string& s, ShogiMove& mv) {
  if (!s.empty() && s[0] == 'D') {
    if (s.size() < 3) return false;
    char lc = s.back();
    uint8_t kind = 0;
    if (!letterKind(lc, kind)) return false;
    int to = 0;
    for (size_t i = 1; i + 1 < s.size(); ++i) {
      if (s[i] < '0' || s[i] > '9') return false;
      to = to * 10 + (s[i] - '0');
    }
    if (to < 0 || to >= 81) return false;
    mv = ShogiMove{true, -1, to, false, kind};
    return true;
  }
  bool promo = !s.empty() && s.back() == '+';
  std::string core = promo ? s.substr(0, s.size() - 1) : s;
  if (core.size() != 4) return false;
  int fr = 0;
  int to = 0;
  if (!parseTwo(core, 0, fr) || !parseTwo(core, 2, to)) return false;
  mv = ShogiMove{false, fr, to, promo, 0};
  return true;
}

bool shogiKingInCheck(const ShogiBoard& board, uint8_t color) {
  int kus = -1;
  for (int sq = 0; sq < 81; ++sq) {
    ShogiPiece p = board.at(sq);
    if (p.present && p.kind == kKing && p.color == color) {
      kus = sq;
      break;
    }
  }
  if (kus < 0) return true;
  for (int asq = 0; asq < 81; ++asq) {
    ShogiPiece p = board.at(asq);
    if (!p.present || p.color == color) continue;
    if (shogiAttacks(board, asq, kus)) return true;
  }
  return false;
}

bool shogiLegalMoves(const std::string& sfen, std::vector<std::string>& out) {
  out.clear();
  ShogiBoard board;
  if (!board.setSfen(sfen)) return false;
  uint8_t stm = board.sideToMove();
  std::vector<ShogiMove> pseudo;
  shogiPseudoMoves(board, pseudo);
  for (const auto& mv : pseudo) {
    if (mv.drop && mv.piece == kPawn && pawnDropMate(board, mv.to)) continue;
    ShogiBoard nb = board;
    if (!doMove(nb, mv, stm)) continue;
    if (shogiKingInCheck(nb, stm)) continue;
    out.push_back(shogiEncodeMove(mv));
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return true;
}

bool shogiApplyMove(const std::string& sfen, const std::string& mv, std::string& out) {
  ShogiMove m;
  if (!shogiDecodeMove(mv, m)) return false;
  ShogiBoard board;
  if (!board.setSfen(sfen)) return false;
  uint8_t stm = board.sideToMove();
  std::vector<ShogiMove> pseudo;
  shogiPseudoMoves(board, pseudo);
  if (!inPseudo(pseudo, m)) return false;
  if (m.drop && m.piece == kPawn && pawnDropMate(board, m.to)) return false;
  ShogiBoard nb = board;
  if (!doMove(nb, m, stm)) return false;
  if (shogiKingInCheck(nb, stm)) return false;
  uint8_t nstm = (stm == kBlack) ? kWhite : kBlack;
  std::string side = (nstm == kBlack) ? "b" : "w";
  out = encodeGrid(nb) + " " + side + " " + encodeHands(nb) + " " +
        std::to_string(board.moveNo() + 1);
  return true;
}

std::string shogiGameResult(const std::string& sfen) {
  std::vector<std::string> moves;
  if (!shogiLegalMoves(sfen, moves) || moves.empty()) {
    ShogiBoard board;
    if (!board.setSfen(sfen)) return "*";
    if (board.sideToMove() == kBlack) return "0-1";
    return "1-0";
  }
  return "*";
}

}
}
