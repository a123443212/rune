#include "core/board/board.h"

#include <cctype>
#include <sstream>

namespace rune {

namespace {

int dirRank(Color c) { return c == Color::White ? 1 : -1; }

bool attacksSquare(const std::array<Piece, 64>& s, int from, int target) {
  Piece p = s[from];
  if (p.empty()) return false;
  int df = fileOf(target) - fileOf(from);
  int dr = rankOf(target) - rankOf(from);
  int adf = df < 0 ? -df : df;
  int adr = dr < 0 ? -dr : dr;
  switch (p.type) {
    case PieceType::Pawn: {
      int d = dirRank(p.color);
      return adr == 1 && adf == 1 && dr == d;
    }
    case PieceType::Knight:
      return (adf == 1 && adr == 2) || (adf == 2 && adr == 1);
    case PieceType::Bishop:
      if (adf != adr) return false;
      break;
    case PieceType::Rook:
      if (df != 0 && dr != 0) return false;
      break;
    case PieceType::Queen:
      if (!((adf == adr) || df == 0 || dr == 0)) return false;
      break;
    case PieceType::King:
      return adf <= 1 && adr <= 1;
    default:
      return false;
  }
  int stepF = (df == 0) ? 0 : (df > 0 ? 1 : -1);
  int stepR = (dr == 0) ? 0 : (dr > 0 ? 1 : -1);
  int f = fileOf(from) + stepF;
  int r = rankOf(from) + stepR;
  while (f != fileOf(target) || r != rankOf(target)) {
    if (!s[makeSq(f, r)].empty()) return false;
    f += stepF;
    r += stepR;
  }
  return true;
}

}  // namespace

Board::Board() {
  squares_.fill(Piece{});
  setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

Board::Board(const std::string& fen) {
  squares_.fill(Piece{});
  setFen(fen);
}

bool Board::setFen(const std::string& fen) {
  squares_.fill(Piece{});
  history_.clear();
  std::istringstream ss(fen);
  std::string placement, side, castle, ep, half, full;
  if (!(ss >> placement >> side >> castle >> ep >> half >> full)) return false;
  int rank = 7;
  int file = 0;
  for (char c : placement) {
    if (c == '/') {
      rank -= 1;
      file = 0;
    } else if (std::isdigit(static_cast<unsigned char>(c))) {
      file += c - '0';
    } else {
      if (file >= 8 || rank < 0) return false;
      Color col = std::isupper(static_cast<unsigned char>(c)) ? Color::White : Color::Black;
      PieceType t = PieceType::None;
      switch (std::tolower(static_cast<unsigned char>(c))) {
        case 'p': t = PieceType::Pawn; break;
        case 'n': t = PieceType::Knight; break;
        case 'b': t = PieceType::Bishop; break;
        case 'r': t = PieceType::Rook; break;
        case 'q': t = PieceType::Queen; break;
        case 'k': t = PieceType::King; break;
        default: return false;
      }
      squares_[makeSq(file, rank)] = Piece{t, col};
      file += 1;
    }
  }
  side_ = (side == "b") ? Color::Black : Color::White;
  castling_ = 0;
  if (castle != "-") {
    for (char c : castle) {
      if (c == 'K') castling_ |= kCastleWK;
      if (c == 'Q') castling_ |= kCastleWQ;
      if (c == 'k') castling_ |= kCastleBK;
      if (c == 'q') castling_ |= kCastleBQ;
    }
  }
  epSquare_ = -1;
  if (ep != "-") {
    int f = ep[0] - 'a';
    int r = ep[1] - '1';
    if (onBoard(f, r)) epSquare_ = makeSq(f, r);
  }
  halfmove_ = static_cast<uint16_t>(std::stoi(half));
  fullmove_ = static_cast<uint16_t>(std::stoi(full));
  return isLegalPosition();
}

std::string Board::toFen() const {
  std::string out;
  for (int r = 7; r >= 0; --r) {
    int empty = 0;
    for (int f = 0; f < 8; ++f) {
      Piece p = squares_[makeSq(f, r)];
      if (p.empty()) {
        empty += 1;
        continue;
      }
      if (empty > 0) {
        out += static_cast<char>('0' + empty);
        empty = 0;
      }
      char c = '?';
      switch (p.type) {
        case PieceType::Pawn: c = 'p'; break;
        case PieceType::Knight: c = 'n'; break;
        case PieceType::Bishop: c = 'b'; break;
        case PieceType::Rook: c = 'r'; break;
        case PieceType::Queen: c = 'q'; break;
        case PieceType::King: c = 'k'; break;
        default: c = '?'; break;
      }
      if (p.color == Color::White) c = static_cast<char>(std::toupper(c));
      out += c;
    }
    if (empty > 0) out += static_cast<char>('0' + empty);
    if (r > 0) out += '/';
  }
  out += (side_ == Color::White) ? " w " : " b ";
  std::string cs;
  if (castling_ & kCastleWK) cs += 'K';
  if (castling_ & kCastleWQ) cs += 'Q';
  if (castling_ & kCastleBK) cs += 'k';
  if (castling_ & kCastleBQ) cs += 'q';
  if (cs.empty()) cs = "-";
  out += cs + " ";
  if (epSquare_ < 0) {
    out += "-";
  } else {
    out += static_cast<char>('a' + fileOf(epSquare_));
    out += static_cast<char>('1' + rankOf(epSquare_));
  }
  out += " " + std::to_string(halfmove_) + " " + std::to_string(fullmove_);
  return out;
}

int Board::kingSquare(Color c) const {
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = squares_[sq];
    if (!p.empty() && p.type == PieceType::King && p.color == c) return sq;
  }
  return -1;
}

bool Board::isAttacked(int sq, Color by) const {
  for (int from = 0; from < 64; ++from) {
    Piece p = squares_[from];
    if (!p.empty() && p.color == by && attacksSquare(squares_, from, sq)) return true;
  }
  return false;
}

bool Board::inCheck(Color c) const {
  int k = kingSquare(c);
  if (k < 0) return false;
  return isAttacked(k, opposite(c));
}

void Board::addPawnMoves(int sq, std::vector<Move>& out) const {
  Piece p = squares_[sq];
  int d = dirRank(p.color);
  int f = fileOf(sq);
  int r = rankOf(sq);
  int startRank = (p.color == Color::White) ? 1 : 6;
  int promoRank = (p.color == Color::White) ? 7 : 0;
  auto pushPromo = [&](int from, int to) {
    if (rankOf(to) == promoRank) {
      for (PieceType t : {PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight}) {
        out.push_back(Move{static_cast<uint8_t>(from), static_cast<uint8_t>(to), t});
      }
    } else {
      out.push_back(Move{static_cast<uint8_t>(from), static_cast<uint8_t>(to), PieceType::None});
    }
  };
  if (onBoard(f, r + d) && squares_[makeSq(f, r + d)].empty()) {
    pushPromo(sq, makeSq(f, r + d));
    if (r == startRank && squares_[makeSq(f, r + 2 * d)].empty()) {
      out.push_back(Move{static_cast<uint8_t>(sq), static_cast<uint8_t>(makeSq(f, r + 2 * d)), PieceType::None});
    }
  }
  for (int df : {-1, 1}) {
    if (!onBoard(f + df, r + d)) continue;
    int to = makeSq(f + df, r + d);
    if (!squares_[to].empty() && squares_[to].color != p.color) pushPromo(sq, to);
    if (to == epSquare_) out.push_back(Move{static_cast<uint8_t>(sq), static_cast<uint8_t>(to), PieceType::None});
  }
}

void Board::addSlidingMoves(int sq, std::vector<Move>& out, bool diag, bool straight) const {
  Piece p = squares_[sq];
  for (int df = -1; df <= 1; ++df) {
    for (int dr = -1; dr <= 1; ++dr) {
      if (df == 0 && dr == 0) continue;
      bool isDiag = (df != 0 && dr != 0);
      if (isDiag && !diag) continue;
      if (!isDiag && !straight) continue;
      int f = fileOf(sq) + df;
      int r = rankOf(sq) + dr;
      while (onBoard(f, r)) {
        int to = makeSq(f, r);
        if (squares_[to].empty()) {
          out.push_back(Move{static_cast<uint8_t>(sq), static_cast<uint8_t>(to), PieceType::None});
        } else {
          if (squares_[to].color != p.color) {
            out.push_back(Move{static_cast<uint8_t>(sq), static_cast<uint8_t>(to), PieceType::None});
          }
          break;
        }
        f += df;
        r += dr;
      }
    }
  }
}

void Board::generatePseudoLegalMoves(std::vector<Move>& out) const {
  out.clear();
  static const int kKnight[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = squares_[sq];
    if (p.empty() || p.color != side_) continue;
    int f = fileOf(sq);
    int r = rankOf(sq);
    if (p.type == PieceType::Pawn) {
      addPawnMoves(sq, out);
    } else if (p.type == PieceType::Knight) {
      for (auto d : kKnight) {
        if (!onBoard(f + d[0], r + d[1])) continue;
        int to = makeSq(f + d[0], r + d[1]);
        if (squares_[to].empty() || squares_[to].color != p.color) {
          out.push_back(Move{static_cast<uint8_t>(sq), static_cast<uint8_t>(to), PieceType::None});
        }
      }
    } else if (p.type == PieceType::King) {
      for (int df = -1; df <= 1; ++df) {
        for (int dr = -1; dr <= 1; ++dr) {
          if (df == 0 && dr == 0) continue;
          if (!onBoard(f + df, r + dr)) continue;
          int to = makeSq(f + df, r + dr);
          if (squares_[to].empty() || squares_[to].color != p.color) {
            out.push_back(Move{static_cast<uint8_t>(sq), static_cast<uint8_t>(to), PieceType::None});
          }
        }
      }
      if (p.color == Color::White && sq == makeSq(4, 0)) {
        if ((castling_ & kCastleWK) && squares_[makeSq(5, 0)].empty() && squares_[makeSq(6, 0)].empty()) {
          out.push_back(Move{4, 6, PieceType::None});
        }
        if ((castling_ & kCastleWQ) && squares_[makeSq(3, 0)].empty() && squares_[makeSq(2, 0)].empty() &&
            squares_[makeSq(1, 0)].empty()) {
          out.push_back(Move{4, 2, PieceType::None});
        }
      }
      if (p.color == Color::Black && sq == makeSq(4, 7)) {
        if ((castling_ & kCastleBK) && squares_[makeSq(5, 7)].empty() && squares_[makeSq(6, 7)].empty()) {
          out.push_back(Move{60, 62, PieceType::None});
        }
        if ((castling_ & kCastleBQ) && squares_[makeSq(3, 7)].empty() && squares_[makeSq(2, 7)].empty() &&
            squares_[makeSq(1, 7)].empty()) {
          out.push_back(Move{60, 58, PieceType::None});
        }
      }
    } else if (p.type == PieceType::Bishop) {
      addSlidingMoves(sq, out, true, false);
    } else if (p.type == PieceType::Rook) {
      addSlidingMoves(sq, out, false, true);
    } else if (p.type == PieceType::Queen) {
      addSlidingMoves(sq, out, true, true);
    }
  }
}

void Board::generateLegalMoves(std::vector<Move>& out) {
  std::vector<Move> pseudo;
  generatePseudoLegalMoves(pseudo);
  out.clear();
  for (const Move& m : pseudo) {
    if (makeMove(m)) {
      unmakeMove();
      out.push_back(m);
    }
  }
}

void Board::applyMove(const Move& m, Piece& captured) {
  Piece moving = squares_[m.from];
  captured = squares_[m.to];
  bool isEp = (moving.type == PieceType::Pawn && m.to == epSquare_ && captured.empty());
  squares_[m.from] = Piece{};
  Piece placed = moving;
  if (m.promotion != PieceType::None) placed.type = m.promotion;
  squares_[m.to] = placed;
  if (isEp) {
    int capSq = makeSq(fileOf(m.to), rankOf(m.from));
    captured = squares_[capSq];
    squares_[capSq] = Piece{};
  }
  bool isCastle = (moving.type == PieceType::King &&
                   (m.to == m.from + 2 || (m.to + 2 == m.from)));
  if (isCastle) {
    if (m.to == makeSq(6, 0)) {
      squares_[makeSq(5, 0)] = squares_[makeSq(7, 0)];
      squares_[makeSq(7, 0)] = Piece{};
    } else if (m.to == makeSq(2, 0)) {
      squares_[makeSq(3, 0)] = squares_[makeSq(0, 0)];
      squares_[makeSq(0, 0)] = Piece{};
    } else if (m.to == makeSq(6, 7)) {
      squares_[makeSq(5, 7)] = squares_[makeSq(7, 7)];
      squares_[makeSq(7, 7)] = Piece{};
    } else if (m.to == makeSq(2, 7)) {
      squares_[makeSq(3, 7)] = squares_[makeSq(0, 7)];
      squares_[makeSq(0, 7)] = Piece{};
    }
  }
  if (moving.type == PieceType::King) {
    if (moving.color == Color::White) castling_ &= static_cast<uint8_t>(~(kCastleWK | kCastleWQ));
    else castling_ &= static_cast<uint8_t>(~(kCastleBK | kCastleBQ));
  }
  if (moving.type == PieceType::Rook) {
    if (m.from == makeSq(0, 0)) castling_ &= static_cast<uint8_t>(~kCastleWQ);
    if (m.from == makeSq(7, 0)) castling_ &= static_cast<uint8_t>(~kCastleWK);
    if (m.from == makeSq(0, 7)) castling_ &= static_cast<uint8_t>(~kCastleBQ);
    if (m.from == makeSq(7, 7)) castling_ &= static_cast<uint8_t>(~kCastleBK);
  }
  if (captured.type == PieceType::Rook) {
    if (m.to == makeSq(0, 0)) castling_ &= static_cast<uint8_t>(~kCastleWQ);
    if (m.to == makeSq(7, 0)) castling_ &= static_cast<uint8_t>(~kCastleWK);
    if (m.to == makeSq(0, 7)) castling_ &= static_cast<uint8_t>(~kCastleBQ);
    if (m.to == makeSq(7, 7)) castling_ &= static_cast<uint8_t>(~kCastleBK);
  }
  epSquare_ = -1;
  if (moving.type == PieceType::Pawn && (m.to == m.from + 16 || m.from == m.to + 16)) {
    epSquare_ = (m.from + m.to) / 2;
  }
  if (moving.color == Color::Black) fullmove_ += 1;
  if (moving.type == PieceType::Pawn || !captured.empty()) halfmove_ = 0;
  else halfmove_ += 1;
  side_ = opposite(side_);
}

bool Board::makeMove(const Move& m) {
  Piece moving = squares_[m.from];
  if (moving.empty() || moving.color != side_) return false;
  bool isCastle = (moving.type == PieceType::King && rankOf(m.from) == rankOf(m.to) &&
                   (m.to == m.from + 2 || m.to + 2 == m.from));
  if (isCastle) {
    int pass = (m.to > m.from) ? m.from + 1 : m.from - 1;
    Color enemy = opposite(side_);
    if (inCheck(side_) || isAttacked(pass, enemy)) return false;
  }
  BoardSnapshot snap;
  snap.move = m;
  snap.captured = squares_[m.to];
  snap.castling = castling_;
  snap.epSquarePrev = epSquare_;
  snap.halfmove = halfmove_;
  snap.fullmove = fullmove_;
  snap.isCastle = isCastle;
  snap.isEp = (moving.type == PieceType::Pawn && m.to == epSquare_ && squares_[m.to].empty() &&
               fileOf(m.from) != fileOf(m.to));
  Color us = side_;
  Piece ignored;
  std::array<Piece, 64> saved = squares_;
  uint8_t savedCastling = castling_;
  int savedEp = epSquare_;
  uint16_t savedHalf = halfmove_;
  uint16_t savedFull = fullmove_;
  applyMove(m, ignored);
  if (inCheck(us)) {
    squares_ = saved;
    castling_ = savedCastling;
    epSquare_ = savedEp;
    halfmove_ = savedHalf;
    fullmove_ = savedFull;
    side_ = us;
    return false;
  }
  history_.push_back(snap);
  return true;
}

void Board::unmakeMove() {
  if (history_.empty()) return;
  BoardSnapshot snap = history_.back();
  history_.pop_back();
  side_ = opposite(side_);
  Move m = snap.move;
  Piece moving = squares_[m.to];
  if (m.promotion != PieceType::None) moving.type = PieceType::Pawn;
  squares_[m.from] = moving;
  squares_[m.to] = Piece{};
  if (snap.isCastle) {
    if (m.to == makeSq(6, 0)) {
      squares_[makeSq(7, 0)] = squares_[makeSq(5, 0)];
      squares_[makeSq(5, 0)] = Piece{};
    } else if (m.to == makeSq(2, 0)) {
      squares_[makeSq(0, 0)] = squares_[makeSq(3, 0)];
      squares_[makeSq(3, 0)] = Piece{};
    } else if (m.to == makeSq(6, 7)) {
      squares_[makeSq(7, 7)] = squares_[makeSq(5, 7)];
      squares_[makeSq(5, 7)] = Piece{};
    } else if (m.to == makeSq(2, 7)) {
      squares_[makeSq(0, 7)] = squares_[makeSq(3, 7)];
      squares_[makeSq(3, 7)] = Piece{};
    }
  } else if (snap.isEp) {
    squares_[m.to] = Piece{};
    squares_[makeSq(fileOf(m.to), rankOf(m.from))] = snap.captured;
  } else {
    squares_[m.to] = snap.captured;
  }
  castling_ = snap.castling;
  halfmove_ = snap.halfmove;
  fullmove_ = snap.fullmove;
  epSquare_ = snap.epSquarePrev;
}

bool Board::isLegalPosition() const {
  int wk = 0;
  int bk = 0;
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = squares_[sq];
    if (p.empty()) continue;
    if (p.type == PieceType::King) {
      if (p.color == Color::White) wk += 1;
      else bk += 1;
    }
    if (p.type == PieceType::Pawn && (rankOf(sq) == 0 || rankOf(sq) == 7)) return false;
  }
  if (wk != 1 || bk != 1) return false;
  if (inCheck(opposite(side_))) return false;
  return true;
}

uint64_t Board::hashKey() const {
  uint64_t h = 1469598103934665603ULL;
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = squares_[sq];
    uint64_t v = static_cast<uint64_t>(static_cast<int>(p.type) * 2 + static_cast<int>(p.color));
    h ^= v + static_cast<uint64_t>(sq) * 1099511628211ULL;
    h *= 1099511628211ULL;
  }
  h ^= static_cast<uint64_t>(side_);
  h *= 1099511628211ULL;
  h ^= castling_;
  h *= 1099511628211ULL;
  return h;
}

int Board::gamePhase() const {
  int nonPawn = 0;
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = squares_[sq];
    if (p.empty() || p.type == PieceType::Pawn || p.type == PieceType::King) continue;
    nonPawn += 1;
  }
  if (nonPawn >= 12) return 0;
  if (nonPawn >= 6) return 1;
  return 2;
}

int Board::pieceCount() const {
  int n = 0;
  for (int sq = 0; sq < 64; ++sq) {
    if (!squares_[sq].empty()) n += 1;
  }
  return n;
}

}  // namespace rune
