#include "core/board/board.h"

#include <cctype>
#include <sstream>

#include "core/board/board_int.h"

namespace rune {

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

}
