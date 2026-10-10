#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace rune {
namespace xiangqi {

constexpr uint8_t kRed = 0;
constexpr uint8_t kBlack = 1;

constexpr uint8_t kPawn = 0;
constexpr uint8_t kHorse = 1;
constexpr uint8_t kRook = 2;
constexpr uint8_t kCannon = 3;
constexpr uint8_t kAdvisor = 4;
constexpr uint8_t kElephant = 5;
constexpr uint8_t kKing = 6;

struct XiangqiPiece {
  uint8_t kind = 0;
  uint8_t color = 0;
  bool present = false;
};

inline int sqFile(int sq) { return sq % 9; }
inline int sqRank(int sq) { return sq / 9; }
inline int makeSq(int file, int rank) { return rank * 9 + file; }
inline bool onBoard(int file, int rank) { return file >= 0 && file < 9 && rank >= 0 && rank < 10; }

class XiangqiBoard {
 public:
  XiangqiBoard();
  explicit XiangqiBoard(const std::string& fen);

  bool setFen(const std::string& fen);

  XiangqiPiece at(int sq) const { return squares_[sq]; }
  void setSquare(int sq, XiangqiPiece p) { squares_[sq] = p; }
  uint8_t sideToMove() const { return side_; }
  uint32_t moveNo() const { return moveNo_; }
  uint32_t pieceCount() const;

 private:
  std::array<XiangqiPiece, 90> squares_;
  uint8_t side_ = kRed;
  uint32_t moveNo_ = 1;
};

bool xiangqiAttacks(const XiangqiBoard& board, int from, int to);
bool flyingGenerals(const XiangqiBoard& board);

}
}
