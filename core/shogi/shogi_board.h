#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace rune {

namespace shogi {

constexpr uint8_t kBlack = 0;
constexpr uint8_t kWhite = 1;

constexpr uint8_t kPawn = 0;
constexpr uint8_t kLance = 1;
constexpr uint8_t kKnight = 2;
constexpr uint8_t kSilver = 3;
constexpr uint8_t kGold = 4;
constexpr uint8_t kBishop = 5;
constexpr uint8_t kRook = 6;
constexpr uint8_t kKing = 7;
constexpr uint8_t kProPawn = 8;
constexpr uint8_t kProLance = 9;
constexpr uint8_t kProKnight = 10;
constexpr uint8_t kProSilver = 11;
constexpr uint8_t kHorse = 12;
constexpr uint8_t kDragon = 13;

constexpr int kHandTypes = 7;

struct ShogiPiece {
  uint8_t kind = 0;
  uint8_t color = 0;
  bool present = false;
};

inline int sqFile(int sq) { return sq % 9; }
inline int sqRank(int sq) { return sq / 9; }
inline int makeSq(int file, int rank) { return rank * 9 + file; }
inline bool onBoard(int file, int rank) { return file >= 0 && file < 9 && rank >= 0 && rank < 9; }

class ShogiBoard {
 public:
  ShogiBoard();
  explicit ShogiBoard(const std::string& sfen);

  bool setSfen(const std::string& sfen);

  ShogiPiece at(int sq) const { return squares_[sq]; }
  uint8_t sideToMove() const { return side_; }
  uint8_t hand(uint8_t color, int type) const { return hand_[color][type]; }
  uint32_t moveNo() const { return moveNo_; }

  uint32_t handTotal() const;
  uint32_t promoCount() const;

 private:
  std::array<ShogiPiece, 81> squares_;
  uint8_t side_ = kBlack;
  uint8_t hand_[2][kHandTypes] = {};
  uint32_t moveNo_ = 1;
};

bool shogiAttacks(const ShogiBoard& board, int from, int to);

}
}
