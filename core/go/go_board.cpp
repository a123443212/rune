#include "core/go/go_board.h"

namespace rune {
namespace go {

namespace {

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

std::vector<std::string> splitRows(const std::string& grid) {
  std::vector<std::string> rows;
  std::string cur;
  for (char c : grid) {
    if (c == '/') {
      rows.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  rows.push_back(cur);
  return rows;
}

}

bool validGoSize(int n) { return n == 9 || n == 13 || n == 19; }

GoBoard::GoBoard() : size_(9), stones_(81, 0), side_(kBlack) {}

GoBoard::GoBoard(int size, const std::string& state) : GoBoard() {
  if (validGoSize(size)) size_ = size;
  stones_.assign(static_cast<size_t>(size_ * size_), 0);
  setState(state);
}

bool GoBoard::setState(const std::string& state) {
  std::vector<std::string> parts = splitWs(state);
  if (parts.empty()) return false;
  std::vector<std::string> rows = splitRows(parts[0]);
  int n = static_cast<int>(rows.size());
  if (!validGoSize(n)) return false;
  size_ = n;
  stones_.assign(static_cast<size_t>(n * n), 0);
  side_ = kBlack;
  for (int r = 0; r < n; ++r) {
    if (static_cast<int>(rows[static_cast<size_t>(r)].size()) != n) return false;
    for (int c = 0; c < n; ++c) {
      char ch = rows[static_cast<size_t>(r)][static_cast<size_t>(c)];
      int8_t v = 0;
      if (ch == '.') v = 0;
      else if (ch == 'X') v = 1;
      else if (ch == 'O') v = -1;
      else return false;
      stones_[static_cast<size_t>(r * n + c)] = v;
    }
  }
  if (parts.size() > 1) {
    if (parts[1] == "w" || parts[1] == "O") side_ = kWhite;
    else if (parts[1] == "b" || parts[1] == "X") side_ = kBlack;
    else return false;
  }
  return true;
}

bool GoBoard::setState(int size, const std::string& state) {
  if (!validGoSize(size)) return false;
  size_ = size;
  stones_.assign(static_cast<size_t>(size * size), 0);
  return setState(state);
}

uint32_t GoBoard::stoneCount(int8_t v) const {
  uint32_t n = 0;
  for (int8_t s : stones_) {
    if (s == v) ++n;
  }
  return n;
}

int GoBoard::libertiesOf(int r, int c) const {
  int n = size_;
  int8_t target = at(r, c);
  if (target == 0) return 0;
  std::vector<char> seen(static_cast<size_t>(n * n), 0);
  std::vector<int> stack;
  stack.push_back(r * n + c);
  seen[static_cast<size_t>(r * n + c)] = 1;
  bool libs[361] = {false};
  int count = 0;
  const int dr[4] = {-1, 1, 0, 0};
  const int dc[4] = {0, 0, -1, 1};
  while (!stack.empty()) {
    int cur = stack.back();
    stack.pop_back();
    int cr = cur / n, cc = cur % n;
    for (int k = 0; k < 4; ++k) {
      int nr = cr + dr[k], nc = cc + dc[k];
      if (nr < 0 || nc < 0 || nr >= n || nc >= n) continue;
      int8_t v = at(nr, nc);
      if (v == 0) {
        if (!libs[nr * n + nc]) {
          libs[nr * n + nc] = true;
          ++count;
          if (count >= 8) return 8;
        }
      } else if (v == target && !seen[static_cast<size_t>(nr * n + nc)]) {
        seen[static_cast<size_t>(nr * n + nc)] = 1;
        stack.push_back(nr * n + nc);
      }
    }
  }
  return count;
}

}
}
