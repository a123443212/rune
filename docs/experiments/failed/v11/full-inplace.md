# Failed: full in-place mixer (gate viết đè scores + mixed viết đè tokens)

Thử: viết gate đè lên scores buffer, mixed_raw đè lên tokens input để giảm arena.

Kết quả: giữ một phần (gate_in_scores, mixed_raw_into_mixed_when_alpha_1 trong plan) nhưng không ép mọi op in-place vì aliasing làm SIMD khó tối ưu và differential tests khó đọc. Arena đã đủ nhỏ (7KB), thêm in-place chỉ tiết kiệm <1KB mà rủi ro correctness.

Bài học: in-place chỉ khi pass differential tests và không làm SIMD khó tối ưu.
