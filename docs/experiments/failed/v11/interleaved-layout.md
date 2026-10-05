# Failed: interleaved token layout

Thử: đổi tokens từ row-major sang interleaved để QKV đọc coalesced.

Kết quả: kernels hiện tại đọc per-token row contiguous đã tốt, interleaved làm accumulator phức tạp mà không cải thiện QKV fused đo được. Giữ row-major, chỉ thử layouts có lý do từ kernel.

Bài học: không brute-force layouts, chọn dựa trên actual kernels.
