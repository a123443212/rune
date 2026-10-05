# Failed: transposed weight packing cho 32x32

Thử: lưu thêm bản transposed của Wq/Wk/Wv để matVec đọc contiguous theo chiều khác.

Kết quả: 32x32 đã vừa L1, row-major + unroll 4 đã đủ nhanh. Bản copy transposed tăng artifact size, tăng pressure lúc load, không giảm latency đo được trên scalar path. Giữ packing v1 row-major-aligned32 duy nhất.

Bài học: weight packing chỉ đáng khi kernel thực sự yêu cầu layout khác, không phải cứ pack là nhanh.
