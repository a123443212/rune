# RUNE v12 — v11 audit

Ngày: 2026-10-05
Phạm vi: C++ runtime, Rust runtime, compiler v0.11, compiled artifacts, numerical contract, quantization, adaptive routing, benchmark suite, engine adapter hiện có (`tools/match/play_match.py` greedy 1-ply), model metadata, serialization, toàn bộ experiment reports v01→v11.
Mục đích: chốt KEEP / REMOVE / REWORK / ENGINE-INTEGRATE / FREEZE / UNKNOWN cho v0.12. v0.12 không tạo architecture neural mới, không rewrite alpha-beta, không thêm MCTS/policy/RL.

## 1. Kết luận nhanh

v0.11 để lại một evaluator mạnh trên microbenchmark (fused nhanh hơn generic ~3-4x scalar, parity 0.00e+00, artifact +1.1%) nhưng toàn bộ số liệu đều đo ngoài search: `full_eval` 68-90us, không có node distribution thật, không có root-move nào được so sánh. Khoảng trống lớn nhất nằm ở 5 chỗ: evaluation scale chưa canonical giữa C++/Rust, WDL là output trang trí, adaptive routing chưa từng thấy search-node distribution, quantization mới so static eval, và engine adapter duy nhất là greedy 1-ply trong `play_match.py` nên mọi claim strength đều ngoại suy.

## 2. Bảng phân loại

| Hạng mục | Phán quyết | Lý do |
|---|---|---|
| spec/ Layer 1, numerical contract per-op eps, VERSIONS, IR v1 | FREEZE | Source of truth đã ổn định qua 3 ngôn ngữ. v0.12 chỉ thêm evaluation contract + engine scale, không sửa semantics kernel |
| Model format v2 + compiled artifact (source_hash, plan_hash, kernel_plan, memory_plan) | KEEP | Framing tương thích, generic loader đọc được compiled weights. v0.12 dùng model hash làm cache identity |
| C++ ref kernels + simd dispatcher + smallmat/fused/quant/accum specialized | KEEP | Oracle + fast path đã PASS `rune_tests`. v0.12 không đục kernel, chỉ đo chúng trong search |
| Rust rune-kernel scalar/simd/specialized/fused + safe wrapper isolate unsafe | KEEP | Bit-identical với C++ là tài sản cho search parity. Giữ nguyên |
| C++ Evaluator + CompiledEvaluator arena + Rust Evaluator + CompiledEvaluator | ENGINE-INTEGRATE | Đúng semantics ngoài search nhưng chưa có `load/refresh/make/unmake` contract thống nhất cho search loop. Bọc bằng engine adapter, không sửa lõi |
| Compiler pipeline Python + rune-ir + rune-compiler | KEEP | Deterministic, cache ổn. v0.12 benchmark generic vs compiled ngay trong engine để xem gain có sống sót không |
| Adaptive routing threshold >=, NaN→refine, cheap/refine split | ENGINE-INTEGRATE | Sweep 9 thresholds 100% agreement ngoài search nhưng fixture difficulties cluster [0.058,0.079]. Chưa ai thống kê refinement rate theo root/PV/cut/leaf |
| Quantization int8/int16 half-away, dequant-clip batched | REWORK | Static parity tốt (torch diff <0.01) nhưng chưa đo root-move stability trong search. v0.12 đo FP32 vs INT8 bằng NPS/nodes/move/PV |
| Golden vectors v10 + v11 compiled.json + diff_compiled stage-wise | REWORK | Giữ oracle. Mở rộng thành search-node golden: leaf eval + root score + move trên cùng settings |
| Benchmarks v10/v11 per-stage us + arena bytes + artifact bytes | REWORK | Giữ method same-model same-hardware ranges. Thêm evaluator NPS vs engine NPS vs nodes-to-depth vs time-to-depth |
| `tools/match/play_match.py` greedy 1-ply + EvalStats | REWORK | Là engine adapter duy nhất hiện có. Giữ làm baseline L0, thêm alpha-beta integration layer thật sự phía trên, không xóa |
| `tools/analysis/calibration.py` u-correlation + buckets | ENGINE-INTEGRATE | Đo monotonic u-err ngoài search. Tái sử dụng cho search-native calibration, thêm eval buckets theo scale/material/phase |
| Feature extraction GroupedFeatureSet 93 features startpos | FREEZE | refresh==incremental PASS hai ngôn ngữ. Cấm đổi ở v0.12 |
| Architecture table ATTN-GAB 8x32 + REL-02 + MLP | FREEZE | Không architecture mới. Chỉ đo scale/material/phase consistency |
| Dual scales maps latent trap | REMOVE | Đã phán ở v11, v0.12 loader strict + header hash ở v1.0. Không bàn thêm |
| Pybind greedy match path | KEEP | Không đưa Python vào hot path. Python giữ analysis/training/calibration/orchestration |
| No FFI, boundary là bytes | KEEP | Engine diff qua FEN/PV replay, không FFI |

## 3. Vấn đề tìm thấy

```text
runtime overhead: feature 7.4us + refresh 12.4us + tokenize 0.35us ≈ 20us/eval đứng ngoài mọi tối ưu fused; trong search với hàng nghìn nodes đây mới là số nhân tiền thật
evaluation scale mismatch: raw tanh [-1,1] chưa có canonical mapping sang engine score; C++/Rust tự chuyển đổi rời rạc, nguy cơ boundary flip quanh alpha/beta
search instability: chưa có test make/evaluate/unmake hàng triệu sequences trong search context; incremental vs fresh mới so ngoài search
WDL calibration problems: WDL head tồn tại nhưng chưa từng report calibration/accuracy/confidence theo bucket; value↔WDL mismatch chưa ai kiểm tra
lazy-evaluation opportunities: v0.4 machinery (cheap+refine+threshold) còn nguyên nhưng chưa có L0/L1/L2 ablation trong search; refinement rate theo node type là UNKNOWN
```

## 4. RQ mapping

RQ1 static→strength: cần static quality + search quality hai trục riêng, đo ranking/delta/sign error chứ không chỉ MSE.
RQ2 scale/calibration: cần canonical scale doc + calibration theo low/medium/high/win/loss/equal buckets.
RQ3 uncertainty/adaptive trong tree: cần stats theo root/PV/cut/leaf + search-native validation subset.
RQ4 latency vs strength: cần evaluator NPS + engine NPS + strength/node + strength/second, không suy từ microbenchmark.
RQ5 lazy eval an toàn: cần L0/L1/L2 ablation với deterministic threshold + bounded error + max refinement + fallback.
RQ6 consistency: cần determinism test + transposition/incremental stress + C++/Rust search parity tới first divergence.

## 5. Quyết định đóng băng

FREEZE: feature IDs, tensor layout, quant rounding, framing+hash, routing compare, per-stage eps, IR v1, kernel ids, packing v1, architecture table.
ENGINE-INTEGRATE nhưng không sửa semantics: evaluator cores, adaptive machinery, calibration analysis, match harness.
REMOVE duy nhất: dual-scales ambiguity (đã phán từ v11).
