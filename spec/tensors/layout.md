# Tensor and memory layout

## Logical shapes

- Grouped path: tokens T=8, dim D=32. Token g is row g of an 8x32 matrix.
- Flex path: T in {6,8,10}, D in {24,32,40}. Token count and dim come
  from model header and must match architecture config.
- Dense/adaptive variable path: per-group width w[g]. Grouped widths
  table `token_dims[8]` in header; shared-pooling models use
  `shared_width` for embedding fetch.
- Context vector: 8 x float32.

## In-memory order

Row-major everywhere: element (row,col) at offset row*cols+col.
Token buffer is T*D floats, token 0 first. Embedding table for group
g is vocab[g] rows x width floats, row = feature index, column = dim.
Matrix weights are rows x cols row-major: `matVec` reads row r at
`mat + r*cols`.

## Accumulator layout

- fp32 path: `float acc[8][D]`, zero-initialised on refresh.
- int8/int16 path: `int32 acc[8][D]`, zero-initialised, sums of int8
  or int16 embedding rows. Accumulation width is exactly 32 bits
  signed. Overflow wraps only in the sense that inputs are bounded:
  worst case vocab fan-in keeps sums inside int32 by construction;
  loaders must reject models whose fan-in * 32767 exceeds int32.

## Alignment and padding

No padding in the file. In memory, implementations may align rows
to 16/32 bytes for SIMD but must preserve logical row-major values.
SIMD kernels must handle tail columns scalar.

## Token post step

After accumulation, tokenise by elementwise clip01:
`tok = min(max(acc,0),1)`. For quantised paths dequantise first
(`float(acc)*scale`), then clip01. Clip bounds are inclusive.
