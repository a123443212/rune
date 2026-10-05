# `.rune` model format — RUNE-10 contract (format version 2)

## Framing

```text
offset 0: 4 bytes magic "RUNE" (0x52 0x55 0x4E 0x45)
offset 4: 4 bytes u32 LE header_len
offset 8: header_len bytes UTF-8 JSON header
offset 8+header_len: payload bytes in tensor order
```

`header_len` <= 1_000_000. Readers must reject longer headers,
short reads, trailing garbage beyond declared tensor bytes is an
error, missing bytes are an error. Payload floats are `<f4`,
int8 raw, int16 `<i2`. File endianness is little-endian always.

## Required header fields

```text
format            u32 == 2
architecture_id   string, e.g. "RUNE-ATTN-GAB"
architecture_version string, e.g. "0.2.0"
feature_version   string == "grouped_hkav2_fullthreats_v01"
tokens            int
token_dim         int
attention, geometric_bias, head, quantization, gate, alpha, ...
tensor_metadata   array [{name, shape:[...], dtype:"float32"|"int8"|"int16"}]
quantization_metadata {mode:"symmetric", bound, scales:{emb0..emb7}}
model_hash        hex fnv1a64 over payload bytes
```

Legacy aliases `arch`, `arch_version`, `feature_set`, `tensors`,
`scales`, `checksum` are accepted on read for format 1 files but
never written by v0.10 exporters. Writers always emit the canonical
names above AND the legacy aliases for backward compatibility.
Canonical dequant scale source is
`quantization_metadata.scales`; writers MUST write identical values
into legacy `scales`. Current readers resolve differently when the
two maps disagree (C++/Rust: quantization_metadata value by parse
order; Python reference paths: `scales`), so disagreeing maps are a
latent divergence trap even though no shipped file has them. A
strict equality check on load is required at v1.0.

## Tensor order and shapes

Order: `emb0..emb7` then arch `export_order` for the architecture
id. Shapes: embeddings `[vocab[g], width]`, arch tensors per
`architecture-spec.json`. Any deviation (unknown name, wrong order,
wrong element count) is a load error.

## Checksum / hash

`model_hash` = lowercase hex of FNV-1a 64 over exactly the payload
bytes (offset 8+header_len to EOF), init 1469598103934665603, prime
1099511628211, masked to 64 bits. Python, C++, Rust must agree byte
for byte. Files with mismatched hash fail closed. Format 1 files
carrying `checksum` only for dense/adaptive verify the same way.

Known limitation (measured, all three loaders): the hash covers
the payload ONLY. Header fields (scales, thresholds, arch ids)
can be altered without breaking the hash — demonstrated: doubling
`emb0` scale on an int8 model changes eval output while both C++
and Rust loaders exit 0. The hash therefore detects transport
corruption of weights, not semantic integrity of the model. Any
workflow that treats equal hash as equal behavior (caches, audit
trails) is unsound until the header is covered. Required at v1.0:
hash header+payload, or a second header checksum.

## Compatibility

- Readers accept format 1 and 2 for all architectures shipped in
  v0.9. Format 0 does not exist.
- Writers emit format 2 only.
- Unknown `architecture_id` fails closed with
  `unsupported-architecture`. Unknown `quantization` fails closed.
- `feature_version` mismatch against runtime or dataset fails closed.
- There is exactly one logical model per file: no
  `network-cpp.rune` / `network-rust.rune` split. Both runtimes load
  the same bytes.

## Size guards

Loaders must check before allocating: header_len bound, per-tensor
element count * dtype size does not overflow size_t, total payload
length equals sum of tensor sizes, vocab*width matches declared
shape, dims within architecture table. Oversized or inconsistent
declarations fail without allocating.
