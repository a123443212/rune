// RUNE — Relational Unified Neural Evaluator
// Copyright (C) 2026 a123443212
//
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// This project is dual-licensed under the MIT License and the
// Apache License, Version 2.0. You may choose either license
// when using, copying, modifying, or distributing this software.
//
// MIT License: https://opensource.org/license/mit
// Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, this
// software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
// OR CONDITIONS OF ANY KIND, either express or implied.

pub const FNV_OFFSET: u64 = 1469598103934665603;
pub const FNV_PRIME: u64 = 1099511628211;

pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

pub fn fnv1a_str(s: &str) -> u64 {
    fnv1a(s.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_known_vector() {
        assert_eq!(fnv1a(b""), 1469598103934665603);
        assert_ne!(fnv1a(b"a"), fnv1a(b"b"));
    }

    #[test]
    fn fnv_deterministic() {
        assert_eq!(fnv1a_str("rnbqkbnr"), fnv1a_str("rnbqkbnr"));
    }
}
