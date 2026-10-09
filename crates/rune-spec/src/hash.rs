pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for b in data {
        h ^= *b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

pub fn fnv1a_str(s: &str) -> u64 {
    fnv1a64(s.as_bytes())
}
