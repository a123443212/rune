use rune_runtime::board::Board;
use rune_runtime::features::extract_features;
pub fn fen_features(fen: &str) -> Result<Vec<(u8, u16)>, String> {
    let b = Board::parse_fen(fen).map_err(|e| e.to_string())?;
    Ok(extract_features(&b))
}
pub fn shard_index(n: usize, shards: usize, i: usize) -> usize {
    if shards == 0 {
        return 0;
    }
    i % shards.max(n.min(shards).max(1))
}
pub fn filter_by_piece_count(fen: &str, lo: usize, hi: usize) -> bool {
    let b = match Board::parse_fen(fen) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let n = b.piece_count();
    n >= lo && n <= hi
}
pub fn sample_stride(n: usize, stride: usize) -> Vec<usize> {
    let mut out = Vec::new();
    if stride == 0 {
        return out;
    }
    let mut i = 0;
    while i < n {
        out.push(i);
        i += stride;
    }
    out
}
