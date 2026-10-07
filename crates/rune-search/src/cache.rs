use std::collections::HashMap;

pub fn board_part(fen: &str) -> String {
    let p: Vec<&str> = fen.split(' ').collect();
    let b = p.first().copied().unwrap_or(fen);
    let s = p.get(1).copied().unwrap_or("w");
    let c = p.get(2).copied().unwrap_or("-");
    let e = p.get(3).copied().unwrap_or("-");
    format!("{b} {s} {c} {e}")
}

pub fn cache_key(fen: &str, model_hash: &str, mode: &str) -> String {
    let raw = format!("{}|{}|{}", board_part(fen), model_hash, mode);
    format!("{:016x}", rune_spec::fnv1a64(raw.as_bytes()))
}

pub struct EvalCache {
    model_hash: String,
    mode: String,
    cap: usize,
    map: HashMap<String, f32>,
    order: Vec<String>,
    hits: usize,
    misses: usize,
}

impl EvalCache {
    pub fn new(model_hash: &str, mode: &str, cap: usize) -> Self {
        EvalCache { model_hash: model_hash.to_string(), mode: mode.to_string(), cap, map: HashMap::new(), order: Vec::new(), hits: 0, misses: 0 }
    }

    pub fn get(&mut self, fen: &str) -> Option<f32> {
        let k = cache_key(fen, &self.model_hash, &self.mode);
        match self.map.get(&k) {
            Some(v) => {
                self.hits += 1;
                Some(*v)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    pub fn put(&mut self, fen: &str, value: f32) {
        if self.cap == 0 {
            return;
        }
        let k = cache_key(fen, &self.model_hash, &self.mode);
        if self.map.contains_key(&k) {
            self.map.insert(k, value);
            return;
        }
        if self.map.len() >= self.cap && !self.order.is_empty() {
            let old = self.order.remove(0);
            self.map.remove(&old);
        }
        self.map.insert(k.clone(), value);
        self.order.push(k);
    }

    pub fn hit_rate(&self) -> f64 {
        let n = self.hits + self.misses;
        if n == 0 {
            0.0
        } else {
            self.hits as f64 / n as f64
        }
    }
}
