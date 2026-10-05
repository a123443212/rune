pub struct Arena {
    buf: Vec<f32>,
}

impl Arena {
    pub fn new(bytes: usize) -> Self {
        let elems = bytes.div_ceil(4).max(1);
        Arena { buf: vec![0.0; elems] }
    }

    pub fn bytes(&self) -> usize {
        self.buf.len() * 4
    }

    pub fn slice(&self, offset_bytes: usize, elems: usize) -> &[f32] {
        let o = offset_bytes / 4;
        &self.buf[o..o + elems]
    }

    pub fn slice_mut(&mut self, offset_bytes: usize, elems: usize) -> &mut [f32] {
        let o = offset_bytes / 4;
        &mut self.buf[o..o + elems]
    }

    pub fn clear(&mut self) {
        for v in self.buf.iter_mut() {
            *v = 0.0;
        }
    }
}

pub fn arena_bytes_for(tokens: usize, dim: usize, h1: usize, h2: usize) -> usize {
    let live = [
        tokens * dim,
        tokens * dim,
        tokens * dim,
        tokens * dim,
        tokens * tokens,
        tokens * dim,
        h1,
        h2,
        tokens * dim,
    ];
    let mut total = 0;
    for e in live {
        total += e * 4 + 32;
    }
    total
}
