use rune_kernel as kernel;
pub struct Tables {
    pub dim: usize,
    pub data: [Vec<f32>; 8],
}
impl Tables {
    pub fn zeros(dim: usize, vocabs: [usize; 8]) -> Tables {
        Tables {
            dim,
            data: [
                vec![0.0; vocabs[0] * dim],
                vec![0.0; vocabs[1] * dim],
                vec![0.0; vocabs[2] * dim],
                vec![0.0; vocabs[3] * dim],
                vec![0.0; vocabs[4] * dim],
                vec![0.0; vocabs[5] * dim],
                vec![0.0; vocabs[6] * dim],
                vec![0.0; vocabs[7] * dim],
            ],
        }
    }
    pub fn row(&self, g: usize, idx: usize) -> &[f32] {
        let base = idx * self.dim;
        &self.data[g][base..base + self.dim]
    }
}
pub struct Accumulator {
    pub dim: usize,
    acc: Vec<f32>,
}
impl Accumulator {
    pub fn new(dim: usize) -> Accumulator {
        Accumulator { dim, acc: vec![0.0; 8 * dim] }
    }
    pub fn refresh(&mut self, tables: &Tables, feats: &[(u8, u16)]) {
        for v in self.acc.iter_mut() {
            *v = 0.0;
        }
        let (added, empty): (Vec<(u8, u16)>, Vec<(u8, u16)>) = (feats.to_vec(), Vec::new());
        self.apply_diff(tables, &added, &empty);
    }
    pub fn apply_diff(&mut self, tables: &Tables, added: &[(u8, u16)], removed: &[(u8, u16)]) {
        for (g, idx) in added {
            let row = tables.row(*g as usize, *idx as usize);
            let base = *g as usize * self.dim;
            for d in 0..self.dim {
                self.acc[base + d] += row[d];
            }
        }
        for (g, idx) in removed {
            let row = tables.row(*g as usize, *idx as usize);
            let base = *g as usize * self.dim;
            for d in 0..self.dim {
                self.acc[base + d] -= row[d];
            }
        }
    }
    pub fn tokens(&self, out: &mut [f32]) {
        kernel::tokens_clip(&self.acc, out);
    }
    pub fn raw(&self) -> &[f32] {
        &self.acc
    }
}
pub struct IntAccumulator {
    pub dim: usize,
    acc: Vec<i32>,
    scales: [f32; 8],
}
impl IntAccumulator {
    pub fn new(dim: usize, scales: [f32; 8]) -> IntAccumulator {
        IntAccumulator { dim, acc: vec![0; 8 * dim], scales }
    }
    pub fn refresh(&mut self, qtables: &[Vec<i32>; 8], feats: &[(u8, u16)]) {
        for v in self.acc.iter_mut() {
            *v = 0;
        }
        self.apply_diff(qtables, feats, &[]);
    }
    pub fn apply_diff(&mut self, qtables: &[Vec<i32>; 8], added: &[(u8, u16)], removed: &[(u8, u16)]) {
        for (g, idx) in added {
            let base_q = *idx as usize * self.dim;
            let base_a = *g as usize * self.dim;
            for d in 0..self.dim {
                self.acc[base_a + d] += qtables[*g as usize][base_q + d];
            }
        }
        for (g, idx) in removed {
            let base_q = *idx as usize * self.dim;
            let base_a = *g as usize * self.dim;
            for d in 0..self.dim {
                self.acc[base_a + d] -= qtables[*g as usize][base_q + d];
            }
        }
    }
    pub fn tokens(&self, out: &mut [f32]) {
        for g in 0..8 {
            for d in 0..self.dim {
                let v = self.acc[g * self.dim + d] as f32 * self.scales[g];
                out[g * self.dim + d] = kernel::clipped_relu(v);
            }
        }
    }
}
