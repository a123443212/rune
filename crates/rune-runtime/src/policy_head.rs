use rune_kernel as kernel;

#[derive(Debug, Clone)]
pub struct PolicyWeights {
    pub input: usize,
    pub output: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

impl PolicyWeights {
    pub fn forward(&self, flat: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; self.output];
        kernel::mat_vec(&self.w, flat, Some(&self.b), &mut logits, self.output, self.input);
        let mut probs = vec![0.0f32; self.output];
        kernel::softmax(&logits, &mut probs);
        probs
    }

    pub fn logits(&self, flat: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.output];
        kernel::mat_vec(&self.w, flat, Some(&self.b), &mut out, self.output, self.input);
        out
    }
}
