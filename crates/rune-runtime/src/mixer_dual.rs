use crate::mixer::MixerWeights;

#[derive(Debug, Clone)]
pub struct DualMixer {
    pub first: MixerWeights,
    pub second: MixerWeights,
}

impl DualMixer {
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        let n = x.len();
        let mut tmp = vec![0.0_f32; n];
        self.first.forward(x, None, &mut tmp, None);
        self.second.forward(&tmp, None, out, None);
    }
}
