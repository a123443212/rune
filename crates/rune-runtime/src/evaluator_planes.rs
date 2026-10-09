use crate::evaluator::Evaluator;

impl Evaluator {
    pub fn is_resnet(&self) -> bool {
        self.arch_id().starts_with("RUNE-RESNET")
    }

    pub fn resnet_board(&self) -> usize {
        self.tokens
    }

    pub fn resnet_channels(&self) -> usize {
        self.dim
    }

    pub fn evaluate_planes(&self, planes: &[f32]) -> crate::evaluator::EvalResult {
        match &self.resnet {
            Some(rw) => {
                let (value, wdl, policy) = rw.forward(planes);
                crate::evaluator::EvalResult { value, wdl, refine: false, difficulty: 0.0, policy, score_mean: 0.0 }
            }
            None => crate::evaluator::EvalResult { value: 0.0, wdl: [0.0, 1.0, 0.0], refine: false, difficulty: 1.0, policy: Vec::new(), score_mean: 0.0 },
        }
    }
}
