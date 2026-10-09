use crate::evaluator::Evaluator;
use crate::resnet::ResnetWeights;

fn is_fast(rw: &ResnetWeights) -> bool {
    let b = rw.cfg.board;
    let c = rw.cfg.channels;
    if rw.stem_w.len() != c * 9 || rw.stem_b.len() != c {
        return false;
    }
    let _ = b;
    for i in 0..rw.cfg.blocks {
        if rw.block_w1[i].len() != c * c * 9 || rw.block_w2[i].len() != c * c * 9 {
            return false;
        }
        if rw.block_b1[i].len() != c || rw.block_b2[i].len() != c {
            return false;
        }
    }
    true
}

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

    pub fn set_resnet_for_bench(&mut self, rw: crate::resnet::ResnetWeights) {
        self.resnet = Some(rw);
    }

    pub fn evaluate_planes(&self, planes: &[f32]) -> crate::evaluator::EvalResult {
        match &self.resnet {
            Some(rw) => {
                if is_fast(rw) {
                    let mut s = self.scratch.borrow_mut();
                    let (value, wdl, policy) = crate::resnet_scratch::forward_fast(rw, &mut s, planes);
                    crate::evaluator::EvalResult { value, wdl, refine: false, difficulty: 0.0, policy, score_mean: 0.0 }
                } else {
                    let (value, wdl, policy) = rw.forward(planes);
                    crate::evaluator::EvalResult { value, wdl, refine: false, difficulty: 0.0, policy, score_mean: 0.0 }
                }
            }
            None => crate::evaluator::EvalResult { value: 0.0, wdl: [0.0, 1.0, 0.0], refine: false, difficulty: 1.0, policy: Vec::new(), score_mean: 0.0 },
        }
    }
}
