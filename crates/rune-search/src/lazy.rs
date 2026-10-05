#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LazyMode {
    L0,
    L1,
    L2,
}

#[derive(Debug, Clone)]
pub struct LazyConfig {
    pub mode: LazyMode,
    pub margin: f32,
    pub max_refine: usize,
}

impl Default for LazyConfig {
    fn default() -> Self {
        LazyConfig { mode: LazyMode::L0, margin: 0.08, max_refine: 1 }
    }
}

pub fn should_refine(cheap: f32, alpha: f32, beta: f32, uncertainty: f32, threshold: f32, cfg: &LazyConfig) -> bool {
    match cfg.mode {
        LazyMode::L0 => true,
        LazyMode::L1 => {
            if cheap.is_nan() {
                return true;
            }
            cheap >= threshold
        }
        LazyMode::L2 => {
            if cheap.is_nan() {
                return true;
            }
            if cheap <= alpha - cfg.margin || cheap >= beta + cfg.margin {
                return false;
            }
            if uncertainty >= 0.5 {
                return true;
            }
            true
        }
    }
}
