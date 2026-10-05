use crate::relational_cache::{IncrWeights, RelationalCache};

pub fn dense_baseline(weights: &IncrWeights, tokens: &[f32]) -> Vec<f32> {
    let mut c = RelationalCache::configure(weights.clone(), usize::MAX);
    c.rebuild(tokens, None);
    c.out().to_vec()
}

pub fn b1_token_only_cost(_tokens: usize, dim: usize, changed: usize) -> usize {
    changed * dim * dim * 3
}

pub fn b2_cells_saved(tokens: usize, changed: usize) -> usize {
    tokens * tokens - (2 * changed * tokens - changed * changed)
}
