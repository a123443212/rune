pub const GRAPH_VERSION: &str = "v13-graph-01";
pub const INVALIDATION_VERSION: &str = "v13-inv-01";

#[derive(Debug, Clone)]
pub struct InteractionGraph {
    pub tokens: usize,
    pub edges: Vec<(usize, usize)>,
}

impl InteractionGraph {
    pub fn dense(tokens: usize) -> InteractionGraph {
        let mut edges = Vec::with_capacity(tokens * tokens);
        for a in 0..tokens {
            for b in 0..tokens {
                edges.push((a, b));
            }
        }
        InteractionGraph { tokens, edges }
    }

    pub fn pruned(&self, keep: &[(usize, usize)]) -> InteractionGraph {
        InteractionGraph {
            tokens: self.tokens,
            edges: self.edges.iter().copied().filter(|e| keep.contains(e)).collect(),
        }
    }

    pub fn is_dense(&self) -> bool {
        self.edges.len() == self.tokens * self.tokens
    }

    pub fn affected_edges(&self, changed: &[usize]) -> Vec<(usize, usize)> {
        let mut mark = vec![false; self.tokens];
        for t in changed {
            if *t < self.tokens {
                mark[*t] = true;
            }
        }
        self.edges.iter().copied().filter(|(a, b)| mark[*a] || mark[*b]).collect()
    }

    pub fn score_cells_for_changed(tokens: usize, changed_len: usize) -> usize {
        let changed = changed_len.min(tokens);
        2 * changed * tokens - changed * changed
    }
}
