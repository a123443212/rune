use crate::game::GameState;
use crate::mcts::Node;

pub fn select_child<S: GameState>(nodes: &[Node], idx: usize, cpuct: f32, fpu: f32) -> usize {
    let node = &nodes[idx];
    let mut total_visits = 0u32;
    for c in node.children.iter() {
        total_visits += nodes[*c].visits;
    }
    let total = (total_visits as f32).sqrt();
    let base = node.value + fpu;
    let mut best = node.children[0];
    let mut best_score = f32::NEG_INFINITY;
    for c in node.children.iter() {
        let child = &nodes[*c];
        let q = if child.visits == 0 { base } else { child.total / child.visits as f32 };
        let s = q + cpuct * child.prior * total / (1.0 + child.visits as f32);
        if s > best_score {
            best_score = s;
            best = *c;
        }
    }
    best
}
