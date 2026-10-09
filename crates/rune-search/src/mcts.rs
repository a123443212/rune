use std::collections::HashMap;
use crate::game::GameState;

#[derive(Debug, Clone)]
pub struct MctsConfig {
    pub simulations: usize,
    pub cpuct: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_eps: f32,
    pub fpu: f32,
    pub seed: u64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig { simulations: 800, cpuct: 1.25, dirichlet_alpha: 0.3, dirichlet_eps: 0.25, fpu: 0.0, seed: 1 }
    }
}

#[derive(Debug, Clone)]
pub struct MctsStats {
    pub simulations: usize,
    pub root_visits: usize,
    pub best_visit: usize,
    pub entropy: f32,
}

struct Node {
    visits: u32,
    total: f32,
    prior: f32,
    children: Vec<usize>,
    moves: Vec<usize>,
    expanded: bool,
    value: f32,
}

pub struct Mcts<S: GameState> {
    cfg: MctsConfig,
    nodes: Vec<Node>,
    keys: Vec<u64>,
    parents: Vec<usize>,
    states: Vec<S>,
    rng: u64,
}

impl<S: GameState> Mcts<S> {
    pub fn new(cfg: MctsConfig) -> Self {
        Mcts { cfg, nodes: Vec::new(), keys: Vec::new(), parents: Vec::new(), states: Vec::new(), rng: 0x853c49e6748fea9b }
    }

    fn next_rand(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        ((self.rng >> 11) as f32) / (u64::MAX >> 11) as f32
    }

    fn select_child(&self, idx: usize) -> usize {
        let node = &self.nodes[idx];
        let total_visits: u32 = node.children.iter().map(|c| self.nodes[*c].visits).sum();
        let total = (total_visits as f32).sqrt();
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for c in node.children.iter() {
            let child = &self.nodes[*c];
            let q = if child.visits == 0 { node.value + self.cfg.fpu } else { child.total / child.visits as f32 };
            let u = self.cfg.cpuct * child.prior * total / (1.0 + child.visits as f32);
            let s = q + u;
            if s > best_score {
                best_score = s;
                best = *c;
            }
        }
        best
    }

    pub fn search<F>(&mut self, root: &S, eval: &mut F) -> (Option<S::Move>, MctsStats)
    where
        F: FnMut(&S) -> (f32, Vec<f32>),
    {
        self.nodes.clear();
        self.keys.clear();
        self.parents.clear();
        self.states.clear();
        self.rng = self.cfg.seed ^ 0x853c49e6748fea9b;
        let legal = root.legal_moves();
        if legal.is_empty() {
            return (None, MctsStats { simulations: 0, root_visits: 0, best_visit: 0, entropy: 0.0 });
        }
        let (v0, p0) = eval(root);
        let mut priors: Vec<f32> = vec![1.0 / legal.len().max(1) as f32; legal.len()];
        if p0.len() == legal.len() && !p0.is_empty() {
            priors.clone_from_slice(&p0);
        }
        if self.cfg.dirichlet_eps > 0.0 {
            let mut noise = vec![0.0f32; legal.len()];
            let mut s = 0.0f32;
            for i in 0..legal.len() {
                let g = -(-self.next_rand().max(1e-6)).ln().max(1e-6);
                noise[i] = g;
                s += g;
            }
            for i in 0..legal.len() {
                noise[i] /= s.max(1e-9);
                priors[i] = (1.0 - self.cfg.dirichlet_eps) * priors[i] + self.cfg.dirichlet_eps * noise[i];
            }
        }
        self.nodes.push(Node { visits: 0, total: 0.0, prior: 0.0, children: Vec::new(), moves: Vec::new(), expanded: true, value: v0 });
        self.keys.push(root.key());
        self.parents.push(usize::MAX);
        self.states.push(root.clone());
        let mut move_idx: Vec<usize> = (0..legal.len()).collect();
        let mut child_ids: Vec<usize> = Vec::new();
        for (i, m) in legal.iter().enumerate() {
            let st = root.apply(m);
            self.nodes.push(Node { visits: 0, total: 0.0, prior: priors[i], children: Vec::new(), moves: Vec::new(), expanded: false, value: 0.0 });
            self.keys.push(st.key());
            self.parents.push(0);
            self.states.push(st);
            child_ids.push(self.nodes.len() - 1);
            move_idx.push(i);
        }
        self.nodes[0].children = child_ids;
        self.nodes[0].moves = move_idx[legal.len()..].to_vec();
        let root_moves = legal;
        for _ in 0..self.cfg.simulations {
            let mut idx = 0usize;
            let mut path: Vec<usize> = vec![0];
            while self.nodes[idx].expanded && !self.nodes[idx].children.is_empty() {
                idx = self.select_child(idx);
                path.push(idx);
            }
            let leaf_value = if self.nodes[idx].expanded {
                let st = self.states[idx].clone();
                if st.is_terminal() {
                    st.terminal_value()
                } else {
                    self.nodes[idx].value
                }
            } else {
                let st = self.states[idx].clone();
                if st.is_terminal() {
                    let tv = st.terminal_value();
                    self.nodes[idx].expanded = true;
                    self.nodes[idx].value = tv;
                    tv
                } else {
                    let (vv, pp) = eval(&st);
                    let lm = st.legal_moves();
                    let mut pr = vec![1.0 / lm.len().max(1) as f32; lm.len()];
                    if pp.len() == lm.len() && !pp.is_empty() {
                        pr.clone_from_slice(&pp);
                    }
                    for (mi, m) in lm.iter().enumerate() {
                        let cst = st.apply(m);
                        self.nodes.push(Node { visits: 0, total: 0.0, prior: pr[mi], children: Vec::new(), moves: Vec::new(), expanded: false, value: 0.0 });
                        self.keys.push(cst.key());
                        self.parents.push(idx);
                        self.states.push(cst);
                        let cid = self.nodes.len() - 1;
                        self.nodes[idx].children.push(cid);
                        self.nodes[idx].moves.push(mi);
                    }
                    self.nodes[idx].expanded = true;
                    self.nodes[idx].value = vv;
                    vv
                }
            };
            let mut val = leaf_value;
            for pi in (0..path.len()).rev() {
                let ni = path[pi];
                self.nodes[ni].visits += 1;
                self.nodes[ni].total += val;
                val = -val;
            }
        }
        let root_node = &self.nodes[0];
        let mut best_i = 0usize;
        let mut best_v = 0u32;
        for (i, c) in root_node.children.iter().enumerate() {
            let vv = self.nodes[*c].visits;
            if i == 0 || vv > best_v {
                best_v = vv;
                best_i = i;
            }
        }
        let mut visits_sum = 0u32;
        for c in root_node.children.iter() {
            visits_sum += self.nodes[*c].visits;
        }
        let mut ent = 0.0f32;
        if visits_sum > 0 {
            for c in root_node.children.iter() {
                let p = self.nodes[*c].visits as f32 / visits_sum as f32;
                if p > 0.0 {
                    ent -= p * p.ln();
                }
            }
        }
        let stats = MctsStats { simulations: self.cfg.simulations, root_visits: visits_sum as usize, best_visit: best_v as usize, entropy: ent };
        let _ = HashMap::<u64, u64>::new();
        (root_moves.get(best_i).cloned(), stats)
    }
}
