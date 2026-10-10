// RUNE — Relational Unified Neural Evaluator
// Copyright (C) 2026 a123443212
//
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// This project is dual-licensed under the MIT License and the
// Apache License, Version 2.0. You may choose either license
// when using, copying, modifying, or distributing this software.
//
// MIT License: https://opensource.org/license/mit
// Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, this
// software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
// OR CONDITIONS OF ANY KIND, either express or implied.

use crate::game::GameState;
use crate::select::select_child;

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

pub(crate) struct Node {
    pub(crate) visits: u32,
    pub(crate) total: f32,
    pub(crate) prior: f32,
    pub(crate) children: Vec<usize>,
    pub(crate) expanded: bool,
    pub(crate) value: f32,
}

pub struct Mcts<S: GameState> {
    cfg: MctsConfig,
    nodes: Vec<Node>,
    states: Vec<S>,
    path: Vec<usize>,
    rng: u64,
}

impl<S: GameState> Mcts<S> {
    pub fn new(cfg: MctsConfig) -> Self {
        Mcts { cfg, nodes: Vec::new(), states: Vec::new(), path: Vec::new(), rng: 0x853c49e6748fea9b }
    }

    fn next_rand(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        ((self.rng >> 11) as f32) / (u64::MAX >> 11) as f32
    }

    fn expand(&mut self, idx: usize, eval: &mut impl FnMut(&S) -> (f32, Vec<f32>)) -> f32 {
        let st = self.states[idx].clone();
        if st.is_terminal() {
            let tv = st.terminal_value();
            self.nodes[idx].expanded = true;
            self.nodes[idx].value = tv;
            return tv;
        }
        let (vv, pp) = eval(&st);
        let lm = st.legal_moves();
        let n = lm.len();
        self.nodes.reserve(n);
        self.states.reserve(n);
        let base = self.nodes.len();
        for (mi, m) in lm.iter().enumerate() {
            let pr = if pp.len() == n && n > 0 { pp[mi] } else if n > 0 { 1.0 / n as f32 } else { 0.0 };
            let cst = st.apply(m);
            self.nodes.push(Node { visits: 0, total: 0.0, prior: pr, children: Vec::new(), expanded: false, value: 0.0 });
            self.states.push(cst);
            self.nodes[idx].children.push(base + mi);
        }
        self.nodes[idx].expanded = true;
        self.nodes[idx].value = vv;
        vv
    }

    pub fn search<F>(&mut self, root: &S, eval: &mut F) -> (Option<S::Move>, MctsStats)
    where
        F: FnMut(&S) -> (f32, Vec<f32>),
    {
        self.nodes.clear();
        self.states.clear();
        self.path.clear();
        self.rng = self.cfg.seed ^ 0x853c49e6748fea9b;
        let legal = root.legal_moves();
        if legal.is_empty() {
            return (None, MctsStats { simulations: 0, root_visits: 0, best_visit: 0, entropy: 0.0 });
        }
        let (v0, p0) = eval(root);
        let n = legal.len();
        let mut priors: Vec<f32> = vec![1.0 / n as f32; n];
        if p0.len() == n && n > 0 {
            priors.clone_from_slice(&p0);
        }
        if self.cfg.dirichlet_eps > 0.0 {
            let mut s = 0.0f32;
            for i in 0..n {
                let g = -(-self.next_rand().max(1e-6)).ln().max(1e-6);
                priors[i] = g;
                s += g;
            }
            let inv = 1.0 / s.max(1e-9);
            for i in 0..n {
                priors[i] *= inv;
            }
            if p0.len() == n && n > 0 {
                for i in 0..n {
                    priors[i] = (1.0 - self.cfg.dirichlet_eps) * p0[i] + self.cfg.dirichlet_eps * priors[i];
                }
            } else {
                let u = 1.0 / n as f32;
                for i in 0..n {
                    priors[i] = (1.0 - self.cfg.dirichlet_eps) * u + self.cfg.dirichlet_eps * priors[i];
                }
            }
        }
        let cap = n * 4 + self.cfg.simulations * 2;
        self.nodes.reserve(cap);
        self.states.reserve(cap);
        self.path.reserve(64);
        self.nodes.push(Node { visits: 0, total: 0.0, prior: 0.0, children: Vec::with_capacity(n), expanded: true, value: v0 });
        self.states.push(root.clone());
        for (i, m) in legal.iter().enumerate() {
            let st = root.apply(m);
            self.nodes.push(Node { visits: 0, total: 0.0, prior: priors[i], children: Vec::new(), expanded: false, value: 0.0 });
            self.states.push(st);
            self.nodes[0].children.push(i + 1);
        }
        let root_moves = legal;
        for _ in 0..self.cfg.simulations {
            let mut idx = 0usize;
            self.path.clear();
            self.path.push(0);
            while self.nodes[idx].expanded && !self.nodes[idx].children.is_empty() {
                idx = select_child(&self.nodes, idx, self.cfg.cpuct, self.cfg.fpu);
                self.path.push(idx);
            }
            let leaf_value = if self.nodes[idx].expanded {
                let st = &self.states[idx];
                if st.is_terminal() {
                    st.terminal_value()
                } else {
                    self.nodes[idx].value
                }
            } else {
                self.expand(idx, eval)
            };
            let mut val = leaf_value;
            for pi in (0..self.path.len()).rev() {
                let ni = self.path[pi];
                self.nodes[ni].visits += 1;
                self.nodes[ni].total += val;
                val = -val;
            }
        }
        let mut best_i = 0usize;
        let mut best_v = 0u32;
        let mut visits_sum = 0u32;
        for (i, c) in self.nodes[0].children.iter().enumerate() {
            let vv = self.nodes[*c].visits;
            visits_sum += vv;
            if i == 0 || vv > best_v {
                best_v = vv;
                best_i = i;
            }
        }
        let mut ent = 0.0f32;
        if visits_sum > 0 {
            let inv = 1.0 / visits_sum as f32;
            for c in self.nodes[0].children.iter() {
                let p = self.nodes[*c].visits as f32 * inv;
                if p > 0.0 {
                    ent -= p * p.ln();
                }
            }
        }
        let stats = MctsStats { simulations: self.cfg.simulations, root_visits: visits_sum as usize, best_visit: best_v as usize, entropy: ent };
        (root_moves.get(best_i).cloned(), stats)
    }
}
