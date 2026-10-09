use rune_search::game::GameState;
use rune_search::mcts::{Mcts, MctsConfig};

#[derive(Debug, Clone)]
pub struct StringGame {
    pub key: u64,
    pub moves: Vec<String>,
    pub terminal: bool,
    pub tv: f32,
}

impl GameState for StringGame {
    type Move = String;
    fn legal_moves(&self) -> Vec<String> {
        self.moves.clone()
    }
    fn apply(&self, m: &String) -> StringGame {
        StringGame { key: self.key.wrapping_mul(31).wrapping_add(m.len() as u64), moves: Vec::new(), terminal: true, tv: 0.0 }
    }
    fn is_terminal(&self) -> bool {
        self.terminal
    }
    fn terminal_value(&self) -> f32 {
        self.tv
    }
    fn key(&self) -> u64 {
        self.key
    }
}

pub fn run_mcts_demo(sims: usize, cpuct: f32, seed: u64) -> i32 {
    let cfg = MctsConfig { simulations: sims, cpuct, dirichlet_alpha: 0.3, dirichlet_eps: 0.25, fpu: 0.0, seed };
    let mut mcts = Mcts::new(cfg);
    let root = StringGame { key: 1, moves: vec!["a".to_string(), "b".to_string()], terminal: false, tv: 0.0 };
    let mut eval = |s: &StringGame| -> (f32, Vec<f32>) {
        if s.key == 1 {
            (0.0, vec![0.6, 0.4])
        } else {
            (if s.key % 2 == 0 { 0.5 } else { -0.5 }, Vec::new())
        }
    };
    let (mv, stats) = mcts.search(&root, &mut eval);
    println!("move {}", mv.unwrap_or_default());
    println!("sims {}", stats.simulations);
    println!("root_visits {}", stats.root_visits);
    println!("best_visit {}", stats.best_visit);
    println!("entropy {:.4}", stats.entropy);
    0
}
