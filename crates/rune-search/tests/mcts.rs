use rune_search::game::GameState;
use rune_search::mcts::{Mcts, MctsConfig};

#[derive(Debug, Clone)]
struct Toy {
    key: u64,
    moves: Vec<String>,
    terminal: bool,
    tv: f32,
}

impl GameState for Toy {
    type Move = String;
    fn legal_moves(&self) -> Vec<String> {
        self.moves.clone()
    }
    fn apply(&self, m: &String) -> Toy {
        Toy { key: self.key * 31 + m.len() as u64, moves: Vec::new(), terminal: true, tv: if m == "win" { 1.0 } else { -1.0 } }
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

#[test]
fn mcts_finds_better_prior() {
    let cfg = MctsConfig { simulations: 200, cpuct: 1.25, dirichlet_alpha: 0.3, dirichlet_eps: 0.0, fpu: 0.0, seed: 1 };
    let mut m = Mcts::new(cfg);
    let root = Toy { key: 7, moves: vec!["win".to_string(), "loss".to_string()], terminal: false, tv: 0.0 };
    let mut eval = |s: &Toy| -> (f32, Vec<f32>) {
        if s.key == 7 {
            (0.0, vec![0.8, 0.2])
        } else {
            (s.tv, Vec::new())
        }
    };
    let (mv, stats) = m.search(&root, &mut eval);
    assert_eq!(mv.unwrap(), "win");
    assert_eq!(stats.simulations, 200);
    assert!(stats.best_visit > stats.root_visits / 2);
}

#[test]
fn mcts_empty_returns_none() {
    let mut m = Mcts::new(MctsConfig::default());
    let root = Toy { key: 1, moves: Vec::new(), terminal: true, tv: 0.0 };
    let mut eval = |_: &Toy| (0.0, Vec::new());
    let (mv, _) = m.search(&root, &mut eval);
    assert!(mv.is_none());
}

#[test]
fn mcts_deterministic_same_seed() {
    let run = || {
        let mut m = Mcts::new(MctsConfig { simulations: 100, cpuct: 1.25, dirichlet_alpha: 0.3, dirichlet_eps: 0.25, fpu: 0.0, seed: 9 });
        let root = Toy { key: 3, moves: vec!["a".to_string(), "b".to_string(), "c".to_string()], terminal: false, tv: 0.0 };
        let mut eval = |s: &Toy| -> (f32, Vec<f32>) {
            if s.key == 3 {
                (0.0, vec![0.5, 0.3, 0.2])
            } else {
                (0.1, Vec::new())
            }
        };
        m.search(&root, &mut eval).0.unwrap()
    };
    assert_eq!(run(), run());
}
