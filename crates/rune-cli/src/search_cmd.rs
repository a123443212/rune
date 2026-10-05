use std::cell::Cell;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;
use rune_runtime::board::Board;
use rune_runtime::evaluator::Evaluator;
use rune_search::lazy::{LazyConfig, LazyMode};

pub fn run(model: &str, fen: &str, depth: usize, lazy: &str) -> i32 {
    let p = PathBuf::from(model);
    let mut ev = match Evaluator::load(&p) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("load failed: {}", e);
            return 1;
        }
    };
    let cfg = LazyConfig {
        mode: match lazy {
            "L1" => LazyMode::L1,
            "L2" => LazyMode::L2,
            _ => LazyMode::L0,
        },
        margin: 0.08,
        max_refine: 1,
    };
    let count = Rc::new(Cell::new(0usize));
    let cc = count.clone();
    let mut eval_fn = move |f: &str| {
        cc.set(cc.get() + 1);
        match Board::parse_fen(f) {
            Ok(b) => ev.evaluate_board(&b).value,
            Err(_) => 0.0,
        }
    };
    let t0 = Instant::now();
    let mut ab = rune_search::AlphaBeta::new(&mut eval_fn, cfg);
    let (score, mv) = ab.search(fen, depth);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("score {:.4}", score);
    println!("engine_score {}", rune_search::contract::engine_score(score));
    println!("move {}", mv.unwrap_or_default());
    println!("nodes {}", ab.stats.nodes);
    println!("evals {}", count.get());
    println!("cutoffs {}", ab.stats.cutoffs);
    println!("ms {:.2}", ms);
    println!("nps {:.0}", ab.stats.nps());
    println!("root {} pv {} cut {} leaf {}", ab.stats.root_count, ab.stats.pv_count, ab.stats.cut_count, ab.stats.leaf_count);
    0
}
