use std::path::PathBuf;
use std::time::Instant;
use rune_runtime::board::Board;
use rune_runtime::compiled::CompiledEvaluator;
use rune_runtime::evaluator::Evaluator;
use rune_runtime::features::{compute_context, extract_features};

pub fn run(model: &str, compiled: &str, iters: usize) -> i32 {
    let mp = PathBuf::from(model);
    let cp = PathBuf::from(compiled);
    let mut ev_g = match Evaluator::load(&mp) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("generic load failed: {}", e);
            return 1;
        }
    };
    let mut ev_c = match CompiledEvaluator::load(&cp) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("compiled load failed: {}", e);
            return 1;
        }
    };
    let b = Board::startpos();
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = ev_g.evaluate_board(&b);
    }
    let g_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = ev_c.evaluate_board(&b);
    }
    let c_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let g_us = g_ms * 1000.0 / iters as f64;
    let c_us = c_ms * 1000.0 / iters as f64;
    println!("iters {}", iters);
    println!("generic_us_per_eval {:.3}", g_us);
    println!("compiled_us_per_eval {:.3}", c_us);
    println!("generic_eval_per_sec {:.0}", 1000000.0 / g_us);
    println!("compiled_eval_per_sec {:.0}", 1000000.0 / c_us);
    println!("speedup {:.2}x", g_us / c_us);
    println!("compiled_isa {}", ev_c.target_isa());
    println!("arena_bytes {}", ev_c.arena_bytes());
    let mut b1 = Board::startpos();
    let _ = b1.apply_uci("e2e4");
    let f0 = extract_features(&b);
    let f1 = extract_features(&b1);
    let c1 = compute_context(&b1);
    let t2 = Instant::now();
    for _ in 0..iters {
        ev_g.refresh(&b);
        ev_g.update_incremental(&f0, &f1, &c1);
        let _ = ev_g.evaluate();
    }
    let i_us = t2.elapsed().as_secs_f64() * 1000000.0 / iters as f64;
    println!("incremental_us_per_eval {:.3}", i_us);
    println!("incremental_eval_per_sec {:.0}", 1000000.0 / i_us);
    ev_g.refresh(&b);
    let t3 = Instant::now();
    for _ in 0..iters {
        let _ = ev_g.evaluate_value_only();
    }
    let v_us = t3.elapsed().as_secs_f64() * 1000000.0 / iters as f64;
    println!("value_only_us_per_eval {:.3}", v_us);
    println!("value_only_eval_per_sec {:.0}", 1000000.0 / v_us);
    0
}
