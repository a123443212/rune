use std::path::PathBuf;
use rune_runtime::board::Board;
use rune_runtime::compiled::CompiledEvaluator;
use rune_runtime::evaluator::Evaluator;

pub fn run(generic: &str, compiled: &str, positions: &str, tol: f32) -> i32 {
    let data = match std::fs::read_to_string(positions) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("positions failed: {}", e);
            return 1;
        }
    };
    let gp = PathBuf::from(generic);
    let cp = PathBuf::from(compiled);
    let mut ev_g = match Evaluator::load(&gp) {
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
    let mut maxd: f32 = 0.0;
    let mut n = 0;
    for line in data.lines() {
        let fen = line.trim();
        if fen.is_empty() {
            continue;
        }
        let b = match Board::parse_fen(fen) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("bad fen {}: {}", fen, e);
                return 1;
            }
        };
        let r1 = ev_g.evaluate_board(&b);
        let r2 = ev_c.evaluate_board(&b);
        let d = (r1.value - r2.value).abs();
        if d > maxd {
            maxd = d;
        }
        println!("pos {} generic {:.6} compiled {:.6} diff {:.2e}", n, r1.value, r2.value, d);
        n += 1;
    }
    println!("compared {} max_abs_diff {:.9} tol {:.9} {}", n, maxd, tol, if maxd <= tol { "PASS" } else { "FAIL" });
    if maxd <= tol {
        0
    } else {
        2
    }
}
