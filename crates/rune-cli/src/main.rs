use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use rune_runtime::board::Board;
use rune_runtime::evaluator::Evaluator;
use rune_runtime::features::extract_features;
fn arg_val(args: &[String], key: &str) -> Option<String> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == key && i + 1 < args.len() {
            return Some(args[i + 1].clone());
        }
        i += 1;
    }
    None
}
fn cmd_model_info(model: &str) -> i32 {
    let p = PathBuf::from(model);
    match rune_model::load(&p) {
        Ok(m) => {
            println!("architecture_id {}", m.header.architecture_id);
            println!("architecture_version {}", m.header.architecture_version);
            println!("feature_version {}", m.header.feature_version);
            println!("tokens {} dim {}", m.header.tokens, m.header.token_dim);
            println!("quantization {}", m.header.quantization);
            println!("format {}", m.header.format);
            let mut n: usize = 0;
            for t in &m.header.tensors {
                let mut c: usize = 1;
                for d in &t.shape {
                    c *= *d;
                }
                n += c;
            }
            println!("tensors {} params {}", m.header.tensors.len(), n);
            for t in &m.header.tensors {
                println!("tensor {} {:?} {}", t.name, t.shape, t.dtype);
            }
            println!("model_hash {}", m.header.model_hash);
            0
        }
        Err(e) => {
            eprintln!("load failed: {}", e);
            1
        }
    }
}
fn cmd_eval(model: &str, fen: &str, kernel: &str) -> i32 {
    match kernel {
        "scalar" => rune_kernel::set_path_for_test(rune_kernel::KernelPath::Scalar),
        "simd" => rune_kernel::set_path_for_test(rune_kernel::KernelPath::Simd),
        _ => rune_kernel::clear_path_for_test(),
    }    let p = PathBuf::from(model);
    let mut ev = match Evaluator::load(&p) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("load failed: {}", e);
            return 1;
        }
    };
    if ev.game_id() == "shogi" {
        let r = match ev.evaluate_sfen(fen) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("bad sfen: {}", e);
                return 1;
            }
        };
        println!("value {:.6}", r.value);
        println!("wdl {:.6} {:.6} {:.6}", r.wdl[0], r.wdl[1], r.wdl[2]);
        println!("arch {}", ev.arch_id());
        println!("path {}", ev.kernel_path());
        return 0;
    }
    if ev.game_id() == "xiangqi" {
        let r = match ev.evaluate_xiangqi(fen) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("bad xiangqi fen: {}", e);
                return 1;
            }
        };
        println!("value {:.6}", r.value);
        println!("wdl {:.6} {:.6} {:.6}", r.wdl[0], r.wdl[1], r.wdl[2]);
        println!("arch {}", ev.arch_id());
        println!("path {}", ev.kernel_path());
        return 0;
    }
    let b = match Board::parse_fen(fen) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("bad fen: {}", e);
            return 1;
        }
    };
    let r = ev.evaluate_board(&b);
    println!("value {:.6}", r.value);
    println!("wdl {:.6} {:.6} {:.6}", r.wdl[0], r.wdl[1], r.wdl[2]);
    println!("arch {}", ev.arch_id());
    println!("path {}", ev.kernel_path());
    0
}
fn cmd_inspect(fen: &str) -> i32 {
    let b = match Board::parse_fen(fen) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("bad fen: {}", e);
            return 1;
        }
    };
    let f = extract_features(&b);
    println!("fen {}", b.to_fen());
    println!("pieces {}", b.piece_count());
    println!("phase {}", b.game_phase());
    println!("features {}", f.len());
    for (g, i) in f {
        println!("f {} {}", g, i);
    }
    0
}
fn cmd_bench(model: &str, iters: usize, kernel: &str) -> i32 {
    match kernel {
        "scalar" => rune_kernel::set_path_for_test(rune_kernel::KernelPath::Scalar),
        "simd" => rune_kernel::set_path_for_test(rune_kernel::KernelPath::Simd),
        _ => rune_kernel::clear_path_for_test(),
    }    let p = PathBuf::from(model);
    let mut ev = match Evaluator::load(&p) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("load failed: {}", e);
            return 1;
        }
    };
    let b = Board::startpos();
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = ev.evaluate_board(&b);
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let us = ms * 1000.0 / iters as f64;
    println!("iters {}", iters);
    println!("total_ms {:.2}", ms);
    println!("us_per_eval {:.3}", us);
    println!("eval_per_sec {:.0}", 1000000.0 / us);
    println!("path {}", ev.kernel_path());
    0
}
fn cmd_diff(model: &str, positions: &str, tol: f32) -> i32 {
    let data = match fs::read_to_string(positions) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("positions read failed: {}", e);
            return 1;
        }
    };
    let p = PathBuf::from(model);
    let mut ev = match Evaluator::load(&p) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("load failed: {}", e);
            return 1;
        }
    };
    let mut maxd: f32 = 0.0;
    let mut bad = false;
    let mut n = 0;
    for line in data.lines() {
        let fen = line.trim();
        if fen.is_empty() {
            continue;
        }
        let b = match Board::parse_fen(fen) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("bad fen {} : {}", fen, e);
                return 1;
            }
        };
        let r1 = ev.evaluate_board(&b);
        let before = ev.current_features().to_vec();
        let mut ev2_tokens = {
            let tr = ev.trace();
            tr.tokens.clone()
        };
        let r2 = ev.forward_tokens(&ev2_tokens);
        let d = (r1.value - r2.value).abs();
        if d.is_nan() {
            bad = true;
        } else if d > maxd {
            maxd = d;
        }
        let _ = before;
        let _ = &mut ev2_tokens;
        println!("pos {} value {:.6} wdl {:.4} {:.4} {:.4}", n, r1.value, r1.wdl[0], r1.wdl[1], r1.wdl[2]);
        n += 1;
    }
    let pass = !bad && maxd <= tol;
    println!("compared {} max_abs_diff {:.9} tol {:.9} {}", n, maxd, tol, if pass { "PASS" } else { "FAIL" });
    if pass {
        0
    } else {
        2
    }
}
mod bench_cmd;
mod compile_cmd;
mod diff_cmd;
mod inspect_cmd;
mod search_cmd;
mod search_diff_cmd;
fn usage() {
    println!("rune eval --model <path> --fen <fen> --kernel <auto|scalar|simd>");
    println!("rune inspect --fen <fen>");
    println!("rune bench --model <path> --iters <n> --kernel <auto|scalar|simd>");
    println!("rune diff --model <path> --positions <file> --tol <f>");
    println!("rune model-info --model <path>");
    println!("rune compile --model <path> --out <path> --isa <portable|avx2|avx512> --cpu <name>");
    println!("rune bench-compile --model <path> --compiled <path> --iters <n>");
    println!("rune inspect-compiled --model <path>");
    println!("rune diff-compiled --generic <path> --compiled <path> --positions <file> --tol <f>");
    println!("rune search --model <path> --fen <fen> --depth <n> --lazy <L0|L1|L2>");
    println!("rune search-diff --model-a <path> --model-b <path> --fen <fen> --depth <n> --tol <f>");
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
        std::process::exit(1);
    }
    let cmd = args[1].clone();
    let rest = args[2..].to_vec();
    let code = if cmd == "model-info" {
        match arg_val(&rest, "--model") {
            Some(m) => cmd_model_info(&m),
            None => {
                usage();
                1
            }
        }
    } else if cmd == "eval" {
        let m = arg_val(&rest, "--model");
        let f = arg_val(&rest, "--fen");
        let k = arg_val(&rest, "--kernel").unwrap_or_else(|| "auto".to_string());
        match (m, f) {
            (Some(mm), Some(ff)) => cmd_eval(&mm, &ff, &k),
            _ => {
                usage();
                1
            }
        }
    } else if cmd == "inspect" {
        match arg_val(&rest, "--fen") {
            Some(ff) => cmd_inspect(&ff),
            None => {
                usage();
                1
            }
        }
    } else if cmd == "bench" {
        let m = arg_val(&rest, "--model").unwrap_or_default();
        let it = arg_val(&rest, "--iters").and_then(|x| x.parse::<usize>().ok()).unwrap_or(2000);
        let k = arg_val(&rest, "--kernel").unwrap_or_else(|| "auto".to_string());
        if m.is_empty() {
            usage();
            1
        } else {
            cmd_bench(&m, it, &k)
        }
    } else if cmd == "diff" {
        let m = arg_val(&rest, "--model").unwrap_or_default();
        let pos = arg_val(&rest, "--positions").unwrap_or_default();
        let tol = arg_val(&rest, "--tol").and_then(|x| x.parse::<f32>().ok()).unwrap_or(1e-5);
        if m.is_empty() || pos.is_empty() {
            usage();
            1
        } else {
            cmd_diff(&m, &pos, tol)
        }
    } else if cmd == "compile" {
        let m = arg_val(&rest, "--model").unwrap_or_default();
        let o = arg_val(&rest, "--out").unwrap_or_default();
        let isa = arg_val(&rest, "--isa").unwrap_or_else(|| "portable".to_string());
        let cpu = arg_val(&rest, "--cpu").unwrap_or_else(|| "generic-x86-64".to_string());
        if m.is_empty() || o.is_empty() {
            usage();
            1
        } else {
            compile_cmd::run(&m, &o, &isa, &cpu)
        }
    } else if cmd == "bench-compile" {
        let m = arg_val(&rest, "--model").unwrap_or_default();
        let c = arg_val(&rest, "--compiled").unwrap_or_default();
        let it = arg_val(&rest, "--iters").and_then(|x| x.parse::<usize>().ok()).unwrap_or(2000);
        if m.is_empty() || c.is_empty() {
            usage();
            1
        } else {
            bench_cmd::run(&m, &c, it)
        }
    } else if cmd == "inspect-compiled" {
        let m = arg_val(&rest, "--model").unwrap_or_default();
        if m.is_empty() {
            usage();
            1
        } else {
            inspect_cmd::run(&m)
        }
    } else if cmd == "diff-compiled" {
        let g = arg_val(&rest, "--generic").unwrap_or_default();
        let c = arg_val(&rest, "--compiled").unwrap_or_default();
        let pos = arg_val(&rest, "--positions").unwrap_or_default();
        let tol = arg_val(&rest, "--tol").and_then(|x| x.parse::<f32>().ok()).unwrap_or(2e-5);
        if g.is_empty() || c.is_empty() || pos.is_empty() {
            usage();
            1
        } else {
            diff_cmd::run(&g, &c, &pos, tol)
        }
    } else if cmd == "search" {
        let m = arg_val(&rest, "--model").unwrap_or_default();
        let f = arg_val(&rest, "--fen").unwrap_or_else(|| "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string());
        let d = arg_val(&rest, "--depth").and_then(|x| x.parse::<usize>().ok()).unwrap_or(2);
        let l = arg_val(&rest, "--lazy").unwrap_or_else(|| "L0".to_string());
        let t = arg_val(&rest, "--threshold").and_then(|x| x.parse::<f32>().ok()).unwrap_or(0.5);
        if m.is_empty() {
            usage();
            1
        } else {
            search_cmd::run(&m, &f, d, &l, t)
        }
    } else if cmd == "search-diff" {
        let a = arg_val(&rest, "--model-a").unwrap_or_default();
        let b = arg_val(&rest, "--model-b").unwrap_or_default();
        let f = arg_val(&rest, "--fen").unwrap_or_else(|| "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string());
        let d = arg_val(&rest, "--depth").and_then(|x| x.parse::<usize>().ok()).unwrap_or(2);
        let t = arg_val(&rest, "--tol").and_then(|x| x.parse::<f32>().ok()).unwrap_or(1e-6);
        if a.is_empty() || b.is_empty() {
            usage();
            1
        } else {
            search_diff_cmd::run(&a, &b, &f, d, t)
        }
    } else {
        usage();
        1
    };
    std::process::exit(code);
}
