use std::path::PathBuf;
use std::time::Instant;

fn bench_planes(ev: &rune_runtime::evaluator::Evaluator, planes: &[f32], iters: usize) -> f64 {
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = ev.evaluate_planes(planes);
    }
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn main() {
    let p = PathBuf::from("spec/test-vectors/models/resnet9-fp32.rune");
    let m = rune_model::load(&p).expect("fixture");
    let ev = rune_runtime::evaluator::Evaluator::from_model(&m).expect("evaluator");
    let go = rune_runtime::go::GoBoard::parse(
        "........./........./........./....X..../........./........./........./........./......... b",
    )
    .expect("parse");
    let planes = rune_runtime::go::extract_planes(&go);
    for iters in [20usize, 100] {
        let ms = bench_planes(&ev, &planes, iters);
        println!("resnet9x8x1 iters {} ms_per_eval {:.3}", iters, ms);
    }
    let big = synthetic(19, 64, 6);
    let planes19 = vec![0.0f32; 19 * 19];
    for iters in [3usize, 10] {
        let ms = bench_planes(&big, &planes19, iters);
        println!("resnet19x64x6 iters {} ms_per_eval {:.3}", iters, ms);
    }
}

fn synthetic(board: usize, channels: usize, blocks: usize) -> rune_runtime::evaluator::Evaluator {
    use rune_runtime::{ResnetConfig, ResnetWeights};
    let hw = board * board;
    let policy = board * board + 1;
    let h2 = 64;
    let mut rng = 1234567u64;
    let mut gen = |n: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push((((rng >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.2);
        }
        v
    };
    let wt = ResnetWeights {
        cfg: ResnetConfig { board, channels, blocks, policy_size: policy, value_h2: h2, in_planes: 1 },
        stem_w: gen(channels * 9),
        stem_b: vec![0.0; channels],
        block_w1: (0..blocks).map(|_| gen(channels * channels * 9)).collect(),
        block_b1: (0..blocks).map(|_| vec![0.0; channels]).collect(),
        block_w2: (0..blocks).map(|_| gen(channels * channels * 9)).collect(),
        block_b2: (0..blocks).map(|_| vec![0.0; channels]).collect(),
        vh1: gen(h2 * channels),
        bh1: vec![0.0; h2],
        wv: gen(h2),
        bv: 0.0,
        wwdl: gen(3 * h2),
        bwdl: vec![0.0; 3],
        policy: rune_runtime::policy_head::PolicyWeights {
            input: channels * hw,
            output: policy,
            w: gen(policy * channels * hw),
            b: vec![0.0; policy],
        },
    };
    let _ = &wt;
    let p = PathBuf::from("spec/test-vectors/models/resnet9-fp32.rune");
    let m = rune_model::load(&p).expect("fixture");
    let mut ev = rune_runtime::evaluator::Evaluator::from_model(&m).expect("evaluator");
    ev.set_resnet_for_bench(wt);
    ev
}
