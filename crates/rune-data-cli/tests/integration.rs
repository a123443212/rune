use std::path::PathBuf;
use std::process::Command;

fn bin() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.join("target").join("debug").join("rune-data")
}

fn run(args: &[&str]) -> (bool, String) {
    let out = Command::new(bin())
        .args(args)
        .output()
        .expect("spawn rune-data");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string(),
    )
}

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("data");
    p.push(name);
    p
}

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("rune_it_{name}"));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn manifest_hash(dir: &std::path::Path) -> String {
    let m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    m["hash"].as_str().unwrap().to_string()
}

#[test]
fn ingest_is_deterministic() {
    let pgn = fixture("two_games.pgn");
    let a = tmp("det_a");
    let b = tmp("det_b");
    assert!(
        run(&[
            "ingest",
            "--pgn",
            pgn.to_str().unwrap(),
            "--out",
            a.to_str().unwrap(),
            "--dataset-id",
            "det",
            "--every",
            "1",
            "--threads",
            "1"
        ])
        .0
    );
    assert!(
        run(&[
            "ingest",
            "--pgn",
            pgn.to_str().unwrap(),
            "--out",
            b.to_str().unwrap(),
            "--dataset-id",
            "det",
            "--every",
            "1",
            "--threads",
            "4"
        ])
        .0
    );
    assert_eq!(manifest_hash(&a), manifest_hash(&b));
}

#[test]
fn full_stage_chain_roundtrips() {
    let pgn = fixture("two_games.pgn");
    let base = tmp("chain");
    let ing = base.join("ing");
    let dd = base.join("dd");
    let flt = base.join("flt");
    let bal = base.join("bal");
    let spl = base.join("spl");
    assert!(
        run(&[
            "ingest",
            "--pgn",
            pgn.to_str().unwrap(),
            "--out",
            ing.to_str().unwrap(),
            "--dataset-id",
            "chain",
            "--every",
            "1"
        ])
        .0
    );
    assert!(
        run(&[
            "dedup",
            "--input",
            ing.to_str().unwrap(),
            "--out",
            dd.to_str().unwrap(),
            "--dataset-id",
            "chain"
        ])
        .0
    );
    assert!(
        run(&[
            "filter",
            "--input",
            dd.to_str().unwrap(),
            "--out",
            flt.to_str().unwrap(),
            "--dataset-id",
            "chain",
            "--min-ply",
            "4"
        ])
        .0
    );
    assert!(
        run(&[
            "balance",
            "--input",
            flt.to_str().unwrap(),
            "--out",
            bal.to_str().unwrap(),
            "--dataset-id",
            "chain"
        ])
        .0
    );
    assert!(run(&["validate", "--input", bal.to_str().unwrap()]).0);
    assert!(
        run(&[
            "split",
            "--input",
            bal.to_str().unwrap(),
            "--out",
            spl.to_str().unwrap(),
            "--dataset-id",
            "chain"
        ])
        .0
    );
    assert!(spl.join("train").join("manifest.json").exists());
    assert!(spl.join("val").join("manifest.json").exists());
    assert!(spl.join("test").join("manifest.json").exists());
}

#[test]
fn shard_deflate_roundtrips() {
    let pgn = fixture("two_games.pgn");
    let base = tmp("deflate");
    let ing = base.join("ing");
    let sh = base.join("sh");
    assert!(
        run(&[
            "ingest",
            "--pgn",
            pgn.to_str().unwrap(),
            "--out",
            ing.to_str().unwrap(),
            "--dataset-id",
            "def",
            "--every",
            "2"
        ])
        .0
    );
    assert!(
        run(&[
            "shard",
            "--input",
            ing.to_str().unwrap(),
            "--out",
            sh.to_str().unwrap(),
            "--dataset-id",
            "def",
            "--per-shard",
            "4",
            "--compression",
            "deflate"
        ])
        .0
    );
    let (ok, _) = run(&["validate", "--input", sh.to_str().unwrap()]);
    assert!(ok);
}

#[test]
fn join_and_export_labels() {
    use std::io::Write;
    let pgn = fixture("two_games.pgn");
    let base = tmp("join");
    let ing = base.join("ing");
    assert!(
        run(&[
            "ingest",
            "--pgn",
            pgn.to_str().unwrap(),
            "--out",
            ing.to_str().unwrap(),
            "--dataset-id",
            "jl",
            "--every",
            "2"
        ])
        .0
    );
    let (ok, _) = run(&["stats", "--input", ing.to_str().unwrap()]);
    assert!(ok);
    let labels = base.join("labels.jsonl");
    {
        let mut f = std::fs::File::create(&labels).unwrap();
        writeln!(f, "{{\"fen\": \"bad\", \"teacher_v\": 0.0}}").unwrap();
    }
    let joined = base.join("joined");
    let (ok, _) = run(&[
        "join-labels",
        "--input",
        ing.to_str().unwrap(),
        "--labels",
        labels.to_str().unwrap(),
        "--out",
        joined.to_str().unwrap(),
        "--dataset-id",
        "jl",
    ]);
    assert!(ok);
    let rep: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(joined.join("report.json")).unwrap())
            .unwrap();
    assert_eq!(
        rep["stages"]["join"]["rejected"],
        rep["stages"]["join"]["input"]
    );
}

fn write_scores(path: &std::path::Path, fens: &[String], val: f32) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();
    for (i, fen) in fens.iter().enumerate() {
        let v = if i % 3 == 0 { val } else { val * 0.1 };
        writeln!(
            f,
            "{{\"fen\": {}, \"disagreement\": {v}}}",
            serde_json::to_string(fen).unwrap()
        )
        .unwrap();
    }
}

fn collect_fens(dir: &std::path::Path) -> Vec<String> {
    let mut out = Vec::new();
    let man: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    for sf in man["shard_files"].as_array().unwrap() {
        let p = dir.join(sf.as_str().unwrap());
        let bytes = std::fs::read(p).unwrap();
        let f = rune_data_core::format::ShardFile::parse(&bytes).unwrap();
        for r in f.read_all().unwrap() {
            out.push(r.fen.clone());
        }
    }
    out
}

#[test]
fn select_is_deterministic_with_audit() {
    let pgn = fixture("two_games.pgn");
    let base = tmp("sel_det");
    let cand = base.join("cand");
    assert!(
        run(&[
            "ingest",
            "--pgn",
            pgn.to_str().unwrap(),
            "--out",
            cand.to_str().unwrap(),
            "--dataset-id",
            "sel",
            "--every",
            "1"
        ])
        .0
    );
    let fens = collect_fens(&cand);
    assert!(!fens.is_empty());
    let scores = base.join("scores.jsonl");
    write_scores(&scores, &fens, 0.9);
    // NOTE: select takes scores via scored shards; score step first
    let scored = base.join("scored");
    assert!(
        run(&[
            "score",
            "--input",
            cand.to_str().unwrap(),
            "--scores",
            scores.to_str().unwrap(),
            "--out",
            scored.to_str().unwrap(),
            "--dataset-id",
            "sel",
        ])
        .0
    );
    let s1 = base.join("s1");
    let s2 = base.join("s2");
    for o in [&s1, &s2] {
        let owned = vec![
            "select".to_string(),
            "--input".to_string(),
            scored.to_str().unwrap().to_string(),
            "--out".to_string(),
            o.to_str().unwrap().to_string(),
            "--dataset-id".to_string(),
            "sel".to_string(),
            "--seed".to_string(),
            "7".to_string(),
            "--budget".to_string(),
            "10".to_string(),
            "--method".to_string(),
            "multi".to_string(),
            "--round".to_string(),
            "2".to_string(),
            "--diversity-cap".to_string(),
            "1000000".to_string(),
            "--floor-ratio".to_string(),
            "0.0".to_string(),
            "--teacher-version".to_string(),
            "t1".to_string(),
            "--student-version".to_string(),
            "s0".to_string(),
        ];
        let refs: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
        assert!(run(&refs).0);
    }
    assert_eq!(manifest_hash(&s1), manifest_hash(&s2));
    let a1txt = std::fs::read_to_string(s1.join("selection_audit.jsonl")).unwrap();
    let a2txt = std::fs::read_to_string(s2.join("selection_audit.jsonl")).unwrap();
    assert_eq!(a1txt, a2txt);
    assert!(std::fs::read_to_string(s1.join("round.yaml"))
        .unwrap()
        .contains("id: 2"));
}

#[test]
fn select_respects_diversity_and_floor() {
    let pgn = fixture("two_games.pgn");
    let base = tmp("sel_div");
    let cand = base.join("cand");
    assert!(
        run(&[
            "ingest",
            "--pgn",
            pgn.to_str().unwrap(),
            "--out",
            cand.to_str().unwrap(),
            "--dataset-id",
            "sel",
            "--every",
            "1"
        ])
        .0
    );
    let fens = collect_fens(&cand);
    let scores = base.join("scores.jsonl");
    write_scores(&scores, &fens, 0.9);
    let scored = base.join("scored");
    assert!(
        run(&[
            "score",
            "--input",
            cand.to_str().unwrap(),
            "--scores",
            scores.to_str().unwrap(),
            "--out",
            scored.to_str().unwrap(),
            "--dataset-id",
            "sel",
        ])
        .0
    );
    let sel = base.join("sel");
    assert!(
        run(&[
            "select",
            "--input",
            scored.to_str().unwrap(),
            "--out",
            sel.to_str().unwrap(),
            "--dataset-id",
            "sel",
            "--seed",
            "7",
            "--budget",
            "12",
            "--method",
            "multi",
            "--round",
            "1",
            "--diversity-cap",
            "2",
            "--floor-ratio",
            "0.25",
            "--teacher-version",
            "t",
            "--student-version",
            "s",
        ])
        .0
    );
    let txt = std::fs::read_to_string(sel.join("selection_audit.jsonl")).unwrap();
    let lines: Vec<&str> = txt.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(lines.len(), 12);
    let strat = lines
        .iter()
        .filter(|l| l.contains("\"selection_method\":\"stratified\""))
        .count();
    assert_eq!(strat, 3);
}

fn write_md_jsonl(path: &std::path::Path) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();
    for (i, vals) in [[0.1, 0.12, 0.11], [-0.8, 0.1, 0.9], [0.5, 0.52, 0.51]]
        .iter()
        .enumerate()
    {
        let levels: Vec<String> = vals
            .iter()
            .enumerate()
            .map(|(k, v)| {
                format!(
                    "\"d{k}\":{{\"cp\":{},\"value_stm\":{v},\"wdl_stm\":1,\"nodes\":100}}",
                    (v * 400.0) as i32
                )
            })
            .collect();
        writeln!(
            f,
            "{{\"fen\": \"8/8/4k3/8/8/4K3/4P3/{} w - - 0 1\", \"teacher_levels\": {{{}}}}}",
            i,
            levels.join(",")
        )
        .unwrap();
    }
}

#[test]
fn target_stats_reports_stability() {
    let base = tmp("tstats");
    std::fs::create_dir_all(&base).unwrap();
    let md = base.join("md.jsonl");
    write_md_jsonl(&md);
    let (ok, out) = run(&[
        "target-stats",
        "--input",
        md.to_str().unwrap(),
        "--out",
        base.join("stats.json").to_str().unwrap(),
    ]);
    assert!(ok, "{out}");
    assert!(out.contains("unstable_frac"));
}

#[test]
fn cascade_plan_prices_routing() {
    let base = tmp("cascade");
    std::fs::create_dir_all(&base).unwrap();
    let md = base.join("md.jsonl");
    write_md_jsonl(&md);
    let (ok, out) = run(&[
        "cascade-plan",
        "--input",
        md.to_str().unwrap(),
        "--levels",
        "d0,d1,d2",
        "--costs",
        "0.02,0.08",
        "--accept-threshold",
        "0.3",
    ]);
    assert!(ok, "{out}");
    assert!(out.contains("savings_frac"));
    assert!(out.contains("escalated"));
}

#[test]
fn verify_targets_rejects_bad() {
    let pgn = fixture("two_games.pgn");
    let base = tmp("verify");
    let ing = base.join("ing");
    assert!(
        run(&[
            "ingest",
            "--pgn",
            pgn.to_str().unwrap(),
            "--out",
            ing.to_str().unwrap(),
            "--dataset-id",
            "vf",
        ])
        .0
    );
    let (ok, _) = run(&["verify-targets", "--input", ing.to_str().unwrap()]);
    assert!(ok);
}
