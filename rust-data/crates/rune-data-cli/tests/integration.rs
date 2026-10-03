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
