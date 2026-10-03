use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};
use rune_data_core::error::{Error, Result};
use rune_data_core::format::{Compression, ShardFile};
use rune_data_core::pipeline::{
    FilterConfig, PipelineConfig, PipelineReport, ScoreComponents, ScoreWeights, ScoredRecord,
    SelectionConfig, StageStats, attach_scores, dedup_external, dedup_in_memory, filter_records,
    final_score, game_split, ingest_pgn_threads, join_labels, near_duplicate_stats, phase_balance,
    phase_counts, pool_to_records, read_jsonl_pool, read_shard_file, sample_stratified,
    sample_uniform, select_topk, write_report, write_shards, rss_mb_estimate,
};
use rune_data_core::record::{Record, SelMethod};

#[derive(Parser)]
#[command(name = "rune-data", version, about = "RUNE Data Engine")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Ingest(IngestArgs),
    FromJsonl(FromJsonlArgs),
    Validate(ValidateArgs),
    Dedup(DedupArgs),
    Filter(FilterArgs),
    Balance(BalanceArgs),
    Sample(SampleArgs),
    Shard(ShardArgs),
    Stats(StatsArgs),
    Inspect(InspectArgs),
    JoinLabels(JoinArgs),
    ExportLabels(ExportLabelsArgs),
    Split(SplitArgs),
    Score(ScoreArgs),
    Select(SelectArgs),
    Benchmark(BenchArgs),
}

#[derive(Args)]
struct Common {
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    dataset_id: Option<String>,
}

fn base_cfg(c: &Common) -> PipelineConfig {    let mut cfg = PipelineConfig::default();
    if let Some(s) = c.seed {
        cfg.seed = s;
    }
    if let Some(d) = &c.dataset_id {
        cfg.dataset_id = d.clone();
    }
    cfg
}

#[derive(Args)]
struct IngestArgs {
    #[arg(long)]
    pgn: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[command(flatten)]
    common: Common,
    #[arg(long, default_value_t = 4)]
    every: usize,
    #[arg(long, default_value_t = 120)]
    max_plies: usize,
    #[arg(long)]
    strict_identity: bool,
    #[arg(long, default_value_t = 1)]
    threads: usize,
}

#[derive(Args)]
struct FromJsonlArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[command(flatten)]
    common: Common,
    #[arg(long)]
    strict_identity: bool,
}

#[derive(Args)]
struct ValidateArgs {
    #[arg(long)]
    input: PathBuf,
}

#[derive(Args)]
struct DedupArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[command(flatten)]
    common: Common,
    #[arg(long)]
    external: bool,
    #[arg(long, default_value_t = 64)]
    partitions: usize,
}

#[derive(Args)]
struct FilterArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[command(flatten)]
    common: Common,
    #[arg(long)]
    require_teacher: bool,
    #[arg(long, default_value_t = 4)]
    min_ply: u32,
    #[arg(long)]
    min_abs_value: Option<f32>,
    #[arg(long)]
    phases: Option<String>,
    #[arg(long)]
    max_imbalance: Option<u8>,
}

#[derive(Args)]
struct BalanceArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[command(flatten)]
    common: Common,
    #[arg(long, default_value_t = 2.0)]
    max_ratio: f64,
}

#[derive(Args)]
struct SampleArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[command(flatten)]
    common: Common,
    #[arg(long)]
    n: usize,
    #[arg(long, default_value = "uniform")]
    mode: String,
}

#[derive(Args)]
struct ShardArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[command(flatten)]
    common: Common,
    #[arg(long, default_value_t = 100000)]
    per_shard: usize,
    #[arg(long, default_value = "raw")]
    compression: String,
}

#[derive(Args)]
struct StatsArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Args)]
struct InspectArgs {
    #[arg(long)]
    shard: PathBuf,
    #[arg(long)]
    index: Option<usize>,
    #[arg(long)]
    features: bool,
    #[arg(long)]
    limit: Option<usize>,
}

#[derive(Args)]
struct JoinArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    labels: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[command(flatten)]
    common: Common,
}

#[derive(Args)]
struct ExportLabelsArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Args)]
struct SplitArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[command(flatten)]
    common: Common,
    #[arg(long, default_value_t = 0.9)]
    train_frac: f64,
    #[arg(long, default_value_t = 0.05)]
    val_frac: f64,
}

#[derive(Args)]
struct ScoreArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    scores: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[command(flatten)]
    common: Common,
    #[arg(long, default_value_t = 1.0)]
    w_disagreement: f32,
    #[arg(long, default_value_t = 0.0)]
    w_uncertainty: f32,
    #[arg(long, default_value_t = 0.0)]
    w_instability: f32,
    #[arg(long, default_value_t = 0.0)]
    w_rarity: f32,
}

#[derive(Args)]
struct SelectArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[command(flatten)]
    common: Common,
    #[arg(long)]
    budget: usize,
    #[arg(long, default_value = "multi")]
    method: String,
    #[arg(long, default_value_t = 0)]
    round: u32,
    #[arg(long, default_value_t = 1.0)]
    w_disagreement: f32,
    #[arg(long, default_value_t = 0.0)]
    w_uncertainty: f32,
    #[arg(long, default_value_t = 0.0)]
    w_instability: f32,
    #[arg(long, default_value_t = 0.0)]
    w_rarity: f32,
    #[arg(long, default_value_t = 1000000)]
    diversity_cap: usize,
    #[arg(long, default_value_t = 0.2)]
    floor_ratio: f64,
    #[arg(long, default_value_t = 0)]
    shard_topk: usize,
    #[arg(long, default_value = "")]
    teacher_version: String,
    #[arg(long, default_value = "")]
    student_version: String,
}

#[derive(Args)]
struct BenchArgs {
    #[arg(long)]
    pool: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    #[arg(long, default_value_t = 2000)]
    n: usize,
}

fn parse_compression(s: &str) -> Result<Compression> {
    match s {
        "raw" => Ok(Compression::Raw),
        "deflate" => Ok(Compression::Deflate),
        _ => Err(Error::BadConfig(format!("unknown compression {s}"))),
    }
}

fn load_input(path: &Path) -> Result<(Vec<Record>, String, String)> {
    if path.is_dir() {
        let man_path = path.join("manifest.json");
        let man: serde_json::Value =
            serde_json::from_reader(File::open(&man_path).map_err(|e| Error::Io(e.to_string()))?)
                .map_err(|e| Error::BadFormat(e.to_string()))?;
        let files = man
            .get("shard_files")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::BadFormat("manifest missing shard_files".to_string()))?;
        let mut out = Vec::new();
        for f in files {
            let name = f
                .as_str()
                .ok_or_else(|| Error::BadFormat("bad shard name".to_string()))?;
            out.extend(read_shard_file(&path.join(name))?);
        }
        let ds = man
            .get("dataset")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let fv = man
            .get("feature_version")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        Ok((out, ds, fv))
    } else {
        let recs = read_shard_file(path)?;
        Ok((recs, String::new(), String::new()))
    }
}

fn read_score_components(path: &Path) -> Result<HashMap<u64, ScoreComponents>> {
    let f = File::open(path)?;
    let mut map = HashMap::new();
    for line in BufReader::new(f).lines() {
        let line = line.map_err(|e| Error::Io(e.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(&line).map_err(|e| Error::BadFormat(e.to_string()))?;
        let id = if let Some(h) = v.get("identity").and_then(|x| x.as_str()) {
            u64::from_str_radix(h.trim_start_matches("0x"), 16)
                .map_err(|_| Error::BadFormat("bad identity hex".to_string()))?
        } else if let Some(fen) = v.get("fen").and_then(|x| x.as_str()) {
            rune_data_core::board::canonical_identity(fen, false)
        } else {
            return Err(Error::BadFormat("score line needs identity or fen".to_string()));
        };
        map.insert(
            id,
            ScoreComponents {
                disagreement: v.get("disagreement").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
                uncertainty: v.get("uncertainty").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
                instability: v.get("instability").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
                rank_disagreement: v
                    .get("rank_disagreement")
                    .and_then(|x| x.as_f64())
                    .unwrap_or(0.0) as f32,
                rarity: 0.0,
            },
        );
    }
    Ok(map)
}

fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn save_stage(
    out_dir: &Path,
    records: &[Record],
    cfg: &PipelineConfig,
    stages: HashMap<String, StageStats>,
    elapsed: f64,
) -> Result<()> {
    let manifest = write_shards(
        records,
        out_dir,
        &cfg.dataset_id,
        &cfg.feature_version,
        cfg.compression,
        100000,
        cfg.seed,
    )?;
    let phases = phase_counts(records);
    let mat: f64 = if records.is_empty() {
        0.0
    } else {
        records.iter().map(|r| (r.piece_count) as f64).sum::<f64>() / records.len() as f64
    };
    let total_in: usize = stages
        .values()
        .map(|s| s.input)
        .sum::<usize>()
        .max(records.len());
    let rate = total_in as f64 / elapsed.max(1e-9);
    let report = PipelineReport {
        dataset_id: cfg.dataset_id.clone(),
        seed: cfg.seed,
        stages,
        phase_counts: phases,
        material_mean: mat,
        throughput_pos_per_sec: rate,
        elapsed_sec: elapsed,
        estimated_rss_mb: rss_mb_estimate(),
    };
    let rpath = out_dir.join("report");
    write_report(&rpath, &report)?;
    let mpath = out_dir.join("manifest.json");
    let mj =
        serde_json::to_string_pretty(&manifest).map_err(|e| Error::BadFormat(e.to_string()))?;
    std::fs::write(&mpath, mj)?;
    let total_in: usize = report
        .stages
        .values()
        .map(|s| s.input)
        .sum::<usize>()
        .max(records.len());
    println!(
        "wrote {} records to {} ({} shards), {:.0} pos/s",
        records.len(),
        out_dir.display(),
        manifest.shards,
        total_in as f64 / report.elapsed_sec.max(1e-9)
    );
    Ok(())
}

fn stage_map(name: &str, stats: StageStats) -> HashMap<String, StageStats> {
    let mut m = HashMap::new();
    m.insert(name.to_string(), stats);
    m
}

fn read_pool_labels(path: &Path) -> Result<Vec<(u64, f32, u8, i32)>> {
    let f = File::open(path)?;
    let mut out = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line.map_err(|e| Error::Io(e.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(&line).map_err(|e| Error::BadFormat(e.to_string()))?;
        let fen = v
            .get("fen")
            .and_then(|x| x.as_str())
            .ok_or_else(|| Error::BadFormat("label missing fen".to_string()))?;
        let id = rune_data_core::board::canonical_identity(fen, false);
        out.push((
            id,
            v.get("teacher_v").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
            v.get("teacher_w").and_then(|x| x.as_u64()).unwrap_or(1) as u8,
            v.get("teacher_cp").and_then(|x| x.as_i64()).unwrap_or(0) as i32,
        ));
    }
    Ok(out)
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Ingest(a) => {
            let mut cfg = base_cfg(&a.common);
            cfg.every_plies = a.every;
            cfg.max_plies = a.max_plies;
            cfg.strict_identity = a.strict_identity;
            let t0 = std::time::Instant::now();
            let mut st = StageStats::default();
            let recs = ingest_pgn_threads(&a.pgn, &cfg, a.threads, &mut st)?;
            let dt = t0.elapsed().as_secs_f64();
            save_stage(&a.out, &recs, &cfg, stage_map("ingest", st), dt)?;
        }
        Cmd::FromJsonl(a) => {
            let mut cfg = base_cfg(&a.common);
            cfg.strict_identity = a.strict_identity;
            let t0 = std::time::Instant::now();
            let pool = read_jsonl_pool(&a.input)?;
            let mut st = StageStats::default();
            let recs = pool_to_records(&pool, &cfg, &mut st);
            let dt = t0.elapsed().as_secs_f64();
            save_stage(&a.out, &recs, &cfg, stage_map("from_jsonl", st), dt)?;
        }
        Cmd::Validate(a) => {
            let (recs, ds, _) = load_input(&a.input)?;
            let mut bad = 0usize;
            for r in &recs {
                if rune_data_core::board::Board::parse_fen(&r.fen).is_err() {
                    bad += 1;
                }
            }
            println!("dataset={ds} records={} invalid_fen={bad}", recs.len());
            if bad > 0 {
                return Err(Error::BadFormat(format!("{bad} invalid records")));
            }
        }
        Cmd::Dedup(a) => {
            let cfg = base_cfg(&a.common);
            let t0 = std::time::Instant::now();
            let (recs, _, _) = load_input(&a.input)?;
            let mut st = StageStats::default();
            let out = if a.external {
                let tmp = a.out.join("tmp_parts");
                dedup_external(recs, a.partitions, &tmp, &mut st)?
            } else {
                dedup_in_memory(recs, &mut st)
            };
            let dt = t0.elapsed().as_secs_f64();
            let _ = std::fs::remove_dir_all(a.out.join("tmp_parts"));
            save_stage(&a.out, &out, &cfg, stage_map("dedup", st), dt)?;
        }
        Cmd::Filter(a) => {
            let cfg = base_cfg(&a.common);
            let t0 = std::time::Instant::now();
            let (recs, _, _) = load_input(&a.input)?;
            let phases = a.phases.map(|s| {
                s.split(',')
                    .filter_map(|x| x.trim().parse::<u8>().ok())
                    .collect::<Vec<u8>>()
            });
            let fc = FilterConfig {
                require_teacher: a.require_teacher,
                min_ply: a.min_ply,
                min_abs_value: a.min_abs_value,
                phases,
                max_imbalance_bucket: a.max_imbalance,
            };
            let mut st = StageStats::default();
            let out = filter_records(recs, &fc, &mut st);
            let dt = t0.elapsed().as_secs_f64();
            save_stage(&a.out, &out, &cfg, stage_map("filter", st), dt)?;
        }
        Cmd::Balance(a) => {
            let cfg = base_cfg(&a.common);
            let t0 = std::time::Instant::now();
            let (recs, _, _) = load_input(&a.input)?;
            let mut st = StageStats::default();
            let out = phase_balance(recs, a.max_ratio, &mut st);
            let dt = t0.elapsed().as_secs_f64();
            save_stage(&a.out, &out, &cfg, stage_map("balance", st), dt)?;
        }
        Cmd::Sample(a) => {
            let cfg = base_cfg(&a.common);
            let t0 = std::time::Instant::now();
            let (recs, _, _) = load_input(&a.input)?;
            let seed = cfg.seed;
            let dt = t0.elapsed().as_secs_f64();
            let mut st = StageStats {
                input: recs.len(),
                ..Default::default()
            };
            let out = match a.mode.as_str() {
                "stratified" => sample_stratified(recs, a.n, seed),
                "uniform" => sample_uniform(recs, a.n, seed),
                _ => return Err(Error::BadConfig(format!("unknown sample mode {}", a.mode))),
            };
            st.output = out.len();
            save_stage(&a.out, &out, &cfg, stage_map("sample", st), dt)?;
        }
        Cmd::Shard(a) => {
            let mut cfg = base_cfg(&a.common);
            cfg.compression = parse_compression(&a.compression)?;
            let t0 = std::time::Instant::now();
            let (recs, ds, fv) = load_input(&a.input)?;
            if !ds.is_empty() {
                cfg.dataset_id = ds;
            }
            if !fv.is_empty() {
                cfg.feature_version = fv;
            }
            let manifest = write_shards(
                &recs,
                &a.out,
                &cfg.dataset_id,
                &cfg.feature_version,
                cfg.compression,
                a.per_shard,
                cfg.seed,
            )?;
            let dt = t0.elapsed().as_secs_f64();
            let mpath = a.out.join("manifest.json");
            let mj = serde_json::to_string_pretty(&manifest)
                .map_err(|e| Error::BadFormat(e.to_string()))?;
            std::fs::write(&mpath, mj)?;
            let st = StageStats {
                input: recs.len(),
                output: recs.len(),
                ..Default::default()
            };
            let mut stages = HashMap::new();
            stages.insert("shard".to_string(), st);
            let report = PipelineReport {
                dataset_id: cfg.dataset_id.clone(),
                seed: cfg.seed,
                stages,
                phase_counts: phase_counts(&recs),
                material_mean: 0.0,
                throughput_pos_per_sec: recs.len() as f64 / dt.max(1e-9),
                elapsed_sec: dt,
                estimated_rss_mb: rss_mb_estimate(),
            };
            write_report(&a.out.join("report"), &report)?;
            println!(
                "sharded {} records into {} shards",
                recs.len(),
                manifest.shards
            );
        }
        Cmd::Stats(a) => {
            let (recs, ds, _) = load_input(&a.input)?;
            let phases = phase_counts(&recs);
            let vals: Vec<f32> = recs
                .iter()
                .filter(|r| r.has_teacher)
                .map(|r| r.teacher_value)
                .collect();
            let teach_n = vals.len();
            let (exact_dup, local_dup, clusters) = near_duplicate_stats(&recs);
            let mut hist = [0usize; 10];
            for &v in &vals {
                let b = ((v + 1.0) * 5.0).floor().clamp(0.0, 9.0) as usize;
                hist[b] += 1;
            }
            let body = serde_json::json!({
                "dataset": ds,
                "records": recs.len(),
                "teacher_labeled": teach_n,
                "phases": phases,
                "value_hist10": hist,
                "exact_duplicates": exact_dup,
                "same_game_local_duplicates": local_dup,
                "repeated_position_clusters": clusters,
            });
            println!("{}", serde_json::to_string_pretty(&body).unwrap());
            if let Some(p) = a.out {
                std::fs::write(p, serde_json::to_string_pretty(&body).unwrap())?;
            }
        }
        Cmd::Inspect(a) => {
            let data = std::fs::read(&a.shard)?;
            let f = ShardFile::parse(&data)?;
            println!(
                "dataset={} shard={}/{} records={} compression={:?}",
                f.header.dataset_id,
                f.header.shard_id,
                f.header.shard_count,
                f.header.record_count,
                f.header.compression
            );
            let offs = f.offsets()?;
            let idxs: Vec<usize> = match a.index {
                Some(i) => vec![i],
                None => {
                    (0..f.header.record_count.min(a.limit.unwrap_or(3) as u64) as usize).collect()
                }
            };
            for i in idxs {
                let r = f.read_at(&offs, i)?;
                println!(
                    "--- [{}] phase={} stm={} pieces={} id={:016x} game={:016x} ply={}",
                    i, r.phase, r.stm, r.piece_count, r.identity, r.game_hash, r.ply
                );
                println!("fen: {}", r.fen);
                if r.has_teacher {
                    println!(
                        "teacher: value={:.4} wdl={} cp={}",
                        r.teacher_value, r.teacher_wdl, r.teacher_cp
                    );
                }
                if a.features {
                    let feats: Vec<String> =
                        r.features.iter().map(|(g, x)| format!("{g}:{x}")).collect();
                    println!("features({}): {}", feats.len(), feats.join(","));
                }
            }
        }
        Cmd::JoinLabels(a) => {
            let cfg = base_cfg(&a.common);
            let t0 = std::time::Instant::now();
            let (recs, _, _) = load_input(&a.input)?;
            let labels = read_pool_labels(&a.labels)?;
            let mut st = StageStats::default();
            let (out, missing, dups) = join_labels(recs, &labels, &mut st);
            let dt = t0.elapsed().as_secs_f64();
            let n = out.len();
            save_stage(&a.out, &out, &cfg, stage_map("join", st), dt)?;
            println!("joined {n} missing={missing} dup_labels={dups}");
        }
        Cmd::ExportLabels(a) => {
            let (recs, _, _) = load_input(&a.input)?;
            let f = File::create(&a.out)?;
            let mut w = BufWriter::new(f);
            let mut n = 0usize;
            for r in &recs {
                if !r.has_teacher {
                    continue;
                }
                let o = serde_json::json!({
                    "fen": r.fen, "teacher_v": r.teacher_value, "teacher_w": r.teacher_wdl,
                    "teacher_cp": r.teacher_cp,
                });
                writeln!(w, "{}", o)?;
                n += 1;
            }
            w.flush()?;
            println!("exported {n} labels");
        }
        Cmd::Split(a) => {
            let cfg = base_cfg(&a.common);
            let (recs, _, _) = load_input(&a.input)?;
            let (train, val, test) = game_split(recs, a.train_frac, a.val_frac, cfg.seed);
            for (name, set) in [("train", train), ("val", val), ("test", test)] {
                let dir = a.out.join(name);
                let st = StageStats {
                    input: set.len(),
                    output: set.len(),
                    ..Default::default()
                };
                save_stage(&dir, &set, &cfg, stage_map("split", st), 0.0)?;
            }
            println!("split written to {}", a.out.display());
        }
        Cmd::Score(a) => {
            let cfg = base_cfg(&a.common);
            let t0 = std::time::Instant::now();
            let (recs, _, _) = load_input(&a.input)?;
            let scores = read_score_components(&a.scores)?;
            let weights = ScoreWeights {
                disagreement: a.w_disagreement,
                uncertainty: a.w_uncertainty,
                instability: a.w_instability,
                rarity: a.w_rarity,
            };
            let mut st = StageStats::default();
            let (scored, missing) = attach_scores(recs, &scores, &weights, &mut st);
            let dt = t0.elapsed().as_secs_f64();
            let scored_recs: Vec<Record> =
                scored.iter().map(|s| s.record.clone()).collect();
            let comp_path = a.out.join("score_components.jsonl");
            std::fs::create_dir_all(&a.out)?;
            {
                let f = File::create(&comp_path)?;
                let mut w = BufWriter::new(f);
                for s in &scored {
                    let o = serde_json::json!({
                        "identity": format!("{:016x}", s.record.identity),
                        "disagreement": s.components.disagreement,
                        "uncertainty": s.components.uncertainty,
                        "instability": s.components.instability,
                        "rank_disagreement": s.components.rank_disagreement,
                        "rarity": s.components.rarity,
                        "final": s.final_score,
                    });
                    writeln!(w, "{o}")?;
                }
                w.flush()?;
            }
            save_stage(&a.out, &scored_recs, &cfg, stage_map("score", st), dt)?;
            println!("scored {} records, missing scores: {missing}", scored.len());
        }
        Cmd::Select(a) => {
            let cfg = base_cfg(&a.common);
            let method = SelMethod::from_name(&a.method)
                .ok_or_else(|| Error::BadConfig(format!("unknown method {}", a.method)))?;
            let t0 = std::time::Instant::now();
            let shard_paths = rune_data_core::pipeline::shard_files(&a.input)?;
            let weights = ScoreWeights {
                disagreement: a.w_disagreement,
                uncertainty: a.w_uncertainty,
                instability: a.w_instability,
                rarity: a.w_rarity,
            };
            let sel_cfg = SelectionConfig {
                budget: a.budget,
                method,
                round: a.round,
                weights,
                diversity_cap: a.diversity_cap,
                floor_ratio: a.floor_ratio,
                seed: cfg.seed,
                shard_topk: a.shard_topk,
            };
            let mut winners: Vec<ScoredRecord> = Vec::new();
            let mut st = StageStats::default();
            let local_k = if a.shard_topk > 0 { a.shard_topk } else { a.budget };
            for sp in &shard_paths {
                let recs = read_shard_file(sp)?;
                let mut local: Vec<ScoredRecord> = recs
                    .into_iter()
                    .map(|mut r| {
                        let c = ScoreComponents {
                            disagreement: r.sel_score,
                            ..Default::default()
                        };
                        let final_score = final_score(&c, &sel_cfg.weights);
                        r.sel_score = final_score;
                        ScoredRecord { record: r, components: c, final_score }
                    })
                    .collect();
                for _ in &local {
                    st.input += 1;
                }
                if shard_paths.len() > 1 && local.len() > local_k {
                    use rune_data_core::pipeline::topk_by_score;

                    let keep = topk_by_score(&local, local_k);
                    let mut reduced = Vec::with_capacity(keep.len());
                    for i in keep {
                        reduced.push(local[i].clone());
                    }
                    local = reduced;
                }
                winners.extend(local);
            }
            let selected = select_topk(winners, &sel_cfg, &mut st);
            let dt = t0.elapsed().as_secs_f64();
            let out_recs: Vec<Record> =
                selected.iter().map(|s| s.record.clone()).collect();
            let requested = out_recs.iter().filter(|r| !r.has_teacher).count();
            let reused = out_recs.len() - requested;
            save_stage(&a.out, &out_recs, &cfg, stage_map("select", st), dt)?;
            {
                let f = File::create(a.out.join("selection_audit.jsonl"))?;
                let mut w = BufWriter::new(f);
                for s in &selected {
                    let o = serde_json::json!({
                        "position_id": format!("{:016x}", s.record.identity),
                        "fen": s.record.fen,
                        "game_hash": format!("{:016x}", s.record.game_hash),
                        "selection_method": s.record.sel_method.name(),
                        "score_components": {
                            "disagreement": s.components.disagreement,
                            "uncertainty": s.components.uncertainty,
                            "instability": s.components.instability,
                            "rank_disagreement": s.components.rank_disagreement,
                            "rarity": s.components.rarity,
                        },
                        "final_score": s.final_score,
                        "round": s.record.active_round,
                        "teacher_version": a.teacher_version,
                        "student_version": a.student_version,
                    });
                    writeln!(w, "{o}")?;
                }
                w.flush()?;
            }
            let round_doc = format!(
                "round:\n  id: {}\n  parent_round: {}\n\ndataset:\n  candidate_id: {}\n  base_training_id: {}\n\nmodel:\n  student_id: {}\n  teacher_id: {}\n\nselection:\n  method: {}\n  seed: {}\n  budget: {}\n\nteacher:\n  version: {}\n  labels_requested: {}\n  labels_reused: {}\n\nsystem:\n  git_commit: {}\n  hardware: {}\n",
                a.round,
                if a.round > 0 { a.round - 1 } else { 0 },
                a.input.display(),
                a.input.display(),
                a.student_version,
                a.teacher_version,
                method.name(),
                cfg.seed,
                a.budget,
                a.teacher_version,
                requested,
                reused,
                git_commit(),
                std::env::consts::ARCH,
            );
            std::fs::write(a.out.join("round.yaml"), round_doc)?;
            println!(
                "selected {} (requested {requested}, reused {reused})",
                out_recs.len()
            );
        }
        Cmd::Benchmark(a) => {
            run_benchmark(a)?;
        }
    }
    Ok(())
}

fn run_benchmark(a: BenchArgs) -> Result<()> {
    use rune_data_core::board::Board;
    use rune_data_core::features::extract_features;

    let pool_path = a
        .pool
        .unwrap_or_else(|| PathBuf::from("/tmp/rune-real/pool.jsonl"));
    let pool = read_jsonl_pool(&pool_path)?;
    let fens: Vec<String> = pool
        .iter()
        .take(a.n.max(1))
        .map(|r| r.fen.clone())
        .collect();
    let t0 = std::time::Instant::now();
    let mut nfeat = 0usize;
    for f in &fens {
        if let Ok(b) = Board::parse_fen(f) {
            nfeat += extract_features(&b).len();
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "extract_features: {:.0} pos/s ({} positions, {} features total)",
        fens.len() as f64 / dt.max(1e-9),
        fens.len(),
        nfeat
    );

    let t0 = std::time::Instant::now();
    let cfg = PipelineConfig::default();
    let mut st = StageStats::default();
    let tpool: Vec<rune_data_core::pipeline::PoolRecord> = pool
        .iter()
        .take(a.n.max(1))
        .map(|r| rune_data_core::pipeline::PoolRecord {
            fen: r.fen.clone(),
            game_id: r.game_id.clone(),
            ply: r.ply,
            source_id: 0,
            teacher_value: None,
            teacher_wdl: None,
            teacher_cp: None,
            perspective_stm: true,
        })
        .collect();
    let recs = pool_to_records(&tpool, &cfg, &mut st);
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "pool_to_records: {:.0} pos/s ({} records)",
        recs.len() as f64 / dt.max(1e-9),
        recs.len()
    );

    let t0 = std::time::Instant::now();
    let mut st2 = StageStats::default();
    let dd = dedup_in_memory(recs.clone(), &mut st2);
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "dedup_in_memory: {:.0} pos/s ({} -> {})",
        dd.len() as f64 / dt.max(1e-9),
        st2.input,
        dd.len()
    );

    for c in [Compression::Raw, Compression::Deflate] {
        let t0 = std::time::Instant::now();
        let mut buf = Vec::new();
        rune_data_core::format::write_shard(
            &rune_data_core::format::ShardHeader {
                schema: 2,
                feature_version: "bench".to_string(),
                compression: c,
                record_count: dd.len() as u64,
                shard_id: 0,
                shard_count: 1,
                dataset_id: "bench".to_string(),
            },
            &dd,
            &mut buf,
        )?;
        let wdt = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let f = ShardFile::parse(&buf)?;
        let back = f.read_all()?;
        let rdt = t1.elapsed().as_secs_f64();
        println!(
            "write {:?}: {:.0} pos/s, {} bytes ({:.1} B/pos); read_all: {:.0} pos/s ({} records)",
            c,
            dd.len() as f64 / wdt.max(1e-9),
            buf.len(),
            buf.len() as f64 / dd.len().max(1) as f64,
            back.len() as f64 / rdt.max(1e-9),
            back.len()
        );
    }

    let t0 = std::time::Instant::now();
    let mut st3 = StageStats::default();
    let tmp = std::env::temp_dir().join("rune_dedup_bench");
    let dd2 = dedup_external(dd.clone(), 16, &tmp, &mut st3)?;
    let dt = std::time::Instant::now().duration_since(t0).as_secs_f64();
    println!(
        "dedup_external(16 parts): {:.0} pos/s ({} records)",
        dd2.len() as f64 / dt.max(1e-9),
        dd2.len()
    );
    let _ = std::fs::remove_dir_all(&tmp);

    let mut wbuf = Vec::new();
    rune_data_core::format::write_shard(
        &rune_data_core::format::ShardHeader {
                schema: 2,
            feature_version: "bench".to_string(),
            compression: rune_data_core::format::Compression::Raw,
            record_count: dd.len() as u64,
            shard_id: 0,
            shard_count: 1,
            dataset_id: "bench".to_string(),
        },
        &dd,
        &mut wbuf,
    )?;
    let bench_path = std::env::temp_dir().join("rune_mmap_bench.bin");
    std::fs::write(&bench_path, &wbuf)?;
    let t0 = std::time::Instant::now();
    let buf = std::fs::read(&bench_path)?;
    let f = ShardFile::parse(&buf)?;
    let back = f.read_all()?;
    let dt = std::time::Instant::now().duration_since(t0).as_secs_f64();
    println!(
        "read buffered: {:.0} pos/s ({} records)",
        back.len() as f64 / dt.max(1e-9),
        back.len()
    );
    let t0 = std::time::Instant::now();
    let file = std::fs::File::open(&bench_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file).map_err(|e| Error::Io(e.to_string()))? };
    let f = ShardFile::parse(&mmap)?;
    let back = f.read_all()?;
    let dt = std::time::Instant::now().duration_since(t0).as_secs_f64();
    println!(
        "read mmap: {:.0} pos/s ({} records)",
        back.len() as f64 / dt.max(1e-9),
        back.len()
    );
    let _ = std::fs::remove_file(&bench_path);

    println!("threads configured: {} (pipeline v1 is single-threaded streaming; worker scaling is a planned benchmark)", a.threads);
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("rune-data: error: {e}");
        std::process::exit(1);
    }
}
