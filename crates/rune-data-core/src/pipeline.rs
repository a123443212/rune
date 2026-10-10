use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::board::Board;
use crate::error::{Error, Result};
use crate::format::{write_shard, Compression, ShardFile, ShardHeader};
use crate::hash;
use crate::record::Record;

mod ingest;

pub use ingest::{ingest_pgn, ingest_pgn_threads};

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub seed: u64,
    pub strict_identity: bool,
    pub every_plies: usize,
    pub max_plies: usize,
    pub phase_cap_ratio: f64,
    pub source_id: u32,
    pub dataset_id: String,
    pub feature_version: String,
    pub compression: Compression,
    pub deterministic: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            seed: 42,
            strict_identity: false,
            every_plies: 4,
            max_plies: 120,
            phase_cap_ratio: 2.0,
            source_id: 0,
            dataset_id: "rune-clean-v07".to_string(),
            feature_version: "grouped_hkav2_fullthreats_v02".to_string(),
            compression: Compression::Raw,
            deterministic: true,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StageStats {
    pub input: usize,
    pub output: usize,
    pub rejected: usize,
    pub reasons: HashMap<String, usize>,
}

impl StageStats {
    pub fn reject(&mut self, reason: &str) {
        self.rejected += 1;
        *self.reasons.entry(reason.to_string()).or_insert(0) += 1;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineReport {
    pub dataset_id: String,
    pub seed: u64,
    pub stages: HashMap<String, StageStats>,
    pub phase_counts: [usize; 3],
    pub material_mean: f64,
    pub throughput_pos_per_sec: f64,
    pub elapsed_sec: f64,
    pub estimated_rss_mb: f64,
}

pub fn read_jsonl_pool(path: &Path) -> Result<Vec<PoolRecord>> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| Error::Io(e.to_string()))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).map_err(|e| Error::BadFormat(e.to_string()))?;
        out.push(PoolRecord::from_json(&v)?);
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct PoolRecord {
    pub fen: String,
    pub game_id: String,
    pub ply: u16,
    pub source_id: u32,
    pub teacher_value: Option<f32>,
    pub teacher_wdl: Option<u8>,
    pub teacher_cp: Option<i32>,
    pub perspective_stm: bool,
}

impl PoolRecord {
    pub fn from_json(v: &serde_json::Value) -> Result<PoolRecord> {
        let get = |k: &str| {
            v.get(k)
                .and_then(|x| x.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| Error::BadFormat(format!("pool record missing {k}")))
        };
        Ok(PoolRecord {
            fen: get("fen")?,
            game_id: v
                .get("game_id")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string(),
            ply: v.get("ply").and_then(|x| x.as_u64()).unwrap_or(8) as u16,
            source_id: v.get("source_id").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
            teacher_value: v
                .get("teacher_v")
                .or_else(|| v.get("teacher_value"))
                .and_then(|x| x.as_f64())
                .map(|x| x as f32),
            teacher_wdl: v
                .get("teacher_w")
                .or_else(|| v.get("teacher_wdl"))
                .and_then(|x| x.as_u64())
                .map(|x| x as u8),
            teacher_cp: v
                .get("teacher_cp")
                .and_then(|x| x.as_i64())
                .map(|x| x as i32),
            perspective_stm: v
                .get("value_perspective")
                .and_then(|x| x.as_str())
                .map(|s| s == "side_to_move")
                .unwrap_or(true),
        })
    }
}

pub fn pool_to_records(
    pool: &[PoolRecord],
    cfg: &PipelineConfig,
    stats: &mut StageStats,
) -> Vec<Record> {
    pool_to_records_with_ply(pool, cfg, 0, stats)
}

pub fn pool_to_records_with_ply(
    pool: &[PoolRecord],
    cfg: &PipelineConfig,
    min_ply: u32,
    stats: &mut StageStats,
) -> Vec<Record> {
    let mut out = Vec::new();
    for r in pool {
        stats.input += 1;
        if (r.ply as u32) < min_ply {
            stats.reject("min_ply");
            continue;
        }
        let board = match Board::parse_fen(&r.fen) {
            Ok(b) => b,
            Err(_) => {
                stats.reject("fen_parse");
                continue;
            }
        };
        if !valid_kings(&board) {
            stats.reject("king_count");
            continue;
        }
        if has_backrank_pawn(&board) {
            stats.reject("pawn_backrank");
            continue;
        }
        let gid = if r.game_id.is_empty() {
            &r.fen
        } else {
            &r.game_id
        };
        let mut rec = match Record::from_fen(&r.fen, gid, r.ply, r.source_id, cfg.strict_identity) {
            Ok(rec) => rec,
            Err(_) => {
                stats.reject("record_build");
                continue;
            }
        };
        if let (Some(v), Some(w)) = (r.teacher_value, r.teacher_wdl) {
            rec = rec.with_teacher(v, w, r.teacher_cp.unwrap_or(0), r.perspective_stm);
        }
        out.push(rec);
    }
    stats.output = out.len();
    out
}

fn valid_kings(b: &Board) -> bool {
    let mut wk = 0;
    let mut bk = 0;
    for c in b.sq.iter().flatten() {
        if c.kind == crate::board::KING {
            if c.color == crate::board::WHITE {
                wk += 1;
            } else {
                bk += 1;
            }
        }
    }
    wk == 1 && bk == 1
}

fn has_backrank_pawn(b: &Board) -> bool {
    for sq in 0..64 {
        if let Some(c) = b.sq[sq] {
            if c.kind == crate::board::PAWN {
                let r = crate::board::sq_rank(sq);
                if r == 0 || r == 7 {
                    return true;
                }
            }
        }
    }
    false
}

pub fn dedup_in_memory(records: Vec<Record>, stats: &mut StageStats) -> Vec<Record> {
    let mut seen: HashSet<u64> = HashSet::with_capacity(records.len());
    let mut out = Vec::with_capacity(records.len());
    for r in records {
        stats.input += 1;
        if seen.insert(r.identity) {
            out.push(r);
        } else {
            stats.reject("duplicate");
        }
    }
    stats.output = out.len();
    out
}

pub fn dedup_external(
    records: Vec<Record>,
    partitions: usize,
    tmp: &Path,
    stats: &mut StageStats,
) -> Result<Vec<Record>> {
    std::fs::create_dir_all(tmp)?;
    let mut writers: Vec<BufWriter<File>> = Vec::with_capacity(partitions);
    for p in 0..partitions {
        writers.push(BufWriter::new(File::create(
            tmp.join(format!("part-{p:04}.bin")),
        )?));
    }
    for r in &records {
        stats.input += 1;
        let p = (r.identity % partitions as u64) as usize;
        let mut buf = Vec::new();
        write_shard(
            &ShardHeader {
                schema: 2,
                feature_version: String::new(),
                compression: Compression::Raw,
                record_count: 1,
                shard_id: 0,
                shard_count: 1,
                dataset_id: String::new(),
            },
            std::slice::from_ref(r),
            &mut buf,
        )?;
        let len = buf.len() as u64;
        writers[p].write_all(&len.to_le_bytes())?;
        writers[p].write_all(&buf)?;
    }
    for w in &mut writers {
        w.flush()?;
    }
    drop(writers);
    let mut out = Vec::new();
    for p in 0..partitions {
        let path = tmp.join(format!("part-{p:04}.bin"));
        let data = std::fs::read(&path)?;
        let mut pos = 0usize;
        let mut seen: HashSet<u64> = HashSet::new();
        while pos + 8 <= data.len() {
            let mut lb = [0u8; 8];
            lb.copy_from_slice(&data[pos..pos + 8]);
            let len = u64::from_le_bytes(lb) as usize;
            pos += 8;
            if pos + len > data.len() {
                return Err(Error::BadFormat("partition truncated".to_string()));
            }
            let f = ShardFile::parse(&data[pos..pos + len])?;
            pos += len;
            let recs = f.read_all()?;
            for r in recs {
                if seen.insert(r.identity) {
                    out.push(r);
                } else {
                    stats.reject("duplicate");
                }
            }
        }
        let _ = std::fs::remove_file(&path);
    }
    out.sort_by_key(|a| (a.game_hash, a.ply, a.identity));
    stats.output = out.len();
    Ok(out)
}

#[derive(Debug, Clone, Default)]
pub struct FilterConfig {
    pub require_teacher: bool,
    pub min_abs_value: Option<f32>,
    pub phases: Option<Vec<u8>>,
    pub max_imbalance_bucket: Option<u8>,
    pub min_ply: u32,
}

pub fn filter_records(
    records: Vec<Record>,
    cfg: &FilterConfig,
    stats: &mut StageStats,
) -> Vec<Record> {
    let mut out = Vec::new();
    for r in records {
        stats.input += 1;
        if (r.ply as u32) < cfg.min_ply {
            stats.reject("min_ply");
            continue;
        }
        if cfg.require_teacher && !r.has_teacher {
            stats.reject("no_teacher");
            continue;
        }
        if let Some(m) = cfg.min_abs_value {
            if r.teacher_value.abs() < m {
                stats.reject("value_range");
                continue;
            }
        }
        if let Some(ph) = &cfg.phases {
            if !ph.contains(&r.phase) {
                stats.reject("phase");
                continue;
            }
        }
        if let Some(mb) = cfg.max_imbalance_bucket {
            if r.imbalance_bucket > mb {
                stats.reject("material");
                continue;
            }
        }
        out.push(r);
    }
    stats.output = out.len();
    out
}

pub fn phase_balance(records: Vec<Record>, max_ratio: f64, stats: &mut StageStats) -> Vec<Record> {
    let mut buckets: [Vec<Record>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for r in records {
        stats.input += 1;
        buckets[r.phase.min(2) as usize].push(r);
    }
    let counts = [buckets[0].len(), buckets[1].len(), buckets[2].len()];
    let nonzero: Vec<usize> = counts.iter().cloned().filter(|&c| c > 0).collect();
    if nonzero.is_empty() {
        stats.output = 0;
        return Vec::new();
    }
    let smallest = *nonzero.iter().min().unwrap();
    let target = *nonzero.iter().max().unwrap();
    let cap = if target as f64 > smallest as f64 * max_ratio {
        (smallest as f64 * max_ratio) as usize
    } else {
        target
    };
    let mut out = Vec::new();
    for b in buckets.iter_mut() {
        if b.len() > cap {
            stats.rejected += b.len() - cap;
            *stats.reasons.entry("balance_cap".to_string()).or_insert(0) += b.len() - cap;
        }
        out.extend(b.drain(..cap.min(b.len())));
    }
    stats.output = out.len();
    out
}

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }
    pub fn step(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.step() % (n as u64).max(1)) as usize
    }
}

pub fn sample_uniform(records: Vec<Record>, n: usize, seed: u64) -> Vec<Record> {
    let mut rng = Rng::new(seed);
    let mut idx: Vec<usize> = (0..records.len()).collect();
    for i in (1..idx.len()).rev() {
        let j = rng.below(i + 1);
        idx.swap(i, j);
    }
    let mut out = Vec::with_capacity(n.min(records.len()));
    let recs = records;
    for k in idx.into_iter().take(n.min(recs.len())) {
        out.push(recs[k].clone());
    }
    let _ = recs;
    out
}

pub fn sample_stratified(records: Vec<Record>, n: usize, seed: u64) -> Vec<Record> {
    let total = records.len();
    let want = n.min(total);
    let mut buckets: [Vec<Record>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for r in records {
        buckets[r.phase.min(2) as usize].push(r);
    }
    let per = want / 3;
    let mut out = Vec::new();
    let mut rng = Rng::new(seed);
    for b in buckets.iter_mut() {
        for i in (1..b.len()).rev() {
            let j = rng.below(i + 1);
            b.swap(i, j);
        }
        out.extend(b.drain(..per.min(b.len())));
    }
    let mut rest: Vec<Record> = buckets.into_iter().flatten().collect();
    for i in (1..rest.len()).rev() {
        let j = rng.below(i + 1);
        rest.swap(i, j);
    }
    out.extend(rest.into_iter().take(want.saturating_sub(out.len())));
    for i in (1..out.len()).rev() {
        let j = rng.below(i + 1);
        out.swap(i, j);
    }
    out
}

pub fn game_split(
    records: Vec<Record>,
    train_frac: f64,
    val_frac: f64,
    seed: u64,
) -> (Vec<Record>, Vec<Record>, Vec<Record>) {
    let mut games: HashMap<u64, Vec<Record>> = HashMap::new();
    for r in records {
        games.entry(r.game_hash).or_default().push(r);
    }
    let mut gids: Vec<u64> = games.keys().cloned().collect();
    gids.sort_unstable();
    let mut train = Vec::new();
    let mut val = Vec::new();
    let mut test = Vec::new();
    for gid in gids {
        let h = hash::fnv1a(&[seed.to_le_bytes(), gid.to_le_bytes()].concat());
        let x = (h % 10000) as f64 / 10000.0;
        let bucket = if x < train_frac {
            &mut train
        } else if x < train_frac + val_frac {
            &mut val
        } else {
            &mut test
        };
        bucket.extend(games.remove(&gid).unwrap());
    }
    if val.is_empty()
        && train
            .iter()
            .map(|r| r.game_hash)
            .collect::<HashSet<_>>()
            .len()
            > 1
    {
        let mut seen_order: Vec<u64> = Vec::new();
        for r in &train {
            if !seen_order.contains(&r.game_hash) {
                seen_order.push(r.game_hash);
            }
        }
        if let Some(&last) = seen_order.last() {
            let mut rest = Vec::new();
            for r in train {
                if r.game_hash == last {
                    val.push(r);
                } else {
                    rest.push(r);
                }
            }
            train = rest;
        }
    }
    (train, val, test)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub dataset: String,
    pub schema: u32,
    pub feature_version: String,
    pub shards: usize,
    pub total_records: usize,
    pub hash: String,
    pub compression: String,
    pub seed: u64,
    pub shard_files: Vec<String>,
}

pub fn write_shards(
    records: &[Record],
    out_dir: &Path,
    dataset_id: &str,
    feature_version: &str,
    compression: Compression,
    per_shard: usize,
    seed: u64,
) -> Result<Manifest> {
    std::fs::create_dir_all(out_dir)?;
    let mut order: Vec<usize> = (0..records.len()).collect();
    order.sort_by_key(|&i| records[i].identity);
    let mut files = Vec::new();
    let mut shard_hashes: Vec<u64> = Vec::new();
    let n_shards = (records.len() + per_shard - 1) / per_shard.max(1);
    for (s, chunk) in order.chunks(per_shard.max(1)).enumerate() {
        let recs: Vec<Record> = chunk.iter().map(|&i| records[i].clone()).collect();
        let header = ShardHeader {
            schema: 2,
            feature_version: feature_version.to_string(),
            compression,
            record_count: recs.len() as u64,
            shard_id: s as u32,
            shard_count: n_shards as u32,
            dataset_id: dataset_id.to_string(),
        };
        let mut buf = Vec::new();
        write_shard(&header, &recs, &mut buf)?;
        let name = format!("shard-{s:05}");
        std::fs::write(out_dir.join(&name), &buf)?;
        shard_hashes.push(hash::fnv1a(&buf));
        files.push(name);
    }
    let mut acc = String::new();
    for h in &shard_hashes {
        acc.push_str(&format!("{h:016x}"));
    }
    acc.push_str(dataset_id);
    acc.push_str(&seed.to_string());
    Ok(Manifest {
        dataset: dataset_id.to_string(),
        schema: crate::format::SCHEMA_VERSION,
        feature_version: feature_version.to_string(),
        shards: n_shards,
        total_records: records.len(),
        hash: format!("{:016x}", hash::fnv1a(acc.as_bytes())),
        compression: format!("{compression:?}").to_lowercase(),
        seed,
        shard_files: files,
    })
}

pub fn read_shard_file(path: &Path) -> Result<Vec<Record>> {
    let data = std::fs::read(path)?;
    let f = ShardFile::parse(&data)?;
    f.read_all()
}

pub fn shard_files(path: &Path) -> Result<Vec<PathBuf>> {
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
            out.push(path.join(name));
        }
        Ok(out)
    } else {
        Ok(vec![path.to_path_buf()])
    }
}

pub fn phase_counts(records: &[Record]) -> [usize; 3] {
    let mut c = [0usize; 3];
    for r in records {
        c[r.phase.min(2) as usize] += 1;
    }
    c
}

pub fn near_duplicate_stats(records: &[Record]) -> (usize, usize, usize) {
    let mut by_game: HashMap<u64, Vec<u64>> = HashMap::new();
    for r in records {
        by_game.entry(r.game_hash).or_default().push(r.identity);
    }
    let mut local_dup = 0usize;
    let mut clusters = 0usize;
    for ids in by_game.values() {
        let mut seen = HashSet::new();
        let mut dup = false;
        for id in ids {
            if !seen.insert(*id) {
                local_dup += 1;
                dup = true;
            }
        }
        if dup {
            clusters += 1;
        }
    }
    let mut counts: HashMap<u64, usize> = HashMap::new();
    for r in records {
        *counts.entry(r.identity).or_default() += 1;
    }
    let exact_dup: usize = counts.values().map(|&c| c.saturating_sub(1)).sum();
    (exact_dup, local_dup, clusters)
}

pub fn join_labels(
    records: Vec<Record>,
    labels: &[(u64, f32, u8, i32)],
    stats: &mut StageStats,
) -> (Vec<Record>, usize, usize) {
    let mut map: HashMap<u64, Vec<(f32, u8, i32)>> = HashMap::new();
    for (id, v, w, cp) in labels {
        map.entry(*id).or_default().push((*v, *w, *cp));
    }
    let mut out = Vec::new();
    let mut missing = 0usize;
    let mut dup_labels = 0usize;
    for mut r in records {
        stats.input += 1;
        match map.get(&r.identity) {
            None => {
                missing += 1;
                stats.reject("missing_label");
            }
            Some(v) if v.len() > 1 => {
                dup_labels += 1;
                stats.reject("duplicate_label");
            }
            Some(v) => {
                r = r.with_teacher(v[0].0, v[0].1, v[0].2, true);
                out.push(r);
            }
        }
    }
    stats.output = out.len();
    (out, missing, dup_labels)
}

pub fn write_report(path: &Path, report: &PipelineReport) -> Result<()> {
    let json = serde_json::to_string_pretty(report).map_err(|e| Error::BadFormat(e.to_string()))?;
    std::fs::write(path.with_extension("json"), &json)?;
    let mut md = String::new();
    md.push_str(&format!("# Pipeline report: {}\n\n", report.dataset_id));
    md.push_str(&format!("- seed: {}\n", report.seed));
    md.push_str(&format!(
        "- elapsed: {:.2}s, throughput: {:.0} pos/s\n",
        report.elapsed_sec, report.throughput_pos_per_sec
    ));
    md.push_str(&format!(
        "- estimated RSS: {:.1} MB\n",
        report.estimated_rss_mb
    ));
    md.push_str(&format!("- phases: {:?}\n", report.phase_counts));
    md.push_str(&format!("- mean material: {:.1}\n", report.material_mean));
    for (stage, s) in &report.stages {
        md.push_str(&format!(
            "\n## {stage}: in={} out={} rejected={} reasons={:?}\n",
            s.input, s.output, s.rejected, s.reasons
        ));
    }
    std::fs::write(path.with_extension("md"), md)?;
    Ok(())
}

pub fn rss_mb_estimate() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            if let Some(kb) = rest
                .split_whitespace()
                .next()
                .and_then(|x| x.parse::<f64>().ok())
            {
                return kb / 1024.0;
            }
        }
    }
    0.0
}

#[derive(Debug, Clone, Default)]
pub struct ScoreComponents {
    pub disagreement: f32,
    pub uncertainty: f32,
    pub instability: f32,
    pub rank_disagreement: f32,
    pub rarity: f32,
}

#[derive(Debug, Clone)]
pub struct ScoreWeights {
    pub disagreement: f32,
    pub uncertainty: f32,
    pub instability: f32,
    pub rarity: f32,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        ScoreWeights {
            disagreement: 1.0,
            uncertainty: 0.0,
            instability: 0.0,
            rarity: 0.0,
        }
    }
}

fn clip01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

pub fn final_score(c: &ScoreComponents, w: &ScoreWeights) -> f32 {
    (w.disagreement * clip01(c.disagreement / 2.0)
        + w.uncertainty * clip01(c.uncertainty)
        + w.instability * clip01(c.instability / 2.0)
        + w.rarity * clip01(c.rarity))
        / (w.disagreement + w.uncertainty + w.instability + w.rarity).max(1e-9)
}

#[derive(Debug, Clone, Default)]
pub struct SelectionConfig {
    pub budget: usize,
    pub method: crate::record::SelMethod,
    pub round: u32,
    pub weights: ScoreWeights,
    pub diversity_cap: usize,
    pub floor_ratio: f64,
    pub seed: u64,
    pub shard_topk: usize,
}

#[derive(Debug, Clone)]
pub struct ScoredRecord {
    pub record: Record,
    pub components: ScoreComponents,
    pub final_score: f32,
}

pub fn attach_scores(
    records: Vec<Record>,
    scores: &HashMap<u64, ScoreComponents>,
    weights: &ScoreWeights,
    stats: &mut StageStats,
) -> (Vec<ScoredRecord>, usize) {
    let mut out = Vec::with_capacity(records.len());
    let mut missing = 0usize;
    let mut buckets: HashMap<(u8, u8, u8), usize> = HashMap::new();
    for r in &records {
        stats.input += 1;
        let key = (
            r.phase.min(2),
            r.imbalance_bucket.min(7),
            (r.teacher_value + 1.0).clamp(0.0, 2.0) as u8,
        );
        *buckets.entry(key).or_insert(0) += 1;
    }
    let peak = buckets.values().cloned().max().unwrap_or(1) as f32;
    for mut r in records {
        match scores.get(&r.identity) {
            None => {
                missing += 1;
                stats.reject("missing_score");
            }
            Some(c) => {
                let mut comp = c.clone();
                let key = (
                    r.phase.min(2),
                    r.imbalance_bucket.min(7),
                    (r.teacher_value + 1.0).clamp(0.0, 2.0) as u8,
                );
                comp.rarity = 1.0 - buckets.get(&key).cloned().unwrap_or(1) as f32 / peak;
                let final_score = final_score(&comp, weights);
                r.sel_score = final_score;
                out.push(ScoredRecord {
                    record: r,
                    components: comp,
                    final_score,
                });
            }
        }
    }
    stats.output = out.len();
    (out, missing)
}

pub fn select_topk(
    scored: Vec<ScoredRecord>,
    cfg: &SelectionConfig,
    stats: &mut StageStats,
) -> Vec<ScoredRecord> {
    for _ in &scored {
        stats.input += 1;
    }
    let top = topk_by_score(&scored, cfg.budget);
    let k = cfg.budget.min(scored.len());
    let mut accepted: Vec<usize> = Vec::new();
    let mut taken: HashSet<usize> = HashSet::new();
    let mut bucket_count: HashMap<(u8, u8), usize> = HashMap::new();
    for &idx in &top {
        let r = &scored[idx].record;
        let key = (r.phase.min(2), r.imbalance_bucket.min(7));
        let c = bucket_count.entry(key).or_insert(0);
        if *c < cfg.diversity_cap.max(1) {
            *c += 1;
            accepted.push(idx);
            taken.insert(idx);
        } else {
            stats.reject("diversity_cap");
        }
        if accepted.len() >= k {
            break;
        }
    }
    if accepted.len() < k {
        for idx in 0..scored.len() {
            if accepted.len() >= k {
                break;
            }
            if taken.contains(&idx) {
                continue;
            }
            accepted.push(idx);
        }
    }
    stats.output = accepted.len();
    let mut rng = Rng::new(cfg.seed);
    let floor_n = ((k as f64 * cfg.floor_ratio).round() as usize).min(accepted.len());
    let mut out: Vec<ScoredRecord> = Vec::with_capacity(accepted.len());
    let mut floor_idx: HashSet<usize> = HashSet::new();
    while floor_idx.len() < floor_n && floor_idx.len() < accepted.len() {
        floor_idx.insert(rng.below(accepted.len()));
    }
    for (pos, idx) in accepted.into_iter().enumerate() {
        let mut s = scored[idx].clone();
        if floor_idx.contains(&pos) {
            s.record.sel_method = crate::record::SelMethod::Stratified;
        } else {
            s.record.sel_method = cfg.method;
        }
        s.record.active_round = cfg.round;
        out.push(s);
    }
    out
}

pub fn topk_by_score(scored: &[ScoredRecord], k: usize) -> Vec<usize> {
    use std::cmp::{Ordering, Reverse};
    use std::collections::BinaryHeap;
    struct Item {
        score: f32,
        ident: u64,
        idx: usize,
    }
    impl PartialEq for Item {
        fn eq(&self, other: &Self) -> bool {
            self.score == other.score && self.ident == other.ident
        }
    }
    impl Eq for Item {}
    impl PartialOrd for Item {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Item {
        fn cmp(&self, other: &Self) -> Ordering {
            self.score
                .total_cmp(&other.score)
                .then_with(|| other.ident.cmp(&self.ident))
        }
    }
    let k = k.min(scored.len());
    if k == 0 {
        return Vec::new();
    }
    let mut heap: BinaryHeap<Reverse<Item>> = BinaryHeap::with_capacity(k + 1);
    for (idx, s) in scored.iter().enumerate() {
        heap.push(Reverse(Item {
            score: s.final_score,
            ident: s.record.identity,
            idx,
        }));
        if heap.len() > k {
            heap.pop();
        }
    }
    heap.into_sorted_vec()
        .into_iter()
        .map(|it| it.0.idx)
        .collect()
}

#[derive(Debug, Clone, Default)]
pub struct LevelEval {
    pub cp: i32,
    pub value_stm: f32,
    pub wdl_stm: u8,
    pub nodes: u64,
    pub uncertainty: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct MultiDepth {
    pub fen: String,
    pub levels: Vec<(String, LevelEval)>,
}

pub fn parse_multidepth_jsonl(path: &Path) -> Result<Vec<MultiDepth>> {
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
            .ok_or_else(|| Error::BadFormat("multidepth record missing fen".to_string()))?
            .to_string();
        let lv = v.get("teacher_levels").ok_or_else(|| {
            Error::BadFormat("multidepth record missing teacher_levels".to_string())
        })?;
        let obj = lv
            .as_object()
            .ok_or_else(|| Error::BadFormat("teacher_levels not an object".to_string()))?;
        let mut levels: Vec<(String, LevelEval)> = Vec::new();
        for (k, e) in obj {
            levels.push((
                k.clone(),
                LevelEval {
                    cp: e.get("cp").and_then(|x| x.as_i64()).unwrap_or(0) as i32,
                    value_stm: e.get("value_stm").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
                    wdl_stm: e.get("wdl_stm").and_then(|x| x.as_u64()).unwrap_or(1) as u8,
                    nodes: e.get("nodes").and_then(|x| x.as_u64()).unwrap_or(0),
                    uncertainty: e
                        .get("uncertainty")
                        .and_then(|x| x.as_f64())
                        .map(|x| x as f32),
                },
            ));
        }
        levels.sort_by(|a, b| a.0.cmp(&b.0));
        out.push(MultiDepth { fen, levels });
    }
    Ok(out)
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct StabilityReport {
    pub n: usize,
    pub max_delta_mean: f32,
    pub max_delta_p90: f32,
    pub wdl_flip_rate: f32,
    pub unstable_frac: f32,
    pub unstable_threshold: f32,
}

pub fn stability_report(records: &[MultiDepth], unstable_above: f32) -> StabilityReport {
    let mut deltas = Vec::new();
    let mut flips = 0usize;
    let mut unstable = 0usize;
    for r in records {
        let vals: Vec<f32> = r.levels.iter().map(|(_, e)| e.value_stm).collect();
        if vals.len() < 2 {
            continue;
        }
        let mut mx = 0.0f32;
        let mut fl = 0usize;
        for w in vals.windows(2) {
            mx = mx.max((w[1] - w[0]).abs());
            let ca = if w[0].abs() < 0.15 {
                1
            } else if w[0] > 0.0 {
                0
            } else {
                2
            };
            let cb = if w[1].abs() < 0.15 {
                1
            } else if w[1] > 0.0 {
                0
            } else {
                2
            };
            fl += usize::from(ca != cb);
        }
        deltas.push(mx);
        flips += usize::from(fl > 0);
        unstable += usize::from(mx > unstable_above);
    }
    deltas.sort_by(|a, b| a.total_cmp(b));
    let n = deltas.len();
    StabilityReport {
        n: records.len(),
        max_delta_mean: if n > 0 {
            deltas.iter().sum::<f32>() / n as f32
        } else {
            0.0
        },
        max_delta_p90: if n > 0 {
            deltas[(n * 9 / 10).min(n - 1)]
        } else {
            0.0
        },
        wdl_flip_rate: flips as f32 / n.max(1) as f32,
        unstable_frac: unstable as f32 / n.max(1) as f32,
        unstable_threshold: unstable_above,
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CascadePlan {
    pub levels: Vec<String>,
    pub per_level_cost_sec: Vec<f64>,
    pub accept_threshold: f32,
    pub accepted_low: usize,
    pub escalated: usize,
    pub total_cost_sec: f64,
    pub all_high_cost_sec: f64,
    pub savings_frac: f64,
    pub residual_unstable: usize,
}

pub fn cascade_plan(
    records: &[MultiDepth],
    level_order: &[String],
    per_level_cost_sec: &[f64],
    accept_threshold: f32,
) -> CascadePlan {
    let mut accepted_low = 0usize;
    let mut escalated = 0usize;
    let mut residual_unstable = 0usize;
    let mut total = 0.0;
    for r in records {
        let get = |name: &str| r.levels.iter().find(|(k, _)| k == name).map(|(_, e)| e);
        total += per_level_cost_sec.first().cloned().unwrap_or(0.0);
        let mut stable = false;
        if level_order.len() >= 2 {
            if let (Some(lo), Some(hi)) = (get(&level_order[0]), get(&level_order[1])) {
                stable = (hi.value_stm - lo.value_stm).abs() <= accept_threshold;
            }
        }
        if stable {
            accepted_low += 1;
        } else {
            escalated += 1;
            total += per_level_cost_sec.iter().skip(1).sum::<f64>();
            if let (Some(first), Some(last)) =
                (get(&level_order[0]), get(level_order.last().unwrap()))
            {
                if (last.value_stm - first.value_stm).abs() > accept_threshold {
                    residual_unstable += 1;
                }
            }
        }
    }
    let all_high = records.len() as f64 * per_level_cost_sec.iter().sum::<f64>();
    CascadePlan {
        levels: level_order.to_vec(),
        per_level_cost_sec: per_level_cost_sec.to_vec(),
        accept_threshold,
        accepted_low,
        escalated,
        total_cost_sec: total,
        all_high_cost_sec: all_high,
        savings_frac: if all_high > 0.0 {
            1.0 - total / all_high
        } else {
            0.0
        },
        residual_unstable,
    }
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TargetVerifyReport {
    pub records: usize,
    pub ok: usize,
    pub missing_teacher: usize,
    pub bad_range: usize,
    pub bad_wdl: usize,
    pub perspective_mismatch: usize,
    pub stated_perspective_stm: bool,
}

pub fn verify_target_records(records: &[Record], expect_stm: bool) -> TargetVerifyReport {
    let mut rep = TargetVerifyReport {
        records: records.len(),
        stated_perspective_stm: expect_stm,
        ..Default::default()
    };
    for r in records {
        if !r.has_teacher {
            rep.missing_teacher += 1;
            continue;
        }
        if !(-1.0..=1.0).contains(&r.teacher_value) {
            rep.bad_range += 1;
            continue;
        }
        if r.teacher_wdl > 2 {
            rep.bad_wdl += 1;
            continue;
        }
        if r.perspective_stm != expect_stm {
            rep.perspective_mismatch += 1;
            continue;
        }
        rep.ok += 1;
    }
    rep
}

#[cfg(test)]
mod target_tests {
    use super::*;

    fn md(vals: &[f32]) -> MultiDepth {
        MultiDepth {
            fen: "x".to_string(),
            levels: vals
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    (
                        format!("d{i}"),
                        LevelEval {
                            cp: (v * 400.0) as i32,
                            value_stm: *v,
                            wdl_stm: 1,
                            nodes: 100,
                            uncertainty: None,
                        },
                    )
                })
                .collect(),
        }
    }

    #[test]
    fn stability_stable_vs_flip() {
        let r = stability_report(&[md(&[0.1, 0.12, 0.11])], 0.3);
        assert_eq!(r.wdl_flip_rate, 0.0);
        assert_eq!(r.unstable_frac, 0.0);
        let r = stability_report(&[md(&[-0.8, 0.1, 0.9])], 0.3);
        assert!(r.unstable_frac > 0.0);
        assert!(r.wdl_flip_rate > 0.0);
    }

    #[test]
    fn cascade_saves_when_stable() {
        let recs = vec![md(&[0.1, 0.11]), md(&[0.1, 0.12]), md(&[-0.9, 0.9])];
        let plan = cascade_plan(
            &recs,
            &["d0".to_string(), "d1".to_string()],
            &[0.02, 0.08],
            0.3,
        );
        assert_eq!(plan.accepted_low, 2);
        assert_eq!(plan.escalated, 1);
        assert!(plan.savings_frac > 0.0);
        assert!(plan.total_cost_sec < plan.all_high_cost_sec);
    }

    #[test]
    fn verify_targets_counts() {
        let good = Record::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "g",
            8,
            0,
            false,
        )
        .unwrap()
        .with_teacher(0.2, 1, 50, true);
        let mut bad = good.clone();
        bad.teacher_value = 5.0;
        let mut unlabeled = good.clone();
        unlabeled.has_teacher = false;
        let rep = verify_target_records(&[good, bad, unlabeled], true);
        assert_eq!((rep.ok, rep.bad_range, rep.missing_teacher), (1, 1, 1));
    }

    fn scored_record(score: f32, ident: u64) -> ScoredRecord {
        ScoredRecord {
            record: crate::record::Record {
                fen: String::new(),
                identity: ident,
                phase: 0,
                stm: 0,
                piece_count: 0,
                pawn_count: 0,
                queens: 0,
                imbalance_bucket: 0,
                teacher_value: 0.0,
                teacher_wdl: 0,
                has_teacher: false,
                teacher_cp: 0,
                perspective_stm: true,
                game_hash: 0,
                ply: 0,
                source_id: 0,
                features: Vec::new(),
                active_round: 0,
                sel_method: crate::record::SelMethod::None,
                sel_score: 0.0,
            },
            components: ScoreComponents {
                disagreement: 0.0,
                uncertainty: 0.0,
                instability: 0.0,
                rank_disagreement: 0.0,
                rarity: 0.0,
            },
            final_score: score,
        }
    }

    #[test]
    fn topk_keeps_highest_scores() {
        let v: Vec<ScoredRecord> = [1.0, 5.0, 3.0, 4.0, 2.0]
            .into_iter()
            .enumerate()
            .map(|(i, s)| scored_record(s, i as u64))
            .collect();
        let top = topk_by_score(&v, 2);
        assert_eq!(top.len(), 2);
        let scores: Vec<f32> = top.iter().map(|i| v[*i].final_score).collect();
        assert_eq!(scores, vec![5.0, 4.0]);
        assert_eq!(topk_by_score(&v, 0).len(), 0);
        assert_eq!(topk_by_score(&v, 99).len(), 5);
    }

    #[test]
    fn stratified_fills_remainder() {
        let mk = |fen: &str, phase: u8| {
            let mut r = crate::record::Record::from_fen(fen, "g", 8, 0, false).unwrap();
            r.phase = phase;
            r
        };
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let end = "7k/8/8/8/8/8/8/K6R w - - 0 1";
        let mut v = Vec::new();
        for _ in 0..4 {
            v.push(mk(start, 0));
        }
        for _ in 0..4 {
            v.push(mk(start, 1));
        }
        for _ in 0..4 {
            v.push(mk(end, 2));
        }
        assert_eq!(sample_stratified(v.clone(), 10, 7).len(), 10);
        assert_eq!(sample_stratified(v.clone(), 9, 7).len(), 9);
        assert_eq!(sample_stratified(v.clone(), 99, 7).len(), 12);
        assert_eq!(sample_stratified(v, 0, 7).len(), 0);
    }
}
