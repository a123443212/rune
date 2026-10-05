use std::fs;
use std::path::{Path, PathBuf};

pub fn cache_file(cache_dir: &Path, key: &str) -> PathBuf {
    cache_dir.join(format!("rune-{}.json", key))
}

pub fn cache_lookup(cache_dir: &Path, key: &str) -> Option<serde_json::Value> {
    let p = cache_file(cache_dir, key);
    let data = fs::read(p).ok()?;
    serde_json::from_slice(&data).ok()
}

pub fn cache_store(cache_dir: &Path, key: &str, doc: &serde_json::Value) -> Result<PathBuf, String> {
    fs::create_dir_all(cache_dir).map_err(|e| e.to_string())?;
    let p = cache_file(cache_dir, key);
    let data = serde_json::to_vec_pretty(doc).map_err(|e| e.to_string())?;
    fs::write(&p, data).map_err(|e| e.to_string())?;
    Ok(p)
}
