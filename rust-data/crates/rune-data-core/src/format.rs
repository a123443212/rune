use crate::error::{Error, Result};
use crate::record::Record;

pub const MAGIC: &[u8; 8] = b"RUNEDATA";
pub const FORMAT_VERSION: u32 = 1;
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Compression {
    Raw = 0,
    Deflate = 1,
}

impl Compression {
    pub fn from_u8(v: u8) -> Result<Compression> {
        match v {
            0 => Ok(Compression::Raw),
            1 => Ok(Compression::Deflate),
            _ => Err(Error::BadFormat(format!("unknown compression {v}"))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShardHeader {
    pub schema: u32,
    pub feature_version: String,
    pub compression: Compression,
    pub record_count: u64,
    pub shard_id: u32,
    pub shard_count: u32,
    pub dataset_id: String,
}

fn put_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_str(out: &mut Vec<u8>, s: &str) {
    put_u16(out, s.len().min(65535) as u16);
    out.extend_from_slice(&s.as_bytes()[..s.len().min(65535)]);
}

const RECORD_FIXED: usize = 42;

fn encode_record(r: &Record, out: &mut Vec<u8>, schema_v2: bool) {
    let base = out.len();
    put_u64(out, r.identity);
    out.push(r.phase);
    out.push(r.stm);
    out.push(r.piece_count);
    out.push(r.pawn_count);
    out.push(r.queens);
    out.push(r.imbalance_bucket);
    out.push(r.teacher_wdl);
    let mut flags = 0u8;
    if r.has_teacher {
        flags |= 1;
    }
    if r.perspective_stm {
        flags |= 2;
    }
    out.push(flags);
    put_f32(out, r.teacher_value);
    put_i32(out, r.teacher_cp);
    put_u64(out, r.game_hash);
    put_u16(out, r.ply);
    put_u32(out, r.source_id);
    put_u16(out, r.features.len().min(65535) as u16);
    put_u16(out, r.fen.len().min(65535) as u16);
    if schema_v2 {
        put_u32(out, r.active_round);
        put_f32(out, r.sel_score);
        out.push(r.sel_method as u8);
    }
    debug_assert_eq!(out.len() - base, fixed_prefix_len(schema_v2));
    for &(g, i) in &r.features {
        put_u16(out, g as u16);
        put_u16(out, i);
    }
    out.extend_from_slice(&r.fen.as_bytes()[..r.fen.len().min(65535)]);
}

fn fixed_prefix_len(schema_v2: bool) -> usize {
    RECORD_FIXED + if schema_v2 { 9 } else { 0 }
}

pub struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(data: &'a [u8]) -> Cursor<'a> {
        Cursor { data, pos: 0 }
    }
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(Error::Truncated);
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    pub fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    pub fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }
    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?[..4].try_into().unwrap()))
    }
    pub fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?[..4].try_into().unwrap()))
    }
    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.take(n)
    }
    pub fn string(&mut self) -> Result<String> {
        let n = self.u16()? as usize;
        let b = self.take(n)?;
        String::from_utf8(b.to_vec()).map_err(|_| Error::BadFormat("bad utf8".to_string()))
    }
}

fn decode_record(cur: &mut Cursor, schema: u32) -> Result<Record> {
    let start = cur.pos;
    let identity = cur.u64()?;
    let phase = cur.u8()?;
    let stm = cur.u8()?;
    let piece_count = cur.u8()?;
    let pawn_count = cur.u8()?;
    let queens = cur.u8()?;
    let imbalance_bucket = cur.u8()?;
    let teacher_wdl = cur.u8()?;
    let flags = cur.u8()?;
    let teacher_value = cur.f32()?;
    let teacher_cp = cur.i32()?;
    let game_hash = cur.u64()?;
    let ply = cur.u16()?;
    let source_id = cur.u32()?;
    let feat_count = cur.u16()? as usize;
    let fen_len = cur.u16()? as usize;
    let (active_round, sel_score, sel_method) = if schema >= 2 {
        let r = cur.u32()?;
        let s = cur.f32()?;
        let m = cur.u8()?;
        (
            r,
            s,
            crate::record::SelMethod::from_u8(m)
                .ok_or_else(|| Error::BadFormat(format!("bad sel method {m}")))?,
        )
    } else {
        (0, 0.0, crate::record::SelMethod::None)
    };
    debug_assert_eq!(cur.pos - start, fixed_prefix_len(schema >= 2));
    let mut features = Vec::with_capacity(feat_count);
    for _ in 0..feat_count {
        let g = cur.u16()?;
        let i = cur.u16()?;
        if g > 7 {
            return Err(Error::BadFormat(format!("bad group {g}")));
        }
        features.push((g as u8, i));
    }
    let fen_b = cur.bytes(fen_len)?;
    let fen = String::from_utf8(fen_b.to_vec())
        .map_err(|_| Error::BadFormat("bad fen utf8".to_string()))?;
    Ok(Record {
        fen,
        identity,
        phase,
        stm,
        piece_count,
        pawn_count,
        queens,
        imbalance_bucket,
        teacher_value,
        teacher_wdl,
        has_teacher: flags & 1 != 0,
        teacher_cp,
        perspective_stm: flags & 2 != 0,
        game_hash,
        ply,
        source_id,
        features,
        active_round,
        sel_method,
        sel_score,
    })
}

fn compress_payload(raw: &[u8], c: Compression) -> Vec<u8> {
    match c {
        Compression::Raw => raw.to_vec(),
        Compression::Deflate => {
            use flate2::write::DeflateEncoder;
            use flate2::Compression as Fl;
            use std::io::Write;
            let mut e = DeflateEncoder::new(Vec::new(), Fl::default());
            e.write_all(raw).expect("deflate write");
            e.finish().expect("deflate finish")
        }
    }
}

fn decompress_payload(stored: &[u8], c: Compression, expect_len: u64) -> Result<Vec<u8>> {
    match c {
        Compression::Raw => Ok(stored.to_vec()),
        Compression::Deflate => {
            use flate2::read::DeflateDecoder;
            use std::io::Read;
            let mut d = DeflateDecoder::new(stored);
            let mut out = Vec::with_capacity(expect_len.min(1 << 31) as usize);
            d.read_to_end(&mut out)
                .map_err(|e| Error::BadFormat(format!("deflate error: {e}")))?;
            if out.len() as u64 != expect_len {
                return Err(Error::BadFormat("decompressed length mismatch".to_string()));
            }
            Ok(out)
        }
    }
}

pub fn write_shard(header: &ShardHeader, records: &[Record], out: &mut Vec<u8>) -> Result<()> {
    let v2 = header.schema >= 2;
    let mut raw: Vec<u8> = Vec::new();
    let mut offsets: Vec<u64> = Vec::with_capacity(records.len());
    let mut recbuf: Vec<u8> = Vec::new();
    for r in records {
        offsets.push(recbuf.len() as u64);
        let before = recbuf.len();
        encode_record(r, &mut recbuf, v2);
        debug_assert!(recbuf.len() > before);
    }
    for o in &offsets {
        put_u64(&mut raw, *o);
    }
    raw.extend_from_slice(&recbuf);

    let stored = compress_payload(&raw, header.compression);
    let mut h: Vec<u8> = Vec::new();
    h.extend_from_slice(MAGIC);
    put_u32(&mut h, FORMAT_VERSION);
    put_u32(&mut h, header.schema);
    put_str(&mut h, &header.feature_version);
    h.push(header.compression as u8);
    put_u64(&mut h, header.record_count);
    put_u32(&mut h, header.shard_id);
    put_u32(&mut h, header.shard_count);
    put_str(&mut h, &header.dataset_id);
    put_u64(&mut h, raw.len() as u64);
    let hcrc = crc32fast::hash(&h);
    put_u32(&mut h, hcrc);
    let pcrc = crc32fast::hash(&stored);
    out.extend_from_slice(&h);
    out.extend_from_slice(&stored);
    put_u32(out, pcrc);
    Ok(())
}

#[derive(Debug)]
pub struct ShardFile {
    pub header: ShardHeader,
    pub payload: Vec<u8>,
    pub uncompressed_len: u64,
    pub schema: u32,
}

impl ShardFile {
    pub fn parse(data: &[u8]) -> Result<ShardFile> {
        let mut cur = Cursor::new(data);
        let magic = cur.bytes(8)?;
        if magic != MAGIC {
            return Err(Error::BadFormat("bad magic".to_string()));
        }
        let header_start = 0usize;
        let fver = cur.u32()?;
        if fver != FORMAT_VERSION {
            return Err(Error::UnsupportedVersion {
                want: FORMAT_VERSION,
                got: fver,
            });
        }
        let _schema = cur.u32()?;
        if _schema != 1 && _schema != 2 {
            return Err(Error::UnsupportedVersion { want: 2, got: _schema });
        }
        let schema = _schema;
        let feature_version = cur.string()?;
        let compression = Compression::from_u8(cur.u8()?)?;
        let record_count = cur.u64()?;
        let shard_id = cur.u32()?;
        let shard_count = cur.u32()?;
        let dataset_id = cur.string()?;
        let uncompressed_len = cur.u64()?;
        let header_end = cur.pos;
        let want_crc = cur.u32()?;
        let got_crc = crc32fast::hash(&data[header_start..header_end]);
        if want_crc != got_crc {
            return Err(Error::BadFormat("header crc mismatch".to_string()));
        }
        let payload_end = data.len().checked_sub(4).ok_or(Error::Truncated)?;
        if payload_end < cur.pos {
            return Err(Error::Truncated);
        }
        let stored = &data[cur.pos..payload_end];
        let mut pcrc_b = [0u8; 4];
        pcrc_b.copy_from_slice(&data[payload_end..]);
        let want_p = u32::from_le_bytes(pcrc_b);
        let got_p = crc32fast::hash(stored);
        if want_p != got_p {
            return Err(Error::ChecksumMismatch {
                want: want_p as u64,
                got: got_p as u64,
            });
        }
        let payload = decompress_payload(stored, compression, uncompressed_len)?;
        Ok(ShardFile {
            header: ShardHeader {
                schema,
                feature_version,
                compression,
                record_count,
                shard_id,
                shard_count,
                dataset_id,
            },
            payload,
            uncompressed_len,
            schema,
        })
    }

    pub fn offsets(&self) -> Result<Vec<u64>> {
        let n = self.header.record_count as usize;
        let mut cur = Cursor::new(&self.payload);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(cur.u64()?);
        }
        Ok(out)
    }

    pub fn read_all(&self) -> Result<Vec<Record>> {
        let n = self.header.record_count as usize;
        let mut cur = Cursor::new(&self.payload);
        for _ in 0..n {
            let _ = cur.u64()?;
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(decode_record(&mut cur, self.schema)?);
        }
        if out.len() != n {
            return Err(Error::BadFormat("record count mismatch".to_string()));
        }
        Ok(out)
    }

    pub fn read_at(&self, offsets: &[u64], idx: usize) -> Result<Record> {
        if idx >= offsets.len() {
            return Err(Error::BadFormat("index out of range".to_string()));
        }
        let table_bytes = offsets.len() * 8;
        let start = table_bytes + offsets[idx] as usize;
        if start >= self.payload.len() {
            return Err(Error::BadFormat("offset out of range".to_string()));
        }
        let mut cur = Cursor::new(&self.payload[start..]);
        decode_record(&mut cur, self.schema)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Record;

    fn sample() -> Vec<Record> {
        vec![
            Record::from_fen(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "g1",
                0,
                0,
                false,
            )
            .unwrap()
            .with_teacher(0.1, 1, 35, true),
            Record::from_fen("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1", "g2", 40, 0, false).unwrap(),
        ]
    }

    fn roundtrip(c: Compression) {
        let h = ShardHeader {
            schema: 2,
            feature_version: "grouped_hkav2_fullthreats_v01".to_string(),
            compression: c,
            record_count: 2,
            shard_id: 0,
            shard_count: 1,
            dataset_id: "test".to_string(),
        };
        let recs = sample();
        let mut buf = Vec::new();
        write_shard(&h, &recs, &mut buf).unwrap();
        let f = ShardFile::parse(&buf).unwrap();
        assert_eq!(f.header.record_count, 2);
        let back = f.read_all().unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].fen, recs[0].fen);
        assert_eq!(back[0].features, recs[0].features);
        assert!((back[0].teacher_value - 0.1).abs() < 1e-6);
        assert!(!back[1].has_teacher);
        let offs = f.offsets().unwrap();
        let r1 = f.read_at(&offs, 1).unwrap();
        assert_eq!(r1.fen, recs[1].fen);
    }

    #[test]
    fn roundtrip_raw() {
        roundtrip(Compression::Raw);
    }

    #[test]
    fn roundtrip_deflate() {
        roundtrip(Compression::Deflate);
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(ShardFile::parse(b"BADMAGICxxxxxxxx").is_err());
    }

    #[test]
    fn rejects_tamper() {
        let h = ShardHeader {
            schema: 2,
            feature_version: "f".to_string(),
            compression: Compression::Raw,
            record_count: 1,
            shard_id: 0,
            shard_count: 1,
            dataset_id: "t".to_string(),
        };
        let mut buf = Vec::new();
        write_shard(&h, &sample()[..1], &mut buf).unwrap();
        let last = buf.len() - 1;
        buf[last] ^= 0xFF;
        assert!(ShardFile::parse(&buf).is_err());
    }

    #[test]
    fn schema_v1_defaults_selection() {
        let h = ShardHeader {
            schema: 1,
            feature_version: "f".to_string(),
            compression: Compression::Raw,
            record_count: 1,
            shard_id: 0,
            shard_count: 1,
            dataset_id: "t".to_string(),
        };
        let mut buf = Vec::new();
        write_shard(&h, &sample()[..1], &mut buf).unwrap();
        let f = ShardFile::parse(&buf).unwrap();
        assert_eq!(f.schema, 1);
        let back = f.read_all().unwrap();
        assert_eq!(back[0].active_round, 0);
        assert_eq!(back[0].sel_score, 0.0);
        assert_eq!(back[0].sel_method, crate::record::SelMethod::None);
    }

    #[test]
    fn schema_v2_selection_roundtrip() {
        let h = ShardHeader {
            schema: 2,
            feature_version: "f".to_string(),
            compression: Compression::Raw,
            record_count: 1,
            shard_id: 0,
            shard_count: 1,
            dataset_id: "t".to_string(),
        };
        let mut recs = sample()[..1].to_vec();
        recs[0].active_round = 3;
        recs[0].sel_method = crate::record::SelMethod::Multi;
        recs[0].sel_score = 0.75;
        let mut buf = Vec::new();
        write_shard(&h, &recs, &mut buf).unwrap();
        let f = ShardFile::parse(&buf).unwrap();
        let back = f.read_all().unwrap();
        assert_eq!(back[0].active_round, 3);
        assert_eq!(back[0].sel_method, crate::record::SelMethod::Multi);
        assert!((back[0].sel_score - 0.75).abs() < 1e-6);
    }

    #[test]
    fn rejects_unknown_schema() {
        let h = ShardHeader {
            schema: 2,
            feature_version: "f".to_string(),
            compression: Compression::Raw,
            record_count: 1,
            shard_id: 0,
            shard_count: 1,
            dataset_id: "t".to_string(),
        };
        let mut buf = Vec::new();
        write_shard(&h, &sample()[..1], &mut buf).unwrap();
        let schema_pos = 8 + 4;
        buf[schema_pos] = 99;
        buf[schema_pos + 1] = 0;
        buf[schema_pos + 2] = 0;
        buf[schema_pos + 3] = 0;
        let crc_pos = 8 + 4 + 4 + 2 + 1 + 1 + 8 + 4 + 4 + 2 + 1 + 8;
        let fixed = crc32fast::hash(&buf[..crc_pos]);
        buf[crc_pos..crc_pos + 4].copy_from_slice(&fixed.to_le_bytes());
        let err = ShardFile::parse(&buf).unwrap_err();
        assert!(matches!(err, Error::UnsupportedVersion { .. }));
    }
}
