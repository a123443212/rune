use std::collections::HashMap;
use rune_kernel as kernel;
use crate::error::{Result, RuntimeError};
use crate::policy_head::PolicyWeights;

#[derive(Debug, Clone)]
pub struct ResnetConfig {
    pub board: usize,
    pub channels: usize,
    pub blocks: usize,
    pub policy_size: usize,
    pub value_h2: usize,
    pub in_planes: usize,
}

#[derive(Debug, Clone)]
pub struct ResnetWeights {
    pub cfg: ResnetConfig,
    pub stem_w: Vec<f32>,
    pub stem_b: Vec<f32>,
    pub block_w1: Vec<Vec<f32>>,
    pub block_b1: Vec<Vec<f32>>,
    pub block_w2: Vec<Vec<f32>>,
    pub block_b2: Vec<Vec<f32>>,
    pub vh1: Vec<f32>,
    pub bh1: Vec<f32>,
    pub wv: Vec<f32>,
    pub bv: f32,
    pub wwdl: Vec<f32>,
    pub bwdl: Vec<f32>,
    pub policy: PolicyWeights,
}

fn need_vec(arrays: &HashMap<String, Vec<f32>>, name: &str) -> Result<Vec<f32>> {
    arrays.get(name).cloned().ok_or_else(|| RuntimeError::TensorMissing(name.to_string()))
}

fn scalar_of(arrays: &HashMap<String, Vec<f32>>, name: &str) -> Result<f32> {
    let v = need_vec(arrays, name)?;
    if v.is_empty() {
        return Err(RuntimeError::Shape(name.to_string()));
    }
    Ok(v[0])
}

impl ResnetWeights {
    pub fn from_arrays(arrays: &HashMap<String, Vec<f32>>, raw: &serde_json::Value) -> Result<ResnetWeights> {
        let board = raw.get("board_size").and_then(|v| v.as_u64()).unwrap_or(19) as usize;
        let channels = raw.get("channels").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
        let blocks = raw.get("num_blocks").and_then(|v| v.as_u64()).unwrap_or(6) as usize;
        let policy_size = raw.get("policy_size").and_then(|v| v.as_u64()).unwrap_or((board * board + 1) as u64) as usize;
        let h2 = raw.get("head_h2").and_then(|v| v.as_u64()).unwrap_or(256) as usize;
        let in_planes = raw.get("in_planes").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        if board == 0 || board > 19 {
            return Err(RuntimeError::Shape("board_size".to_string()));
        }
        if channels == 0 || channels > 256 {
            return Err(RuntimeError::Shape("channels".to_string()));
        }
        if in_planes == 0 || in_planes > 16 {
            return Err(RuntimeError::Shape("in_planes".to_string()));
        }
        let stem_w = need_vec(arrays, "stem_w")?;
        let stem_b = need_vec(arrays, "stem_b")?;
        let mut block_w1 = Vec::new();
        let mut block_b1 = Vec::new();
        let mut block_w2 = Vec::new();
        let mut block_b2 = Vec::new();
        for b in 0..blocks {
            block_w1.push(need_vec(arrays, &format!("b{}_w1", b))?);
            block_b1.push(need_vec(arrays, &format!("b{}_b1", b))?);
            block_w2.push(need_vec(arrays, &format!("b{}_w2", b))?);
            block_b2.push(need_vec(arrays, &format!("b{}_b2", b))?);
        }
        let vh1 = need_vec(arrays, "vh1")?;
        let bh1 = need_vec(arrays, "bh1")?;
        let wv = need_vec(arrays, "wv")?;
        let bv = scalar_of(arrays, "bv")?;
        let wwdl = need_vec(arrays, "wwdl")?;
        let bwdl = need_vec(arrays, "bwdl")?;
        let wpol = need_vec(arrays, "wpol")?;
        let bpol = need_vec(arrays, "bpol")?;
        let policy = PolicyWeights { input: channels * board * board, output: policy_size, w: wpol, b: bpol };
        if policy.w.len() != policy.input * policy.output {
            return Err(RuntimeError::Shape("wpol".to_string()));
        }
        if stem_w.len() != channels * in_planes * 9 {
            return Err(RuntimeError::Shape("stem_w".to_string()));
        }
        Ok(ResnetWeights { cfg: ResnetConfig { board, channels, blocks, policy_size, value_h2: h2, in_planes }, stem_w, stem_b, block_w1, block_b1, block_w2, block_b2, vh1, bh1, wv, bv, wwdl, bwdl, policy })
    }

    pub fn forward(&self, planes: &[f32]) -> (f32, [f32; 3], Vec<f32>) {
        let b = self.cfg.board;
        let c = self.cfg.channels;
        let p = self.cfg.in_planes;
        let hw = b * b;
        let mut cur = vec![0.0f32; c * hw];
        kernel::conv2d_nchw(planes, &self.stem_w, Some(&self.stem_b), &mut cur, 1, p, c, b, b, 3, 3, 1, 1);
        kernel::relu_inplace(&mut cur);
        let mut tmp = vec![0.0f32; c * hw];
        let mut tmp2 = vec![0.0f32; c * hw];
        let mut out = vec![0.0f32; c * hw];
        for i in 0..self.cfg.blocks {
            kernel::conv2d_nchw(&cur, &self.block_w1[i], Some(&self.block_b1[i]), &mut tmp, 1, c, c, b, b, 3, 3, 1, 1);
            kernel::relu_inplace(&mut tmp);
            kernel::conv2d_nchw(&tmp, &self.block_w2[i], Some(&self.block_b2[i]), &mut tmp2, 1, c, c, b, b, 3, 3, 1, 1);
            kernel::residual_add_relu(&cur, &tmp2, &mut out);
            cur.clone_from_slice(&out);
        }
        let mut pooled = vec![0.0f32; c];
        kernel::global_avg_pool(&cur, &mut pooled, 1, c, b, b);
        let h2n = self.bh1.len();
        let mut h = vec![0.0f32; h2n];
        kernel::mat_vec(&self.vh1, &pooled, Some(&self.bh1), &mut h, h2n, c);
        for v in h.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            } else if *v > 1.0 {
                *v = 1.0;
            }
        }
        let mut vv = self.bv;
        for i in 0..h2n.min(self.wv.len()) {
            vv += self.wv[i] * h[i];
        }
        let value = vv.tanh();
        let mut wdl = [0.0f32; 3];
        kernel::mat_vec(&self.wwdl, &h, Some(&self.bwdl), &mut wdl, 3, h2n);
        let policy = self.policy.forward(&cur);
        (value, wdl, policy)
    }
}
