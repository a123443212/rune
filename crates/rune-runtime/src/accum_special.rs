use crate::accumulator::Tables;

pub struct GroupedSpec {
    pub dim: usize,
    pub offsets: [usize; 8],
}

impl GroupedSpec {
    pub fn fixed_8x32() -> Self {
        let mut offsets = [0usize; 8];
        for g in 0..8 {
            offsets[g] = g * 32;
        }
        GroupedSpec { dim: 32, offsets }
    }

    pub fn for_dim(dim: usize) -> Self {
        let mut offsets = [0usize; 8];
        for g in 0..8 {
            offsets[g] = g * dim;
        }
        GroupedSpec { dim, offsets }
    }
}

pub fn refresh_grouped(acc: &mut [f32], tables: &Tables, feats: &[(u8, u16)], spec: &GroupedSpec) {
    for v in acc.iter_mut() {
        *v = 0.0;
    }
    apply_grouped(acc, tables, feats, &[], spec);
}

pub fn apply_grouped(acc: &mut [f32], tables: &Tables, added: &[(u8, u16)], removed: &[(u8, u16)], spec: &GroupedSpec) {
    let dim = spec.dim;
    let mut by_group_add: [Vec<u16>; 8] = Default::default();
    let mut by_group_rm: [Vec<u16>; 8] = Default::default();
    for (g, idx) in added {
        by_group_add[*g as usize].push(*idx);
    }
    for (g, idx) in removed {
        by_group_rm[*g as usize].push(*idx);
    }
    for g in 0..8 {
        if by_group_add[g].is_empty() && by_group_rm[g].is_empty() {
            continue;
        }
        let base = spec.offsets[g];
        for idx in &by_group_add[g] {
            let row = tables.row(g, *idx as usize);
            let dst = &mut acc[base..base + dim];
            let mut d = 0;
            while d < dim {
                dst[d] += row[d];
                if d + 1 < dim {
                    dst[d + 1] += row[d + 1];
                }
                d += 2;
            }
        }
        for idx in &by_group_rm[g] {
            let row = tables.row(g, *idx as usize);
            let dst = &mut acc[base..base + dim];
            let mut d = 0;
            while d < dim {
                dst[d] -= row[d];
                if d + 1 < dim {
                    dst[d + 1] -= row[d + 1];
                }
                d += 2;
            }
        }
    }
}

pub fn pack_feature_ids(feats: &[(u8, u16)], out_groups: &mut [u8], out_idx: &mut [u16]) -> usize {
    let n = feats.len().min(out_groups.len()).min(out_idx.len());
    for i in 0..n {
        out_groups[i] = feats[i].0;
        out_idx[i] = feats[i].1;
    }
    n
}
