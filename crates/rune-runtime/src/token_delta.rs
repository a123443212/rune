// RUNE — Relational Unified Neural Evaluator
// Copyright (C) 2026 a123443212
//
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// This project is dual-licensed under the MIT License and the
// Apache License, Version 2.0. You may choose either license
// when using, copying, modifying, or distributing this software.
//
// MIT License: https://opensource.org/license/mit
// Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, this
// software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
// OR CONDITIONS OF ANY KIND, either express or implied.

#[derive(Debug, Clone, Default)]
pub struct GroupDelta {
    pub added: Vec<Vec<(u8, u16)>>,
    pub removed: Vec<Vec<(u8, u16)>>,
    pub changed_groups: Vec<usize>,
}

pub fn detect_changed_groups(before: &[(u8, u16)], after: &[(u8, u16)]) -> GroupDelta {
    let mut i = 0;
    let mut j = 0;
    let mut added_all: Vec<(u8, u16)> = Vec::new();
    let mut removed_all: Vec<(u8, u16)> = Vec::new();
    while i < before.len() && j < after.len() {
        let kb = ((before[i].0 as u32) << 16) | before[i].1 as u32;
        let ka = ((after[j].0 as u32) << 16) | after[j].1 as u32;
        if kb == ka {
            i += 1;
            j += 1;
        } else if kb < ka {
            removed_all.push(before[i]);
            i += 1;
        } else {
            added_all.push(after[j]);
            j += 1;
        }
    }
    while i < before.len() {
        removed_all.push(before[i]);
        i += 1;
    }
    while j < after.len() {
        added_all.push(after[j]);
        j += 1;
    }
    let mut out = GroupDelta {
        added: vec![Vec::new(); 8],
        removed: vec![Vec::new(); 8],
        changed_groups: Vec::new(),
    };
    let mut touched = [false; 8];
    for f in added_all {
        out.added[f.0 as usize].push(f);
        touched[f.0 as usize] = true;
    }
    for f in removed_all {
        out.removed[f.0 as usize].push(f);
        touched[f.0 as usize] = true;
    }
    for g in 0..8 {
        if touched[g] {
            out.changed_groups.push(g);
        }
    }
    out
}

pub fn changed_tokens_from_groups(delta: &GroupDelta, token_of_group: &[usize], num_tokens: usize) -> Vec<usize> {
    let mut seen = vec![false; num_tokens];
    for g in &delta.changed_groups {
        let t = token_of_group[*g];
        if t < num_tokens {
            seen[t] = true;
        }
    }
    seen.iter().enumerate().filter_map(|(t, s)| if *s { Some(t) } else { None }).collect()
}

pub fn changed_tokens_by_compare(tok_old: &[f32], tok_new: &[f32], num_tokens: usize, dim: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for t in 0..num_tokens {
        let mut same = true;
        for d in 0..dim {
            if tok_old[t * dim + d] != tok_new[t * dim + d] {
                same = false;
                break;
            }
        }
        if !same {
            out.push(t);
        }
    }
    out
}
