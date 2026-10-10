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

use std::cell::RefCell;

use crate::error::Result;
use crate::evaluator::EvalResult;
use crate::resnet::ResnetWeights;
use crate::resnet_scratch::{self, Scratch};
use rune_model::RuneModel;

pub(crate) struct SpatialExecutor {
    weights: ResnetWeights,
    scratch: RefCell<Scratch>,
}

impl SpatialExecutor {
    pub(crate) fn from_model(model: &RuneModel) -> Result<Self> {
        let weights = ResnetWeights::from_arrays(&model.arrays, &model.header.raw)?;
        Ok(Self::new(weights))
    }

    pub(crate) fn new(weights: ResnetWeights) -> Self {
        Self {
            weights,
            scratch: RefCell::new(Scratch::default()),
        }
    }

    pub(crate) fn replace_weights(&mut self, weights: ResnetWeights) {
        self.weights = weights;
    }

    pub(crate) fn evaluate_planes(&self, planes: &[f32]) -> EvalResult {
        let (value, wdl, policy) = if is_fast(&self.weights) {
            resnet_scratch::forward_fast(&self.weights, &mut self.scratch.borrow_mut(), planes)
        } else {
            self.weights.forward(planes)
        };
        EvalResult {
            value,
            wdl,
            refine: false,
            difficulty: 0.0,
            policy,
            score_mean: 0.0,
        }
    }
}

fn is_fast(weights: &ResnetWeights) -> bool {
    let channels = weights.cfg.channels;
    let in_planes = weights.cfg.in_planes;
    if weights.stem_w.len() != channels * in_planes * 9 || weights.stem_b.len() != channels {
        return false;
    }
    for block in 0..weights.cfg.blocks {
        if weights.block_w1[block].len() != channels * channels * 9
            || weights.block_w2[block].len() != channels * channels * 9
            || weights.block_b1[block].len() != channels
            || weights.block_b2[block].len() != channels
        {
            return false;
        }
    }
    true
}
