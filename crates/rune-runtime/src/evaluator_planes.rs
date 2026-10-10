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

use crate::evaluator::Evaluator;

impl Evaluator {
    pub fn is_resnet(&self) -> bool {
        self.arch_id().starts_with("RUNE-RESNET")
    }

    pub fn resnet_board(&self) -> usize {
        self.tokens
    }

    pub fn resnet_channels(&self) -> usize {
        self.dim
    }

    pub fn set_resnet_for_bench(&mut self, rw: crate::resnet::ResnetWeights) {
        match &mut self.spatial {
            Some(spatial) => spatial.replace_weights(rw),
            None => self.spatial = Some(crate::evaluator_spatial::SpatialExecutor::new(rw)),
        }
    }

    pub fn evaluate_planes(&self, planes: &[f32]) -> crate::evaluator::EvalResult {
        match &self.spatial {
            Some(spatial) => spatial.evaluate_planes(planes),
            None => crate::evaluator::EvalResult { value: 0.0, wdl: [0.0, 1.0, 0.0], refine: false, difficulty: 1.0, policy: Vec::new(), score_mean: 0.0 },
        }
    }
}
