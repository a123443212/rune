use rune_model::RuneModel;
use rune_spec as spec;

use crate::error::{Result, RuntimeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExecutionTarget {
    Reference,
    Compiled,
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedModelConfig {
    pub architecture_id: String,
    pub game: String,
    pub tokens: usize,
    pub dim: usize,
    pub vocabs: [usize; 9],
    pub is_resnet: bool,
}

pub(crate) fn resolve_model_config(
    model: &RuneModel,
    target: ExecutionTarget,
) -> Result<ResolvedModelConfig> {
    let game = model.header.game.clone();
    let expected_feature = spec::game_feature_version(&game)
        .ok_or_else(|| RuntimeError::GameMismatch(game.clone()))?;
    let accepted = spec::accepted_feature_versions(&game);
    let feature_ok = model.header.feature_version == expected_feature
        || accepted.contains(&model.header.feature_version.as_str());
    if !feature_ok {
        return Err(RuntimeError::FeatureMismatch(
            model.header.feature_version.clone(),
        ));
    }

    let architecture_id = model.header.architecture_id.clone();
    let is_resnet = architecture_id.starts_with("RUNE-RESNET");
    let is_supported = matches!(
        architecture_id.as_str(),
        "RUNE-SFNN" | "RUNE-MLP" | "RUNE-ATTN" | "RUNE-ATTN-GAB" | "RUNE-ATTN-MH4" | "RUNE-REL-02"
    ) || (is_resnet && target == ExecutionTarget::Reference);
    if !is_supported {
        return Err(RuntimeError::UnsupportedArch(architecture_id));
    }
    if target == ExecutionTarget::Compiled && game != spec::GAME_CHESS {
        return Err(RuntimeError::GameMismatch(game));
    }

    let tokens = model.header.tokens;
    let dim = model.header.token_dim;
    if is_resnet {
        if tokens == 0 || tokens > 19 {
            return Err(RuntimeError::Shape("board".to_string()));
        }
        if dim == 0 || dim > 256 {
            return Err(RuntimeError::Shape("channels".to_string()));
        }
    } else if architecture_id == "RUNE-REL-02" {
        if !matches!(tokens, 6 | 8 | 10) {
            return Err(RuntimeError::Shape("tokens".to_string()));
        }
        if !matches!(dim, 24 | 32 | 40) {
            return Err(RuntimeError::Shape("token dim".to_string()));
        }
    } else if tokens != 8 || dim != 32 {
        return Err(RuntimeError::Shape("tokens".to_string()));
    }

    let vocabs = match game.as_str() {
        spec::GAME_SHOGI => crate::shogi::SHOGI_VOCABS,
        spec::GAME_XIANGQI => crate::xiangqi::XIANGQI_VOCABS,
        _ => spec::VOCAB_SIZES,
    };

    Ok(ResolvedModelConfig {
        architecture_id,
        game,
        tokens,
        dim,
        vocabs,
        is_resnet,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rune_model::{ModelHeader, RuneModel};

    use super::{resolve_model_config, ExecutionTarget};
    use crate::error::RuntimeError;

    fn model(game: &str, architecture: &str, feature: &str) -> RuneModel {
        RuneModel {
            header: ModelHeader {
                format: 2,
                architecture_id: architecture.to_string(),
                architecture_version: "0.1.0".to_string(),
                feature_version: feature.to_string(),
                game: game.to_string(),
                tokens: 8,
                token_dim: 32,
                quantization: "fp32".to_string(),
                scales: HashMap::new(),
                tensors: Vec::new(),
                model_hash: String::new(),
                raw: serde_json::Value::Null,
            },
            arrays: HashMap::new(),
            scales_f32: [1.0; 9],
            payload_bytes: Vec::new(),
        }
    }

    #[test]
    fn resolves_game_specific_vocabulary() {
        let model = model("shogi", "RUNE-MLP", "shogi_raw_v01");
        let resolved = resolve_model_config(&model, ExecutionTarget::Reference).unwrap();
        assert_eq!(resolved.game, "shogi");
        assert_eq!(resolved.vocabs, crate::shogi::SHOGI_VOCABS);
    }

    #[test]
    fn rejects_feature_version_mismatch() {
        let model = model("chess", "RUNE-MLP", "wrong-feature-version");
        let result = resolve_model_config(&model, ExecutionTarget::Reference);
        assert!(matches!(result, Err(RuntimeError::FeatureMismatch(_))));
    }

    #[test]
    fn compiled_target_rejects_non_chess_models() {
        let model = model("go", "RUNE-MLP", "go_planes_v01");
        let result = resolve_model_config(&model, ExecutionTarget::Compiled);
        assert!(matches!(result, Err(RuntimeError::GameMismatch(_))));
    }

    #[test]
    fn compiled_target_rejects_spatial_architectures() {
        let model = model("chess", "RUNE-RESNET-01", "grouped_hkav2_fullthreats_v02");
        let result = resolve_model_config(&model, ExecutionTarget::Compiled);
        assert!(matches!(result, Err(RuntimeError::UnsupportedArch(_))));
    }
}
