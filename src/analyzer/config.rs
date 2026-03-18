use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::types::ArchitectureInfo;

pub fn load_config_file(path: &Path) -> Result<Value> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read config file: {}", path.display()))?;
    let value: Value = serde_json::from_str(&raw)
        .with_context(|| format!("invalid config json: {}", path.display()))?;
    Ok(value)
}

pub fn architecture_from_config(config: &Value) -> ArchitectureInfo {
    let scopes = candidate_scopes(config);

    let model_type = get_string_alias(&scopes, &["model_type"]).or_else(|| {
        scopes.iter().find_map(|scope| {
            scope
                .get("architectures")
                .and_then(Value::as_array)
                .and_then(|arr| arr.first())
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
    });

    let hidden_size = get_u64_alias(&scopes, &["hidden_size", "n_embd", "d_model"]);
    let num_layers = get_u64_alias(&scopes, &["num_hidden_layers", "n_layer", "num_layers"]);
    let num_heads = get_u64_alias(&scopes, &["num_attention_heads", "n_head"]);
    let num_key_value_heads = get_u64_alias(
        &scopes,
        &["num_key_value_heads", "num_kv_heads", "n_head_kv"],
    );

    ArchitectureInfo {
        model_type,
        hidden_size,
        num_layers,
        num_heads,
        num_key_value_heads,
        attention_type: None,
    }
}

fn candidate_scopes<'a>(config: &'a Value) -> Vec<&'a Value> {
    let mut scopes = vec![config];
    for key in ["model_config", "config", "model", "transformer_config", "training_config"] {
        if let Some(v) = config.get(key) {
            scopes.push(v);
        }
    }
    scopes
}

fn get_u64_alias(values: &[&Value], keys: &[&str]) -> Option<u64> {
    values
        .iter()
        .find_map(|value| keys.iter().find_map(|k| value.get(*k).and_then(value_to_u64)))
}

fn get_string_alias(values: &[&Value], keys: &[&str]) -> Option<String> {
    values.iter().find_map(|value| {
        keys.iter()
            .find_map(|k| value.get(*k).and_then(Value::as_str).map(ToString::to_string))
    })
}

fn value_to_u64(value: &Value) -> Option<u64> {
    if let Some(v) = value.as_u64() {
        return Some(v);
    }
    if let Some(v) = value.as_i64() {
        return (v >= 0).then_some(v as u64);
    }
    if let Some(v) = value.as_f64() {
        return (v >= 0.0).then_some(v as u64);
    }
    None
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn extracts_aliases_and_model_type() {
        let cfg = json!({
            "architectures": ["LlamaForCausalLM"],
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "n_head_kv": 4
        });

        let arch = architecture_from_config(&cfg);
        assert_eq!(arch.model_type.as_deref(), Some("LlamaForCausalLM"));
        assert_eq!(arch.hidden_size, Some(768));
        assert_eq!(arch.num_layers, Some(12));
        assert_eq!(arch.num_heads, Some(12));
        assert_eq!(arch.num_key_value_heads, Some(4));
    }

    #[test]
    fn extracts_from_nested_model_config() {
        let cfg = json!({
            "schema_version": "1.0",
            "model_config": {
                "model_type": "gqa_transformer",
                "hidden_size": 768,
                "num_hidden_layers": 18,
                "num_attention_heads": 16,
                "num_key_value_heads": 8
            },
            "training_config": {
                "n_embd": 123,
                "n_layer": 456
            }
        });

        let arch = architecture_from_config(&cfg);
        assert_eq!(arch.model_type.as_deref(), Some("gqa_transformer"));
        assert_eq!(arch.hidden_size, Some(768));
        assert_eq!(arch.num_layers, Some(18));
        assert_eq!(arch.num_heads, Some(16));
        assert_eq!(arch.num_key_value_heads, Some(8));
    }
}
