use crate::analyzer::classifier::classify_tensor_for_model;
use crate::analyzer::safetensor::TensorMetaRaw;
use crate::types::{
    ArchitectureInfo, AttentionInfo, CategoryTotals, LayerCategory, ParamStats, TensorMeta,
};

pub fn summarize_tensors(
    raw_tensors: &[TensorMetaRaw],
    architecture: &ArchitectureInfo,
) -> (Vec<TensorMeta>, ParamStats, AttentionInfo) {
    let mut categories = CategoryTotals::default();
    let mut total_params = 0u64;
    let mut tensors = Vec::with_capacity(raw_tensors.len());

    let mut q_proj_params = 0u64;
    let mut k_proj_params = 0u64;
    let mut v_proj_params = 0u64;
    let mut o_proj_params = 0u64;
    let mut inferred_k_proj_dim = None;

    for raw in raw_tensors {
        let param_count = param_count_from_shape(&raw.shape);
        let category = classify_tensor_for_model(&raw.name, architecture.model_type.as_deref());

        match category {
            LayerCategory::Attention => {
                categories.attention = categories.attention.saturating_add(param_count)
            }
            LayerCategory::FeedForward => {
                categories.feedforward = categories.feedforward.saturating_add(param_count)
            }
            LayerCategory::Embedding => {
                categories.embedding = categories.embedding.saturating_add(param_count)
            }
            LayerCategory::Normalization => {
                categories.normalization = categories.normalization.saturating_add(param_count)
            }
            LayerCategory::OutputHead => {
                categories.output_head = categories.output_head.saturating_add(param_count)
            }
            LayerCategory::Other => categories.other = categories.other.saturating_add(param_count),
        }

        let lname = raw.name.to_ascii_lowercase();
        if lname.contains("q_proj") || lname.contains("query") {
            q_proj_params = q_proj_params.saturating_add(param_count);
        }
        if lname.contains("k_proj") || lname.contains("key") {
            k_proj_params = k_proj_params.saturating_add(param_count);
            if inferred_k_proj_dim.is_none() && !raw.shape.is_empty() {
                inferred_k_proj_dim = Some(raw.shape[0]);
            }
        }
        if lname.contains("v_proj") || lname.contains("value") {
            v_proj_params = v_proj_params.saturating_add(param_count);
        }
        if lname.contains("o_proj") || lname.contains("out_proj") {
            o_proj_params = o_proj_params.saturating_add(param_count);
        }

        total_params = total_params.saturating_add(param_count);

        tensors.push(TensorMeta {
            name: raw.name.clone(),
            shape: raw.shape.clone(),
            dtype: raw.dtype.clone(),
            data_offsets: raw.data_offsets,
            param_count,
            category,
        });
    }

    let mut attention = AttentionInfo {
        q_proj_params,
        k_proj_params,
        v_proj_params,
        o_proj_params,
        ..AttentionInfo::default()
    };

    if let (Some(hidden_size), Some(num_heads)) = (architecture.hidden_size, architecture.num_heads)
    {
        if num_heads != 0 {
            let head_dim = hidden_size / num_heads;
            attention.head_dim = Some(head_dim);
            attention.num_heads = Some(num_heads);

            let kv_heads = architecture.num_key_value_heads.or_else(|| {
                if head_dim > 0 {
                    inferred_k_proj_dim.map(|kdim| kdim / head_dim)
                } else {
                    None
                }
            });
            attention.kv_heads = kv_heads;

            attention.attention_type = kv_heads.map(|kv| {
                if kv == num_heads {
                    "MHA".to_string()
                } else if kv == 1 {
                    "MQA".to_string()
                } else {
                    "GQA".to_string()
                }
            });
        }
    }

    (
        tensors,
        ParamStats {
            total_params,
            categories,
        },
        attention,
    )
}

fn param_count_from_shape(shape: &[u64]) -> u64 {
    if shape.is_empty() {
        return 1;
    }

    let product = shape
        .iter()
        .fold(1u128, |acc, d| acc.saturating_mul(*d as u128));
    if product > u64::MAX as u128 {
        u64::MAX
    } else {
        product as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(name: &str, shape: &[u64]) -> TensorMetaRaw {
        TensorMetaRaw {
            name: name.to_string(),
            shape: shape.to_vec(),
            dtype: "F32".to_string(),
            data_offsets: (0, 0),
        }
    }

    #[test]
    fn counts_parameters_and_detects_attention_type() {
        let tensors = vec![
            tensor("model.layers.0.self_attn.q_proj.weight", &[8, 8]),
            tensor("model.layers.0.self_attn.k_proj.weight", &[4, 8]),
            tensor("model.layers.0.self_attn.v_proj.weight", &[4, 8]),
            tensor("model.layers.0.self_attn.o_proj.weight", &[8, 8]),
            tensor("model.layers.0.mlp.up_proj.weight", &[8, 16]),
            tensor("model.embed_tokens.weight", &[128, 8]),
            tensor("model.norm.weight", &[8]),
            tensor("lm_head.weight", &[128, 8]),
        ];

        let arch = ArchitectureInfo {
            hidden_size: Some(8),
            num_heads: Some(4),
            ..ArchitectureInfo::default()
        };

        let (_, stats, attention) = summarize_tensors(&tensors, &arch);
        assert_eq!(stats.categories.attention, 64 + 32 + 32 + 64);
        assert_eq!(stats.categories.feedforward, 128);
        assert_eq!(stats.categories.embedding, 1024);
        assert_eq!(stats.categories.normalization, 8);
        assert_eq!(stats.categories.output_head, 1024);
        assert_eq!(attention.kv_heads, Some(2));
        assert_eq!(attention.attention_type.as_deref(), Some("GQA"));
    }
}
