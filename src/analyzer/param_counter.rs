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
    let mut total_params_all = 0u64;
    let mut tensors = Vec::with_capacity(raw_tensors.len());

    let mut q_proj_params = 0u64;
    let mut k_proj_params = 0u64;
    let mut v_proj_params = 0u64;
    let mut o_proj_params = 0u64;
    let mut q_proj_shapes = Vec::new();
    let mut k_proj_shapes = Vec::new();
    let mut o_proj_shapes = Vec::new();
    let mut inferred_num_heads_from_q_scale = None;

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
            q_proj_shapes.push(raw.shape.clone());
        }
        if lname.contains("k_proj") || lname.contains("key") {
            k_proj_params = k_proj_params.saturating_add(param_count);
            k_proj_shapes.push(raw.shape.clone());
        }
        if lname.contains("v_proj") || lname.contains("value") {
            v_proj_params = v_proj_params.saturating_add(param_count);
        }
        if lname.contains("o_proj") || lname.contains("out_proj") {
            o_proj_params = o_proj_params.saturating_add(param_count);
            o_proj_shapes.push(raw.shape.clone());
        }
        if lname.contains("q_scale") && !raw.shape.is_empty() && inferred_num_heads_from_q_scale.is_none() {
            inferred_num_heads_from_q_scale = Some(raw.shape[0]);
        }

        total_params_all = total_params_all.saturating_add(param_count);
        if category != LayerCategory::OutputHead {
            total_params = total_params.saturating_add(param_count);
        }

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

    let hidden_size = architecture
        .hidden_size
        .or_else(|| infer_hidden_size_from_projections(&q_proj_shapes, &o_proj_shapes));
    let num_heads = architecture
        .num_heads
        .filter(|v| *v > 0)
        .or(inferred_num_heads_from_q_scale.filter(|v| *v > 0));

    if let Some(num_heads) = num_heads {
        attention.num_heads = Some(num_heads);
    }

    if let (Some(hidden_size), Some(num_heads)) = (hidden_size, num_heads) {
        if num_heads > 0 && hidden_size % num_heads == 0 {
            let head_dim = hidden_size / num_heads;
            attention.head_dim = Some(head_dim);

            let kv_heads = architecture
                .num_key_value_heads
                .filter(|v| *v > 0)
                .or_else(|| infer_kv_heads_from_k_shapes(&k_proj_shapes, head_dim, num_heads));
            attention.kv_heads = kv_heads;

            if let Some(kv) = kv_heads {
                attention.attention_type = Some(
                    if kv == num_heads {
                        "MHA".to_string()
                    } else if kv == 1 {
                        "MQA".to_string()
                    } else if kv < num_heads {
                        "GQA".to_string()
                    } else {
                        "UNKNOWN".to_string()
                    },
                );
            }
        }
    }

    (
        tensors,
        ParamStats {
            total_params,
            total_params_all,
            categories,
        },
        attention,
    )
}

fn infer_hidden_size_from_projections(
    q_proj_shapes: &[Vec<u64>],
    o_proj_shapes: &[Vec<u64>],
) -> Option<u64> {
    for shape in o_proj_shapes {
        if shape.len() >= 2 && shape[0] == shape[1] && shape[0] > 0 {
            return Some(shape[0]);
        }
    }
    for shape in q_proj_shapes {
        if shape.len() >= 2 && shape[1] > 0 {
            return Some(shape[1]);
        }
    }
    None
}

fn infer_kv_heads_from_k_shapes(
    k_proj_shapes: &[Vec<u64>],
    head_dim: u64,
    num_heads: u64,
) -> Option<u64> {
    if head_dim == 0 || num_heads == 0 {
        return None;
    }

    let mut candidates = Vec::new();
    for shape in k_proj_shapes {
        for dim in shape {
            if *dim >= head_dim && *dim % head_dim == 0 {
                let kv = *dim / head_dim;
                if kv > 0 && kv <= num_heads {
                    candidates.push(kv);
                }
            }
        }
    }

    if candidates.is_empty() {
        return None;
    }

    candidates.sort_unstable();
    let mut best = (0u64, 0usize);
    let mut current = candidates[0];
    let mut count = 1usize;
    for kv in candidates.into_iter().skip(1) {
        if kv == current {
            count += 1;
        } else {
            if count > best.1 {
                best = (current, count);
            }
            current = kv;
            count = 1;
        }
    }
    if count > best.1 {
        best = (current, count);
    }
    Some(best.0)
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

    #[test]
    fn infers_kv_heads_from_transposed_k_proj_shape() {
        let tensors = vec![
            tensor("attn.q_proj.weight", &[768, 768]),
            tensor("attn.k_proj.weight", &[768, 384]),
            tensor("attn.o_proj.weight", &[768, 768]),
        ];

        let arch = ArchitectureInfo {
            hidden_size: Some(768),
            num_heads: Some(16),
            ..ArchitectureInfo::default()
        };

        let (_, _, attention) = summarize_tensors(&tensors, &arch);
        assert_eq!(attention.head_dim, Some(48));
        assert_eq!(attention.kv_heads, Some(8));
        assert_eq!(attention.attention_type.as_deref(), Some("GQA"));
    }

    #[test]
    fn infers_num_heads_from_q_scale_when_missing_in_config() {
        let tensors = vec![
            tensor("attn.q_proj.weight", &[768, 768]),
            tensor("attn.k_proj.weight", &[384, 768]),
            tensor("attn.o_proj.weight", &[768, 768]),
            tensor("attn.q_scale", &[16, 1, 1]),
        ];

        let arch = ArchitectureInfo {
            hidden_size: Some(768),
            ..ArchitectureInfo::default()
        };

        let (_, _, attention) = summarize_tensors(&tensors, &arch);
        assert_eq!(attention.num_heads, Some(16));
        assert_eq!(attention.head_dim, Some(48));
        assert_eq!(attention.kv_heads, Some(8));
    }
}
