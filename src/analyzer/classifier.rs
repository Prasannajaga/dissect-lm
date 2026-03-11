use crate::types::LayerCategory;

pub fn classify_tensor(name: &str) -> LayerCategory {
    let n = name.to_ascii_lowercase();
    if n.contains("self_attn") || n.contains("attention") {
        return LayerCategory::Attention;
    }
    if n.contains("mlp") || n.contains("ffn") {
        return LayerCategory::FeedForward;
    }
    if n.contains("embed") {
        return LayerCategory::Embedding;
    }
    if n.contains("norm") {
        return LayerCategory::Normalization;
    }
    LayerCategory::Other
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_maps_expected_patterns() {
        assert_eq!(
            classify_tensor("model.layers.0.self_attn.q_proj.weight"),
            LayerCategory::Attention
        );
        assert_eq!(
            classify_tensor("decoder.mlp.up_proj.weight"),
            LayerCategory::FeedForward
        );
        assert_eq!(
            classify_tensor("model.embed_tokens.weight"),
            LayerCategory::Embedding
        );
        assert_eq!(
            classify_tensor("model.norm.weight"),
            LayerCategory::Normalization
        );
        assert_eq!(classify_tensor("lm_head.weight"), LayerCategory::Other);
    }
}
