use crate::types::LayerCategory;

pub fn classify_tensor(name: &str) -> LayerCategory {
    classify_tensor_for_model(name, None)
}

pub fn classify_tensor_for_model(name: &str, model_type: Option<&str>) -> LayerCategory {
    let n = name.to_ascii_lowercase();
    if let Some(category) = classify_with_model_overrides(&n, model_type) {
        return category;
    }
    if is_output_head(&n) {
        return LayerCategory::OutputHead;
    }
    if is_attention(&n) {
        return LayerCategory::Attention;
    }
    if is_feedforward(&n) {
        return LayerCategory::FeedForward;
    }
    if is_embedding(&n) {
        return LayerCategory::Embedding;
    }
    if is_normalization(&n) {
        return LayerCategory::Normalization;
    }
    LayerCategory::Other
}

fn classify_with_model_overrides(name: &str, model_type: Option<&str>) -> Option<LayerCategory> {
    let mt = model_type?.to_ascii_lowercase();

    if mt.contains("qwen") {
        if name.contains("mlp.gate_proj")
            || name.contains("mlp.up_proj")
            || name.contains("mlp.down_proj")
        {
            return Some(LayerCategory::FeedForward);
        }
        if name.contains("self_attn.") || name.contains("attn.") {
            return Some(LayerCategory::Attention);
        }
        if name.contains("model.embed_tokens") || name.contains("transformer.wte") {
            return Some(LayerCategory::Embedding);
        }
    }

    if mt.contains("llama") || mt.contains("mistral") {
        if name.contains("mlp.gate_proj")
            || name.contains("mlp.up_proj")
            || name.contains("mlp.down_proj")
        {
            return Some(LayerCategory::FeedForward);
        }
        if name.contains("self_attn.") {
            return Some(LayerCategory::Attention);
        }
        if name.contains("embed_tokens") {
            return Some(LayerCategory::Embedding);
        }
    }

    if mt.contains("gpt2") || mt.contains("gptj") || mt.contains("gpt_neox") {
        if name.contains(".attn.") {
            return Some(LayerCategory::Attention);
        }
        if name.contains(".mlp.") {
            return Some(LayerCategory::FeedForward);
        }
        if name.contains("wte") || name.contains("wpe") {
            return Some(LayerCategory::Embedding);
        }
    }

    if mt.contains("bert") || mt.contains("roberta") || mt.contains("distilbert") {
        if name.contains("intermediate.dense") || name.contains("output.dense") {
            return Some(LayerCategory::FeedForward);
        }
        if name.contains(".attention.") {
            return Some(LayerCategory::Attention);
        }
        if name.contains("embeddings.") {
            return Some(LayerCategory::Embedding);
        }
    }

    None
}

fn is_output_head(name: &str) -> bool {
    name.contains("lm_head")
        || name.ends_with("output.weight")
        || name.contains("word_embeddings.weight")
}

fn is_attention(name: &str) -> bool {
    name.contains("self_attn")
        || name.contains("attention")
        || name.contains(".attn.")
        || name.contains("q_proj")
        || name.contains("k_proj")
        || name.contains("v_proj")
        || name.contains("o_proj")
        || name.contains("query")
        || name.contains("key")
        || name.contains("value")
        || name.contains("out_proj")
}

fn is_feedforward(name: &str) -> bool {
    name.contains("mlp")
        || name.contains("ffn")
        || name.contains("gate_proj")
        || name.contains("up_proj")
        || name.contains("down_proj")
        || name.contains("wi")
        || name.contains("wo")
        || name.contains("dense_h_to_4h")
        || name.contains("dense_4h_to_h")
        || name.contains("experts")
}

fn is_embedding(name: &str) -> bool {
    name.contains("embed")
        || name.contains("token_emb")
        || name.contains("tok_emb")
        || name.contains("token_embedding")
        || name.contains("input_embedding")
        || name.contains("wte")
        || name.contains("tok_embeddings")
        || name.contains("embed_tokens")
        || name.contains("word_embeddings")
}

fn is_normalization(name: &str) -> bool {
    name.contains("norm")
        || name.contains("ln_")
        || name.contains("layer_norm")
        || name.contains("rms_norm")
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
            classify_tensor("token_emb.weight"),
            LayerCategory::Embedding
        );
        assert_eq!(
            classify_tensor("model.norm.weight"),
            LayerCategory::Normalization
        );
        assert_eq!(classify_tensor("lm_head.weight"), LayerCategory::OutputHead);
        assert_eq!(
            classify_tensor("transformer.wte.weight"),
            LayerCategory::Embedding
        );
        assert_eq!(
            classify_tensor("decoder.layers.0.attn.out_proj.weight"),
            LayerCategory::Attention
        );
    }

    #[test]
    fn model_type_overrides_apply_before_generic() {
        assert_eq!(
            classify_tensor_for_model("transformer.h.0.mlp.c_fc.weight", Some("GPT2LMHeadModel")),
            LayerCategory::FeedForward
        );
        assert_eq!(
            classify_tensor_for_model(
                "bert.encoder.layer.0.attention.self.query.weight",
                Some("BertForMaskedLM")
            ),
            LayerCategory::Attention
        );
    }
}
