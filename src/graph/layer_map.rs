/// Maps model_type strings to known block patterns with component detail.
///
/// Returns `(block_label, components)` where:
/// - `block_label` is the human-readable block name (e.g. "Transformer Block")
/// - `components` is a list of sub-layer names inside the block.
pub fn block_layout(model_type: Option<&str>) -> (&'static str, Vec<&'static str>) {
    let mt = model_type
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    if mt.contains("bert") {
        (
            "Encoder Block",
            vec![
                "Multi-Head Self-Attention",
                "Add & Norm",
                "Feed-Forward Network",
                "Add & Norm",
            ],
        )
    } else if mt.contains("t5") || mt.contains("bart") || mt.contains("marian") {
        (
            "Enc-Dec Block",
            vec![
                "Self-Attention",
                "LayerNorm",
                "Cross-Attention",
                "LayerNorm",
                "Feed-Forward",
                "LayerNorm",
            ],
        )
    } else if mt.contains("gpt2") || mt.contains("gpt_neo") {
        (
            "Transformer Block",
            vec![
                "LayerNorm",
                "Multi-Head Attention",
                "Residual Add",
                "LayerNorm",
                "MLP (GELU)",
                "Residual Add",
            ],
        )
    } else if mt.contains("mistral") || mt.contains("mixtral") {
        (
            "Transformer Block",
            vec![
                "RMSNorm",
                "Grouped-Query Attention",
                "Residual Add",
                "RMSNorm",
                "SwiGLU MLP",
                "Residual Add",
            ],
        )
    } else if mt.contains("gemma") {
        (
            "Transformer Block",
            vec![
                "RMSNorm",
                "Multi-Query Attention",
                "Residual Add",
                "RMSNorm",
                "GeGLU MLP",
                "Residual Add",
            ],
        )
    } else if mt.contains("phi") {
        (
            "Transformer Block",
            vec![
                "LayerNorm",
                "Parallel Attention + MLP",
                "Residual Add",
            ],
        )
    } else if mt.contains("mamba") || mt.contains("ssm") {
        (
            "SSM Block",
            vec![
                "Norm",
                "Selective State Space",
                "Residual Add",
                "Norm",
                "MLP",
                "Residual Add",
            ],
        )
    } else {
        // Default: LLaMA-style pre-norm transformer
        (
            "Transformer Block",
            vec![
                "RMSNorm",
                "Self-Attention",
                "Residual Add",
                "RMSNorm",
                "Feed-Forward (SwiGLU)",
                "Residual Add",
            ],
        )
    }
}

/// Legacy function kept for backward compatibility but now delegates to `block_layout`.
pub fn block_pattern(model_type: Option<&str>) -> &'static str {
    let (label, _) = block_layout(model_type);
    label
}
