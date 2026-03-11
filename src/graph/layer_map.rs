pub fn block_pattern(model_type: Option<&str>) -> &'static str {
    if let Some(mt) = model_type {
        let lowered = mt.to_ascii_lowercase();
        if lowered.contains("bert") {
            return "Attention -> FFN -> Norm";
        }
    }
    "Attention -> Norm -> FFN"
}
