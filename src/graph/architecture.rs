use crate::graph::layer_map::block_pattern;
use crate::types::ArchitectureInfo;

pub fn build_architecture_graph(architecture: &ArchitectureInfo) -> String {
    let block = block_pattern(architecture.model_type.as_deref());
    let repeat = architecture
        .num_layers
        .map(|n| n.to_string())
        .unwrap_or_else(|| "N".to_string());

    format!("Embedding\n↓\n[{block}] x {repeat}\n↓\nLM Head")
}
