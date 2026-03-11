use crate::types::{CompareReport, ModelReport};

#[derive(Debug, Clone, Copy, Default)]
pub struct RenderOptions {
    pub show_params: bool,
    pub show_graph: bool,
    pub show_attention_breakdown: bool,
}

pub fn render_model(report: &ModelReport, options: &RenderOptions) -> String {
    let mut out = String::new();

    out.push_str("MODEL SUMMARY\n");
    out.push_str("────────────────────────\n\n");
    out.push_str(&format!("Model: {}\n", report.model));
    out.push_str(&format!("Source: {}\n", report.source.location));
    out.push_str(&format!(
        "Total params: {}\n\n",
        human_params(report.params.total_params)
    ));

    out.push_str("Layer distribution\n\n");
    out.push_str(&format_layer_row(
        "FeedForward",
        report.params.categories.feedforward,
        report.params.pct(report.params.categories.feedforward),
    ));
    out.push_str(&format_layer_row(
        "Attention",
        report.params.categories.attention,
        report.params.pct(report.params.categories.attention),
    ));
    out.push_str(&format_layer_row(
        "Embedding",
        report.params.categories.embedding,
        report.params.pct(report.params.categories.embedding),
    ));
    out.push_str(&format_layer_row(
        "Normalization",
        report.params.categories.normalization,
        report.params.pct(report.params.categories.normalization),
    ));
    out.push_str(&format_layer_row(
        "Other",
        report.params.categories.other,
        report.params.pct(report.params.categories.other),
    ));

    out.push_str("\nArchitecture\n\n");
    out.push_str(&format!(
        "Layers: {}\n",
        opt_u64(report.architecture.num_layers)
    ));
    out.push_str(&format!(
        "Hidden size: {}\n",
        opt_u64(report.architecture.hidden_size)
    ));
    out.push_str(&format!(
        "Heads: {}\n",
        opt_u64(report.architecture.num_heads)
    ));
    out.push_str(&format!(
        "KV heads: {}\n",
        opt_u64(
            report
                .architecture
                .num_key_value_heads
                .or(report.attention.kv_heads)
        )
    ));
    out.push_str(&format!(
        "Attention: {}\n",
        report
            .architecture
            .attention_type
            .as_deref()
            .or(report.attention.attention_type.as_deref())
            .unwrap_or("-")
    ));

    if options.show_attention_breakdown {
        out.push_str("\nAttention breakdown\n\n");
        out.push_str(&format!(
            "Q proj params: {}\n",
            human_params(report.attention.q_proj_params)
        ));
        out.push_str(&format!(
            "K proj params: {}\n",
            human_params(report.attention.k_proj_params)
        ));
        out.push_str(&format!(
            "V proj params: {}\n",
            human_params(report.attention.v_proj_params)
        ));
        out.push_str(&format!(
            "O proj params: {}\n",
            human_params(report.attention.o_proj_params)
        ));
    }

    if options.show_graph {
        out.push_str("\nArchitecture graph\n\n");
        out.push_str(report.graph.as_deref().unwrap_or("Graph unavailable"));
        out.push('\n');
    }

    if options.show_params {
        out.push_str("\nTensor stats\n\n");
        out.push_str(&format!("Tensors indexed: {}\n", report.tensor_count));
        out.push_str("Top tensors by parameter count:\n");

        if let Some(tensors) = &report.tensors {
            let mut sorted = tensors.clone();
            sorted.sort_by(|a, b| b.param_count.cmp(&a.param_count));
            for tensor in sorted.iter().take(20) {
                out.push_str(&format!(
                    "{:<60} {:>12}\n",
                    truncate(&tensor.name, 60),
                    human_params(tensor.param_count)
                ));
            }
        }
    }

    if let Some(deep) = &report.deep {
        out.push_str("\nDeep inspection\n\n");
        out.push_str(&format!("{}\n", deep));
    }

    if !report.warnings.is_empty() {
        out.push_str("\nWarnings\n\n");
        for w in &report.warnings {
            out.push_str(&format!("- {w}\n"));
        }
    }

    out
}

pub fn render_compare(report: &CompareReport) -> String {
    let mut out = String::new();
    out.push_str("MODEL COMPARISON\n");
    out.push_str("────────────────────────\n\n");
    out.push_str(&format!("Left:  {}\n", report.left.model));
    out.push_str(&format!("Right: {}\n\n", report.right.model));

    for diff in &report.diffs {
        out.push_str(&format!(
            "{:<14} {} -> {}\n",
            diff.metric, diff.left, diff.right
        ));
    }

    out
}

fn format_layer_row(name: &str, value: u64, pct: f64) -> String {
    format!("{:<14} {:>6.1}% ({})\n", name, pct, human_params(value))
}

fn human_params(value: u64) -> String {
    const K: f64 = 1_000.0;
    const M: f64 = 1_000_000.0;
    const B: f64 = 1_000_000_000.0;
    const T: f64 = 1_000_000_000_000.0;

    let v = value as f64;
    if v >= T {
        format!("{:.2}T", v / T)
    } else if v >= B {
        format!("{:.2}B", v / B)
    } else if v >= M {
        format!("{:.2}M", v / M)
    } else if v >= K {
        format!("{:.2}K", v / K)
    } else {
        value.to_string()
    }
}

fn opt_u64(value: Option<u64>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        return s.to_string();
    }

    let mut out = s.chars().take(max_len - 1).collect::<String>();
    out.push('…');
    out
}
