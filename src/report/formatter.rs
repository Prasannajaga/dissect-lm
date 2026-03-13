use console::style;

use crate::types::{CompareReport, ModelReport};

const PANEL_WIDTH: usize = 96;
const BAR_WIDTH: usize = 24;

#[derive(Debug, Clone, Copy, Default)]
pub struct RenderOptions {
    pub show_params: bool,
    pub show_graph: bool,
    pub show_attention_breakdown: bool,
}

pub fn render_model(report: &ModelReport, options: &RenderOptions) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "{}\n{}\n\n",
        style("DISSECTLM MODEL REPORT").bold().cyan(),
        style("=".repeat(PANEL_WIDTH)).dim()
    ));

    let summary_rows = vec![
        ("Model", report.model.clone()),
        ("Source", report.source.location.clone()),
        ("Total params", human_params(report.params.total_params)),
        ("Tensors indexed", report.tensor_count.to_string()),
    ];
    out.push_str(&kv_panel("Summary", &summary_rows));

    let distribution_rows = vec![
        (
            "FeedForward",
            report.params.categories.feedforward,
            report.params.pct(report.params.categories.feedforward),
        ),
        (
            "Attention",
            report.params.categories.attention,
            report.params.pct(report.params.categories.attention),
        ),
        (
            "Embedding",
            report.params.categories.embedding,
            report.params.pct(report.params.categories.embedding),
        ),
        (
            "Normalization",
            report.params.categories.normalization,
            report.params.pct(report.params.categories.normalization),
        ),
        (
            "OutputHead",
            report.params.categories.output_head,
            report.params.pct(report.params.categories.output_head),
        ),
        (
            "Other",
            report.params.categories.other,
            report.params.pct(report.params.categories.other),
        ),
    ];
    out.push_str(&distribution_panel(
        "Layer Distribution",
        &distribution_rows,
    ));

    let architecture_rows = vec![
        ("Layers", opt_u64(report.architecture.num_layers)),
        ("Hidden size", opt_u64(report.architecture.hidden_size)),
        ("Heads", opt_u64(report.architecture.num_heads)),
        (
            "KV heads",
            opt_u64(
                report
                    .architecture
                    .num_key_value_heads
                    .or(report.attention.kv_heads),
            ),
        ),
        (
            "Attention type",
            report
                .architecture
                .attention_type
                .as_deref()
                .or(report.attention.attention_type.as_deref())
                .unwrap_or("-")
                .to_string(),
        ),
    ];
    out.push_str(&kv_panel("Architecture", &architecture_rows));

    if options.show_attention_breakdown {
        let attn_rows = vec![
            (
                "Q proj params",
                human_params(report.attention.q_proj_params),
            ),
            (
                "K proj params",
                human_params(report.attention.k_proj_params),
            ),
            (
                "V proj params",
                human_params(report.attention.v_proj_params),
            ),
            (
                "O proj params",
                human_params(report.attention.o_proj_params),
            ),
        ];
        out.push_str(&kv_panel("Attention Breakdown", &attn_rows));
    }

    if options.show_graph {
        let graph = report
            .graph
            .as_deref()
            .unwrap_or("Graph unavailable")
            .lines()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        out.push_str(&text_panel("Architecture Graph", &graph));
    }

    if options.show_params {
        let mut lines = Vec::new();
        lines.push(format!("{:<4} {:<62} {:>14}", "#", "Tensor", "Params"));
        lines.push("-".repeat(PANEL_WIDTH - 6));

        if let Some(tensors) = &report.tensors {
            let mut sorted = tensors.clone();
            sorted.sort_by(|a, b| b.param_count.cmp(&a.param_count));
            for (idx, tensor) in sorted.iter().take(20).enumerate() {
                lines.push(format!(
                    "{:<4} {:<62} {:>14}",
                    idx + 1,
                    truncate(&tensor.name, 62),
                    human_params(tensor.param_count)
                ));
            }
        }

        out.push_str(&text_panel("Top Tensors", &lines));
    }

    if let Some(deep) = &report.deep {
        let deep_lines = vec![deep.to_string()];
        out.push_str(&text_panel("Deep Inspection", &deep_lines));
    }

    if !report.warnings.is_empty() {
        let warning_lines = report
            .warnings
            .iter()
            .map(|w| format!("! {w}"))
            .collect::<Vec<_>>();
        out.push_str(&text_panel(
            &format!("{}", style("Warnings").yellow().bold()),
            &warning_lines,
        ));
    }

    out
}

pub fn render_compare(report: &CompareReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{}\n{}\n\n",
        style("DISSECTLM MODEL COMPARISON").bold().magenta(),
        style("=".repeat(PANEL_WIDTH)).dim()
    ));

    let header_rows = vec![
        ("Left", report.left.model.clone()),
        ("Right", report.right.model.clone()),
    ];
    out.push_str(&kv_panel("Models", &header_rows));

    let mut lines = Vec::new();
    lines.push(format!("{:<22} {:<26} {:<26}", "Metric", "Left", "Right"));
    lines.push("-".repeat(PANEL_WIDTH - 6));
    for diff in &report.diffs {
        lines.push(format!(
            "{:<22} {:<26} {:<26}",
            truncate(&diff.metric, 22),
            truncate(&diff.left, 26),
            truncate(&diff.right, 26)
        ));
    }
    out.push_str(&text_panel("Comparison", &lines));

    out
}

fn kv_panel(title: &str, rows: &[(&str, String)]) -> String {
    let lines = rows
        .iter()
        .map(|(k, v)| format!("{:<20} {}", k, v))
        .collect::<Vec<_>>();
    text_panel(title, &lines)
}

fn distribution_panel(title: &str, rows: &[(&str, u64, f64)]) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "{:<14} {:>8} {:>14}  {:<24}",
        "Category", "Share", "Params", "Distribution"
    ));
    lines.push("-".repeat(PANEL_WIDTH - 6));

    for (name, count, pct) in rows {
        lines.push(format!(
            "{:<14} {:>7.1}% {:>14}  {}",
            name,
            pct,
            human_params(*count),
            pct_bar(*pct)
        ));
    }

    text_panel(title, &lines)
}

fn text_panel(title: &str, lines: &[String]) -> String {
    let mut out = String::new();
    out.push_str(&format!("┌{}┐\n", "─".repeat(PANEL_WIDTH - 2)));
    out.push_str(&format!(
        "│ {:<width$} │\n",
        format!("[ {} ]", title),
        width = PANEL_WIDTH - 4
    ));
    out.push_str(&format!("├{}┤\n", "─".repeat(PANEL_WIDTH - 2)));

    for line in lines {
        for wrapped in wrap_line(line, PANEL_WIDTH - 4) {
            out.push_str(&format!(
                "│ {:<width$} │\n",
                wrapped,
                width = PANEL_WIDTH - 4
            ));
        }
    }

    out.push_str(&format!("└{}┘\n\n", "─".repeat(PANEL_WIDTH - 2)));
    out
}

fn pct_bar(pct: f64) -> String {
    let clamped = pct.clamp(0.0, 100.0);
    let filled = ((clamped / 100.0) * BAR_WIDTH as f64).round() as usize;
    let filled = filled.min(BAR_WIDTH);
    let empty = BAR_WIDTH.saturating_sub(filled);
    format!("{}{}", "#".repeat(filled), ".".repeat(empty))
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

    let mut out = s
        .chars()
        .take(max_len.saturating_sub(1))
        .collect::<String>();
    out.push('~');
    out
}

fn wrap_line(s: &str, width: usize) -> Vec<String> {
    if s.chars().count() <= width {
        return vec![s.to_string()];
    }

    let mut out = Vec::new();
    let mut current = String::new();

    for word in s.split_whitespace() {
        let candidate_len = if current.is_empty() {
            word.chars().count()
        } else {
            current.chars().count() + 1 + word.chars().count()
        };

        if candidate_len > width && !current.is_empty() {
            out.push(current);
            current = word.to_string();
        } else if current.is_empty() {
            current.push_str(word);
        } else {
            current.push(' ');
            current.push_str(word);
        }
    }

    if !current.is_empty() {
        out.push(current);
    }

    if out.is_empty() {
        vec![truncate(s, width)]
    } else {
        out
    }
}
