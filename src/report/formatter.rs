use std::env;

use console::{Term, style};
use serde_json::Value;

use crate::types::{CompareReport, ModelReport, ModelSourceKind};

const DEFAULT_PANEL_WIDTH: usize = 96;
const MIN_PANEL_WIDTH: usize = 40;
const MAX_TOP_TENSORS: usize = 20;

#[derive(Debug, Clone, Copy, Default)]
pub struct RenderOptions {
    pub show_params: bool,
    pub show_graph: bool,
    pub show_attention_breakdown: bool,
}

#[derive(Debug, Clone, Copy)]
struct RenderLayout {
    panel_width: usize,
    inner_width: usize,
    bar_width: usize,
}

impl RenderLayout {
    fn detect() -> Self {
        let panel_width = compute_panel_width(detect_terminal_columns());
        let inner_width = panel_width.saturating_sub(4);
        let bar_width = ((inner_width as f64) * 0.28).round() as usize;
        let bar_width = bar_width.clamp(8, 40);

        Self {
            panel_width,
            inner_width,
            bar_width,
        }
    }
}

pub fn render_model(report: &ModelReport, options: &RenderOptions) -> String {
    let layout = RenderLayout::detect();
    let mut out = String::new();

    // out.push_str(&render_header("DISSECTLM MODEL REPORT", &layout));

    let summary_rows = vec![
        ("Model".to_string(), report.model.clone()),
        ("Source kind".to_string(), source_kind_label(&report.source.kind)),
        ("Source".to_string(), report.source.location.clone()),
        (
            "Total params (excl head)".to_string(),
            human_params(report.params.total_params),
        ),
        (
            "Tensor files found".to_string(),
            report.tensor_files_found.to_string(),
        ),
        (
            "Model size".to_string(),
            report
                .model_size_bytes
                .map(human_bytes)
                .unwrap_or_else(|| "-".to_string()),
        ),
        (
            "Tensor dtypes".to_string(),
            if report.tensor_dtypes.is_empty() {
                "-".to_string()
            } else {
                report.tensor_dtypes.join(", ")
            },
        ),
        (
            "Config keys".to_string(),
            report.config_key_count.to_string(),
        ),
        (
            "Tensors indexed".to_string(),
            report.tensor_count.to_string(),
        ),
    ];
    out.push_str(&kv_panel("Summary", &summary_rows, &layout));

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
        &layout,
    ));

    let mut architecture_rows = vec![
        ("Layers".to_string(), opt_u64(report.architecture.num_layers)),
        (
            "Hidden size".to_string(),
            opt_u64(report.architecture.hidden_size),
        ),
        ("Heads".to_string(), opt_u64(report.architecture.num_heads)),
        (
            "KV heads".to_string(),
            opt_u64(
                report
                    .architecture
                    .num_key_value_heads
                    .or(report.attention.kv_heads),
            ),
        ),
        (
            "Attention type".to_string(),
            report
                .architecture
                .attention_type
                .as_deref()
                .or(report.attention.attention_type.as_deref())
                .unwrap_or("-")
                .to_string(),
        ),
    ];
    if let Some(config) = &report.config {
        architecture_rows.extend(flatten_config_fields("cfg", config));
    }
    out.push_str(&kv_panel("Architecture", &architecture_rows, &layout));

    if options.show_attention_breakdown {
        let attn_rows = vec![
            (
                "Q proj params".to_string(),
                human_params(report.attention.q_proj_params),
            ),
            (
                "K proj params".to_string(),
                human_params(report.attention.k_proj_params),
            ),
            (
                "V proj params".to_string(),
                human_params(report.attention.v_proj_params),
            ),
            (
                "O proj params".to_string(),
                human_params(report.attention.o_proj_params),
            ),
        ];
        out.push_str(&kv_panel("Attention Breakdown", &attn_rows, &layout));
    }

    if options.show_graph {
        let graph = report
            .graph
            .as_deref()
            .unwrap_or("Graph unavailable")
            .lines()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        out.push_str(&text_panel("Architecture Graph", &graph, &layout));
    }

    if options.show_params {
        let mut lines = Vec::new();
        let rank_width = 4;
        let params_width = 14.min(layout.inner_width.saturating_sub(rank_width + 10));
        let params_width = params_width.max(8);
        let tensor_width = layout
            .inner_width
            .saturating_sub(rank_width + params_width + 2)
            .max(8);

        lines.push(format!(
            "{:<rank_width$} {:<tensor_width$} {:>params_width$}",
            "#", "Tensor", "Params"
        ));
        lines.push("─".repeat(layout.inner_width));

        if let Some(tensors) = &report.tensors {
            let mut sorted = tensors.clone();
            sorted.sort_by(|a, b| b.param_count.cmp(&a.param_count));
            for (idx, tensor) in sorted.iter().take(MAX_TOP_TENSORS).enumerate() {
                lines.push(format!(
                    "{:<rank_width$} {:<tensor_width$} {:>params_width$}",
                    idx + 1,
                    truncate(&tensor.name, tensor_width),
                    human_params(tensor.param_count)
                ));
            }
        }

        out.push_str(&text_panel("Top Tensors", &lines, &layout));
    }

    if let Some(deep) = &report.deep {
        let deep_lines = vec![deep.to_string()];
        out.push_str(&text_panel("Deep Inspection", &deep_lines, &layout));
    }

    if !report.warnings.is_empty() {
        let warning_lines = report
            .warnings
            .iter()
            .map(|w| format!("! {w}"))
            .collect::<Vec<_>>();
        out.push_str(&text_panel("Warnings", &warning_lines, &layout));
    }

    out.push_str(&render_footer(&layout));

    out
}

pub fn render_compare(report: &CompareReport) -> String {
    let layout = RenderLayout::detect();
    let mut out = String::new();

    out.push_str(&render_header("DISSECTLM MODEL COMPARISON", &layout));

    let changed_count = report
        .diffs
        .iter()
        .filter(|d| d.left != d.right)
        .count();
    let total_count = report.diffs.len();
    let header_rows = vec![
        ("Left".to_string(), report.left.model.clone()),
        ("Right".to_string(), report.right.model.clone()),
        (
            "Metrics changed".to_string(),
            format!("{changed_count} / {total_count}"),
        ),
    ];
    out.push_str(&kv_panel("Models", &header_rows, &layout));

    let status_width = 3;
    let available = layout.inner_width.saturating_sub(status_width + 3);
    let metric_width = ((available as f64) * 0.34).round() as usize;
    let metric_width = metric_width.clamp(12, 26).min(available.saturating_sub(16));
    let left_width = (available.saturating_sub(metric_width)) / 2;
    let right_width = available.saturating_sub(metric_width + left_width);

    let mut lines = Vec::new();
    lines.push(format!(
        "{:<status_width$} {:<metric_width$} {:<left_width$} {:<right_width$}",
        " ", "Metric", "Left", "Right"
    ));
    lines.push("─".repeat(layout.inner_width));

    for diff in &report.diffs {
        let marker = if diff.left != diff.right {
            "≠"
        } else {
            "="
        };
        lines.push(format!(
            "{:<status_width$} {:<metric_width$} {:<left_width$} {:<right_width$}",
            marker,
            truncate(&diff.metric, metric_width),
            truncate(&diff.left, left_width),
            truncate(&diff.right, right_width)
        ));
    }

    out.push_str(&text_panel("Comparison", &lines, &layout));
    out.push_str(&render_footer(&layout));

    out
}

fn render_header(title: &str, layout: &RenderLayout) -> String {
    let mut out = String::new();
    out.push('\n');
    out.push_str(&format!(
        "  {}  {}\n",
        style("◆").cyan().bold(),
        style(title).bold().cyan()
    ));
    out.push_str(&format!(
        "  {}\n\n",
        style("━".repeat(layout.panel_width.saturating_sub(2)))
            .cyan()
            .dim()
    ));
    out
}

fn render_footer(layout: &RenderLayout) -> String {
    let mut out = String::new();
    let version = env!("CARGO_PKG_VERSION");
    let footer_text = format!("dissectlm v{version}");
    let padding = layout.panel_width.saturating_sub(footer_text.len() + 4);
    out.push_str(&format!(
        "  {}{}\n\n",
        " ".repeat(padding),
        style(footer_text).dim()
    ));
    out
}

fn kv_panel(title: &str, rows: &[(String, String)], layout: &RenderLayout) -> String {
    let key_width = ((layout.inner_width as f64) * 0.28).round() as usize;
    let key_width = key_width.clamp(12, 24);

    let lines = rows
        .iter()
        .map(|(k, v)| format!("{:<key_width$} {}", k, v))
        .collect::<Vec<_>>();
    text_panel(title, &lines, layout)
}

fn flatten_config_fields(prefix: &str, value: &Value) -> Vec<(String, String)> {
    let mut out = Vec::new();
    flatten_config_fields_inner(prefix, value, &mut out);
    out
}

fn flatten_config_fields_inner(prefix: &str, value: &Value, out: &mut Vec<(String, String)>) {
    match value {
        Value::Object(map) => {
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort();
            for key in keys {
                if let Some(v) = map.get(key) {
                    flatten_config_fields_inner(&format!("{prefix}.{key}"), v, out);
                }
            }
        }
        Value::Array(arr) => {
            for (idx, v) in arr.iter().enumerate() {
                flatten_config_fields_inner(&format!("{prefix}[{idx}]"), v, out);
            }
        }
        _ => out.push((prefix.to_string(), format_config_leaf(value))),
    }
}

fn format_config_leaf(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "null".to_string(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| value.to_string()),
    }
}

fn distribution_panel(
    title: &str,
    rows: &[(&str, u64, f64)],
    layout: &RenderLayout,
) -> String {
    let mut lines = Vec::new();
    let category_width = 14.min(layout.inner_width.saturating_sub(28)).max(9);
    let pct_width = 7;
    let params_width = 12;
    let used_without_bar = category_width + pct_width + params_width + 4;
    let available_bar = layout.inner_width.saturating_sub(used_without_bar);
    let bar_width = available_bar.min(layout.bar_width);

    if bar_width >= 6 {
        lines.push(format!(
            "{:<category_width$} {:>pct_width$} {:>params_width$}  {}",
            "Category", "Share", "Params", "Distribution"
        ));
    } else {
        lines.push(format!(
            "{:<category_width$} {:>pct_width$} {:>params_width$}",
            "Category", "Share", "Params"
        ));
    }
    lines.push("─".repeat(layout.inner_width));

    for (name, count, pct) in rows {
        if bar_width >= 6 {
            lines.push(format!(
                "{:<category_width$} {:>6.1}% {:>params_width$}  {}",
                name,
                pct,
                human_params(*count),
                pct_bar(*pct, bar_width)
            ));
        } else {
            lines.push(format!(
                "{:<category_width$} {:>6.1}% {:>params_width$}",
                name,
                pct,
                human_params(*count),
            ));
        }
    }

    text_panel(title, &lines, layout)
}

fn text_panel(title: &str, lines: &[String], layout: &RenderLayout) -> String {
    let mut out = String::new();
    let pw = layout.panel_width;
    let iw = layout.inner_width;

    // Top border
    out.push_str(&format!(
        "  {}{}{}",
        style("╭").dim(),
        style("─".repeat(pw - 2)).dim(),
        style("╮").dim()
    ));
    out.push('\n');

    // Title row
    let title_display = format!(" {} ", title);
    let title_pad = iw.saturating_sub(title_display.chars().count());
    out.push_str(&format!(
        "  {} {}{} {}",
        style("│").dim(),
        style(&title_display).bold().cyan(),
        " ".repeat(title_pad),
        style("│").dim()
    ));
    out.push('\n');

    // Title separator
    out.push_str(&format!(
        "  {}{}{}",
        style("├").dim(),
        style("─".repeat(pw - 2)).dim(),
        style("┤").dim()
    ));
    out.push('\n');

    // Content lines — plain text, proper char-count padding
    for line in lines {
        for wrapped in wrap_line(line, iw) {
            let char_count = wrapped.chars().count();
            let pad = iw.saturating_sub(char_count);
            out.push_str(&format!(
                "  {} {}{} {}",
                style("│").dim(),
                wrapped,
                " ".repeat(pad),
                style("│").dim()
            ));
            out.push('\n');
        }
    }

    // Bottom border
    out.push_str(&format!(
        "  {}{}{}",
        style("╰").dim(),
        style("─".repeat(pw - 2)).dim(),
        style("╯").dim()
    ));
    out.push_str("\n\n");
    out
}

fn pct_bar(pct: f64, width: usize) -> String {
    let clamped = pct.clamp(0.0, 100.0);
    let filled_exact = (clamped / 100.0) * width as f64;
    let filled_full = filled_exact.floor() as usize;
    let remainder = filled_exact - filled_full as f64;

    let filled_full = filled_full.min(width);

    let partial_char = if remainder >= 0.75 {
        "▓"
    } else if remainder >= 0.5 {
        "▒"
    } else if remainder >= 0.25 {
        "░"
    } else {
        ""
    };

    let partial_count = if !partial_char.is_empty() && filled_full < width {
        1
    } else {
        0
    };

    let empty = width.saturating_sub(filled_full + partial_count);

    format!(
        "{}{}{}",
        "█".repeat(filled_full),
        partial_char,
        "·".repeat(empty)
    )
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

fn human_bytes(value: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const TB: f64 = 1024.0 * 1024.0 * 1024.0 * 1024.0;

    let v = value as f64;
    if v >= TB {
        format!("{:.2} TB", v / TB)
    } else if v >= GB {
        format!("{:.2} GB", v / GB)
    } else if v >= MB {
        format!("{:.2} MB", v / MB)
    } else if v >= KB {
        format!("{:.2} KB", v / KB)
    } else {
        format!("{value} B")
    }
}

fn source_kind_label(kind: &ModelSourceKind) -> String {
    match kind {
        ModelSourceKind::LocalPath => "local_path".to_string(),
        ModelSourceKind::HuggingFace => "hugging_face".to_string(),
    }
}

fn opt_u64(value: Option<u64>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn truncate(s: &str, max_len: usize) -> String {
    if max_len == 0 {
        return String::new();
    }

    if s.chars().count() <= max_len {
        return s.to_string();
    }

    let mut out = s
        .chars()
        .take(max_len.saturating_sub(1))
        .collect::<String>();
    out.push('…');
    out
}

fn wrap_line(s: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }

    if s.chars().count() <= width {
        return vec![s.to_string()];
    }

    let mut out = Vec::new();
    let mut current = String::new();

    for word in s.split_whitespace() {
        let word_len = word.chars().count();
        if word_len > width {
            if !current.is_empty() {
                out.push(current);
                current = String::new();
            }
            out.extend(split_word(word, width));
            continue;
        }

        let current_len = current.chars().count();
        let candidate_len = if current.is_empty() {
            word_len
        } else {
            current_len + 1 + word_len
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
        split_word(s, width)
    } else {
        out
    }
}

fn split_word(word: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for ch in word.chars() {
        if current.chars().count() == width {
            chunks.push(current);
            current = String::new();
        }
        current.push(ch);
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

fn detect_terminal_columns() -> Option<usize> {
    if let Some(cols) = env::var("COLUMNS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|v| *v > 0)
    {
        return Some(cols);
    }

    let (_, cols) = Term::stdout().size();
    if cols > 0 { Some(cols as usize) } else { None }
}

fn compute_panel_width(columns: Option<usize>) -> usize {
    columns
        .map(|cols| cols.saturating_sub(4).max(MIN_PANEL_WIDTH))
        .unwrap_or(DEFAULT_PANEL_WIDTH)
}

#[cfg(test)]
mod tests {
    use super::{compute_panel_width, human_params, split_word, truncate, wrap_line};

    #[test]
    fn panel_width_uses_full_terminal() {
        assert_eq!(compute_panel_width(Some(200)), 196);
        assert_eq!(compute_panel_width(Some(80)), 76);
        assert_eq!(compute_panel_width(Some(20)), 40);
        assert_eq!(compute_panel_width(None), 96);
    }

    #[test]
    fn wrap_line_splits_long_words() {
        let wrapped = wrap_line("aaaaaaaaaaaa", 5);
        assert_eq!(wrapped, vec!["aaaaa", "aaaaa", "aa"]);
    }

    #[test]
    fn split_word_handles_exact_chunks() {
        let chunks = split_word("abcdef", 3);
        assert_eq!(chunks, vec!["abc", "def"]);
    }

    #[test]
    fn human_params_formats_correctly() {
        assert_eq!(human_params(0), "0");
        assert_eq!(human_params(500), "500");
        assert_eq!(human_params(1_500), "1.50K");
        assert_eq!(human_params(1_500_000), "1.50M");
        assert_eq!(human_params(7_000_000_000), "7.00B");
    }

    #[test]
    fn truncate_uses_ellipsis() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 6), "hello…");
    }
}
