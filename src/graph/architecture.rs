use crate::graph::layer_map::block_layout;
use crate::types::ArchitectureInfo;

/// Maximum inner width for the graph box (excluding the border characters).
const GRAPH_INNER_WIDTH: usize = 52;

/// Characters used for the box-drawing.
const TOP_LEFT: &str = "╭";
const TOP_RIGHT: &str = "╮";
const BOTTOM_LEFT: &str = "╰";
const BOTTOM_RIGHT: &str = "╯";
const HORIZONTAL: &str = "─";
const VERTICAL: &str = "│";
const T_LEFT: &str = "├";
const T_RIGHT: &str = "┤";
const ARROW_DOWN: &str = "▼"; 
const REPEAT_LEFT: &str = "┊";
const REPEAT_RIGHT: &str = "┊";

/// Build a visually rich ASCII architecture graph from the model metadata.
///
/// The graph is structured as:
/// ```text
/// ╭────────────────────────────────────────────────────╮
/// │             Token Embedding Layer                  │
/// │           (hidden_size = 4096)                     │
/// ╰────────────────────────────────────────────────────╯
///                        ▼
/// ┊ ╭──────────────────────────────────────────────╮ ┊
/// ┊ │          Transformer Block  x 32             │ ┊
/// ┊ ├──────────────────────────────────────────────┤ ┊
/// ┊ │   ○ RMSNorm                                  │ ┊
/// ┊ │   ○ Self-Attention (GQA, 32 heads, 8 kv)     │ ┊
/// ┊ │   ○ Residual Add                             │ ┊
/// ┊ │   ○ RMSNorm                                  │ ┊
/// ┊ │   ○ Feed-Forward (SwiGLU)                    │ ┊
/// ┊ │   ○ Residual Add                             │ ┊
/// ┊ ╰──────────────────────────────────────────────╯ ┊
///                        ▼
/// ╭────────────────────────────────────────────────────╮
/// │               Final RMSNorm                       │
/// ╰────────────────────────────────────────────────────╯
///                        ▼
/// ╭────────────────────────────────────────────────────╮
/// │              LM Head (Output)                     │
/// ╰────────────────────────────────────────────────────╯
/// ```
pub fn build_architecture_graph(architecture: &ArchitectureInfo) -> String {
    let iw = GRAPH_INNER_WIDTH;
    let (block_label, components) = block_layout(architecture.model_type.as_deref());

    let repeat = architecture
        .num_layers
        .map(|n| n.to_string())
        .unwrap_or_else(|| "N".to_string());

    let mut lines: Vec<String> = Vec::new();

    // ── Embedding layer ──────────────────────────────────
    let emb_title = "Token Embedding Layer";
    let emb_sub = architecture
        .hidden_size
        .map(|h| format!("(hidden_size = {})", h))
        .unwrap_or_default();

    lines.extend(single_box(emb_title, if emb_sub.is_empty() { None } else { Some(&emb_sub) }, iw));

    // ── Arrow ──────────────────────────────────────
    lines.push(center_text(ARROW_DOWN, iw + 2));

    // ── Repeated block ──────────────────────────────────
    let block_title = format!("{}  x {}", block_label, repeat);

    // Build attention detail string
    let attn_detail = build_attention_detail(architecture);

    lines.extend(repeated_block(&block_title, &components, &attn_detail, iw));

    // ── Arrow ──────────────────────────────────────
    lines.push(center_text(ARROW_DOWN, iw + 2));

    // ── Final norm ──────────────────────────────────────
    lines.extend(single_box("Final RMSNorm", None, iw));

    // ── Arrow ──────────────────────────────────────
    lines.push(center_text(ARROW_DOWN, iw + 2));

    // ── LM Head ──────────────────────────────────────
    lines.extend(single_box("LM Head (Output)", None, iw));

    lines.join("\n")
}

/// Build a short annotation string for the attention component.
fn build_attention_detail(arch: &ArchitectureInfo) -> String {
    let mut parts = Vec::new();

    if let Some(ref at) = arch.attention_type {
        parts.push(at.clone());
    }

    if let Some(h) = arch.num_heads {
        parts.push(format!("{} heads", h));
    }

    if let Some(kv) = arch.num_key_value_heads {
        parts.push(format!("{} kv", kv));
    }

    if parts.is_empty() {
        String::new()
    } else {
        format!("({})", parts.join(", "))
    }
}

/// Render a simple single-line box, optionally with a subtitle.
fn single_box(title: &str, subtitle: Option<&str>, inner_width: usize) -> Vec<String> {
    let mut out = Vec::new();

    // top border
    out.push(format!(
        "{}{}{}",
        TOP_LEFT,
        HORIZONTAL.repeat(inner_width),
        TOP_RIGHT
    ));

    // title
    out.push(format!(
        "{}{}{}",
        VERTICAL,
        center_pad(title, inner_width),
        VERTICAL
    ));

    // subtitle
    if let Some(sub) = subtitle {
        out.push(format!(
            "{}{}{}",
            VERTICAL,
            center_pad(sub, inner_width),
            VERTICAL
        ));
    }

    // bottom border
    out.push(format!(
        "{}{}{}",
        BOTTOM_LEFT,
        HORIZONTAL.repeat(inner_width),
        BOTTOM_RIGHT
    ));

    out
}

/// Render the repeated block with dotted repeat markers on the sides.
fn repeated_block(
    title: &str,
    components: &[&str],
    attn_detail: &str,
    outer_inner: usize,
) -> Vec<String> {
    let mut out = Vec::new();

    // The inner box is indented by the repeat markers: "┊ " on each side (2 chars each)
    let block_inner = outer_inner.saturating_sub(4);

    // Top repeat + box border
    out.push(format!(
        "{} {}{}{}  {}",
        REPEAT_LEFT,
        TOP_LEFT,
        HORIZONTAL.repeat(block_inner),
        TOP_RIGHT,
        REPEAT_RIGHT
    ));

    // Title row
    out.push(format!(
        "{} {}{}{}  {}",
        REPEAT_LEFT,
        VERTICAL,
        center_pad(title, block_inner),
        VERTICAL,
        REPEAT_RIGHT
    ));

    // Separator
    out.push(format!(
        "{} {}{}{}  {}",
        REPEAT_LEFT,
        T_LEFT,
        HORIZONTAL.repeat(block_inner),
        T_RIGHT,
        REPEAT_RIGHT
    ));

    // Components
    for comp in components {
        let label = if is_attention_component(comp) && !attn_detail.is_empty() {
            format!("○ {} {}", comp, attn_detail)
        } else {
            format!("○ {}", comp)
        };

        let padded = left_pad(&label, block_inner, 3);
        out.push(format!(
            "{} {}{}{}  {}",
            REPEAT_LEFT, VERTICAL, padded, VERTICAL, REPEAT_RIGHT
        ));
    }

    // Bottom repeat + box border
    out.push(format!(
        "{} {}{}{}  {}",
        REPEAT_LEFT,
        BOTTOM_LEFT,
        HORIZONTAL.repeat(block_inner),
        BOTTOM_RIGHT,
        REPEAT_RIGHT
    ));

    out
}

/// Returns true if this component name looks like an attention layer.
fn is_attention_component(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("attention") && !lower.contains("cross")
}

/// Center a string within a given width.
fn center_pad(text: &str, width: usize) -> String {
    let text_len = text.chars().count();
    if text_len >= width {
        return text.chars().take(width).collect();
    }
    let left = (width - text_len) / 2;
    let right = width - text_len - left;
    format!("{}{}{}", " ".repeat(left), text, " ".repeat(right))
}

/// Left-align a string with `indent` spaces, then pad to `width`.
fn left_pad(text: &str, width: usize, indent: usize) -> String {
    let prefix = " ".repeat(indent);
    let content = format!("{}{}", prefix, text);
    let content_len = content.chars().count();
    if content_len >= width {
        return content.chars().take(width).collect();
    }
    let pad = width - content_len;
    format!("{}{}", content, " ".repeat(pad))
}

/// Center a single-char or short string within `width` total characters.
fn center_text(text: &str, width: usize) -> String {
    let text_len = text.chars().count();
    if text_len >= width {
        return text.to_string();
    }
    let left = (width - text_len) / 2;
    format!("{}{}", " ".repeat(left), text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_graph_has_all_sections() {
        let arch = ArchitectureInfo {
            model_type: Some("llama".to_string()),
            hidden_size: Some(4096),
            num_layers: Some(32),
            num_heads: Some(32),
            num_key_value_heads: Some(8),
            attention_type: Some("GQA".to_string()),
        };
        let graph = build_architecture_graph(&arch);

        assert!(graph.contains("Token Embedding Layer"));
        assert!(graph.contains("hidden_size = 4096"));
        assert!(graph.contains("x 32"));
        assert!(graph.contains("Self-Attention"));
        assert!(graph.contains("GQA"));
        assert!(graph.contains("32 heads"));
        assert!(graph.contains("8 kv"));
        assert!(graph.contains("Feed-Forward (SwiGLU)"));
        assert!(graph.contains("LM Head (Output)"));
        assert!(graph.contains("Final RMSNorm"));
    }

    #[test]
    fn bert_graph_shows_encoder_block() {
        let arch = ArchitectureInfo {
            model_type: Some("bert".to_string()),
            hidden_size: Some(768),
            num_layers: Some(12),
            num_heads: Some(12),
            num_key_value_heads: None,
            attention_type: Some("MHA".to_string()),
        };
        let graph = build_architecture_graph(&arch);

        assert!(graph.contains("Encoder Block"));
        assert!(graph.contains("Multi-Head Self-Attention"));
        assert!(graph.contains("Feed-Forward Network"));
        assert!(graph.contains("x 12"));
    }

    #[test]
    fn unknown_model_uses_default_layout() {
        let arch = ArchitectureInfo::default();
        let graph = build_architecture_graph(&arch);

        assert!(graph.contains("Transformer Block"));
        assert!(graph.contains("x N"));
        assert!(graph.contains("LM Head (Output)"));
    }
}
