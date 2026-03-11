use anyhow::Result;

use crate::types::{CompareReport, ModelReport};

pub fn render_model_json(report: &ModelReport) -> Result<String> {
    Ok(serde_json::to_string_pretty(report)?)
}

pub fn render_compare_json(report: &CompareReport) -> Result<String> {
    Ok(serde_json::to_string_pretty(report)?)
}
