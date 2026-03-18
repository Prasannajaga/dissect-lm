use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSourceKind {
    LocalPath,
    HuggingFace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSource {
    pub kind: ModelSourceKind,
    pub location: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum LayerCategory {
    Attention,
    FeedForward,
    Embedding,
    Normalization,
    OutputHead,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub data_offsets: (u64, u64),
    pub param_count: u64,
    pub category: LayerCategory,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CategoryTotals {
    pub attention: u64,
    pub feedforward: u64,
    pub embedding: u64,
    pub normalization: u64,
    pub output_head: u64,
    pub other: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParamStats {
    pub total_params: u64,
    pub categories: CategoryTotals,
}

impl ParamStats {
    pub fn pct(&self, value: u64) -> f64 {
        if self.total_params == 0 {
            return 0.0;
        }
        (value as f64 * 100.0) / self.total_params as f64
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArchitectureInfo {
    pub model_type: Option<String>,
    pub hidden_size: Option<u64>,
    pub num_layers: Option<u64>,
    pub num_heads: Option<u64>,
    pub num_key_value_heads: Option<u64>,
    pub attention_type: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttentionInfo {
    pub head_dim: Option<u64>,
    pub num_heads: Option<u64>,
    pub kv_heads: Option<u64>,
    pub attention_type: Option<String>,
    pub q_proj_params: u64,
    pub k_proj_params: u64,
    pub v_proj_params: u64,
    pub o_proj_params: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelReport {
    pub model: String,
    pub source: ModelSource,
    pub config: Option<Value>,
    pub config_key_count: usize,
    pub architecture: ArchitectureInfo,
    pub params: ParamStats,
    pub attention: AttentionInfo,
    pub tensor_files_found: usize,
    pub model_size_bytes: Option<u64>,
    pub tensor_dtypes: Vec<String>,
    pub tensor_count: usize,
    pub tensors: Option<Vec<TensorMeta>>,
    pub graph: Option<String>,
    pub deep: Option<Value>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareMetricDiff {
    pub metric: String,
    pub left: String,
    pub right: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareReport {
    pub left: ModelReport,
    pub right: ModelReport,
    pub diffs: Vec<CompareMetricDiff>,
}
