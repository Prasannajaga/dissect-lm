use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::Value;

use crate::hf::download::{get_safetensor_header_cached, get_text_cached};

#[derive(Debug, Clone)]
pub struct HfResolvedData {
    pub config: Option<Value>,
    pub headers: Vec<String>,
    pub tensor_files_found: usize,
    pub model_size_bytes: Option<u64>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
struct HfSiblingFile {
    name: String,
    size_bytes: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct HfRepoClient {
    client: Client,
    cache_root: PathBuf,
    api_base: String,
    resolve_base: String,
}

impl HfRepoClient {
    pub fn new() -> Self {
        let client = Client::builder()
            .user_agent("dissectlm/0.1")
            .build()
            .expect("reqwest client should build");

        let cache_root = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("dissectlm");

        Self {
            client,
            cache_root,
            api_base: "https://huggingface.co/api".to_string(),
            resolve_base: "https://huggingface.co".to_string(),
        }
    }

    #[cfg(test)]
    pub fn with_endpoints(api_base: &str, resolve_base: &str, cache_root: PathBuf) -> Self {
        let client = Client::builder()
            .user_agent("dissectlm-test/0.1")
            .build()
            .expect("reqwest client should build");

        Self {
            client,
            cache_root,
            api_base: api_base.trim_end_matches('/').to_string(),
            resolve_base: resolve_base.trim_end_matches('/').to_string(),
        }
    }

    pub async fn resolve(&self, repo_id: &str) -> Result<HfResolvedData> {
        self.resolve_with_progress(repo_id, |_| {}).await
    }

    pub async fn resolve_with_progress<F>(
        &self,
        repo_id: &str,
        mut on_stage: F,
    ) -> Result<HfResolvedData>
    where
        F: FnMut(&str),
    {
        on_stage("Pulling model manifest");
        let repo_cache = self.cache_root.join(sanitize_repo_id(repo_id));
        std::fs::create_dir_all(&repo_cache)
            .with_context(|| format!("failed to create cache dir: {}", repo_cache.display()))?;

        let manifest_url = format!("{}/models/{}", self.api_base, repo_id);

        let manifest_cache = repo_cache.join("manifest.json");
        let manifest_raw = get_text_cached(&self.client, &manifest_url, &manifest_cache).await?;
        let manifest: Value =
            serde_json::from_str(&manifest_raw).context("invalid HuggingFace model manifest")?;

        on_stage("Reading model file index");
        let sibling_files = extract_sibling_files(&manifest);

        let mut warnings = Vec::new();
        let mut headers = Vec::new();

        let config = if sibling_files.iter().any(|f| f.name == "config.json") {
            on_stage("Pulling config.json");
            let cfg_cache = repo_cache.join("config.json");
            let cfg_url = resolve_url(&self.resolve_base, repo_id, "config.json");
            let cfg_raw = get_text_cached(&self.client, &cfg_url, &cfg_cache).await?;
            Some(serde_json::from_str(&cfg_raw).context("invalid config.json from HuggingFace")?)
        } else {
            None
        };

        let mut safetensor_files = BTreeSet::new();
        if sibling_files
            .iter()
            .any(|f| f.name.ends_with("model.safetensors.index.json"))
        {
            on_stage("Pulling safetensors shard index");
            let index_name = sibling_files
                .iter()
                .find(|f| f.name.ends_with("model.safetensors.index.json"))
                .expect("already checked exists");
            let index_cache = repo_cache.join(&index_name.name);
            let index_url = resolve_url(&self.resolve_base, repo_id, &index_name.name);
            let index_raw = get_text_cached(&self.client, &index_url, &index_cache).await?;
            let index_json: Value = serde_json::from_str(&index_raw)
                .with_context(|| format!("invalid safetensor index for {repo_id}"))?;

            if let Some(weight_map) = index_json.get("weight_map").and_then(Value::as_object) {
                for file in weight_map.values().filter_map(Value::as_str) {
                    safetensor_files.insert(file.to_string());
                }
            }
        } else {
            for file in sibling_files
                .iter()
                .filter(|f| f.name.ends_with(".safetensors"))
            {
                safetensor_files.insert(file.name.clone());
            }
        }

        let tensor_files_found = safetensor_files.len();
        let model_size_bytes = {
            let mut all_known = tensor_files_found > 0;
            let mut total = 0u64;
            for file in &safetensor_files {
                let size = sibling_files
                    .iter()
                    .find(|s| s.name == *file)
                    .and_then(|s| s.size_bytes);
                match size {
                    Some(v) => total = total.saturating_add(v),
                    None => all_known = false,
                }
            }
            if all_known { Some(total) } else { None }
        };

        let total = safetensor_files.len();
        for (idx, file) in safetensor_files.into_iter().enumerate() {
            on_stage(&format!(
                "Pulling safetensors header ({}/{})",
                idx + 1,
                total
            ));
            let header_cache = repo_cache
                .join("headers")
                .join(format!("{file}.header.json"));
            let file_url = resolve_url(&self.resolve_base, repo_id, &file);
            let header_json =
                get_safetensor_header_cached(&self.client, &file_url, &header_cache).await?;
            headers.push(header_json);
        }

        if headers.is_empty() {
            let unsupported = sibling_files
                .iter()
                .filter(|f| {
                    f.name.ends_with(".bin")
                        || f.name.ends_with(".pt")
                        || f.name.ends_with(".ckpt")
                        || f.name.ends_with(".h5")
                        || f.name.ends_with(".pb")
                })
                .map(|f| f.name.clone())
                .collect::<Vec<_>>();
            if !unsupported.is_empty() {
                warnings.push(format!(
                    "Only non-safetensors checkpoints found ({}). Use --deep for raw checkpoint inspection.",
                    unsupported.join(", ")
                ));
            }
        }

        Ok(HfResolvedData {
            config,
            headers,
            tensor_files_found,
            model_size_bytes,
            warnings,
        })
    }
}

fn extract_sibling_files(manifest: &Value) -> Vec<HfSiblingFile> {
    manifest
        .get("siblings")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|s| {
            let name = s.get("rfilename").and_then(Value::as_str)?.to_string();
            let size_bytes = s.get("size").and_then(Value::as_u64);
            Some(HfSiblingFile { name, size_bytes })
        })
        .collect()
}

fn sanitize_repo_id(repo_id: &str) -> String {
    repo_id.to_string()
}

fn resolve_url(base: &str, repo_id: &str, file: &str) -> String {
    format!(
        "{}/{}/resolve/main/{}",
        base.trim_end_matches('/'),
        repo_id,
        file
    )
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn sibling_extraction_handles_missing() {
        let manifest = serde_json::json!({"modelId": "x"});
        let files = extract_sibling_files(&manifest);
        assert!(files.is_empty());
    }

    #[tokio::test]
    async fn resolves_manifest_and_header_with_mock_server() {
        use httpmock::prelude::*;

        let server = MockServer::start_async().await;
        let header_json = r#"{"model.layers.0.self_attn.q_proj.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;

        server
            .mock_async(|when, then| {
                when.method(GET).path("/api/models/test-model");
                then.status(200).json_body(serde_json::json!({
                    "siblings": [
                        {"rfilename": "config.json"},
                        {"rfilename": "model.safetensors"}
                    ]
                }));
            })
            .await;

        server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/test-model/resolve/main/config.json");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(r#"{"hidden_size":2,"num_attention_heads":1,"num_hidden_layers":1}"#);
            })
            .await;

        server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/test-model/resolve/main/model.safetensors")
                    .header("range", "bytes=0-7");
                then.status(206)
                    .header("content-type", "application/octet-stream")
                    .body((header_json.len() as u64).to_le_bytes().to_vec());
            })
            .await;

        server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/test-model/resolve/main/model.safetensors")
                    .header("range", format!("bytes=8-{}", 7 + header_json.len()));
                then.status(206)
                    .header("content-type", "application/octet-stream")
                    .body(header_json);
            })
            .await;

        let tmp = tempdir().expect("tempdir");
        let client = HfRepoClient::with_endpoints(
            &format!("{}/api", server.base_url()),
            &server.base_url(),
            tmp.path().to_path_buf(),
        );

        let resolved = client
            .resolve("test-model")
            .await
            .expect("resolve should work");
        assert!(resolved.config.is_some());
        assert_eq!(resolved.headers.len(), 1);
        assert!(resolved.warnings.is_empty());
    }
}
