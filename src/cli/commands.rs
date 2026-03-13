use std::collections::HashSet;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use indicatif::ProgressBar;
use serde_json::Value;
use walkdir::WalkDir;

use crate::analyzer::config::{architecture_from_config, load_config_file};
use crate::analyzer::param_counter::summarize_tensors;
use crate::analyzer::safetensor::{TensorMetaRaw, parse_header_json, read_header_from_file};
use crate::cli::args::{Cli, Commands};
use crate::graph::architecture::build_architecture_graph;
use crate::hf::repo::{HfRepoClient, HfResolvedData};
use crate::python_bridge::runner::run_deep_inspection;
use crate::report::formatter::{RenderOptions, render_compare, render_model};
use crate::report::json::{render_compare_json, render_model_json};
use crate::types::{
    ArchitectureInfo, CompareMetricDiff, CompareReport, ModelReport, ModelSource, ModelSourceKind,
};

#[derive(Debug, Clone, Default)]
pub struct InspectOptions {
    pub show_params: bool,
    pub show_graph: bool,
    pub show_attention_breakdown: bool,
    pub deep: bool,
    pub json: bool,
    pub checkpoint: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct CompareOptions {
    pub deep: bool,
    pub json: bool,
}

struct ResolvedInput {
    source: ModelSource,
    config: Option<Value>,
    tensors: Vec<TensorMetaRaw>,
    warnings: Vec<String>,
}

pub async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Some(Commands::Compare {
            model1,
            model2,
            deep,
        }) => {
            let options = CompareOptions {
                deep,
                json: cli.json,
            };
            let report = compare_models(&model1, &model2, &options).await?;
            if options.json {
                println!("{}", render_compare_json(&report)?);
            } else {
                println!("{}", render_compare(&report));
            }
        }
        None => {
            let model_owned = cli.model.clone().or(cli.checkpoint.clone()).context(
                "model is required when no subcommand is used (or provide --checkpoint with --deep)",
            )?;
            let model = model_owned.as_str();

            let options = InspectOptions {
                show_params: cli.params,
                show_graph: cli.graph,
                show_attention_breakdown: cli.attention_breakdown,
                deep: cli.deep,
                json: cli.json,
                checkpoint: cli.checkpoint.clone(),
            };

            let report = inspect_model(model, &options).await?;
            if options.json {
                println!("{}", render_model_json(&report)?);
            } else {
                let render_options = RenderOptions {
                    show_params: options.show_params,
                    show_graph: options.show_graph,
                    show_attention_breakdown: options.show_attention_breakdown,
                };
                println!("{}", render_model(&report, &render_options));
            }
        }
    }
    Ok(())
}

pub async fn inspect_model(model: &str, options: &InspectOptions) -> Result<ModelReport> {
    let mut spinner = LoadingSpinner::start(format!("Inspecting {model}"));
    spinner.set_message("Resolving model source...");
    let mut resolved = resolve_input(model).await?;

    spinner.set_message("Analyzing metadata...");
    let mut architecture = match &resolved.config {
        Some(cfg) => architecture_from_config(cfg),
        None => ArchitectureInfo::default(),
    };

    let (tensor_metas, param_stats, mut attention_info) =
        summarize_tensors(&resolved.tensors, &architecture);
    if architecture.attention_type.is_none() {
        architecture.attention_type = attention_info.attention_type.clone();
    }

    let graph = if options.show_graph {
        spinner.set_message("Building architecture graph...");
        Some(build_architecture_graph(&architecture))
    } else {
        None
    };

    let deep = if options.deep {
        spinner.set_message("Running deep inspection...");
        Some(
            run_deep_inspection(
                model,
                options.checkpoint.as_deref(),
                Some(spinner.progress_bar()),
            )
            .await?,
        )
    } else {
        None
    };

    if resolved.tensors.is_empty() {
        resolved.warnings.push(
            "No safetensors metadata found. Fast mode is metadata-only; try --deep for raw checkpoint inspection."
                .to_string(),
        );
    }

    if attention_info.attention_type.is_none() {
        attention_info.attention_type = architecture.attention_type.clone();
    }

    let report = ModelReport {
        model: model.to_string(),
        source: resolved.source,
        architecture,
        params: param_stats,
        attention: attention_info,
        tensor_count: tensor_metas.len(),
        tensors: if options.show_params {
            Some(tensor_metas)
        } else {
            None
        },
        graph,
        deep,
        warnings: resolved.warnings,
    };

    spinner.finish("Inspection complete");
    Ok(report)
}

pub async fn compare_models(
    model1: &str,
    model2: &str,
    options: &CompareOptions,
) -> Result<CompareReport> {
    let left_opts = InspectOptions {
        deep: options.deep,
        json: options.json,
        ..InspectOptions::default()
    };
    let right_opts = left_opts.clone();

    let left = inspect_model(model1, &left_opts).await?;
    let right = inspect_model(model2, &right_opts).await?;

    let diffs = vec![
        diff_metric(
            "Layers",
            opt_u64(left.architecture.num_layers),
            opt_u64(right.architecture.num_layers),
        ),
        diff_metric(
            "Hidden size",
            opt_u64(left.architecture.hidden_size),
            opt_u64(right.architecture.hidden_size),
        ),
        diff_metric(
            "Heads",
            opt_u64(left.architecture.num_heads),
            opt_u64(right.architecture.num_heads),
        ),
        diff_metric(
            "KV heads",
            opt_u64(
                left.architecture
                    .num_key_value_heads
                    .or(left.attention.kv_heads),
            ),
            opt_u64(
                right
                    .architecture
                    .num_key_value_heads
                    .or(right.attention.kv_heads),
            ),
        ),
        diff_metric(
            "Attention",
            opt_string(
                left.architecture
                    .attention_type
                    .clone()
                    .or(left.attention.attention_type.clone()),
            ),
            opt_string(
                right
                    .architecture
                    .attention_type
                    .clone()
                    .or(right.attention.attention_type.clone()),
            ),
        ),
        diff_metric(
            "Params",
            left.params.total_params.to_string(),
            right.params.total_params.to_string(),
        ),
        diff_metric(
            "Attention %",
            pct_string(left.params.pct(left.params.categories.attention)),
            pct_string(right.params.pct(right.params.categories.attention)),
        ),
        diff_metric(
            "FeedForward %",
            pct_string(left.params.pct(left.params.categories.feedforward)),
            pct_string(right.params.pct(right.params.categories.feedforward)),
        ),
        diff_metric(
            "Embedding %",
            pct_string(left.params.pct(left.params.categories.embedding)),
            pct_string(right.params.pct(right.params.categories.embedding)),
        ),
        diff_metric(
            "Normalization %",
            pct_string(left.params.pct(left.params.categories.normalization)),
            pct_string(right.params.pct(right.params.categories.normalization)),
        ),
        diff_metric(
            "OutputHead %",
            pct_string(left.params.pct(left.params.categories.output_head)),
            pct_string(right.params.pct(right.params.categories.output_head)),
        ),
    ];

    Ok(CompareReport { left, right, diffs })
}

fn diff_metric(metric: &str, left: String, right: String) -> CompareMetricDiff {
    CompareMetricDiff {
        metric: metric.to_string(),
        left,
        right,
    }
}

fn opt_u64(value: Option<u64>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn opt_string(value: Option<String>) -> String {
    value.unwrap_or_else(|| "-".to_string())
}

fn pct_string(value: f64) -> String {
    format!("{value:.1}%")
}

struct LoadingSpinner {
    pb: ProgressBar,
    finished: bool,
}

impl LoadingSpinner {
    fn start(message: String) -> Self {
        let pb = ProgressBar::new_spinner();
        pb.enable_steady_tick(Duration::from_millis(100));
        pb.set_message(message);
        Self {
            pb,
            finished: false,
        }
    }

    fn set_message(&self, message: &str) {
        self.pb.set_message(message.to_string());
    }

    fn progress_bar(&self) -> ProgressBar {
        self.pb.clone()
    }

    fn finish(&mut self, message: &str) {
        self.finished = true;
        self.pb.finish_with_message(message.to_string());
    }
}

impl Drop for LoadingSpinner {
    fn drop(&mut self) {
        if !self.finished {
            self.pb.finish_and_clear();
        }
    }
}

async fn resolve_input(model: &str) -> Result<ResolvedInput> {
    let path = Path::new(model);
    if path.exists() {
        resolve_local_input(path)
    } else {
        resolve_hf_input(model).await
    }
}

fn resolve_local_input(path: &Path) -> Result<ResolvedInput> {
    let mut tensors = Vec::new();
    let mut warnings = Vec::new();
    let mut config = None;

    if path.is_file() {
        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            if ext.eq_ignore_ascii_case("safetensors") {
                tensors.extend(read_header_from_file(path)?);
                if let Some(parent) = path.parent() {
                    let cfg_path = parent.join("config.json");
                    if cfg_path.exists() {
                        config = Some(load_config_file(&cfg_path)?);
                    }
                }
            } else if ext.eq_ignore_ascii_case("json")
                && path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|n| n == "config.json")
            {
                config = Some(load_config_file(path)?);
            } else {
                warnings.push(
                    "Local checkpoint format is not metadata-readable in fast mode. Use --deep."
                        .to_string(),
                );
            }
        }
    } else if path.is_dir() {
        let cfg_path = path.join("config.json");
        if cfg_path.exists() {
            config = Some(load_config_file(&cfg_path)?);
        }

        for entry in WalkDir::new(path).into_iter().filter_map(Result::ok) {
            if !entry.file_type().is_file() {
                continue;
            }
            let file_path = entry.path();
            if file_path
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
            {
                tensors.extend(read_header_from_file(file_path)?);
            }
        }

        if tensors.is_empty() {
            let unsupported = find_unsupported_checkpoint_files(path)?;
            if !unsupported.is_empty() {
                warnings.push(format!(
                    "Found checkpoint files not inspectable in fast mode: {}. Use --deep.",
                    unsupported.join(", ")
                ));
            }
        }
    }

    Ok(ResolvedInput {
        source: ModelSource {
            kind: ModelSourceKind::LocalPath,
            location: path.display().to_string(),
        },
        config,
        tensors,
        warnings,
    })
}

fn find_unsupported_checkpoint_files(path: &Path) -> Result<Vec<String>> {
    let unsupported_ext = ["bin", "pt", "ckpt", "h5", "pb"];
    let mut files = Vec::new();

    for entry in WalkDir::new(path).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        let entry_path = entry.path();
        if entry_path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|ext| unsupported_ext.iter().any(|u| u.eq_ignore_ascii_case(ext)))
        {
            files.push(entry_path.display().to_string());
        }
    }

    Ok(files)
}

async fn resolve_hf_input(repo_id: &str) -> Result<ResolvedInput> {
    let client = HfRepoClient::new();
    let HfResolvedData {
        config,
        headers,
        warnings,
    } = client.resolve(repo_id).await?;

    let mut tensors = Vec::new();
    for header_json in headers {
        tensors.extend(parse_header_json(&header_json)?);
    }

    if tensors.is_empty() && config.is_none() {
        bail!(
            "Could not resolve metadata for HuggingFace model '{repo_id}'. Provide a local path or use --deep."
        );
    }

    let mut dedup = HashSet::new();
    tensors.retain(|t| dedup.insert(t.name.clone()));

    Ok(ResolvedInput {
        source: ModelSource {
            kind: ModelSourceKind::HuggingFace,
            location: repo_id.to_string(),
        },
        config,
        tensors,
        warnings,
    })
}
